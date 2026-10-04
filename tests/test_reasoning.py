"""Semantic tests: independently check paths, the order theorem, and defaults."""

import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from canonical import EDGES_13, conditional_permutations, from_positions, is_canonical, path_positions
from data import LOW, HIGH, N_FACTS, encode, make_facts, prepare, target_table, validate
from hardware import select_batch
from model import ReasoningTransformer
from train import batch, learning_rate, parser, resident_shard, training_sample_ids


def propagate(perm, query, layers):
    """Literal token-level simulator: pair in block 0, then parallel set union.

    This does not use canonical predicates, the ternary decomposition, or a
    fact-level shortcut. All source and destination token positions are present.
    """
    tokens = [v for fact in perm for v in (int(fact), int(fact) + 1)] + [int(query)]
    state = [1 << value for value in tokens]
    for layer in range(layers):
        updated = state.copy()
        for receiver in range(len(tokens)):
            if layer == 0:
                if receiver < len(tokens) - 1 and receiver % 2:
                    updated[receiver] |= state[receiver - 1]
            else:
                for sender in range(receiver):
                    if state[sender] & state[receiver]:
                        updated[receiver] |= state[sender]
        state = updated
    return state[-1]


class CanonicalTests(unittest.TestCase):
    def test_four_step_matches_reference_for_all_24_orders(self):
        for order in itertools.permutations(range(4)):
            expected = bool(propagate(order, 0, 3) & (1 << 4))
            actual = bool(is_canonical(np.array([order]), np.array([0]), 4)[0])
            self.assertEqual(actual, expected, order)

    def test_thirteen_step_matches_independent_propagation(self):
        rng = np.random.default_rng(811)
        starts = rng.integers(0, 14, 180)
        positive = conditional_permutations(rng, starts, N_FACTS, True)
        negative = conditional_permutations(rng, starts, N_FACTS, False)
        for desired, permutations in ((True, positive), (False, negative)):
            for order, start in zip(permutations, starts):
                self.assertEqual(bool(propagate(order, start, 4) & (1 << (int(start) + 13))), desired)
        random_orders = np.argsort(rng.random((1500, N_FACTS)), axis=1)
        starts = rng.integers(0, 14, len(random_orders))
        actual = is_canonical(random_orders, starts)
        expected = [bool(propagate(order, start, 4) & (1 << (int(start) + 13)))
                    for order, start in zip(random_orders, starts)]
        np.testing.assert_array_equal(actual, expected)

    def test_exact_order_frequency_is_one_in_243(self):
        # Count linear extensions independently using a subset dynamic program.
        prerequisites = [0] * 13
        for earlier, later in EDGES_13:
            prerequisites[later - 1] |= 1 << (earlier - 1)
        counts = [0] * (1 << 13)
        counts[0] = 1
        for mask in range(1 << 13):
            for fact in range(13):
                bit = 1 << fact
                if not mask & bit and mask & prerequisites[fact] == prerequisites[fact]:
                    counts[mask | bit] += counts[mask]
        self.assertEqual(counts[-1] * 243, math.factorial(13))


class DataTests(unittest.TestCase):
    def test_every_task_has_valid_chain_and_disjoint_fact_split(self):
        rng = np.random.default_rng(5)
        for steps in range(7, 14):
            for split in ("train", "test"):
                facts, nodes, start = make_facts(rng, 24, split, steps)
                order = np.argsort(rng.random((24, N_FACTS)), axis=1)
                x, y = encode(facts, nodes, start, order, steps)
                validate(x, y, split, steps)
                self.assertEqual(x.shape, (24, 53))
        tr, nt = target_table("train")
        te, ne = target_table("test")
        for i in range(HIGH - LOW + 1):
            self.assertFalse(set(tr[i, :nt[i]]) & set(te[i, :ne[i]]))
        self.assertEqual((LOW, HIGH), (1, 200))

    def test_canonical_sample_is_from_training_and_paired_test_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare(root, steps=13, train_size=8000, eval_per_group=7,
                    chunk_size=1000, canonical_train_size=7)
            ids = np.load(root / "canonical_train_ids.npy")
            self.assertEqual(len(ids), 7)
            self.assertTrue((np.diff(ids) > 0).all())
            x, y = np.load(root / "train_x.npy"), np.load(root / "train_y.npy")
            validate(x[ids], y[ids], "train", 13, True)
            meta = json.loads((root / "dataset.json").read_text())
            positions, answers = path_positions(x, 13)
            np.testing.assert_array_equal(answers, y)
            self.assertEqual(meta["canonical_train_count"], int(from_positions(positions).sum()))
            ox = np.load(root / "order_x.npy")
            oy = np.load(root / "order_y.npy")
            np.testing.assert_array_equal(oy[:7], oy[7:])
            np.testing.assert_array_equal(ox[:7, -1], ox[7:, -1])
            for a, b in zip(ox[:7], ox[7:]):
                self.assertEqual(set(map(tuple, a[:-1].reshape(-1, 2))),
                                 set(map(tuple, b[:-1].reshape(-1, 2))))
            # Reuse must neither change the data nor silently change its config.
            mtime = (root / "train_x.npy").stat().st_mtime_ns
            prepare(root, steps=13, train_size=8000, eval_per_group=7,
                    chunk_size=1000, canonical_train_size=7)
            self.assertEqual(mtime, (root / "train_x.npy").stat().st_mtime_ns)
            with self.assertRaises(RuntimeError):
                prepare(root, steps=12, train_size=8000, eval_per_group=7,
                        chunk_size=1000, canonical_train_size=7)


class TrainingTests(unittest.TestCase):
    def test_full_model_shape_and_reference_hyperparameters(self):
        with torch.device("meta"):
            model = ReasoningTransformer()
        self.assertEqual(len(model.blocks), 4)
        self.assertEqual(tuple(model.token.weight.shape), (200, 2048))
        self.assertEqual(tuple(model.head.weight.shape), (200, 2048))
        self.assertEqual(tuple(model.position.weight.shape), (53, 2048))
        for block in model.blocks:
            self.assertEqual(tuple(block.ffn[0].weight.shape), (4096, 2048))
            self.assertFalse(block.norm_attention.elementwise_affine)
            self.assertFalse(block.norm_ffn.elementwise_affine)
        args = parser().parse_args(["--data-dir", "data", "--run-dir", "runs"])
        self.assertEqual((args.global_batch, args.lr, args.warmup_epochs, args.epochs,
                          args.eval_every, args.eval_batch, args.seed, args.train_eval_size,
                          args.train_eval_seed, args.expected_gpus),
                         (16000, 1e-4, 20, 2000, 5, 250, 2029, 10000, 2027, 8))

    def test_training_sample_and_schedule(self):
        ids = training_sample_ids(200_000_000, 10000, 8, 2027)
        self.assertEqual(len(np.unique(ids)), 10000)
        self.assertEqual(np.bincount(ids // 25_000_000).tolist(), [1250] * 8)
        np.testing.assert_array_equal(ids, training_sample_ids(200_000_000, 10000, 8, 2027))
        for global_batch in (16000, 32000):
            steps = 200_000_000 // global_batch
            self.assertAlmostEqual(learning_rate(0, steps * 2000, steps * 20, 1e-4), 1e-4 / (steps * 20))
            self.assertEqual(learning_rate(steps * 20 - 1, steps * 2000, steps * 20, 1e-4), 1e-4)
            self.assertEqual(learning_rate(steps * 2000, steps * 2000, steps * 20, 1e-4), 0)

    def test_eight_a100_batch_selection_without_accessing_gpus(self):
        self.assertEqual(select_batch([39.5] * 8), 16000)
        self.assertEqual(select_batch([79.1] * 8), 32000)
        self.assertEqual(select_batch([79.1] * 7 + [39.5]), 16000)
        self.assertEqual(select_batch([79.1] * 8, 8000), 8000)
        for capacities, override in (([39.5] * 7, None), ([16] * 8, None), ([79.1] * 8, 8192)):
            with self.assertRaises(ValueError):
                select_batch(capacities, override)

    def test_token_boundaries_and_resident_shard_without_training(self):
        x = np.tile(np.array([1, 200, 1, 200, 1], dtype=np.uint8), (4, 1))
        y = np.array([1, 200, 200, 1], dtype=np.uint8)
        tx, ty = resident_shard(x, y, 1, 3, torch.device("cpu"))
        a = batch(x, y, np.array([2, 1]), torch.device("cpu"))
        b = batch(tx, ty, torch.tensor([1, 0]), torch.device("cpu"))
        for left, right in zip(a, b):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(batch(x, y, np.array([0, 1]), torch.device("cpu"))[1].tolist(), [0, 199])
        # Inference only: IDs 1 and 200 must both be accepted, and logits must
        # have exactly 200 classes (no spare class for token 0).
        model = ReasoningTransformer(width=8, ffn_width=16, layers=1, length=5)
        model.eval()
        with torch.inference_mode():
            self.assertEqual(tuple(model(a[0]).shape), (2, 200))


if __name__ == "__main__":
    unittest.main()
