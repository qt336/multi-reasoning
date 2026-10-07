"""Non-training checks for data semantics, reference settings and launch flow."""

import csv
import itertools
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from benchmark import choose_best
from canonical import from_positions, path_positions
from data import LOW, HIGH, VOCAB_SIZE, N_FACTS, N_CHAIN, SEQ_LEN, encode, is_canonical, make_facts, prepare, sample_permutations, target_table, validate
from hardware import TRAIN_SIZE, batch_layout, select_batch
from model import ReasoningTransformer
from train import batch, evaluate, learning_rate, parser, record, resident_shard, training_sample_ids

ROOT = Path(__file__).resolve().parents[1]


def propagate(perm, query, layers=3):
    """Independent literal token-level pairing and synchronous causal set union."""
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


class DataTests(unittest.TestCase):
    def test_canonical_matches_all_24_path_orders(self):
        count = 0
        for order in itertools.permutations(range(4)):
            actual = bool(is_canonical(np.array([order]), np.array([0]))[0])
            self.assertEqual(actual, bool(propagate(order, 0) & (1 << 4)))
            count += actual
        self.assertEqual(count, 8)  # Probability exactly 1/3.

    def test_fifteen_fact_chain_and_order_with_context(self):
        self.assertEqual((N_FACTS, N_CHAIN, SEQ_LEN, LOW, HIGH, VOCAB_SIZE), (15, 4, 31, 1, 120, 120))
        rng = np.random.default_rng(802)
        for split in ("train", "test"):
            facts, nodes, start = make_facts(rng, 200, split)
            for desired in (None, True, False):
                order = sample_permutations(rng, len(start), start, desired)
                x, y = encode(facts, nodes, start, order)
                validate(x, y, split, desired)
                positions, recovered = path_positions(x)
                np.testing.assert_array_equal(recovered, y)
                np.testing.assert_array_equal(from_positions(positions), is_canonical(order, start))
                expected = [bool(propagate(p, q) & (1 << (int(q) + 4))) for p, q in zip(order, start)]
                np.testing.assert_array_equal(is_canonical(order, start), expected)

    def test_train_test_fact_pairs_are_disjoint(self):
        train, nt = target_table("train")
        test, ne = target_table("test")
        for i in range(HIGH - LOW + 1):
            self.assertFalse(set(train[i, :nt[i]]) & set(test[i, :ne[i]]))
        with self.assertRaises(ValueError):
            target_table("unknown")

    def test_fixed_dataset_and_paired_test_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare(root, train_size=2000, chunk_size=700)
            x, y = np.load(root / "train_x.npy"), np.load(root / "train_y.npy")
            self.assertEqual(x.shape, (2000, 31))
            validate(x, y, "train")
            np.testing.assert_array_equal(np.unique(x), np.arange(1, 121))
            ox, oy = np.load(root / "eval_x.npy"), np.load(root / "eval_y.npy")
            group_size = 10_000
            self.assertEqual(ox.shape, (2 * group_size, 31))
            self.assertEqual(oy.shape, (2 * group_size,))
            validate(ox[:group_size], oy[:group_size], "test", True)
            validate(ox[group_size:], oy[group_size:], "test", False)
            np.testing.assert_array_equal(oy[:group_size], oy[group_size:])
            np.testing.assert_array_equal(ox[:group_size, -1], ox[group_size:, -1])
            for a, b in zip(ox[:group_size], ox[group_size:]):
                self.assertEqual(set(map(tuple, a[:-1].reshape(-1, 2))),
                                 set(map(tuple, b[:-1].reshape(-1, 2))))
            meta = json.loads((root / "dataset.json").read_text())
            self.assertEqual(meta["eval_per_group"], group_size)
            self.assertEqual((meta["sequence_length"], meta["steps"], meta["vocab_size"]), (31, 4, 120))
            self.assertEqual((meta["token_min"], meta["token_max"], meta["model_token_offset"],
                              meta["format_version"]), (1, 120, 1, 4))
            mtime = (root / "train_x.npy").stat().st_mtime_ns
            prepare(root, train_size=2000, chunk_size=700)
            self.assertEqual(mtime, (root / "train_x.npy").stat().st_mtime_ns)
            with self.assertRaises(RuntimeError):
                prepare(root, train_size=2001, chunk_size=700)
            with self.assertRaises(RuntimeError):
                prepare(root, train_size=2000, eval_per_group=1000, chunk_size=700)


class ConfigurationTests(unittest.TestCase):
    def test_shape_and_unchanged_reference_hyperparameters(self):
        with torch.device("meta"):
            model = ReasoningTransformer()
        self.assertEqual(len(model.blocks), 3)
        self.assertEqual(tuple(model.token.weight.shape), (120, 1024))
        self.assertEqual(tuple(model.head.weight.shape), (120, 1024))
        self.assertEqual(tuple(model.position.weight.shape), (31, 1024))
        for block in model.blocks:
            self.assertEqual(tuple(block.ffn[0].weight.shape), (2048, 1024))
            self.assertFalse(block.norm_attention.elementwise_affine)
            self.assertFalse(block.norm_ffn.elementwise_affine)
        args = parser().parse_args(["--data-dir", "data", "--run-dir", "runs"])
        reference = json.loads((ROOT / "reference_config.json").read_text())
        same = {"width": "width", "layers": "layers", "lr": "learning_rate",
                "warmup_epochs": "warmup_epochs", "epochs": "epochs", "eval_every": "eval_every",
                "seed": "seed", "normalization": "normalization", "initialization": "initialization",
                "train_eval_size": "train_accuracy_sample_size", "train_eval_seed": "train_accuracy_sample_seed"}
        for argument, original in same.items():
            self.assertEqual(getattr(args, argument), reference[original], argument)
        self.assertEqual((args.ffn_width, args.global_batch, args.expected_gpus, args.eval_batch), (2048, 64000, 8, 250))

    def test_fixed_sample_and_epoch_based_schedule(self):
        self.assertEqual(TRAIN_SIZE, 6_500_000)
        ids = training_sample_ids(TRAIN_SIZE, 10000, 8, 2027)
        self.assertEqual(len(np.unique(ids)), 10000)
        self.assertEqual(np.bincount(ids // 812_500).tolist(), [1250] * 8)
        np.testing.assert_array_equal(ids, training_sample_ids(TRAIN_SIZE, 10000, 8, 2027))
        for size, expected_steps in ((16000, 407), (32000, 204), (64000, 102), (128000, 51)):
            _, _, steps = batch_layout(TRAIN_SIZE, size, 8)
            self.assertEqual(steps, expected_steps)
            self.assertAlmostEqual(learning_rate(0, steps * 2000, steps * 20, 1e-4), 1e-4 / (steps * 20))
            self.assertEqual(learning_rate(steps * 20 - 1, steps * 2000, steps * 20, 1e-4), 1e-4)
            self.assertEqual(learning_rate(steps * 2000, steps * 2000, steps * 20, 1e-4), 0)

    def test_partial_batches_cover_all_training_rows_on_eight_ranks(self):
        for size, tail in ((16000, 4000), (32000, 4000), (64000, 36000), (128000, 100000), (65000, 65000)):
            with self.subTest(global_batch=size):
                local_count, local_batch, steps = batch_layout(TRAIN_SIZE, size, 8)
                visits = np.zeros(TRAIN_SIZE, dtype=np.uint8)
                rank_sizes = []
                for rank in range(8):
                    counts = []
                    for left in range(0, local_count, local_batch):
                        right = min(left + local_batch, local_count)
                        visits[rank * local_count + left:rank * local_count + right] += 1
                        counts.append(right - left)
                    self.assertEqual(len(counts), steps)
                    self.assertEqual(sum(counts), 812_500)
                    self.assertEqual(counts[-1] * 8, tail)
                    rank_sizes.append(counts)
                self.assertTrue(np.all(visits == 1))
                self.assertTrue(all(counts == rank_sizes[0] for counts in rank_sizes))
        self.assertEqual(batch_layout(TRAIN_SIZE, 64000, 8), (812_500, 8000, 102))
        for args in ((0, 64000, 8), (TRAIN_SIZE, 0, 8), (TRAIN_SIZE, 64000, 0),
                     (TRAIN_SIZE + 1, 64000, 8), (TRAIN_SIZE, 64001, 8),
                     (TRAIN_SIZE, TRAIN_SIZE + 8, 8)):
            with self.assertRaises(ValueError):
                batch_layout(*args)

    def test_a100_80gb_and_batch_validation_without_gpu_access(self):
        self.assertEqual(select_batch([79.1] * 8), 64000)
        self.assertEqual(select_batch([79.1] * 8, 32000), 32000)
        for capacities, size in (([79.1] * 7, None), ([39.5] * 8, None), ([79.1] * 8, 6145)):
            with self.assertRaises(ValueError):
                select_batch(capacities, size)
        winner = choose_best([dict(status="out_of_memory", global_batch=128000),
                              dict(status="ok", global_batch=64000, examples_per_second=1000),
                              dict(status="ok", global_batch=32000, examples_per_second=1100)])
        self.assertEqual(winner["global_batch"], 32000)

    def test_launcher_only_prepares_the_requested_four_step_task(self):
        # A fake Python executable records arguments. It cannot import torch or
        # train; this catches wrong task size/width/GPU flags in the shell launcher.
        import os
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            stub = tmp / "python"
            calls = tmp / "calls.jsonl"
            stub.write_text("#!/usr/bin/env python3\nimport json, os, sys\n"
                            "with open(os.environ['TEST_CALLS'], 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n")
            stub.chmod(0o755)
            environment = {**os.environ, "PYTHON": str(stub), "TEST_CALLS": str(calls),
                           "DATA_ROOT": str(tmp / "data"), "RUN_ROOT": str(tmp / "runs"),
                           "GLOBAL_BATCH": "64000", "COMPILE_MODEL": "1"}
            subprocess.run(["bash", str(ROOT / "run.sh")], env=environment, check=True)
            commands = [json.loads(line) for line in calls.read_text().splitlines()]
            self.assertEqual(len(commands), 3)
            data = commands[1]
            self.assertEqual(data[data.index("--train-size") + 1], "6500000")
            self.assertEqual(data[data.index("--eval-per-group") + 1], "10000")
            self.assertIn("6p5m", data[data.index("--root") + 1])
            self.assertIn("vocab120", data[data.index("--root") + 1])
            launch = commands[2]
            self.assertIn("6p5m", launch[launch.index("--run-dir") + 1])
            self.assertIn("vocab120", launch[launch.index("--run-dir") + 1])
            self.assertIn("--nproc_per_node=8", launch)
            for flag, value in (("--layers", "3"), ("--width", "1024"), ("--ffn-width", "2048"),
                                ("--global-batch", "64000"), ("--lr", "1e-4")):
                self.assertEqual(launch[launch.index(flag) + 1], value)
            self.assertIn("--compile-model", launch)


class EvaluationTests(unittest.TestCase):
    def test_resident_and_mmap_batches_map_symbols_to_120_classes(self):
        x = np.tile(np.array([1, 120, 1, 120, 1], dtype=np.uint8), (4, 1))
        y = np.array([1, 120, 120, 1], dtype=np.uint8)
        tx, ty = resident_shard(x, y, 1, 3, torch.device("cpu"))
        a = batch(x, y, np.array([2, 1]), torch.device("cpu"))
        b = batch(tx, ty, torch.tensor([1, 0]), torch.device("cpu"))
        for left, right in zip(a, b):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(a[0][0].tolist(), [0, 119, 0, 119, 0])
        self.assertEqual(batch(x, y, np.array([0, 1]), torch.device("cpu"))[1].tolist(), [0, 119])
        np.testing.assert_array_equal(x[0], [1, 120, 1, 120, 1])
        self.assertEqual(tx[0].tolist(), [1, 120, 1, 120, 1])
        model = ReasoningTransformer(width=8, ffn_width=16, layers=1, length=5).eval()
        with torch.inference_mode():
            logits = model(a[0])
            self.assertEqual(tuple(logits.shape), (2, 120))
            self.assertTrue(torch.isfinite(torch.nn.functional.cross_entropy(logits, a[1])))

    def test_uneven_distributed_evaluation_counts_without_training(self):
        class PredictLastToken(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.one_hot(x[:, -1], 120).float()
        for group_size in (11, 10_000):
            x = (torch.arange(2 * group_size) % 120 + 1).to(torch.uint8).reshape(-1, 1).repeat(1, 31)
            y = x[:, -1].clone()
            # Distinct known accuracies ensure the two groups stay separate.
            y[:group_size:2] = y[:group_size:2] % 120 + 1
            y[group_size::4] = y[group_size::4] % 120 + 1
            for offset, wrong in ((0, (group_size + 1) // 2),
                                  (group_size, (group_size + 3) // 4)):
                with self.subTest(group_size=group_size, offset=offset):
                    totals = np.zeros(2, dtype=np.int64)
                    with patch("train.dist.all_reduce"):
                        for rank in range(8):
                            ids = torch.arange(rank, group_size, 8) + offset
                            totals += evaluate(PredictLastToken(), x, y, ids, 250, torch.device("cpu"))
                    self.assertEqual(totals.tolist(), [group_size - wrong, group_size])

    def test_curve_records_train_and_paired_test_without_fake_missing_values(self):
        import time
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scores = dict(train=(6, 10), test=(11, 20), test_canonical=(9, 10), test_noncanonical=(2, 10))
            record(root, 0, scores, None, 0, time.time(), 4)
            record(root, 1, dict(train=(7, 10)), 1.2, 1e-4, time.time(), 4)
            with (root / "accuracy.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["test_accuracy"], "0.55")
            self.assertEqual(rows[1]["test_accuracy"], "")
            self.assertEqual(rows[1]["train_accuracy"], "0.7")
            self.assertGreater((root / "accuracy.png").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
