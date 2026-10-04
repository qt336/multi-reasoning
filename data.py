"""Generate fixed, disk-backed single-chain tasks in bounded-memory chunks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from canonical import EDGES_13, conditional_permutations, from_positions, is_canonical, path_positions


LOW, HIGH = 1, 200
N_FACTS = 26
SEQ_LEN = 53


def target_table(split: str) -> tuple[np.ndarray, np.ndarray]:
    if split not in ("train", "test"):
        raise ValueError(split)
    allowed = {0, 1, 4} if split == "train" else {2, 3}
    choices = [[y for y in range(LOW, HIGH + 1)
                if y != x and (y - x) % 5 in allowed]
               for x in range(LOW, HIGH + 1)]
    counts = np.asarray([len(row) for row in choices], dtype=np.int64)
    table = np.full((HIGH - LOW + 1, max(counts)), LOW, dtype=np.uint8)
    for i, row in enumerate(choices):
        table[i, :len(row)] = row
    return table, counts


def draw_target(rng, src, table, counts):
    row = src.astype(np.int64) - LOW
    col = (rng.random(src.shape) * counts[row]).astype(np.int64)
    return table[row, col]


def make_facts(rng: np.random.Generator, n: int, split: str, steps: int):
    if not 1 <= steps <= N_FACTS:
        raise ValueError("Reasoning steps must be between 1 and 26")
    table, counts = target_table(split)
    nodes = np.empty((n, N_FACTS + 1), dtype=np.uint8)
    nodes[:, 0] = rng.integers(LOW, HIGH + 1, n, dtype=np.uint8)
    for step in range(1, N_FACTS + 1):
        candidate = draw_target(rng, nodes[:, step - 1], table, counts)
        bad = (candidate[:, None] == nodes[:, :step]).any(axis=1)
        while bad.any():
            candidate[bad] = draw_target(rng, nodes[bad, step - 1], table, counts)
            bad = (candidate[:, None] == nodes[:, :step]).any(axis=1)
        nodes[:, step] = candidate
    facts = np.stack((nodes[:, :-1], nodes[:, 1:]), axis=-1)
    start = rng.integers(0, N_FACTS - steps + 1, n, dtype=np.uint8)
    return facts, nodes, start


def encode(facts, nodes, start, perm, steps):
    x = np.empty((len(facts), SEQ_LEN), dtype=np.uint8)
    row = np.arange(len(facts))
    x[:, :-1] = facts[row[:, None], perm].reshape(-1, 2 * N_FACTS)
    x[:, -1] = nodes[row, start]
    return x, nodes[row, start.astype(np.int64) + steps].copy()


def validate(x, y, split, steps, canonical=None):
    """Independently recover all 26 edges and the answer from encoded inputs."""
    assert x.shape == (len(y), SEQ_LEN)
    assert np.all((x >= LOW) & (x <= HIGH))
    pairs = x[:, :-1].reshape(-1, N_FACTS, 2)
    allowed = [0, 1, 4] if split == "train" else [2, 3]
    assert np.isin((pairs[:, :, 1].astype(int) - pairs[:, :, 0]) % 5, allowed).all()
    for facts in pairs:
        edge = {int(a): int(b) for a, b in facts}
        starts = set(edge) - set(edge.values())
        assert len(edge) == N_FACTS and len(starts) == 1
        node = starts.pop()
        seen = {node}
        for _ in range(N_FACTS):
            node = edge[node]
            assert node not in seen
            seen.add(node)
        assert len(seen) == N_FACTS + 1
    positions, recovered = path_positions(x, steps)
    assert np.array_equal(recovered, y)
    if canonical is not None:
        assert (from_positions(positions, steps) == canonical).all()


def save_array(path: Path, value: np.ndarray):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, value)
    os.replace(tmp, path)


def save_json(path: Path, value: dict):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def prepare(root: Path, steps: int = 13, train_size: int = 200_000_000,
            eval_per_group: int = 1000, seed: int = 2027,
            chunk_size: int = 50_000, canonical_train_size: int = 10_000):
    if min(train_size, eval_per_group, chunk_size, canonical_train_size) < 1:
        raise ValueError("Dataset and chunk sizes must be positive")
    if steps not in range(7, 14):
        raise ValueError("This experiment has separate 7 through 13 step tasks")
    root.mkdir(parents=True, exist_ok=True)
    config = dict(format_version=2, steps=steps, train_size=train_size,
                  eval_per_group=eval_per_group, test_size=2 * eval_per_group,
                  seed=seed, chunk_size=chunk_size, sequence_length=SEQ_LEN,
                  n_facts=N_FACTS, token_min=LOW, token_max=HIGH, vocab_size=200,
                  structure="one_continuous_26_fact_chain",
                  train_mod5=[0, 1, 4], test_mod5=[2, 3],
                  canonical_train_sample_size=canonical_train_size if steps == 13 else 0,
                  canonical_edges=EDGES_13 if steps == 13 else None,
                  test_sampling="uniform fact permutations; extra paired order diagnostic at 13 steps")
    # Normalize tuples to the JSON representation before comparing a reused dataset.
    config = json.loads(json.dumps(config))
    names = ["train_x", "train_y", "test_x", "test_y"]
    if steps == 13:
        names += ["order_x", "order_y", "canonical_train_ids"]
    meta_path = root / "dataset.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if all(meta.get(k) == v for k, v in config.items()) and all(
                (root / f"{name}.npy").exists() for name in names):
            print(f"Using existing dataset: {root}", flush=True)
            return
        raise RuntimeError(f"Dataset differs or is incomplete: {root}")
    if any((root / f"{name}.npy").exists() for name in names):
        raise RuntimeError(f"Unfinished dataset in {root}; use a fresh directory")

    # Independent RNGs: diagnostics and reservoir selection never affect training.
    rng = np.random.default_rng(seed)
    sample_rng = np.random.default_rng(np.random.SeedSequence([seed, 13, 1]))
    ids = np.empty(0, dtype=np.int64)
    priorities = np.empty(0)
    canonical_count = 0
    tx = np.lib.format.open_memmap(root / "train_x.npy.tmp", mode="w+",
                                  dtype=np.uint8, shape=(train_size, SEQ_LEN))
    ty = np.lib.format.open_memmap(root / "train_y.npy.tmp", mode="w+",
                                  dtype=np.uint8, shape=(train_size,))
    for left in range(0, train_size, chunk_size):
        right = min(left + chunk_size, train_size)
        facts, nodes, start = make_facts(rng, right - left, "train", steps)
        perm = np.argsort(rng.random((right - left, N_FACTS)), axis=1).astype(np.uint8)
        x, y = encode(facts, nodes, start, perm, steps)
        if left == 0:
            validate(x[:256], y[:256], "train", steps)
        tx[left:right], ty[left:right] = x, y
        if steps == 13:
            new_ids = np.flatnonzero(is_canonical(perm, start)) + left
            canonical_count += len(new_ids)
            ids = np.concatenate((ids, new_ids))
            priorities = np.concatenate((priorities, sample_rng.random(len(new_ids))))
            if len(ids) > canonical_train_size:
                keep = np.argpartition(priorities, canonical_train_size - 1)[:canonical_train_size]
                ids, priorities = ids[keep], priorities[keep]
        if right == train_size or right % (10 * chunk_size) == 0:
            print(f"DATA steps={steps} rows={right:,}/{train_size:,}", flush=True)
    tx.flush()
    ty.flush()
    del tx, ty
    if steps == 13 and len(ids) < canonical_train_size:
        raise RuntimeError(f"Only {len(ids)} canonical training rows; need {canonical_train_size}. "
                           "Use a larger dataset or a smaller --canonical-train-size for smoke tests.")
    for name in ("train_x", "train_y"):
        os.replace(root / f"{name}.npy.tmp", root / f"{name}.npy")
    if steps == 13:
        ids.sort()
        save_array(root / "canonical_train_ids.npy", ids)
        tx = np.load(root / "train_x.npy", mmap_mode="r")
        ty = np.load(root / "train_y.npy", mmap_mode="r")
        validate(tx[ids], ty[ids], "train", 13, True)
        config["canonical_train_count"] = canonical_count
        config["canonical_train_sampling"] = "fixed uniform sample without replacement of actual training IDs, via independent random priorities"

    test_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    facts, nodes, start = make_facts(test_rng, 2 * eval_per_group, "test", steps)
    perm = np.argsort(test_rng.random((len(start), N_FACTS)), axis=1).astype(np.uint8)
    x, y = encode(facts, nodes, start, perm, steps)
    validate(x, y, "test", steps)
    save_array(root / "test_x.npy", x)
    save_array(root / "test_y.npy", y)
    if steps == 13:
        facts, nodes, start = make_facts(test_rng, eval_per_group, "test", steps)
        groups = []
        for desired in (True, False):
            perm = conditional_permutations(test_rng, start, N_FACTS, desired)
            x, y = encode(facts, nodes, start, perm, steps)
            validate(x, y, "test", steps, desired)
            groups.append((x, y))
        assert np.array_equal(groups[0][1], groups[1][1])
        save_array(root / "order_x.npy", np.concatenate([g[0] for g in groups]))
        save_array(root / "order_y.npy", np.concatenate([g[1] for g in groups]))
    # The manifest is the completion marker, published after all arrays validate.
    save_json(meta_path, config)
    print(f"Dataset ready: {root}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=range(7, 14), required=True)
    parser.add_argument("--train-size", type=int, default=200_000_000)
    parser.add_argument("--eval-per-group", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--canonical-train-size", type=int, default=10_000)
    prepare(**vars(parser.parse_args()))
