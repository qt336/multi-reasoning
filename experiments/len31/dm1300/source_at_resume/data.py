"""Reconstruct the paper's four-step symbolic task with held-out fact pairs.

The original experiment generator is not included with the PDF. Every example
has a single 15-fact reasoning chain and a final query token (31 tokens total).
The query selects a four-step subchain. Train/test fact pairs obey the disjoint
modulo-5 rules in the paper. Evaluation uses paired fact sets with two
different orderings.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


LOW, HIGH = 20, 100
N_FACTS = 15
N_CHAIN = 4
SEQ_LEN = 2 * N_FACTS + 1


def target_table(split: str) -> tuple[np.ndarray, np.ndarray]:
    if split not in ("train", "test"):
        raise ValueError(f"Unknown split: {split}")
    allowed = {0, 1, 4} if split == "train" else {2, 3}
    choices = [
        [y for y in range(LOW, HIGH + 1) if y != x and (y - x) % 5 in allowed]
        for x in range(LOW, HIGH + 1)
    ]
    counts = np.asarray([len(row) for row in choices], dtype=np.int64)
    table = np.full((HIGH - LOW + 1, max(counts)), LOW, dtype=np.uint8)
    for i, row in enumerate(choices):
        table[i, : len(row)] = row
    return table, counts


def draw_target(rng: np.random.Generator, src: np.ndarray,
                table: np.ndarray, counts: np.ndarray) -> np.ndarray:
    row = src.astype(np.int64) - LOW
    col = (rng.random(src.shape) * counts[row]).astype(np.int64)
    return table[row, col]


def make_facts(rng: np.random.Generator, n: int, split: str
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    facts = np.empty((n, N_FACTS, 2), dtype=np.uint8)
    facts[:, :, 0] = nodes[:, :-1]
    facts[:, :, 1] = nodes[:, 1:]
    start = rng.integers(0, N_FACTS - N_CHAIN + 1, n, dtype=np.uint8)
    return facts, nodes, start


def sample_permutations(rng: np.random.Generator, n: int, start: np.ndarray,
                        desired: bool | None = None) -> np.ndarray:
    perm = np.argsort(rng.random((n, N_FACTS)), axis=1).astype(np.uint8)
    if desired is None:
        return perm
    canonical = is_canonical(perm, start)
    bad = canonical != desired
    while bad.any():
        perm[bad] = np.argsort(rng.random((int(bad.sum()), N_FACTS)), axis=1)
        canonical = is_canonical(perm, start)
        bad = canonical != desired
    return perm


def is_canonical(perm: np.ndarray, start: np.ndarray) -> np.ndarray:
    positions = np.argsort(perm, axis=1)
    row = np.arange(len(perm))
    return ((positions[row, start + 2] > positions[row, start + 1]) &
            (positions[row, start + 2] > positions[row, start + 3]))


def encode(facts: np.ndarray, nodes: np.ndarray, start: np.ndarray,
           perm: np.ndarray
           ) -> tuple[np.ndarray, np.ndarray]:
    row = np.arange(len(facts))[:, None]
    x = np.empty((len(facts), SEQ_LEN), dtype=np.uint8)
    x[:, :2 * N_FACTS] = facts[row, perm].reshape(-1, 2 * N_FACTS)
    sample = np.arange(len(facts))
    x[:, -1] = nodes[sample, start]
    return x, nodes[sample, start + N_CHAIN].copy()


def validate(x: np.ndarray, y: np.ndarray, split: str,
             canonical: bool | None = None) -> None:
    allowed = {0, 1, 4} if split == "train" else {2, 3}
    pairs = x[:, :-1].reshape(-1, N_FACTS, 2)
    assert x.shape[1] == SEQ_LEN and np.all((x >= LOW) & (x <= HIGH))
    assert np.all(np.isin((pairs[:, :, 1].astype(int) - pairs[:, :, 0]) % 5,
                              list(allowed)))
    assert np.all(pairs[:, :, 0] != pairs[:, :, 1])
    for i in range(len(x)):
        # Independently recover the complete 15-edge chain from the unordered facts.
        starts = set(map(int, pairs[i, :, 0])) - set(map(int, pairs[i, :, 1]))
        assert len(starts) == 1
        node = starts.pop()
        seen = {node}
        for _ in range(N_FACTS):
            hits = np.flatnonzero(pairs[i, :, 0] == node)
            assert len(hits) == 1
            node = int(pairs[i, hits[0], 1])
            assert node not in seen
            seen.add(node)
        assert len(seen) == N_FACTS + 1
        value = int(x[i, -1])
        positions = []
        for _ in range(N_CHAIN):
            hits = np.flatnonzero(pairs[i, :, 0] == value)
            assert len(hits) == 1, (i, value, hits)
            positions.append(int(hits[0]))
            value = int(pairs[i, hits[0], 1])
        assert value == int(y[i])
        if canonical is not None:
            assert (positions[2] > positions[1] and positions[2] > positions[3]) == canonical


def save_array(path: Path, value: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, value)
    os.replace(tmp, path)


def save_json(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def prepare(root: Path, train_size: int = 6_500_000, eval_per_group: int = 10_000,
            seed: int = 2027, chunk_size: int = 50_000) -> None:
    if min(train_size, eval_per_group, chunk_size) < 1:
        raise ValueError("Data sizes must be positive")
    root.mkdir(parents=True, exist_ok=True)
    meta_path = root / "dataset.json"
    config = {"format_version": 3, "steps": N_CHAIN, "n_facts": N_FACTS,
              "token_min": LOW, "token_max": HIGH, "vocab_size": 101,
              "train_size": train_size, "eval_per_group": eval_per_group,
              "seed": seed, "sequence_length": SEQ_LEN,
              "chunk_size": chunk_size,
              "structure": "one_continuous_15_fact_chain_query_four_steps",
              "train_mod5": [0, 1, 4], "test_mod5": [2, 3]}
    paths = [root / f"{name}.npy" for name in
             ("train_x", "train_y", "eval_x", "eval_y")]
    if meta_path.exists():
        current = json.loads(meta_path.read_text())
        if current == config and all(path.exists() for path in paths):
            print(f"Using existing dataset: {root}", flush=True)
            return
        raise RuntimeError(f"Dataset configuration differs from {meta_path}")
    if any(path.exists() for path in paths):
        raise RuntimeError(f"Incomplete dataset in {root}; use a fresh directory")

    rng = np.random.default_rng(seed)
    tmp_x = root / "train_x.npy.tmp"
    tmp_y = root / "train_y.npy.tmp"
    train_x = np.lib.format.open_memmap(tmp_x, mode="w+", dtype=np.uint8,
                                        shape=(train_size, SEQ_LEN))
    train_y = np.lib.format.open_memmap(tmp_y, mode="w+", dtype=np.uint8,
                                        shape=(train_size,))
    for start in range(0, train_size, chunk_size):
        end = min(start + chunk_size, train_size)
        facts, nodes, query_start = make_facts(rng, end - start, "train")
        perm = sample_permutations(rng, end - start, query_start)
        x, y = encode(facts, nodes, query_start, perm)
        if start == 0:
            validate(x[:min(len(x), 256)], y[:min(len(x), 256)], "train")
        train_x[start:end] = x
        train_y[start:end] = y
        if end == train_size or end % (10 * chunk_size) == 0:
            print(f"Data preparation: {end:,}/{train_size:,} training examples", flush=True)
    train_x.flush()
    train_y.flush()
    del train_x, train_y
    os.replace(tmp_x, root / "train_x.npy")
    os.replace(tmp_y, root / "train_y.npy")

    facts, nodes, query_start = make_facts(rng, eval_per_group, "test")
    canonical_perm = sample_permutations(rng, eval_per_group, query_start, True)
    other_perm = sample_permutations(rng, eval_per_group, query_start, False)
    canonical_x, canonical_y = encode(facts, nodes, query_start, canonical_perm)
    other_x, other_y = encode(facts, nodes, query_start, other_perm)
    assert np.array_equal(canonical_y, other_y)
    validate(canonical_x, canonical_y, "test", True)
    validate(other_x, other_y, "test", False)
    assert np.array_equal(np.sort(canonical_x[:, :-1].reshape(-1, N_FACTS, 2), axis=1),
                          np.sort(other_x[:, :-1].reshape(-1, N_FACTS, 2), axis=1))
    for name, array in (("eval_x", np.concatenate((canonical_x, other_x))),
                        ("eval_y", np.concatenate((canonical_y, other_y)))):
        tmp = root / f"{name}.npy.tmp"
        with tmp.open("wb") as handle:
            np.save(handle, array)
        os.replace(tmp, root / f"{name}.npy")
    tmp_meta = meta_path.with_suffix(".json.tmp")
    tmp_meta.write_text(json.dumps(config, indent=2) + "\n")
    os.replace(tmp_meta, meta_path)
    print(f"Dataset ready: {root}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=6_500_000)
    parser.add_argument("--eval-per-group", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()
    prepare(args.root, args.train_size, args.eval_per_group,
            args.seed, args.chunk_size)
