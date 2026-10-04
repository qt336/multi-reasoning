"""Fact-order conditions, relative to the queried subchain (F1 is one-based)."""

from __future__ import annotations

import numpy as np


# (earlier fact, later fact). No constraint is imposed on F1, on facts outside
# the queried path, or between otherwise unrelated branches.
EDGES_4 = ((2, 3), (4, 3))
EDGES_13 = (
    (2, 3), (4, 3),
    (5, 6), (7, 6),
    (8, 9), (10, 9),
    (11, 12), (13, 12),
    (6, 9), (12, 9),
)
CANONICAL_13_PROBABILITY = 1 / 243


def from_positions(positions: np.ndarray, steps: int = 13) -> np.ndarray:
    """positions[:, i-1] is the input fact-slot of Fi along the queried path."""
    if steps not in (4, 13):
        raise ValueError("The canonical diagnostic is defined only for 4 or 13 steps")
    if positions.ndim != 2 or positions.shape[1] < steps:
        raise ValueError("Expected one input position for every required fact")
    edges = EDGES_4 if steps == 4 else EDGES_13
    result = np.ones(len(positions), dtype=bool)
    for earlier, later in edges:
        result &= positions[:, earlier - 1] < positions[:, later - 1]
    return result


def is_canonical(perm: np.ndarray, start: np.ndarray,
                 steps: int = 13) -> np.ndarray:
    """perm lists zero-based chain fact IDs in input order; start is query ID."""
    positions = np.argsort(perm, axis=1)
    path_positions = np.take_along_axis(
        positions, start.astype(np.int64)[:, None] + np.arange(steps), axis=1)
    return from_positions(path_positions, steps)


def path_positions(x: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Recover the queried path from encoded tokens, independent of generation."""
    facts = x[:, :-1].reshape(len(x), -1, 2)
    node = x[:, -1].copy()
    positions = np.empty((len(x), steps), dtype=np.int64)
    row = np.arange(len(x))
    for step in range(steps):
        hits = facts[:, :, 0] == node[:, None]
        if not np.all(hits.sum(axis=1) == 1):
            raise ValueError("Query does not have a unique complete reasoning path")
        slot = hits.argmax(axis=1)
        positions[:, step] = slot
        node = facts[row, slot, 1]
    return positions, node


def conditional_permutations(rng: np.random.Generator, start: np.ndarray,
                             n_facts: int, desired: bool) -> np.ndarray:
    """Uniform conditional permutations by rejection, for small diagnostics only."""
    perm = np.argsort(rng.random((len(start), n_facts)), axis=1).astype(np.uint8)
    bad = is_canonical(perm, start) != desired
    while bad.any():
        perm[bad] = np.argsort(rng.random((int(bad.sum()), n_facts)), axis=1)
        bad = is_canonical(perm, start) != desired
    return perm
