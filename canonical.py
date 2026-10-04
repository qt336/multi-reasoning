"""Four-step order diagnostic; Fi is the i-th fact along the queried path."""

import numpy as np

from data import is_canonical


def path_positions(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Independently recover F1..F4 input slots and the four-step answer."""
    pairs = x[:, :-1].reshape(len(x), -1, 2)
    current = x[:, -1].copy()
    positions = np.empty((len(x), 4), dtype=np.int64)
    row = np.arange(len(x))
    for step in range(4):
        hits = pairs[:, :, 0] == current[:, None]
        if not np.all(hits.sum(1) == 1):
            raise ValueError("Every queried step must have exactly one successor")
        positions[:, step] = hits.argmax(1)
        current = pairs[row, positions[:, step], 1]
    return positions, current


def from_positions(p: np.ndarray) -> np.ndarray:
    return (p[:, 2] > p[:, 1]) & (p[:, 2] > p[:, 3])
