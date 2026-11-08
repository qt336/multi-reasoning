"""Plot four-step fixed training sample and paired held-out test accuracy."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def draw(ax, rows, field, label, color):
    points = [(int(r["epoch"]), 100 * float(r[field]))
              for r in rows if r.get(field) not in (None, "")]
    if points:
        ax.plot(*zip(*points), label=label, color=color, linewidth=1.7,
                marker="o", markersize=2)


def render(csv_path: Path, png_path: Path, steps: int | None = None):
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    if steps is None:
        config_path = csv_path.parent / "config.json"
        if config_path.exists():
            steps = json.loads(config_path.read_text())["data"]["steps"]
    has_order = any(r.get("test_canonical_accuracy") not in (None, "") for r in rows)
    fig, axes = plt.subplots(1, 2 if has_order else 1,
                            figsize=(13, 5) if has_order else (8, 5), squeeze=False)
    ax = axes[0, 0]
    draw(ax, rows, "train_accuracy", "Train (fixed sample)", "#1565C0")
    draw(ax, rows, "test_accuracy", "Test (balanced order groups)", "#D84315")
    if has_order:
        order_ax = axes[0, 1]
        draw(order_ax, rows, "test_canonical_accuracy", "Test canonical (paired)", "#6A1B9A")
        draw(order_ax, rows, "test_noncanonical_accuracy", "Test noncanonical (paired)", "#C62828")
        order_ax.set_title("4-step order: F3 after F2 and F4")
    ax.set_title(f"{steps}-step reasoning" if steps else "Reasoning accuracy")
    for ax in axes.flat:
        ax.set(xlabel="Epoch", ylabel="Accuracy (%)", ylim=(0, 100))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = png_path.with_name(png_path.name + ".tmp")
    fig.savefig(tmp, dpi=160, format="png")
    plt.close(fig)
    os.replace(tmp, png_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("png", type=Path)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    render(args.csv, args.png, args.steps)
