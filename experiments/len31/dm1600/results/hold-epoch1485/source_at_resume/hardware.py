"""Read-only 4xRTX 4080 validation and four-step batch configuration."""

from __future__ import annotations

import argparse
import json

TRAIN_SIZE = 6_500_000
DEFAULT_BATCH = 4_000
EXPECTED_GPUS = 4
MIN_GPU_MEMORY_GIB = 15


def batch_layout(train_size: int, global_batch: int, world: int) -> tuple[int, int, int]:
    """Equal rank shards, retaining the final partial batch on every rank."""
    if min(train_size, global_batch, world) <= 0 or global_batch > train_size:
        raise ValueError("Sizes must be positive and global batch must not exceed training size")
    if train_size % world or global_batch % world:
        raise ValueError("Training size and global batch must both be divisible by workers")
    return train_size // world, global_batch // world, (train_size + global_batch - 1) // global_batch


def select_batch(memory_gib: list[float], override: int | None = None) -> int:
    if len(memory_gib) != EXPECTED_GPUS or min(memory_gib) < MIN_GPU_MEMORY_GIB:
        raise ValueError(f"Exactly {EXPECTED_GPUS} RTX 4080-class 16GB GPUs are required")
    size = DEFAULT_BATCH if override is None else override
    batch_layout(TRAIN_SIZE, size, EXPECTED_GPUS)
    return size


def inspect_hardware(global_batch: int | None = None, learning_rate: float = 1e-4) -> dict:
    import torch
    devices = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
    if len(devices) != EXPECTED_GPUS or any("4080" not in d.name for d in devices):
        raise ValueError(f"Expected {EXPECTED_GPUS} visible NVIDIA RTX 4080 GPUs")
    capacities = [d.total_memory / 2**30 for d in devices]
    size = select_batch(capacities, global_batch)
    return dict(gpus=[dict(name=d.name, memory_gib=m) for d, m in zip(devices, capacities)],
                task=f"one four-step model on all {EXPECTED_GPUS} GPUs", global_batch=size,
                per_gpu_batch=size // EXPECTED_GPUS, gradient_accumulation_steps=1,
                learning_rate=learning_rate,
                steps_per_epoch=batch_layout(TRAIN_SIZE, size, EXPECTED_GPUS)[2],
                note="Global batch 4,000 means 1,000 samples per 16GB GPU with no gradient accumulation")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--global-batch", type=int)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-only", action="store_true")
    args = p.parse_args()
    plan = inspect_hardware(args.global_batch, args.lr)
    print(plan["global_batch"] if args.batch_only else json.dumps(plan, indent=2))
