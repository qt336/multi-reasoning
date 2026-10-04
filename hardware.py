"""Read-only 8xA100 80GB validation and four-step batch configuration."""

from __future__ import annotations

import argparse
import json

TRAIN_SIZE = 32_000_000
DEFAULT_BATCH = 64_000


def select_batch(memory_gib: list[float], override: int | None = None) -> int:
    if len(memory_gib) != 8 or min(memory_gib) < 75:
        raise ValueError("Exactly 8 A100 80GB GPUs are required")
    size = DEFAULT_BATCH if override is None else override
    if size <= 0 or size % 8 or TRAIN_SIZE % size:
        raise ValueError("Global batch must be positive, divisible by 8, and divide 32000000")
    return size


def inspect_hardware(global_batch: int | None = None) -> dict:
    import torch
    devices = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
    if len(devices) != 8 or any("A100" not in d.name for d in devices):
        raise ValueError("Expected 8 visible NVIDIA A100 80GB GPUs")
    capacities = [d.total_memory / 2**30 for d in devices]
    size = select_batch(capacities, global_batch)
    return dict(gpus=[dict(name=d.name, memory_gib=m) for d, m in zip(devices, capacities)],
                task="one four-step model on all eight GPUs", global_batch=size,
                per_gpu_batch=size // 8, learning_rate=1e-4,
                steps_per_epoch=TRAIN_SIZE // size,
                note="Default batch is an unbenchmarked starting configuration; benchmark.py can compare throughput")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--global-batch", type=int)
    p.add_argument("--batch-only", action="store_true")
    args = p.parse_args()
    plan = inspect_hardware(args.global_batch)
    print(plan["global_batch"] if args.batch_only else json.dumps(plan, indent=2))
