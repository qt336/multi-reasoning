"""Read-only 8-A100 launch planning; never runs a model or an optimizer."""

from __future__ import annotations

import argparse
import json


def select_batch(memory_gib: list[float], override: int | None = None) -> int:
    if len(memory_gib) != 8:
        raise ValueError("Exactly 8 visible GPUs are required for one model")
    if min(memory_gib) < 37:
        raise ValueError("Each GPU must have at least the memory of an A100 40GB")
    # Starting configurations, not claims of an A100 throughput benchmark.
    global_batch = override if override is not None else (32000 if min(memory_gib) >= 75 else 16000)
    if global_batch <= 0 or global_batch % 8 or 200_000_000 % global_batch:
        raise ValueError("GLOBAL_BATCH must be positive, divisible by 8, and divide 200000000")
    return global_batch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--global-batch", type=int)
    p.add_argument("--batch-only", action="store_true")
    args = p.parse_args()
    import torch
    devices = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
    if len(devices) != 8 or any("A100" not in d.name for d in devices):
        raise SystemExit("Expected exactly 8 visible NVIDIA A100 GPUs; no training has started")
    capacities = [d.total_memory / 2**30 for d in devices]
    size = select_batch(capacities, args.global_batch)
    if args.batch_only:
        print(size)
    else:
        print(json.dumps(dict(gpus=[dict(name=d.name, memory_gib=m) for d, m in zip(devices, capacities)],
                              models="7,8,9,10,11,12,13 sequentially; all 8 GPUs cooperate",
                              global_batch=size, per_gpu_batch=size // 8, learning_rate=1e-4,
                              steps_per_epoch=200_000_000 // size,
                              note="Capacity-based starting batch, not an A100 throughput measurement"), indent=2))


if __name__ == "__main__":
    main()
