"""Optional short training-throughput comparison, to run on the target 8 A100s.

This script DOES perform synthetic optimizer updates, in isolated subprocesses.
It does not create or modify formal training data, run directories or checkpoints.
Do not run it on the development machine when only preparing/uploading code.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from hardware import inspect_hardware, select_batch, TRAIN_SIZE


def choose_best(results: list[dict]) -> dict:
    good = [r for r in results if r.get("status") == "ok"]
    if not good:
        raise RuntimeError("No benchmark candidate completed successfully")
    return max(good, key=lambda r: (r["examples_per_second"], -r["global_batch"]))


def worker(args):
    import torch
    from torch import distributed as dist
    from torch.nn import functional as F
    from data import LOW, HIGH, save_json
    from model import ReasoningTransformer
    from train import WEIGHT_DECAY, autocast, batch, training_model

    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 8:
        raise ValueError("Benchmark requires eight cooperating ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    torch.manual_seed(2029)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    local_batch = args.global_batch // 8
    # Match the formal run's resident shard and permutation memory footprint.
    # Random tokens and targets are for throughput only, not accuracy reporting.
    x = torch.randint(LOW, HIGH + 1, (TRAIN_SIZE // 8, 31), dtype=torch.uint8, device=device)
    y = torch.randint(LOW, HIGH + 1, (TRAIN_SIZE // 8,), dtype=torch.uint8, device=device)
    order = torch.randperm(TRAIN_SIZE // 8, device=device)
    raw_model = ReasoningTransformer().to(device)
    model = training_model(raw_model, device, local_rank, args.compile_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=WEIGHT_DECAY, fused=True)

    def update(i):
        # Measure full batches consistently, even when the shard has a tail.
        left = (i % (len(order) // local_batch)) * local_batch
        tokens, targets = batch(x, y, order[left:left + local_batch], device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device):
            loss = F.cross_entropy(model(tokens).float(), targets)
        loss.backward()
        optimizer.step()

    for i in range(args.warmup):
        update(i)
    torch.cuda.synchronize(device)
    dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for i in range(args.measure):
        update(args.warmup + i)
    torch.cuda.synchronize(device)
    metrics = torch.tensor([time.perf_counter() - started,
                            torch.cuda.max_memory_allocated(device) / 2**30,
                            torch.cuda.max_memory_reserved(device) / 2**30],
                           dtype=torch.float64, device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    if rank == 0:
        seconds, allocated, reserved = metrics.tolist()
        save_json(args.result, dict(status="ok", global_batch=args.global_batch,
                                    per_gpu_batch=local_batch, compile_model=args.compile_model,
                                    warmup_steps=args.warmup, measured_steps=args.measure,
                                    seconds_per_step=seconds / args.measure,
                                    examples_per_second=args.global_batch * args.measure / seconds,
                                    peak_allocated_gib=allocated, peak_reserved_gib=reserved,
                                    scope="synthetic eight-GPU training throughput; not task accuracy"))
    dist.destroy_process_group()


def main(args):
    from data import save_json
    if args.warmup < 1 or args.measure < 1:
        raise ValueError("Warmup and measured step counts must be positive")
    if args.worker:
        return worker(args)
    plan = inspect_hardware()
    capacities = [d["memory_gib"] for d in plan["gpus"]]
    for size in args.candidates:
        select_batch(capacities, size)
    root = args.output_dir or Path("runs") / ("batch_benchmark_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    root.mkdir(parents=True, exist_ok=False)
    results = []
    for size in args.candidates:
        output = root / f"batch_{size}.json"
        log = root / f"batch_{size}.log"
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
                   str(Path(__file__).resolve()), "--worker", "--global-batch", str(size),
                   "--warmup", str(args.warmup), "--measure", str(args.measure), "--result", str(output)]
        if not args.compile_model:
            command += ["--no-compile-model"]
        print(f"Measuring global batch {size}; log: {log}", flush=True)
        environment = {**os.environ, "OMP_NUM_THREADS": "2"}
        with log.open("w") as handle:
            completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=environment)
        if completed.returncode:
            detail = log.read_text().lower()
            if "out of memory" not in detail and "outofmemoryerror" not in detail:
                raise RuntimeError(f"Candidate {size} failed; inspect {log}")
            result = dict(status="out_of_memory", global_batch=size)
        else:
            result = json.loads(output.read_text())
        results.append(result)
        save_json(root / "results.json", dict(hardware=plan, results=results))
    best = choose_best(results)
    save_json(root / "recommendation.json", best)
    print(json.dumps(best, indent=2))
    compile_prefix = "" if args.compile_model else "COMPILE_MODEL=0 "
    print(f"Recommended launch: {compile_prefix}GLOBAL_BATCH={best['global_batch']} bash run.sh")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", type=int, default=[16000, 32000, 64000, 128000])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measure", type=int, default=20)
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--global-batch", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    main(parser.parse_args())
