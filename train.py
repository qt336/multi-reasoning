"""DDP training with the seed-2029 reference hyperparameters and live plots."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from data import save_array, save_json
from hardware import batch_layout
from model import ReasoningTransformer
from plot import render


METRICS = ("train", "test", "test_canonical", "test_noncanonical")
FIELDS = ("epoch", *(f"{name}_{stat}" for name in METRICS
                     for stat in ("correct", "n", "accuracy")),
          "train_loss", "learning_rate", "elapsed_hours")


def timestamp():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def learning_rate(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))


def training_sample_ids(train_size, sample_size, world, sample_seed):
    """Keep the reference's fixed, equal-per-shard training sample policy."""
    if not 0 < sample_size <= train_size:
        raise ValueError("Training evaluation sample must be in (0, train_size]")
    if train_size % world or sample_size % world:
        raise ValueError("Training size and training sample size must divide GPU count")
    shard_size = train_size // world
    return np.concatenate([
        np.sort(np.random.default_rng(np.random.SeedSequence([sample_seed, rank]))
                .choice(shard_size, sample_size // world, replace=False)) + rank * shard_size
        for rank in range(world)
    ]).astype(np.int64)


def load_xy(root, name):
    return (np.load(root / f"{name}_x.npy", mmap_mode="r"),
            np.load(root / f"{name}_y.npy", mmap_mode="r"))


def batch(x, y, ids, device):
    # Preserve the reference's token/target IDs and all 101 output classes.
    if isinstance(x, torch.Tensor):
        ids = torch.as_tensor(ids, dtype=torch.long, device=x.device)
        return x[ids].long(), y[ids].long()
    return (torch.from_numpy(np.array(x[ids], dtype=np.int64, copy=True)).to(device),
            torch.from_numpy(np.array(y[ids], dtype=np.int64, copy=True)).to(device))


def resident_shard(x, y, left, right, device):
    """Load only this rank's uint8 shard once, using bounded CPU staging memory."""
    tx = torch.empty((right - left, x.shape[1]), dtype=torch.uint8, device=device)
    ty = torch.empty(right - left, dtype=torch.uint8, device=device)
    for offset in range(left, right, 250_000):
        end = min(offset + 250_000, right)
        tx[offset-left:end-left].copy_(torch.from_numpy(np.array(x[offset:end], copy=True)))
        ty[offset-left:end-left].copy_(torch.from_numpy(np.array(y[offset:end], copy=True)))
    return tx, ty


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def training_model(raw_model, device, local_rank, compile_model=True):
    """Compile the inner module; retain raw_model for portable checkpoints/eval.

    The no-cudagraph mode supports DDP graph partitioning in PyTorch 2.5.
    Kernel autotuning happens only when launched on the target GPUs.
    """
    inner = (torch.compile(raw_model, mode="max-autotune-no-cudagraphs", dynamic=False)
             if compile_model else raw_model)
    return DDP(inner, device_ids=[local_rank] if device.type == "cuda" else None,
               broadcast_buffers=False, gradient_as_bucket_view=True, static_graph=True)


@torch.inference_mode()
def evaluate(raw_model, x, y, ids, batch_size, device):
    # Use the underlying module: ranks can have unequal evaluation sample counts
    # and must not enter DDP forward collectives a different number of times.
    raw_model.eval()
    totals = torch.zeros(2, dtype=torch.int64, device=device)
    for left in range(0, len(ids), batch_size):
        tokens, targets = batch(x, y, ids[left:left + batch_size], device)
        with autocast(device):
            logits = raw_model(tokens)
        totals[0] += logits.argmax(-1).eq(targets).sum()
        totals[1] += len(targets)
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    raw_model.train()
    return tuple(map(int, totals.tolist()))


def record(run_dir, epoch, scores, loss, lr, started_at, steps):
    path = run_dir / "accuracy.csv"
    rows = []
    if path.exists():
        with path.open(newline="") as f:
            # On resume, discard diagnostics from epochs after the saved model.
            rows = [r for r in csv.DictReader(f) if int(r["epoch"]) < epoch]
    row = dict(epoch=epoch, train_loss="" if loss is None else loss,
               learning_rate=lr, elapsed_hours=(time.time() - started_at) / 3600)
    for name in METRICS:
        value = scores.get(name)
        if value is not None:
            correct, total = value
            if total <= 0:
                raise ValueError(f"Empty evaluation set: {name}")
            row.update({f"{name}_correct": correct, f"{name}_n": total,
                        f"{name}_accuracy": correct / total})
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(row)
    os.replace(tmp, path)
    render(path, run_dir / "accuracy.png", steps)
    print("EVAL " + json.dumps(row), flush=True)
    return row


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--expected-gpus", type=int, default=8)
    p.add_argument("--global-batch", type=int, default=64000)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--ffn-width", type=int, default=2048)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=20)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-batch", type=int, default=250)
    p.add_argument("--train-eval-size", type=int, default=10000)
    p.add_argument("--train-eval-seed", type=int, default=2027)
    p.add_argument("--seed", type=int, default=2029)
    p.add_argument("--initialization", default="kaiming_uniform_relu_gamma1",
                   choices=("uniform_gamma1", "kaiming_uniform_relu", "kaiming_uniform_relu_gamma1"))
    p.add_argument("--normalization", default="prenorm",
                   choices=("prenorm", "block_end", "prelayernorm"))
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                   help="CPU is provided for explicit small smoke tests")
    p.add_argument("--data-residency", choices=("gpu", "mmap"), default="gpu",
                   help="Keep each rank's uint8 shard on its GPU to avoid per-step disk/CPU copies")
    p.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=True)
    return p


def main(args):
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != args.expected_gpus:
        raise ValueError(f"Expected {args.expected_gpus} workers, found {world}")
    if min(args.epochs, args.eval_every, args.eval_batch, args.global_batch) < 1 or args.warmup_epochs < 0:
        raise ValueError("Invalid training/evaluation sizes")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; run.sh requires 8 A100 GPUs")
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if args.device == "cuda" else torch.device("cpu")
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.run_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.data_dir / "dataset.json").read_text())
    train_size, steps = meta["train_size"], meta["steps"]
    local_count, local_batch, steps_per_epoch = batch_layout(train_size, args.global_batch, world)
    train_x, train_y = load_xy(args.data_dir, "train")
    if train_x.shape != (train_size, 31) or train_y.shape != (train_size,):
        raise ValueError("Training array shape differs from the 31-token manifest")
    if (meta["sequence_length"] != 31 or steps != 4 or meta.get("n_facts") != 15 or
            meta.get("token_min") != 20 or meta.get("token_max") != 100 or
            meta.get("vocab_size") != 101 or meta.get("format_version") != 3):
        raise ValueError("Unsupported task")
    model_config = dict(width=args.width, ffn_width=args.ffn_width, layers=args.layers,
                        vocab=101, length=31, initialization=args.initialization,
                        normalization=args.normalization)
    config = dict(data=meta, model=model_config, gpus=world, global_batch=args.global_batch,
                  learning_rate=args.lr, warmup_epochs=args.warmup_epochs,
                  epochs=args.epochs, eval_every=args.eval_every, eval_batch=args.eval_batch,
                  seed=args.seed, train_accuracy_sample_size=args.train_eval_size,
                  train_accuracy_sample_seed=args.train_eval_seed, train_accuracy_every=1,
                  optimizer="AdamW", betas=[0.9, 0.999], eps=1e-8, weight_decay=0.1,
                  device=args.device, mixed_precision="bfloat16" if args.device == "cuda" else None,
                  data_residency=args.data_residency,
                  data_loading="fixed uint8 shard; full random permutation per epoch",
                  ddp_gradient_as_bucket_view=True, ddp_static_graph=True,
                  compile_model=args.compile_model,
                  compile_mode="max-autotune-no-cudagraphs" if args.compile_model else None)
    config_path = args.run_dir / "config.json"
    previous = json.loads(config_path.read_text()) if config_path.exists() else None
    if previous is not None and any(previous.get(k) != v for k, v in config.items()):
        raise RuntimeError("Run configuration differs; use a new run directory")
    started_at = (datetime.fromisoformat(previous["started_at"]).timestamp()
                  if previous else time.time())

    sample_path = args.run_dir / "train_accuracy_sample_ids.npy"
    expected_ids = training_sample_ids(train_size, args.train_eval_size, world, args.train_eval_seed)
    if sample_path.exists() and not np.array_equal(np.load(sample_path), expected_ids):
        raise RuntimeError("Training accuracy sample IDs differ from the fixed seed/shards")
    sample_ids = expected_ids
    if rank == 0:
        save_array(sample_path, sample_ids)
        save_json(config_path, {**config, "started_at": datetime.fromtimestamp(
            started_at, tz=timezone.utc).isoformat()})
        save_json(args.run_dir / "train_accuracy_sample.json", dict(
            source=str(args.data_dir.resolve()), source_size=train_size,
            sample_size=len(sample_ids), sample_seed=args.train_eval_seed,
            sampling="fixed without replacement, equal per training shard"))
    per_rank = len(sample_ids) // world
    local_train_ids = sample_ids[rank * per_rank:(rank + 1) * per_rank]
    evaluations = {"train": (train_x, train_y, local_train_ids, local_batch)}
    eval_x, eval_y = load_xy(args.data_dir, "eval")
    group_size = meta["eval_per_group"]
    if eval_x.shape != (2 * group_size, 31) or eval_y.shape != (2 * group_size,):
        raise ValueError("Invalid paired evaluation shape")
    for name, offset in (("test_canonical", 0), ("test_noncanonical", group_size)):
        evaluations[name] = (eval_x, eval_y, np.arange(rank, group_size, world) + offset, args.eval_batch)
    # Cache these small fixed subsets once; evaluate without repeated disk reads.
    for name, (x, y, ids, size) in list(evaluations.items()):
        cached_x = torch.from_numpy(np.array(x[ids], copy=True)).to(device)
        cached_y = torch.from_numpy(np.array(y[ids], copy=True)).to(device)
        evaluations[name] = (cached_x, cached_y, torch.arange(len(ids), device=device), size)

    if args.data_residency == "gpu":
        if device.type != "cuda":
            raise ValueError("Use --data-residency mmap with --device cpu")
        train_batch_x, train_batch_y = resident_shard(
            train_x, train_y, rank * local_count, (rank + 1) * local_count, device)
        order_device = device
    else:
        train_batch_x, train_batch_y = train_x, train_y
        order_device = torch.device("cpu")
    raw_model = ReasoningTransformer(**model_config).to(device)
    model = training_model(raw_model, device, local_rank, args.compile_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.1, fused=device.type == "cuda")
    total_steps, warmup_steps = steps_per_epoch * args.epochs, steps_per_epoch * args.warmup_epochs
    checkpoint = args.run_dir / "latest.pt"
    start_epoch = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved["config"] != config:
            raise RuntimeError("Checkpoint configuration differs")
        raw_model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = saved["epoch"]
    elif (args.run_dir / "accuracy.csv").exists():
        raise RuntimeError("Accuracy history exists without a checkpoint; use a new run directory")

    def measure(include_test):
        scores = {name: evaluate(raw_model, x, y, ids, size, device)
                  for name, (x, y, ids, size) in evaluations.items()
                  if include_test or name == "train"}
        if include_test:
            c, n = scores["test_canonical"], scores["test_noncanonical"]
            scores["test"] = (c[0] + n[0], c[1] + n[1])
        return scores

    if rank == 0:
        print(f"TRAIN steps={steps} GPUs={world} parameters={sum(p.numel() for p in raw_model.parameters()):,} "
              f"examples={train_size:,} steps_per_epoch={steps_per_epoch:,} resume_epoch={start_epoch}", flush=True)
    # Re-evaluate the saved model, replacing any CSV rows beyond its epoch.
    scores = measure(True)
    if rank == 0:
        saved_loss = saved.get("train_loss") if start_epoch else None
        record(args.run_dir, start_epoch, scores, saved_loss,
               optimizer.param_groups[0]["lr"] if start_epoch else 0, started_at, steps)
        if not checkpoint.exists():
            tmp = checkpoint.with_name("latest.pt.tmp")
            torch.save(dict(epoch=0, config=config, model=raw_model.state_dict(),
                            optimizer=optimizer.state_dict(), train_loss=None), tmp)
            os.replace(tmp, checkpoint)
    dist.barrier()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_start = time.time()
        # Visit every fixed training row exactly once per epoch.
        generator = torch.Generator(device=order_device)
        generator.manual_seed(args.seed + epoch * 10_007 + rank)
        order = torch.randperm(local_count, generator=generator, device=order_device)
        loss_sum = torch.zeros((), device=device)
        for batch_idx, left in enumerate(range(0, local_count, local_batch)):
            ids = order[left:left + local_batch]
            if args.data_residency == "mmap":
                ids = ids.numpy() + rank * local_count
            tokens, targets = batch(train_batch_x, train_batch_y, ids, device)
            lr = learning_rate((epoch - 1) * steps_per_epoch + batch_idx,
                               total_steps, warmup_steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                logits = model(tokens)
                loss = F.cross_entropy(logits.float(), targets)
            loss.backward()
            optimizer.step()
            # Weight the final smaller batch by its actual number of examples.
            loss_sum += loss.detach() * len(targets)
        del order
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        mean_loss = float(loss_sum) / train_size
        include_test = epoch % args.eval_every == 0 or epoch == args.epochs
        scores = measure(include_test)
        if rank == 0:
            row = record(args.run_dir, epoch, scores, mean_loss, lr, started_at, steps)
            save_json(args.run_dir / "status.json", dict(
                state="complete" if epoch == args.epochs else "training",
                epochs=args.epochs, **row, updated_at=timestamp(),
                seconds_per_epoch=time.time() - epoch_start,
                examples_per_second=train_size / (time.time() - epoch_start),
                accuracy_scope="fixed training sample; paired canonical/noncanonical held-out test"))
            if include_test:
                tmp = checkpoint.with_name("latest.pt.tmp")
                torch.save(dict(epoch=epoch, config=config, model=raw_model.state_dict(),
                                optimizer=optimizer.state_dict(), train_loss=mean_loss), tmp)
                os.replace(tmp, checkpoint)
        dist.barrier()
    if rank == 0:
        print(f"COMPLETE {timestamp()}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main(parser().parse_args())
