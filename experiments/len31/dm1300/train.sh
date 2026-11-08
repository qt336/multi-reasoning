#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "$0")" && pwd)"
data_dir="${DATA_DIR:-$here/../data}"
run_dir="${RUN_DIR:-$here/runs/reproduction}"
python3 "$here/source_at_launch/data.py" --root "$data_dir" --train-size 6500000 --eval-per-group 10000 --seed 2027 --chunk-size 50000
mkdir -p "$run_dir"
python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 "$here/source_at_launch/train.py" \
  --data-dir "$data_dir" --run-dir "$run_dir" --expected-gpus 4 \
  --width 1300 --ffn-width 2600 --layers 3 --normalization prenorm --initialization kaiming_uniform_relu_gamma1 \
  --global-batch 4000 --lr 1e-4 --weight-decay 0.15 --precision fp32 --warmup-epochs 20 --epochs 2000 \
  --eval-every 5 --eval-batch 250 --train-eval-size 10000 --train-eval-seed 2027 --seed 2031 \
  --data-residency gpu --snapshot-epochs 30 31 32 33 35 --compile-model 2>&1 | tee -a "$run_dir/train.log"
