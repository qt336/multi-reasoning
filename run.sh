#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$project_dir"
task="${1:-all}"
mode="${2:-train}"
if [[ "$task" == all ]]; then
  tasks=(7 8 9 10 11 12 13)
elif [[ "$task" =~ ^(7|8|9|10|11|12|13)$ ]]; then
  tasks=("$task")
else
  echo 'Usage: bash run.sh [all|7|8|9|10|11|12|13] [train|prepare]' >&2
  exit 2
fi
if [[ "$mode" != train && "$mode" != prepare ]]; then
  echo 'Mode must be train or prepare' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
python_bin="${PYTHON:-python3}"
workers=8
data_root="${DATA_ROOT:-$project_dir/data}"
run_root="${RUN_ROOT:-$project_dir/runs}"
if [[ "$mode" == train ]]; then
  batch_args=()
  if [[ -n "${GLOBAL_BATCH:-}" ]]; then
    batch_args=(--global-batch "$GLOBAL_BATCH")
  fi
  # This queries hardware only. No trial training or automatic batch sweep.
  global_batch="$("$python_bin" hardware.py --batch-only "${batch_args[@]}")"
  "$python_bin" hardware.py --global-batch "$global_batch"
fi
for steps in "${tasks[@]}"; do
  data_dir="$data_root/chain_${steps}step_200m_len53_vocab200"
  "$python_bin" data.py --root "$data_dir" --steps "$steps" \
    --train-size 200000000 --eval-per-group 1000 --seed 2027 \
    --chunk-size 50000 --canonical-train-size 10000
  if [[ "$mode" == prepare ]]; then
    continue
  fi
  run_dir="$run_root/chain_${steps}step_4layer_dm2048_dff4096_vocab200_batch${global_batch}_seed2029"
  mkdir -p -- "$run_dir"
  "$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$workers" train.py \
    --data-dir "$data_dir" --run-dir "$run_dir" --expected-gpus "$workers" \
    --normalization prenorm --initialization kaiming_uniform_relu_gamma1 \
    --global-batch "$global_batch" --width 2048 --ffn-width 4096 --layers 4 --lr 1e-4 \
    --warmup-epochs 20 --epochs 2000 --eval-every 5 --eval-batch 250 \
    --seed 2029 --train-eval-size 10000 --train-eval-seed 2027 --data-residency gpu \
    2>&1 | tee -a "$run_dir/train.log"
done
