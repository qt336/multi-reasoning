#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$project_dir"
mode="${1:-train}"
if [[ "$mode" != train && "$mode" != prepare ]]; then
  echo 'Usage: bash run.sh [train|prepare]' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
python_bin="${PYTHON:-python3}"
global_batch="${GLOBAL_BATCH:-64000}"
compile_model="${COMPILE_MODEL:-1}"
if [[ "$compile_model" != 0 && "$compile_model" != 1 ]]; then
  echo 'COMPILE_MODEL must be 0 or 1' >&2
  exit 2
fi
compile_args=(--compile-model)
if [[ "$compile_model" == 0 ]]; then
  compile_args=(--no-compile-model)
fi
data_root="${DATA_ROOT:-$project_dir/data}"
run_root="${RUN_ROOT:-$project_dir/runs}"
data_dir="$data_root/chain_4step_6p5m_len31_vocab120_eval10000"
run_dir="$run_root/chain_4step_3layer_dm1024_dff2048_len31_6p5m_vocab120_batch${global_batch}_compile${compile_model}_seed2029_eval10000_wd0p3"
if [[ "$mode" == train ]]; then
  "$python_bin" hardware.py --global-batch "$global_batch"
fi
"$python_bin" data.py --root "$data_dir" --train-size 6500000 \
  --eval-per-group 10000 --seed 2027 --chunk-size 50000
if [[ "$mode" == prepare ]]; then
  exit 0
fi
mkdir -p -- "$run_dir"
"$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 train.py \
  --data-dir "$data_dir" --run-dir "$run_dir" --expected-gpus 8 \
  --normalization prenorm --initialization kaiming_uniform_relu_gamma1 \
  --global-batch "$global_batch" --width 1024 --ffn-width 2048 --layers 3 --lr 1e-4 \
  --warmup-epochs 20 --epochs 2000 --eval-every 5 --eval-batch 250 \
  --seed 2029 --train-eval-size 10000 --train-eval-seed 2027 --data-residency gpu \
  "${compile_args[@]}" 2>&1 | tee -a "$run_dir/train.log"
