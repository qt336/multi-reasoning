#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
source ./server_env_resume_20261027.sh
exec 9>../training.lock
flock -n 9
python3 hardware.py --global-batch 4000 --lr 1e-4
python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 train.py \
  --data-dir /home/qintian/multi-reasoning/data/chain_4step_6p5m_len31_vocab101_eval10000 \
  --run-dir /home/qintian/multi-reasoning/runs/chain_4step_3layer_dm1300_dff2600_len31_6p5m_4x4080_batch4000_compile1_seed2031_eval10000_wd0p15_lr1e-4_fp32_20261027 \
  --expected-gpus 4 --normalization prenorm --initialization kaiming_uniform_relu_gamma1 \
  --global-batch 4000 --width 1300 --ffn-width 2600 --layers 3 \
  --lr 1e-4 --weight-decay 0.15 --precision fp32 --warmup-epochs 20 --epochs 2000 \
  --eval-every 5 --eval-batch 250 --seed 2031 --train-eval-size 10000 --train-eval-seed 2027 \
  --data-residency gpu --snapshot-epochs 30 31 32 33 35 --compile-model \
  2>&1 | tee -a ../train.log
