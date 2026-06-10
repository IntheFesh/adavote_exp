#!/bin/bash
set -e
cd /root/autodl-tmp/adavote_exp
export ADAVOTE_RESULTS=/root/autodl-tmp/adavote_exp/results

for env in simple_spread simple_reference overcooked_v0; do
  for algo in ippo mappo; do
    echo "=== TRAIN $env $algo $(date) ==="
    python -m e2_jaxmarl.train_committee \
        --env $env --algo $algo \
        --n_members 7 --total_timesteps 3000000 --seed 0 \
        --out_dir results/e2/checkpoints
  done
done
echo "=== ALL TRAINING DONE $(date) ==="
