#!/usr/bin/env bash
cd "$(dirname "$0")/pilot" || exit 1
PY=/e/anaconda3/envs/paper/python.exe
LOG=../results/stage0.log
: > "$LOG"
for B in 3 5; do
  for k in 1 2 3 4 5; do
    echo "[$(date +%H:%M:%S)] START r$k b$B" >>"$LOG"
    $PY arm_c.py --data ../data/train.json --task finqa_pot --limit 400 \
      --keys ../results/stage0_keys.json --force_json ../results/stage0_force_r$k.json \
      --conds auto_f@$B --budget_mode fixed --value_first \
      --out ../results/stage0_r${k}_b${B}.json >>"$LOG" 2>&1
    echo "[$(date +%H:%M:%S)] DONE r$k b$B" >>"$LOG"
  done
done
echo "[$(date +%H:%M:%S)] ALL DONE" >>"$LOG"
