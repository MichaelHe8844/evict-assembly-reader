#!/usr/bin/env bash
cd "$(dirname "$0")/pilot" || exit 1
PY=/e/anaconda3/envs/paper/python.exe
LOG=../results/minbind_run.log
: > "$LOG"
run(){ local out="$1"; shift; echo "[$(date +%H:%M:%S)] START $out" >>"$LOG";
  $PY arm_c.py "$@" --out "../results/$out.json" >>"$LOG" 2>&1; echo "[$(date +%H:%M:%S)] DONE $out" >>"$LOG"; }

for B in 3 5 10; do
  run "mb_af$B" --data ../data/dev.json --task finqa_pot --limit 1000 \
      --keys ../results/dev126_keys.json --conds "auto_f@$B" --value_first --min_bind
  run "mb_af${B}_sel" --data ../data/dev.json --task finqa_pot --limit 1000 \
      --keys ../results/dev126_keys.json --force_json ../results/r1_force.json \
      --conds "auto_f@$B" --budget_mode fixed --value_first --min_bind
done
echo "[$(date +%H:%M:%S)] ALL DONE" >>"$LOG"
