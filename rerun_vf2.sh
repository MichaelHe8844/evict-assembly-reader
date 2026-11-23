#!/usr/bin/env bash
cd "$(dirname "$0")/pilot" || exit 1
PY=/e/anaconda3/envs/paper/python.exe
LOG=../results/rerun_vf2.log
: > "$LOG"
run(){ local out="$1"; shift; echo "[$(date +%H:%M:%S)] START $out" >>"$LOG";
  $PY arm_c.py "$@" --out "../results/$out.json" >>"$LOG" 2>&1; echo "[$(date +%H:%M:%S)] DONE $out" >>"$LOG"; }

run vf_numb3 --data ../data/dev.json --task finqa_pot --limit 1000 \
    --keys ../results/dev126_keys.json --conds numb@3 --value_first
run vf_fp3 --data ../data/dev.json --task finqa_pot --limit 1000 \
    --keys ../results/dev126_keys.json --force_json ../results/fp_af3.json \
    --conds auto_f@3 --budget_mode fixed --value_first
run vf_fp5 --data ../data/dev.json --task finqa_pot --limit 1000 \
    --keys ../results/dev126_keys.json --force_json ../results/fp_af5.json \
    --conds auto_f@5 --budget_mode fixed --value_first
run vf_x5_rule --data ../data/fin_test.json --task finqa_pot --limit 1000 \
    --keys ../results/xds_keys_used.json --force_json ../results/xds_force_rule.json \
    --conds auto_f@5 --budget_mode fixed --value_first
run vf_holdout --data ../data/dev.json --task finqa_pot --single_step --skip 400 \
    --keys ../results/holdout_keys.json --limit 500 --conds auto_f@20 --value_first --load_8bit
echo "[$(date +%H:%M:%S)] ALL DONE" >>"$LOG"
