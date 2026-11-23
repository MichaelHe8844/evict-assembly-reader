#!/usr/bin/env bash
# 重训后的 216 题两装配评测：base = 旧装配（表头优先），force = 新装配（钉住一对）。
# 题单与 force 文件与主表完全一致：vf216_new.json（216 个唯一 id）+ ens_force_test.json（182 题有对）。
cd "$(dirname "$0")/pilot" || exit 1
PY=/e/anaconda3/envs/paper/python.exe
LOG=../results/eval_vf216.log
: > "$LOG"

run() {  # run <adapter 名> <输出名>
  echo "[$(date +%H:%M:%S)] START $2" >> "$LOG"
  $PY oracle_ceiling.py --base E:/KV_models/Qwen2.5-3B-Instruct \
      --adapter "../results/$1" --data ../data/fin_test.json \
      --ids ../results/vf216_new.json --budgets 3,5,10,20 \
      --cond base,force --force_json ../results/ens_force_test.json \
      --out "../results/$2.json" >> "$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $2 exit=$?" >> "$LOG"
}

run lora_vf_gold vf216_gold_vf
run lora_vf_rft  vf216_rft_vf
echo "[$(date +%H:%M:%S)] ALL DONE" >> "$LOG"
