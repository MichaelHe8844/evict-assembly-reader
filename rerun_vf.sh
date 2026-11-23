#!/usr/bin/env bash
# 采用「先保数值、后保绑定」(--value_first) 后重跑受影响的全部表。
# 输出统一加 vf_ 前缀，不覆盖旧结果，便于逐表对照。
cd "$(dirname "$0")/pilot" || exit 1
PY=/e/anaconda3/envs/paper/python.exe
LOG=../results/rerun_vf.log
: > "$LOG"

run() {  # run <输出名> <参数...>
  local out="$1"; shift
  echo "[$(date +%H:%M:%S)] START $out" >> "$LOG"
  $PY arm_c.py "$@" --out "../results/$out.json" >> "$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $out exit=$?" >> "$LOG"
}

echo "===== 1) 5.2 主表（7B，117 题，20%） =====" >> "$LOG"
run vf_arith_main --data ../data/dev.json --task finqa_pot --single_step \
    --keys ../results/dev_keys.json --limit 500 --load_8bit --value_first \
    --conds full,auto_f@20,auto_n@20

echo "===== 2) 5.10 dev 基线（3B，126 条，四档） =====" >> "$LOG"
for B in 20 10 5 3; do
  run "vf_af$B" --data ../data/dev.json --task finqa_pot --limit 1000 \
      --keys ../results/dev126_keys.json --conds "auto_f@$B" --value_first
done

echo "===== 3) 5.10 dev 同预算强制（四档） =====" >> "$LOG"
for B in 20 10 5 3; do
  run "vf_af${B}_sel" --data ../data/dev.json --task finqa_pot --limit 1000 \
      --keys ../results/dev126_keys.json --force_json ../results/r1_force.json \
      --conds "auto_f@$B" --budget_mode fixed --value_first
done

echo "===== 4) 5.10 5% 三个对照 + 种子 =====" >> "$LOG"
run vf_af7 --data ../data/dev.json --task finqa_pot --limit 1000 \
    --keys ../results/dev126_keys.json --conds auto_f@7 --value_first
for tag in rand jr; do
  run "vf_af5_$tag" --data ../data/dev.json --task finqa_pot --limit 1000 \
      --keys ../results/dev126_keys.json --force_json "../results/${tag}_force.json" \
      --conds auto_f@5 --budget_mode fixed --value_first
done
run vf_af5_s1 --data ../data/dev.json --task finqa_pot --limit 1000 \
    --keys ../results/dev126_keys.json --force_json ../results/r1_force_s1.json \
    --conds auto_f@5 --budget_mode fixed --value_first

echo "===== 5) 5.11 test（182 条，三档） =====" >> "$LOG"
for B in 3 5 10; do
  run "vf_x$B" --data ../data/fin_test.json --task finqa_pot --limit 1000 \
      --keys ../results/xds_keys_used.json --conds "auto_f@$B" --value_first
  run "vf_x${B}_sel" --data ../data/fin_test.json --task finqa_pot --limit 1000 \
      --keys ../results/xds_keys_used.json --force_json ../results/xds_force.json \
      --conds "auto_f@$B" --budget_mode fixed --value_first
done

echo "[$(date +%H:%M:%S)] ALL DONE" >> "$LOG"
