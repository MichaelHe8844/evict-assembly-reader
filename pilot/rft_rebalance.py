# -*- coding: utf-8 -*-
"""把自采样结果按预算档分层抽到固定条数，保证与 gold 监督的同规模对照。

用法：
  python rft_rebalance.py ../results/vf_rft_rows_raw.json 1893 ../results/vf_rft_rows.json
"""
import json
import random
import sys
from collections import defaultdict


def budget_of(row):
    head = row["prompt"].split("\n")[0]
    return head.split("：")[-1].strip() if "：" in head else "?"


def main():
    src, n, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    rows = json.load(open(src, encoding="utf-8"))
    by = defaultdict(list)
    for r in rows:
        by[budget_of(r)].append(r)
    total = len(rows)
    if total <= n:
        json.dump(rows, open(dst, "w"), ensure_ascii=False)
        print("保留全部 %d 条（未达目标 %d）" % (total, n))
        return

    rng = random.Random(0)
    keys = sorted(by)
    # 按各档占比分配名额，余数补给最大的档
    quota = {k: int(round(n * len(by[k]) / total)) for k in keys}
    diff = n - sum(quota.values())
    if diff:
        quota[max(keys, key=lambda k: len(by[k]))] += diff
    out = []
    for k in keys:
        pool = by[k][:]
        rng.shuffle(pool)
        out.extend(pool[:quota[k]])
        print("  %-8s 池 %4d -> 取 %4d" % (k, len(by[k]), min(quota[k], len(pool))))
    rng.shuffle(out)
    json.dump(out, open(dst, "w"), ensure_ascii=False)
    print("共 %d 条 -> %s（源池 %d）" % (len(out), dst, total))


if __name__ == "__main__":
    main()
