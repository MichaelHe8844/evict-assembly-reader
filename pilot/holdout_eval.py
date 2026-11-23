# -*- coding: utf-8 -*-
"""
留出集评估：规则是在 dev 前 300 条上边看边改的，
这里在从未参与调参的那一段（默认 skip=400）上重测同一批结构指标。

只做结构层，不需要前向，几秒钟出结果。

用法：
  python holdout_eval.py --skip 0   --limit 150   # 调参段
  python holdout_eval.py --skip 400 --limit 150   # 留出段
"""
import json
import re
import argparse

import numpy as np

from common import build_dataset
from anchor_auto import extract, table_index
from analyze_slice import nums
from arm_a import tokenize_sample
from arm_c import fit_budget

SINGLE = re.compile(r"^[a-z]+\s*\([^()]*\)$")
PRIO = {"col_name": 0, "unit": 0, "col_header": 0, "row_header": 1, "cell": 2, "text": 3, "sep": 9}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/dev.json")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--budget", type=float, default=0.20)
    ap.add_argument("--maxlen", type=int, default=1500)
    ap.add_argument("--all_steps", action="store_true", help="不限制为单步")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("E:/KV_models/Qwen2.5-3B-Instruct")

    ds = build_dataset(args.data)
    raw = {e.get("id"): e for e in json.load(open(args.data, encoding="utf-8"))}

    hitV, hitC, op_all, op_cov, cost, anr, goldfull = [], [], [], [], [], [], []
    n = 0
    seen = set()
    for d in ds[args.skip:]:
        if n >= args.limit:
            break
        qa = (raw.get(d["id"], {}) or {}).get("qa") or {}
        if not args.all_steps and not SINGLE.match((qa.get("program") or "").strip()):
            continue
        if not d["group"].get("V"):
            continue
        v = d["group"]["V"][0]
        gr, gc = d["segs"][v]["row"], d["segs"][v]["col"]
        if gr < 0 or gc < 0:
            continue

        _k = (d["id"], d["row"], d["col"])
        if _k in seen:
            continue
        seen.add(_k)
        ids, sp = tokenize_sample(tok, d["segs"])
        L = len(ids)
        if L < 64 or L > args.maxlen:
            continue

        idx, diag = extract(d["segs"], d["question"])
        idxset = set(idx)
        cn, un, rh, cell, _, _ = table_index(d["segs"])

        hitV.append(v in idxset)
        goldfull.append(all(i in idxset for c in "VRCU"
                            for i in d["group"].get(c, []) if sp[i][0] >= 0))
        hitC.append(gc in set(diag["cols"]))

        # 操作数覆盖
        table_nums = set()
        for s in d["segs"]:
            if s["kind"] == "cell":
                table_nums |= nums(s["text"])
        want = nums(qa.get("program") or "") & table_nums
        if want:
            ops = [si for (r, c), si in cell.items() if nums(d["segs"][si]["text"]) & want]
            if ops:
                cov = sum(1 for si in ops if si in idxset) / len(ops)
                op_cov.append(cov)
                op_all.append(cov >= 0.999)

        # 代价：锚点集 token 占文档比例
        t = sum(sp[i][1] - sp[i][0] for i in idx if sp[i][0] >= 0)
        cost.append(t / L)

        # 落地 ANR：按 4.2 的优先级装进预算
        B = max(8, int(args.budget * L))
        kept = set(fit_budget(sorted(idx, key=lambda k: PRIO.get(d["segs"][k]["kind"], 9)),
                              d["segs"], sp, B))
        ok = all(all(t in kept for t in range(*sp[i]))
                 for c in "VRCU" for i in d["group"].get(c, []) if sp[i][0] >= 0)
        anr.append(ok)

        n += 1

    print("区间 skip=%d  n=%d" % (args.skip, n))
    print("  gold 组完全保留  %.3f" % np.mean(goldfull))
    print("  答案格命中 V      %.3f" % np.mean(hitV))
    print("  列名命中   C      %.3f" % np.mean(hitC))
    print("  操作数全覆盖      %.3f" % (np.mean(op_all) if op_all else float("nan")))
    print("  操作数平均覆盖    %.3f" % (np.mean(op_cov) if op_cov else float("nan")))
    print("  代价 中位/均值    %.3f / %.3f" % (np.median(cost), np.mean(cost)))
    print("  落地 ANR@%.0f%%     %.3f" % (args.budget * 100, np.mean(anr)))


if __name__ == "__main__":
    main()
