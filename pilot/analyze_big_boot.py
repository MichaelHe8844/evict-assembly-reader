# -*- coding: utf-8 -*-
"""放大后的主结果：按文档聚类的 bootstrap（替代旧的 83 题版本）。

用法：python analyze_big_boot.py
读 results/arith_big.json、results/holdout_big.json
"""
import json
import random
from collections import defaultdict

random.seed(20261102)
B = 4000


def load(path, cond):
    d = json.load(open(path, encoding="utf-8"))
    return {r["id"]: r["correct"] for r in d["records"] if r["cond"] == cond}


def boot(a, b, label):
    keys = [k for k in b if k in a]
    if not keys:
        print("%-40s 无可配对样本" % label)
        return
    doc = defaultdict(list)
    for k in keys:
        doc[k.rsplit("-", 1)[0]].append(k)
    docs = list(doc)
    deltas = []
    for _ in range(B):
        pick = [docs[random.randrange(len(docs))] for _ in docs]
        ks = [k for d in pick for k in doc[d]]
        deltas.append((sum(1 for k in ks if b[k]) - sum(1 for k in ks if a[k])) / len(ks))
    deltas.sort()
    lo, hi = deltas[int(0.025 * B)], deltas[int(0.975 * B)]
    na = sum(1 for k in keys if a[k]) / len(keys)
    nb = sum(1 for k in keys if b[k]) / len(keys)
    neg = sum(1 for x in deltas if x <= 0) / B
    pos = sum(1 for x in deltas if x >= 0) / B
    print("%-40s n=%-4d doc=%-3d %.3f→%.3f Δ=%+.3f [%+.3f, %+.3f] p=%.3f"
          % (label, len(keys), len(docs), na, nb, nb - na, lo, hi, min(1.0, 2 * min(neg, pos))))


def main():
    d = "../results/arith_big.json"
    full, snap, h2o, auto = (load(d, c) for c in ("full", "snap@20", "h2o@20", "auto_f@20"))
    print("== dev 单步全集（117 题）==")
    boot(snap, auto, "SnapKV@20 → 本文")
    boot(h2o, auto, "H2O@20 → 本文")
    boot(full, auto, "全上下文 → 本文")
    boot(snap, full, "SnapKV@20 → 全上下文")
    try:
        h = "../results/holdout_big.json"
        hf, hs, ha = (load(h, c) for c in ("full", "snap@20", "auto_f@20"))
        print("\n== 留出段单步全集 ==")
        boot(hs, ha, "SnapKV@20 → 本文")
        boot(hf, ha, "全上下文 → 本文")
    except FileNotFoundError:
        print("\n（留出段尚未完成）")


if __name__ == "__main__":
    main()
