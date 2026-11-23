# -*- coding: utf-8 -*-
"""导出「手写规则」（Joint-R）在给定成对数据上的选择，格式同 --force_json。

用途：跨数据集上做「学习式 vs 规则」的端到端对照——两者必须强制同样的东西，
才谈得上比较。
"""
import json
import re
import sys

import numpy as np


def rule_pick(r):
    cs = r["cands"]
    u = np.array([max(1e-6, c["gscore"]) for c in cs])
    q = r["q"].lower()
    grow = bool(re.search(r"change|increase|decrease|growth|from .* to ", q))
    share = bool(re.search(r"percent of|percentage of|share of|portion|ratio of", q))
    best, bp = -1e9, None
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            a, b = cs[i], cs[j]
            sr, sc = a["r"] == b["r"], a["c"] == b["c"]
            bonus = 2.0 if (sr != sc) else 0.2
            if grow and sr and not sc:
                bonus += 1.5
            if share and sc and not sr:
                bonus += 1.5
            ra, rb = set(a.get("row_t") or []), set(b.get("row_t") or [])
            ca, cb = set(a.get("col_t") or []), set(b.get("col_t") or [])
            ov = (len(ra & rb) + len(ca & cb)) / max(1, len(ra | rb) + len(ca | cb))
            v = u[i] / u.max() + u[j] / u.max() + bonus - 0.5 * ov
            if v > best:
                best, bp = v, (a, b)
    return bp


def main():
    src = sys.argv[1]
    out = sys.argv[2]
    rows = json.load(open(src, encoding="utf-8"))
    picks, hit = {}, []
    for r in rows:
        a, b = rule_pick(r)
        ok = set(map(tuple, r["gold"])) <= {(a["r"], a["c"]), (b["r"], b["c"])}
        hit.append(ok)
        picks[r["qid"]] = [[a["r"], a["c"]], [b["r"], b["c"]]]
    print("规则选择：%d 题，整对命中 %.3f（R1 参考 0.511）" % (len(rows), sum(hit) / len(hit)))
    json.dump(picks, open(out, "w"), ensure_ascii=False)
    print("已写", out)


if __name__ == "__main__":
    main()
