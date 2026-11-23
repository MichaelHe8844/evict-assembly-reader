"""预算–可复算覆盖的探针（CPU only）。

问题：给定文档的锚点集，每个锚点的闭包是一组 token（数值 + 行头 + 列名 + 单位，
彼此共享表头）。在预算 B 下选择保留哪些 token，使**完整保留闭包的锚点数**最大。

这是带并集代价的最大覆盖（不是价值背包）：一个锚点只有全部闭包 token 都在
保留集里才算被覆盖。探针回答：
  1) 现有装配（按类型优先级整段装）离最优多远；
  2) 按边际代价贪心能否补上；
  3) 规模是否可精确求解（决定能否写成算法）。
"""
import json

import numpy as np
from scipy.optimize import LinearConstraint, milp
from scipy.sparse import lil_matrix
from transformers import AutoTokenizer

from anchor_auto import extract
from arm_a import tokenize_sample
from common import build_dataset

TOK = AutoTokenizer.from_pretrained("E:/KV_models/Qwen2.5-7B-Instruct")


def closure_masks(segs, spans):
    row_tok, col_tok = {}, {}
    for s, (a, b) in zip(segs, spans):
        if a < 0 or b <= a:
            continue
        m = ((1 << (b - a)) - 1) << a
        k = s.get("kind")
        if k == "row_header":
            row_tok[s.get("row")] = row_tok.get(s.get("row"), 0) | m
        elif k in ("col_name", "col_header", "unit"):
            col_tok[s.get("col")] = col_tok.get(s.get("col"), 0) | m
    out = {}
    for i, (s, (a, b)) in enumerate(zip(segs, spans)):
        if a < 0 or b <= a or s.get("kind") != "cell":
            continue
        m = ((1 << (b - a)) - 1) << a
        m |= row_tok.get(s.get("row"), 0)
        m |= col_tok.get(s.get("col"), 0)
        out[i] = m
    return out


def current_value(segs, spans, masks, B):
    """现有装配：数值优先、按类型整段装入，装不下就跳过。"""
    prio = {"cell": 0, "row_header": 1, "col_name": 2, "col_header": 2,
            "unit": 2, "text": 3, "sep": 9}
    sel = 0
    for i in sorted(range(len(segs)), key=lambda k: prio.get(segs[k].get("kind"), 5)):
        a, b = spans[i]
        if a < 0 or b <= a:
            continue
        if sel.bit_count() + (b - a) > B:
            continue
        sel |= ((1 << (b - a)) - 1) << a
    return sum(1 for m in masks if (m & sel) == m)


def greedy_value(masks, B):
    sel, done, val = 0, set(), 0
    while True:
        best, bi = 0.0, -1
        for i, m in enumerate(masks):
            if i in done:
                continue
            add = m & ~sel
            if not add:
                continue
            c = add.bit_count()
            if sel.bit_count() + c > B:
                continue
            r = 1.0 / c
            if r > best:
                best, bi = r, i
        if bi < 0:
            break
        sel |= masks[bi]
        done.add(bi)
        val = sum(1 for m in masks if (m & sel) == m)
    return val


def exact_value(masks, B):
    """用 MILP 精确求解：max Σy, s.t. y_a ≤ x_t (t ∈ closure_a), Σx ≤ B。"""
    toks = sorted({t for m in masks for t in range(m.bit_length()) if (m >> t) & 1})
    if not toks:
        return 0
    idx = {t: k for k, t in enumerate(toks)}
    nt, na = len(toks), len(masks)
    pairs = [(a, t) for a, m in enumerate(masks) for t in toks if (m >> t) & 1]
    A = lil_matrix((len(pairs), nt + na))
    for r, (a, t) in enumerate(pairs):
        A[r, idx[t]] = -1
        A[r, nt + a] = 1
    cons = [LinearConstraint(A.tocsr(), -np.inf, 0)]
    cap = np.zeros((1, nt + na))
    cap[0, :nt] = 1
    cons.append(LinearConstraint(cap, -np.inf, B))
    c = np.zeros(nt + na)
    c[nt:] = -1
    res = milp(c=c, constraints=cons, integrality=np.ones(nt + na),
               bounds=(0, 1), options={"time_limit": 10})
    return int(round(-res.fun)) if res.x is not None else 0


def main():
    ds = build_dataset("../data/dev.json")
    seen, rows = set(), []
    for d in ds:
        if d["id"] in seen:
            continue
        seen.add(d["id"])
        ids, spans = tokenize_sample(TOK, d["segs"])
        n = len(ids)
        if not (100 <= n <= 1500):
            continue
        try:
            idxs, _ = extract(d["segs"], d["question"],
                              max_cells=max(1, int((0.85 * int(0.20 * n)) / 6)))
        except Exception:
            continue
        cm = closure_masks(d["segs"], spans)
        masks = [cm[i] for i in idxs if i in cm]
        if len(masks) < 2:
            continue
        rows.append((d, spans, n, masks))

    print("文档 %d 个；锚点数 中位 %d、最大 %d"
          % (len(rows), int(np.median([len(r[3]) for r in rows])),
             max(len(r[3]) for r in rows)))
    print("%-5s %-10s %-10s %-10s %-8s" % ("预算", "现有装配", "边际贪心", "精确最优", "现/最优"))
    for pct in (3, 5, 10, 20):
        cur = gre = opt = 0.0
        for d, spans, n, masks in rows:
            B = max(8, int(pct / 100 * n))
            cur += current_value(d["segs"], spans, masks, B)
            gre += greedy_value(masks, B)
            opt += exact_value(masks, B)
        m = len(rows)
        print("%-5s %-10.2f %-10.2f %-10.2f %-8.1f%%"
              % ("%d%%" % pct, cur / m, gre / m, opt / m, 100 * cur / max(1e-9, opt)))


if __name__ == "__main__":
    main()
