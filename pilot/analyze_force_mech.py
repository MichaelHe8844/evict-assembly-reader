# -*- coding: utf-8 -*-
"""R1 强制保留的机制分解：增益是否集中在「那一对本会被丢掉」的样本上。

若增益只出现在 outside 组，说明起作用的是「把会丢的证据钉住」；
若 inside 组也涨，则说明另有来源（多给的 token、提示变短等），需要另找解释。
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from transformers import AutoTokenizer  # noqa: E402
from common import build_dataset  # noqa: E402
from arm_a import tokenize_sample  # noqa: E402
from anchor_auto import extract  # noqa: E402
from arm_c import fit_budget  # noqa: E402


def mcn(a, b):
    import math
    x = sum(1 for p, q in zip(a, b) if p and not q)
    y = sum(1 for p, q in zip(a, b) if q and not p)
    m = x + y
    if not m:
        return x, y, 1.0
    k = min(x, y)
    return x, y, min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)


def main():
    frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    bp = sys.argv[2] if len(sys.argv) > 2 else "../results/e2e_af5_base.json"
    sp = sys.argv[3] if len(sys.argv) > 3 else "../results/e2e_af5_sel.json"
    base = json.load(open(bp, encoding="utf-8"))
    sel = json.load(open(sp, encoding="utf-8"))
    A = {(r["id"], round(float(r["gold"]), 6)): r for r in base["records"]}
    B = {(r["id"], round(float(r["gold"]), 6)): r for r in sel["records"]}
    force = json.load(open("../results/r1_force.json", encoding="utf-8"))

    tok = AutoTokenizer.from_pretrained("E:/KV_models/Qwen2.5-3B-Instruct")
    ds = {}
    for d in build_dataset("../data/dev.json"):
        try:
            ds[d["id"]] = d
        except Exception:
            pass

    INSIDE, OUTSIDE = [], []
    for k in A:
        if k not in B:
            continue
        d = ds.get(k[0])
        if d is None:
            continue
        pair = force.get(k[0])
        if not pair:
            continue
        doc, spans = tokenize_sample(tok, d["segs"])
        Bt = max(8, int(frac * len(doc)))
        mc = max(1, int((0.85 * Bt) / 6))
        idx, _ = extract(d["segs"], d["question"], max_cells=mc)
        keep = set(fit_budget([i for i in idx if spans[i][0] >= 0], d["segs"], spans, Bt))
        kept = set()
        for si, s in enumerate(d["segs"]):
            a, b = spans[si]
            if a >= 0 and all(t in keep for t in range(a, b)):
                kept.add((s.get("row"), s.get("col"), s.get("kind")))
        ok = all((int(r_), int(c_), "cell") in kept for r_, c_ in pair)
        (INSIDE if ok else OUTSIDE).append((A[k]["correct"], B[k]["correct"], k))

    print("预算 %.0f%%，可匹配 %d 条" % (frac * 100, len(INSIDE) + len(OUTSIDE)))
    for name, R in (("基线已保住这一对", INSIDE), ("基线丢了至少一格", OUTSIDE)):
        if not R:
            continue
        a = [x[0] for x in R]
        b = [x[1] for x in R]
        x, y, p = mcn(b, a)
        print("%s n=%d：EM %.3f → %.3f（%+.3f）| +%d/-%d p=%.3g"
              % (name, len(R), np.mean(a), np.mean(b), np.mean(b) - np.mean(a), x, y, p))
    print("分层占比：inside %.3f / outside %.3f"
          % (len(INSIDE) / max(1, len(INSIDE) + len(OUTSIDE)),
             len(OUTSIDE) / max(1, len(INSIDE) + len(OUTSIDE))))


if __name__ == "__main__":
    main()
