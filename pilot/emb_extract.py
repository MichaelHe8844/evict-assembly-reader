# -*- coding: utf-8 -*-
"""抽取器加一个轻量学习组件：用句向量相似度替代/补足词面规则。

变体（同一批样本、同一阈值搜索口径）：
  rule   现行规则（词面 + 前缀模糊 + 年份 + 子串）
  emb    纯向量 cos(问题, 表头)，关掉全部词面规则
  hyb    max(规则, 向量)
  hyb_w  0.6*规则 + 0.4*向量

指标与论文一致：完全召回（V/R/C/U 四成分全中）、各成分命中率、代价（中位/均值）。
阈值只在调参段选，留出段只做验证、不参与任何选择。
"""
import sys
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, ".")
from common import build_dataset  # noqa: E402
from arm_a import tokenize_sample  # noqa: E402
from anchor_auto import extract, table_index  # noqa: E402

M = "E:/KV_models/bge-small-en-v1.5"
INSTR = "Represent this sentence for searching relevant passages: "

_tok = AutoTokenizer.from_pretrained(M)
_model = AutoModel.from_pretrained(M).cuda().eval()


@torch.no_grad()
def encode(texts, batch=64, is_query=False):
    out = []
    for i in range(0, len(texts), batch):
        chunk = [(INSTR + t) if is_query else t for t in texts[i:i + batch]]
        b = _tok(chunk, padding=True, truncation=True, max_length=64,
                 return_tensors="pt").to("cuda")
        h = _model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
        v = torch.nn.functional.normalize(v, dim=-1)
        out.append(v.cpu().numpy().astype(np.float32))
    return np.concatenate(out, 0)


def headers_of(segs):
    cn, un, rh, cell, nr, nc = table_index(segs)
    return [segs[i]["text"] for i in cn.values()] + [segs[i]["text"] for i in rh.values()]


def prepare(limit, skip=0):
    ds = build_dataset("../data/dev.json")[skip:skip + limit]
    rows = []
    for d in ds:
        doc, spans = tokenize_sample(_tok, d["segs"])
        if len(doc) < 64:
            continue
        gold = set()
        for c in "VRCU":
            gold.update(d["group"].get(c, []))
        if gold:
            rows.append(dict(d=d, L=len(doc), gold=gold, spans=spans))
    all_h = sorted({h for r in rows for h in headers_of(r["d"]["segs"]) if h})
    H = encode(all_h)
    hidx = {h: i for i, h in enumerate(all_h)}
    Q = encode([r["d"]["question"] for r in rows], is_query=True)
    return rows, H, hidx, Q


def evaluate(rows, H, hidx, Q, variant, thresh, emb_w=1.0):
    ok = {c: 0 for c in "VRCU"}
    full = 0
    costs = []
    for r, qv in zip(rows, Q):
        d = r["d"]

        def emb(h, qv=qv):
            i = hidx.get(h)
            return 0.0 if i is None else float(max(0.0, float(qv @ H[i])))

        if variant == "rule":
            auto, _ = extract(d["segs"], d["question"], thresh=thresh)
        elif variant == "emb":
            auto, _ = extract(d["segs"], d["question"], thresh=thresh,
                              emb=emb, use_lex=False, cand2=False)
        else:
            auto, _ = extract(d["segs"], d["question"], thresh=thresh,
                              emb=emb, emb_w=emb_w)
        auto = [i for i in (auto or []) if r["spans"][i][0] >= 0]
        atoks = sum(r["spans"][i][1] - r["spans"][i][0] for i in auto)
        auto = set(auto)
        comp = {}
        for c in "VRCU":
            gg = d["group"].get(c, [])
            comp[c] = True if not gg else all(i in auto for i in gg)
            ok[c] += int(comp[c])
        full += int(all(comp.values()))
        costs.append(atoks / r["L"])
    n = len(rows)
    return dict(n=n, full=full / n, **{c: ok[c] / n for c in "VRCU"},
                cost_med=float(np.median(costs)), cost_mean=float(np.mean(costs)))


def show(tag, r):
    print("%-18s n=%d  完全召回 %.3f | V %.3f R %.3f C %.3f U %.3f | 代价 中位 %.1f%% 均值 %.1f%%" % (
        tag, r["n"], r["full"], r["V"], r["R"], r["C"], r["U"],
        100 * r["cost_med"], 100 * r["cost_mean"]))


def flags(rows, H, hidx, Q, variant, thresh, emb_w=1.0):
    """逐样本"完全召回"布尔值，用于配对检验。"""
    out, detail = [], []
    for r, qv in zip(rows, Q):
        d = r["d"]

        def emb(h, qv=qv):
            i = hidx.get(h)
            return 0.0 if i is None else float(max(0.0, float(qv @ H[i])))

        if variant == "rule":
            auto, diag = extract(d["segs"], d["question"], thresh=thresh)
        else:
            auto, diag = extract(d["segs"], d["question"], thresh=thresh,
                                 emb=emb, emb_w=emb_w)
        auto = set(auto or [])
        ok = all(True if not d["group"].get(c) else all(i in auto for i in d["group"][c])
                 for c in "VRCU")
        out.append(ok)
        detail.append((d["question"], diag, ok))
    return out, detail


def mcnemar(b, a):
    x = sum(1 for p, q in zip(b, a) if p and not q)
    y = sum(1 for p, q in zip(b, a) if q and not p)
    m = x + y
    if not m:
        return x, y, 1.0
    k = min(x, y)
    p = min(1.0, 2 * sum(__import__("math").comb(m, i) for i in range(k + 1)) / 2 ** m)
    return x, y, p


if __name__ == "__main__":
    t0 = time.time()
    rows, H, hidx, Q = prepare(300, 0)
    print("调参段 %d 样本，表头 %d 个，编码 %.1fs\n" % (len(rows), len(H), time.time() - t0))

    base = evaluate(rows, H, hidx, Q, "rule", 0.6)
    show("rule (现状)", base)

    # 纯向量：阈值扫描
    best_emb = None
    for th in (0.45, 0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70):
        r = evaluate(rows, H, hidx, Q, "emb", th)
        show("emb th=%.2f" % th, r)
        if best_emb is None or r["full"] > best_emb[1]["full"]:
            best_emb = (th, r)
    print()

    # 混合：max(规则, w*向量)
    best_hyb = None
    for th in (0.55, 0.60, 0.62, 0.65):
        for w in (0.9, 1.0, 1.1):
            r = evaluate(rows, H, hidx, Q, "hyb", th, emb_w=w)
            show("hyb th=%.2f w=%.1f" % (th, w), r)
            if best_hyb is None or r["full"] > best_hyb[1]["full"]:
                best_hyb = (th, r, w)
    print("\n调参段最优：emb th=%.2f 完全召回 %.3f；hyb th=%.2f w=%.1f 完全召回 %.3f" % (
        best_emb[0], best_emb[1]["full"], best_hyb[0], best_hyb[2], best_hyb[1]["full"]))

    print("\n=== 留出段验证（阈值沿用调参段，不重选）===")
    hr, hH, hidx2, hQ = prepare(448, 400)
    show("rule", evaluate(hr, hH, hidx2, hQ, "rule", 0.6))
    show("emb th=%.2f" % best_emb[0], evaluate(hr, hH, hidx2, hQ, "emb", best_emb[0]))
    show("hyb th=%.2f w=%.1f" % (best_hyb[0], best_hyb[2]),
         evaluate(hr, hH, hidx2, hQ, "hyb", best_hyb[0], emb_w=best_hyb[2]))
    print("\n=== 配对检验（留出段，rule vs hyb）===")
    hr, hH, hidx2, hQ = prepare(448, 400)
    fr, dr = flags(hr, hH, hidx2, hQ, "rule", 0.6)
    fh, dh = flags(hr, hH, hidx2, hQ, "hyb", best_hyb[0], emb_w=best_hyb[2])
    x, y, p = mcnemar(fh, fr)
    print("hyb 对/rule 错 %d，rule 对/hyb 错 %d，McNemar p=%.4f" % (x, y, p))
    k = 0
    for (q, diag, okh), okr in zip(dh, fr):
        if okh and not okr and k < 5:
            print("  [补回] %s" % q[:88])
            print("         选中列 %s | 金标列 %s" % (diag["cols"][:4], "—"))
            k += 1

