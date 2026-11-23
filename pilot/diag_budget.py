# -*- coding: utf-8 -*-
"""
预算分配诊断：20% 预算装不下时，现在的规则丢了什么？

现状（arm_c.fit_budget）：按片段类型排优先级（列名/单位 → 行头 → 单元格 → 单位句），
装不下的**整段丢弃**。问题在于这个顺序与"哪个单元格是答案"无关——
预算一紧，最先被丢的恰恰是单元格，也就是数值本身。

本脚本比较三种打包顺序（都保持"整段、不切碎"）：
  A. 现行：按片段类型优先级
  B. 按分排序：行分+列分高的单元格优先，表头按其所属行列的分数排序
  C. 先保最小锚点：先装入最高分（行,列）的 V+R+C+U，再按分填空

指标：答案锚点四成分全部保留的样本比例（即落地 ANR），以及预算占用。
"""
import json
import re
import argparse
import numpy as np

from common import build_dataset
from anchor_auto import extract, table_index

SINGLE = re.compile(r"^[a-z]+\s*\([^()]*\)$")
PRIO = {"col_name": 0, "unit": 0, "col_header": 0, "row_header": 1,
        "cell": 2, "text": 3, "sep": 9}


def pack(order, spans, B):
    """按给定顺序整段装入，装不下就跳过该段（绝不切碎）。"""
    tok = set()
    for i in order:
        a, b = spans[i]
        if a < 0 or b <= a:
            continue
        if len(tok) + (b - a) > B:
            continue
        tok.update(range(a, b))
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/dev.json")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--tokenizer", default="E:/KV_models/Qwen2.5-3B-Instruct")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from arm_a import tokenize_sample
    tokz = AutoTokenizer.from_pretrained(args.tokenizer)

    ds = build_dataset(args.data)
    raw = {e.get("id"): e for e in json.load(open(args.data, encoding="utf-8"))}
    res = {k: [] for k in ("A", "B", "C")}
    used = []
    n = 0

    for d in ds:
        if n >= args.limit:
            break
        qa = (raw.get(d["id"], {}) or {}).get("qa") or {}
        if not SINGLE.match((qa.get("program") or "").strip()):
            continue
        if not d["group"].get("V"):
            continue
        ids, sp = tokenize_sample(tokz, d["segs"])
        L = len(ids)
        if L < 64 or L > 1500:
            continue
        idx, diag = extract(d["segs"], d["question"])
        rs, cs = diag["rscore"], diag["cscore"]
        cn, un, rh, cell, _, _ = table_index(d["segs"])
        B = max(8, int(0.20 * L))

        def w(i):
            """该片段的重要性分数：单元格取行分+列分，表头取所属行列的最高分。"""
            s = d["segs"][i]
            if s["kind"] == "cell":
                return rs.get(s["row"], 0) + cs.get(s["col"], 0) + 1.0
            if s["kind"] == "row_header":
                return rs.get(s["row"], 0) + 1.5
            if s["kind"] == "col_name":
                return cs.get(s["col"], 0) + 1.5
            if s["kind"] == "unit":
                return 1.2
            return 0.5

        orderA = sorted(idx, key=lambda k: PRIO.get(d["segs"][k]["kind"], 5))
        orderB = sorted(idx, key=lambda k: -w(k))

        # 预测的核心：分数最高的行 × 分数最高的列（不偷看 gold）
        pr = max(rs, key=rs.get) if rs else None
        pc = max(cs, key=cs.get) if cs else None
        core_pred = [i for i in (rh.get(pr), cn.get(pc), un.get(pc)) if i is not None]
        if pr is not None and pc is not None and (pr, pc) in cell:
            core_pred.append(cell[(pr, pc)])
        orderC = core_pred + [k for k in orderB if k not in set(core_pred)]
        # C2：保护 top-2 行 × top-2 列的所有核心
        tr = sorted(rs, key=lambda k: -rs[k])[:2]
        tc = sorted(cs, key=lambda k: -cs[k])[:2]
        core2 = set()
        for i in tr:
            for j in tc:
                if (i, j) in cell:
                    core2.add(cell[(i, j)])
                    if i in rh:
                        core2.add(rh[i])
                    if j in cn:
                        core2.add(cn[j])
        core2 = [i for i in core2 if i in set(idx)]
        orderD = sorted(core2, key=lambda k: -w(k)) + \
                 [k for k in orderB if k not in set(core2)]

        # 仅供参考的上界：用 gold 答案格做核心
        v = d["group"]["V"][0]
        gr, gc = d["segs"][v]["row"], d["segs"][v]["col"]
        core = [i for i in (rh.get(gr), cn.get(gc), un.get(gc)) if i is not None]
        core.append(v)
        orderO = core + [k for k in orderB if k not in set(core)]

        for tag, order in (("A", orderA), ("B", orderB), ("C", orderC),
                           ("D", orderD), ("O", orderO)):
            kept = pack(order, sp, B)
            ok = all(all(t in kept for t in range(*sp[i]))
                     for c in "VRCU" for i in d["group"].get(c, [])
                     if sp[i][0] >= 0)
            res.setdefault(tag, []).append(ok)
            if tag == "A":
                used.append(len(kept) / L)
        n += 1

    print("样本 n=%d | 现行规则平均占用 %.3f" % (n, np.mean(used)))
    print("%-34s %10s" % ("打包顺序", "落地 ANR"))
    names = {"A": "A 现行（按片段类型）", "B": "B 按行列分排序",
             "C": "C 先保预测核心（top1 行×列）",
             "D": "D 先保预测核心（top2×top2）",
             "O": "O 先保 gold 核心（上界参考）"}
    for k in ("A", "B", "C", "D", "O"):
        a = np.array(res[k])
        print("%-34s %10.3f" % (names[k], a.mean()))

    # 配对比较
    a = np.array(res["A"])
    for k in ("C", "D"):
        c = np.array(res[k])
        print("\nA→%s：变好 %d，变差 %d，不变 %d" % (
            k, int((c & ~a).sum()), int((a & ~c).sum()), int((a & c).sum())))
    ov = np.array(used) > 0.205
    print("现行规则下占用超 20%% 的样本：%d (%.1f%%)" % (ov.sum(), 100 * ov.mean()))


if __name__ == "__main__":
    main()
