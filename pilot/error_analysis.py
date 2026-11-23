# -*- coding: utf-8 -*-
"""
误差分解：把「抽取是否成功」与「最终是否答对」对齐，并列出可读的失败样例。

人群与 cand2_confirm 一致：单步、len<=1200、前 150 个。
输出：
  1. 2x2 表（抽取 OK / 抽取失败 × 答对 / 答错）
  2. 抽取失败的细分（选错行 / 选错列 / 行列都对但没装进）
  3. 若给 --dump N，打印 N 个失败样例供人工归类
"""
import argparse
import json
import re

import numpy as np

from common import build_dataset
from anchor_auto import extract, table_index
from arm_c import _nums
from arm_a import tokenize_sample

SINGLE = re.compile(r"^[a-z]+\s*\([^()]*\)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="../results/cand2_confirm_rescored.json")
    ap.add_argument("--cond", default="auto_f@20")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--maxlen", type=int, default=1200)
    ap.add_argument("--dump", type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("E:/KV_models/Qwen2.5-3B-Instruct")

    ds = build_dataset("../data/dev.json")
    raw = {e.get("id"): e for e in json.load(open("../data/dev.json", encoding="utf-8"))}

    rows = []
    for d in ds:
        if len(rows) >= args.limit:
            break
        qa = (raw.get(d["id"], {}) or {}).get("qa") or {}
        if not SINGLE.match((qa.get("program") or "").strip()):
            continue
        try:
            float(d["exe_ans"])
        except Exception:
            continue
        ids, sp = tokenize_sample(tok, d["segs"])
        if not (64 <= len(ids) <= args.maxlen):
            continue
        rows.append(d)

    preds = [r for r in json.load(open(args.pred, encoding="utf-8"))["records"]
             if r["cond"] == args.cond]
    n = min(len(rows), len(preds))
    print("对齐样本 n=%d（抽取侧 %d / 结果侧 %d）\n" % (n, len(rows), len(preds)))

    bucket = {"OK": [], "A_col": [], "A_row": [], "A_both": [], "A_nocell": []}
    # 三级分解：V 丢失 / V 在但闭包不全 / 齐全但答错
    lvl = {"L1_V_missing": [], "L2_closure_gap": [], "L3_complete_wrong": [],
           "L3_complete_right": []}
    gaps = []
    detail = []

    for i in range(n):
        d, p = rows[i], preds[i]
        v = d["group"]["V"][0]
        gr, gc = d["segs"][v]["row"], d["segs"][v]["col"]
        idx, diag = extract(d["segs"], d["question"])
        cols, rws = set(diag["cols"]), set(diag["rows"])
        if v in idx:
            cat = "OK"
        elif gr in rws and gc in cols:
            cat = "A_nocell"          # 行列都对，但该格未装入（表结构/预算）
        elif gc in cols:
            cat = "A_row"             # 列对、行错
        elif gr in rws:
            cat = "A_col"             # 行对、列错
        else:
            cat = "A_both"
        bucket[cat].append(p["correct"])

        # 操作数覆盖：程序里出现的数值 → 表格单元格 → 是否都在抽取集内
        cn2, _, _, cmap2, _, _ = table_index(d["segs"])
        want = _nums(qa.get("program") or "")
        ops = [si for (rr, cc), si in cmap2.items() if _nums(d["segs"][si]["text"]) & want]
        covall = all(si in idx for si in ops) if ops else True
        if v not in idx:
            lvl["L1_V_missing"].append(p["correct"])
        elif not covall:
            lvl["L2_closure_gap"].append(p["correct"])
            gaps.append(len([si for si in ops if si not in idx]))
        elif p["correct"]:
            lvl["L3_complete_right"].append(True)
        else:
            lvl["L3_complete_wrong"].append(False)
        detail.append((i, cat, d, p, gr, gc, diag))

    print("== 抽取状态 × 答对 ==")
    ok = np.array(bucket["OK"])
    bad = np.array([x for k in bucket if k != "OK" for x in bucket[k]])
    print("  抽取成功  n=%3d  答对率 %.3f" % (len(ok), ok.mean() if len(ok) else float("nan")))
    print("  抽取失败  n=%3d  答对率 %.3f" % (len(bad), bad.mean() if len(bad) else float("nan")))
    print("\n== 抽取失败的细分 ==")
    for k, name in (("A_col", "选错列"), ("A_row", "选错行"),
                    ("A_both", "行列都错"), ("A_nocell", "行列对但格未装入")):
        v2 = np.array(bucket[k])
        if len(v2):
            print("  %-12s n=%3d  答对率 %.3f" % (name, len(v2), v2.mean()))

    print("\n== 三级分解（按操作数是否齐全） ==")
    for k, name in (("L1_V_missing", "① 答案格未保留"),
                    ("L2_closure_gap", "② 答案格在、但其余操作数缺"),
                    ("L3_complete_right", "③ 操作数齐全且答对"),
                    ("L3_complete_wrong", "④ 操作数齐全但答错")):
        v2 = lvl[k]
        print("  %-22s n=%3d (%.1f%%)" % (name, len(v2), 100.0 * len(v2) / n))
    if gaps:
        print("  ② 平均缺 %.1f 个操作数" % np.mean(gaps))

    print("\n== 失败样例 ==")
    shown = 0
    for (i, cat, d, p, gr, gc, diag) in detail:
        if p["correct"] or shown >= args.dump:
            continue
        cn, un, rh, cell, _, _ = table_index(d["segs"])
        print("[%s] %s" % (cat, d["question"][:88]))
        print("     gold=%s  pred=%s" % (p["gold"], p["pred"]))
        print("     gold 行=%s | 列=%s" % (
            d["segs"][rh[gr]]["text"][:40] if gr in rh else "—",
            d["segs"][cn[gc]]["text"][:40] if gc in cn else "—"))
        print("     选中行=%s | 选中列=%s" % (
            [d["segs"][rh[j]]["text"][:18] for j in list(set(diag["rows"]))[:3] if j in rh],
            [d["segs"][cn[j]]["text"][:18] for j in list(set(diag["cols"]))[:3] if j in cn]))
        shown += 1


if __name__ == "__main__":
    main()
