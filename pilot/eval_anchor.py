# -*- coding: utf-8 -*-
"""评估无监督锚点抽取：与 gold 锚点组比对，并统计 token 代价。"""
import argparse
import numpy as np
from common import build_dataset
from arm_a import tokenize_sample
from anchor_auto import extract, probe_question


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/dev.json")
    ap.add_argument("--model", default="E:/KV_models/Qwen2.5-3B-Instruct")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", default="../results/anchor_auto_eval.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    ds = build_dataset(args.data)[: args.limit]

    res = {"probe": [], "finqa": []}
    shown = 0

    for d in ds:
        doc, spans = tokenize_sample(tok, d["segs"])
        L = len(doc)
        if L < 64:
            continue
        gold = set()
        for c in "VRCU":
            gold.update(d["group"].get(c, []))
        if not gold:
            continue
        gtoks = sum(spans[i][1] - spans[i][0] for i in gold if spans[i][0] >= 0)

        qs = {"probe": probe_question(d["segs"], d["group"]), "finqa": d["question"]}
        for tag, q in qs.items():
            if not q:
                continue
            auto, diag = extract(d["segs"], q)
            if not auto:
                continue
            atoks = sum(spans[i][1] - spans[i][0] for i in auto if spans[i][0] >= 0)
            inter = len(gold & set(auto))
            itoks = sum(spans[i][1] - spans[i][0] for i in (gold & set(auto)) if spans[i][0] >= 0)
            comp = {}
            for c in "VRCU":
                gg = d["group"].get(c, [])
                comp[c] = None if not gg else all(i in auto for i in gg)
            res[tag].append(dict(
                comp=comp,
                recall_seg=inter / len(gold),
                prec_seg=inter / len(auto),
                recall_tok=itoks / max(1, gtoks),
                cost_tok=atoks / L,
                gtoks=gtoks, atoks=atoks, L=L,
                n_rows=len(diag["rows"]), n_cols=len(diag["cols"]),
                n_pairs=diag["n_pairs"],
                fb_row=diag["fallback_row"], fb_col=diag["fallback_col"]))

            if shown < 3 and tag == "probe":
                shown += 1
                print("\n--- 例 %d: %s" % (shown, q))
                print("    选列 %s / 选行 %s / 单元格 %d" % (
                    diag["cols"], diag["rows"][:6], diag["n_pairs"]))
                print("    金标列 %s 行 %s" % (d["group"]["C"], d["group"]["R"]))

    print("\n%-7s %5s %8s %8s %8s %9s %8s" % (
        "task", "n", "recall@seg", "prec@seg", "recall@tok", "cost/doc", "rows"))
    out = {}
    for tag in ("probe", "finqa"):
        r = res[tag]
        if not r:
            continue
        f = lambda k: float(np.median([x[k] for x in r]))
        out[tag] = {k: f(k) for k in
                    ("recall_seg", "prec_seg", "recall_tok", "cost_tok",
                     "n_rows", "n_cols", "n_pairs")}
        out[tag]["n"] = len(r)
        out[tag]["full_recall_rate"] = float(np.mean([x["recall_tok"] >= 0.999 for x in r]))
        out[tag]["fb_row_rate"] = float(np.mean([x["fb_row"] for x in r]))
        out[tag]["comp"] = {}
        for c in "VRCU":
            vals = [x["comp"][c] for x in r if x["comp"][c] is not None]
            out[tag]["comp"][c] = float(np.mean(vals)) if vals else None
        print("%-7s %5d %8.3f %8.3f %8.3f %9.3f %8.1f" % (
            tag, len(r), out[tag]["recall_seg"], out[tag]["prec_seg"],
            out[tag]["recall_tok"], out[tag]["cost_tok"], out[tag]["n_rows"]))
        print("        完全召回率(逐样本) %.3f  行回退率 %.3f" % (
            out[tag]["full_recall_rate"], out[tag]["fb_row_rate"]))
        # 分成分命中率
        line = "        分成分命中:"
        for c in "VRCU":
            vals = [x["comp"][c] for x in r if x["comp"][c] is not None]
            if vals:
                line += "  %s=%.3f(n=%d)" % (c, np.mean(vals), len(vals))
        print(line)

    import json
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n写入", args.out)


if __name__ == "__main__":
    main()
