# -*- coding: utf-8 -*-
"""对抽取器的几个开关做消融，目标：列名召回 → 0.90，同时控制代价。"""
import numpy as np
from common import build_dataset
from arm_a import tokenize_sample
from anchor_auto import extract


def run(ds, spans_of, tok, thresh, fuzzy, use_year, tag, qkey="finqa", hedge=True):
    C = []; V = []; full = []; cost = []
    for d in ds:
        doc, spans = spans_of[d["id"]]
        L = len(doc)
        q = d["question"] if qkey == "finqa" else d.get("_probe")
        if not q:
            continue
        auto, _ = extract(d["segs"], q, thresh=thresh, fuzzy=fuzzy, use_year=use_year, hedge=hedge)
        if not auto:
            continue
        gold = set()
        for c in "VRCU":
            gold.update(d["group"].get(c, []))
        A = set(auto)
        C.append(all(i in A for i in d["group"]["C"]))
        V.append(all(i in A for i in d["group"]["V"]))
        full.append(gold <= A)
        at = sum(spans[i][1] - spans[i][0] for i in A if spans[i][0] >= 0)
        cost.append(at / L)
    print("%-28s n=%3d  C=%.3f  V=%.3f  full=%.3f  cost=%.3f" % (
        tag, len(C), np.mean(C), np.mean(V), np.mean(full), np.mean(cost)))
    return np.mean(C), np.mean(cost)


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("E:/KV_models/Qwen2.5-3B-Instruct")
    ds = build_dataset("../data/dev.json")[:300]
    spans_of = {}
    for d in ds:
        spans_of[d["id"]] = tokenize_sample(tok, d["segs"])

    from anchor_auto import probe_question
    for d in ds:
        d["_probe"] = probe_question(d["segs"], d["group"])

    print("== FinQA 原始问题 ==")
    run(ds, spans_of, tok, 0.6, False, False, "baseline (0.6)", hedge=False)
    run(ds, spans_of, tok, 0.6, True, True, "+fuzzy +year", hedge=False)
    run(ds, spans_of, tok, 0.6, True, True, "+fuzzy +year +hedge")
    run(ds, spans_of, tok, 0.5, True, True, "+fuzzy +year +hedge th=0.5")
    run(ds, spans_of, tok, 0.4, True, True, "+fuzzy +year +hedge th=0.4")

    print("\n== 探针问题（上界，不应退化）==")
    run(ds, spans_of, tok, 0.6, True, True, "+fuzzy +year +hedge", qkey="probe")


if __name__ == "__main__":
    main()
