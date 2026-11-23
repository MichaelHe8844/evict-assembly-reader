# -*- coding: utf-8 -*-
"""把 b1_label_*.json（读者反馈）转成 b1_train.py 能吃的行文件。

输出与 r1_pairs_train_full.json 同构，只把 pairs 的第三项从「覆盖 gold」
换成「读者答对」，并只保留标注过的对。
"""
import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="../results/r1_pairs_train_full.json")
    ap.add_argument("--labels", default="../results/b1_label_train.json")
    ap.add_argument("--out", default="../results/b1_train_rows.json")
    args = ap.parse_args()

    rows = {r["qid"]: r for r in json.load(open(args.pairs, encoding="utf-8"))}
    lab = json.load(open(args.labels, encoding="utf-8"))
    by = defaultdict(dict)
    for x in lab:
        key = (tuple(x["pair"][0]), tuple(x["pair"][1]))
        by[x["qid"]][key] = bool(x["ok"])

    out, npos, ntot, miss = [], 0, 0, 0
    for qid, d in by.items():
        r = rows.get(qid)
        if r is None:
            miss += 1
            continue
        cs = r["cands"]
        idx = {(c["r"], c["c"]): k for k, c in enumerate(cs)}
        ps = []
        for (a, b), ok in d.items():
            if a not in idx or b not in idx:
                continue
            ps.append([idx[a], idx[b], ok])
            npos += ok
            ntot += 1
        if len(ps) < 2:
            continue
        out.append(dict(qid=qid, key=r["key"], doc=r["doc"], q=r["q"],
                        gold=r["gold"], cands=cs, pairs=ps))

    json.dump(out, open(args.out, "w"), ensure_ascii=False)
    print("题 %d，对 %d，正例 %d（%.1f%%），缺行 %d"
          % (len(out), ntot, npos, 100 * npos / max(1, ntot), miss))


if __name__ == "__main__":
    main()
