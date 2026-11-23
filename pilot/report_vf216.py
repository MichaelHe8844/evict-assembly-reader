# -*- coding: utf-8 -*-
"""把 216 题两装配的评测结果汇成表 5 的六列，并给出配对差与按文档聚类的 95% CI。

用法：
  python report_vf216.py vf216_gold_vf vf216_rft_vf
（每个名字对应 results/<name>.json，须含 cond=base/force 的 detail）
"""
import json
import random
import sys

BUD = ("3", "5", "10", "20")


def doc_of(qid):
    return qid.rsplit("-", 1)[0]


def load(name):
    d = json.load(open("../results/%s.json" % name, encoding="utf-8"))
    out = {}
    for b in BUD:
        out[b] = {}
        for r in d["detail"][b]:
            out[b][r["id"]] = dict(cond=r["cond"], ok=bool(r["ok"]))
    # 拆成 {cond: {id: ok}}
    res = {}
    for b in BUD:
        for cond in ("base", "force"):
            res.setdefault(cond, {})[b] = {
                i: v["ok"] for i, v in out[b].items() if v["cond"] == cond}
    return res


def boot(a, b, B=2000, seed=0):
    """按文档聚类的配对 bootstrap，返回 (obs, lo, hi)。a、b: {id: 0/1}"""
    ids = [i for i in a if i in b]
    by = {}
    for i in ids:
        by.setdefault(doc_of(i), []).append((a[i], b[i]))
    docs = list(by)
    rng = random.Random(seed)
    obs = sum(b[i] - a[i] for i in ids) / len(ids)
    vals = []
    for _ in range(B):
        num = den = 0
        for d in (docs[rng.randrange(len(docs))] for _ in docs):
            for x, y in by[d]:
                num += y - x
                den += 1
        vals.append(num / den)
    vals.sort()
    return obs, vals[int(0.025 * B)], vals[int(0.975 * B)]


def em(m, b):
    v = list(m[b].values())
    return sum(v) / len(v) if v else float("nan")


def main():
    names = sys.argv[1:] or ["vf216_gold_vf", "vf216_rft_vf"]
    models = {n: load(n) for n in names}

    hdr = "%-10s" % "预算"
    for n in names:
        hdr += " | %-22s" % n
    print(hdr + "   (旧装配 / 新装配，均值在末行)")
    print("-" * len(hdr))
    for b in BUD:
        line = "%-10s" % (b + "%")
        for n in names:
            m = models[n]
            line += " | %.3f  /  %.3f      " % (em(m["base"], b), em(m["force"], b))
        print(line)
    line = "%-10s" % "平均"
    for n in names:
        m = models[n]
        line += " | %.3f  /  %.3f      " % (
            sum(em(m["base"], b) for b in BUD) / 4,
            sum(em(m["force"], b) for b in BUD) / 4)
    print(line)

    if len(names) == 2:
        a, b = names
        print("\n配对差（%s − %s，同装配同预算；按文档聚类 95% CI）" % (b, a))
        for cond in ("base", "force"):
            print(" [%s]" % ("旧装配" if cond == "base" else "新装配"))
            for bd in BUD:
                da = models[a][cond][bd]
                db = models[b][cond][bd]
                d, lo, hi = boot(da, db)
                ids = [i for i in da if i in db]
                plus = sum(1 for i in ids if db[i] and not da[i])
                minus = sum(1 for i in ids if da[i] and not db[i])
                print("  %-4s %+.3f [%+.3f, %+.3f]  +%d/-%d  n=%d"
                      % (bd + "%", d, lo, hi, plus, minus, len(ids)))

    print("\n各装配的逐档样本量：",
          {n: {b: len(models[n]["base"][b]) for b in BUD} for n in names})


if __name__ == "__main__":
    main()
