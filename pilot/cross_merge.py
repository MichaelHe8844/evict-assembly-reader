# -*- coding: utf-8 -*-
"""合并分块跑出来的 arm_c 结果，并按题配对比较三个条件。

用法：
  python cross_merge.py            # 合并 _pin_c* / _rand_c*，并读 cross_pure216
"""
import glob
import json
import math
import random

BUD = ("3", "5", "10", "20")


def load_chunks(pat):
    recs, n = [], 0
    for f in sorted(glob.glob("../results/" + pat)):
        d = json.load(open(f, encoding="utf-8"))
        recs += d["records"]
        n += d.get("n", 0)
    return recs


def by_cond(recs):
    out = {}
    for r in recs:
        b = r["cond"].split("@")[-1]
        out.setdefault(b, {})[r["id"]] = bool(r["correct"])
    return out


def boot(a, b, B=4000, seed=0):
    ids = [i for i in a if i in b]
    by = {}
    for i in ids:
        by.setdefault(i.rsplit("-", 1)[0], []).append((a[i], b[i]))
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
    return obs, vals[100], vals[3899]


def flip(a, b):
    ids = [i for i in a if i in b]
    plus = sum(1 for i in ids if b[i] and not a[i])
    minus = sum(1 for i in ids if a[i] and not b[i])
    n = plus + minus
    k = min(plus, minus)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    return plus, minus, p


def main():
    pure = by_cond(json.load(open("../results/cross_pure216.json",
                                  encoding="utf-8"))["records"])
    pin = by_cond(load_chunks("_pin_c*.json"))
    rand = by_cond(load_chunks("_rand_c*.json"))
    json.dump({"pure": pure, "pin": pin, "rand": rand},
              open("../results/cross216_merged.json", "w"), ensure_ascii=False)

    print("%-6s %10s %10s %10s" % ("预算", "SnapKV", "+真钉对", "+随机对"))
    for b in BUD:
        row = "%-6s" % (b + "%")
        for d in (pure, pin, rand):
            v = d.get(b, {})
            row += "%10.3f" % (sum(v.values()) / max(1, len(v)))
        print(row)

    print("\n配对差（按文档聚类 95% CI）")
    for b in BUD:
        for name, d in (("真钉对−纯", pin), ("随机对−纯", rand)):
            if b not in d or b not in pure:
                continue
            o, lo, hi = boot(pure[b], d[b])
            plus, minus, p = flip(pure[b], d[b])
            print("  %-4s %-10s %+.3f [%+.3f, %+.3f]  +%d/-%d  p=%.1e  n=%d"
                  % (b + "%", name, o, lo, hi, plus, minus, p,
                     len([i for i in pure[b] if i in d[b]])))


if __name__ == "__main__":
    main()
