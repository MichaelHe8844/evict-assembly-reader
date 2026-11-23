# -*- coding: utf-8 -*-
"""表 5/表 6 的六列与全部配对差（216 题，按文档聚类的 95% CI + McNemar 翻转数）。

来源：
  旧·LM   results/vf216_new.json      （arm_c，表头优先）
  新·LM   results/vf216_sel_new.json  （arm_c，钉住一对）
  旧/新·程序 results/vf216_gold_vf.json、vf216_rft_vf.json（oracle_ceiling，cond=base/force）

用法：python ci216.py
"""
import json
import math
import random

BUD = ("3", "5", "10", "20")


def doc_of(qid):
    return qid.rsplit("-", 1)[0]


def load_armc(path):
    d = json.load(open(path, encoding="utf-8"))["records"]
    out = {b: {} for b in BUD}
    for r in d:
        b = r["cond"].split("@")[-1]
        if b in out:
            out[b][r["id"]] = bool(r["correct"])
    return out


def load_oc(path):
    d = json.load(open(path, encoding="utf-8"))["detail"]
    out = {"base": {b: {} for b in BUD}, "force": {b: {} for b in BUD}}
    for b in BUD:
        for r in d[b]:
            out[r["cond"]][b][r["id"]] = bool(r["ok"])
    return out


def boot(a, b, B=4000, seed=0):
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
        for dd in (docs[rng.randrange(len(docs))] for _ in docs):
            for x, y in by[dd]:
                num += y - x
                den += 1
        vals.append(num / den)
    vals.sort()
    return obs, vals[int(0.025 * B)], vals[int(0.975 * B)]


def flip(a, b):
    ids = [i for i in a if i in b]
    plus = sum(1 for i in ids if b[i] and not a[i])
    minus = sum(1 for i in ids if a[i] and not b[i])
    n = plus + minus
    k = min(plus, minus)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    return plus, minus, p


def em(m):
    v = list(m.values())
    return sum(v) / len(v) if v else float("nan")


def compare(name, a, b):
    print("  %-28s %+.3f [%+.3f, %+.3f]  +%d/-%d  p=%.3f"
          % (name, *boot(a, b)[:1], *boot(a, b)[1:], 0) if False else "", end="")
    d, lo, hi = boot(a, b)
    plus, minus, p = flip(a, b)
    print("  %-26s %+.3f [%+.3f, %+.3f]  +%d/-%d  p=%.3f"
          % (name, d, lo, hi, plus, minus, p))


def main():
    old_lm = load_armc("../results/vf216_new.json")
    new_lm = load_armc("../results/vf216_sel_new.json")
    gold = load_oc("../results/vf216_gold_vf.json")
    efst = load_oc("../results/vf216_rft_vf.json")

    col = {
        "旧·语言模型": {b: old_lm[b] for b in BUD},
        "旧·程序（gold）": {b: gold["base"][b] for b in BUD},
        "旧·程序（EFST）": {b: efst["base"][b] for b in BUD},
        "新·语言模型": {b: new_lm[b] for b in BUD},
        "新·程序（gold）": {b: gold["force"][b] for b in BUD},
        "新·程序（EFST）": {b: efst["force"][b] for b in BUD},
    }
    ks = list(col)
    print("%-6s" % "预算" + "".join("%18s" % k for k in ks))
    for b in BUD:
        print("%-6s" % (b + "%") + "".join("%18.3f" % em(col[k][b]) for k in ks))
    print("%-6s" % "平均" + "".join(
        "%18.3f" % (sum(em(col[k][b]) for b in BUD) / 4) for k in ks))

    print("\n[装配] 旧 -> 新（同一读者/同一模型，只换装配）")
    for k in ("旧·语言模型", "旧·程序（gold）", "旧·程序（EFST）"):
        k2 = k.replace("旧", "新")
        for b in BUD:
            compare("%s %s%%" % (k, b), col[k][b], col[k2][b])

    print("\n[读者] 同装配 LM -> 程序（EFST）")
    for pre in ("旧", "新"):
        for b in BUD:
            compare("%s %s%%" % (pre, b), col[pre + "·语言模型"][b],
                    col[pre + "·程序（EFST）"][b])

    print("\n[监督] gold -> EFST")
    for pre in ("旧", "新"):
        for b in BUD:
            compare("%s %s%%" % (pre, b), col[pre + "·程序（gold）"][b],
                    col[pre + "·程序（EFST）"][b])


if __name__ == "__main__":
    main()
