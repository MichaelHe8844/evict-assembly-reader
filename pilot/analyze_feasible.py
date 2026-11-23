# -*- coding: utf-8 -*-
"""学习组件的三种用法对比：无约束 / 预算可行（BFOPS） / 换打分器。

读：
  results/fp_{af,x}{3,5,10,20}.json        可行选择的强制对
  results/fp_{af,x}free100.json            无约束（取分数最高的一对）
  results/fix_fp{B}.json                   可行选择的端到端
  results/fix_af{B}.json                   无约束（部署版选择器）的端到端
  results/fix_free_{dev,test}5.json        换成单格打分器后的端到端
"""
import json
import math


def load(p, c):
    d = json.load(open(p, encoding="utf-8"))
    return {(r["id"], round(float(r["gold"]), 6)): r for r in d["records"] if r["cond"] == c}


def mcn(a, b):
    x = sum(1 for k in a if b[k] and not a[k])
    y = sum(1 for k in a if a[k] and not b[k])
    m = x + y
    if not m:
        return x, y, 1.0
    k = min(x, y)
    return x, y, min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)


def sel_acc(fp, rows):
    ok = n = 0
    for k, v in fp.items():
        qid, g = k.split("|")
        r = rows.get((qid, float(g)))
        if not r:
            continue
        gold = {(x[0], x[1]) for x in r["gold"]}
        pick = {(x[0], x[1]) for x in v}
        ok += gold <= pick
        n += 1
    return ok, n


def main():
    dev = {(r["qid"], round(float(r["key"][1]), 6)): r
           for r in json.load(open("../results/r1_pairs_dev_full.json", encoding="utf-8"))}
    test = {(r["qid"], round(float(r["key"][1]), 6)): r
            for r in json.load(open("../results/xds_pairs_test.json", encoding="utf-8"))}
    print("== 选择层：无约束 vs 预算可行 ==")
    for tag, rows, name in (("af", dev, "dev"), ("x", test, "test")):
        fp = json.load(open("../results/fp_%sfree100.json" % tag, encoding="utf-8"))
        ok, n = sel_acc(fp, rows)
        print("%s 无约束 %.3f（%d/%d）" % (name, ok / n, ok, n))
        for B in (3, 5, 10):
            p = "../results/fp_%s%d.json" % (tag, B)
            try:
                fp = json.load(open(p, encoding="utf-8"))
            except FileNotFoundError:
                continue
            ok, n = sel_acc(fp, rows)
            print("   %2d%% 可行 %.3f（%d/%d）" % (B, ok / n, ok, n))

    print("\n== 端到端（dev，5%）==")
    base = load("../results/e2e_af5_base.json", "auto_f@5")
    for name, p in (("部署版选择器", "../results/fix_af5.json"),
                    ("单格打分器", "../results/fix_free_dev5.json"),
                    ("可行选择（BFOPS）", "../results/fix_fp5.json")):
        R = load(p, "auto_f@5")
        ks = [k for k in R if k in base]
        em = sum(1 for k in ks if R[k]["correct"]) / len(ks)
        x, y, pv = mcn({k: bool(base[k]["correct"]) for k in ks},
                       {k: bool(R[k]["correct"]) for k in ks})
        print("%-16s EM %.3f（+%d/-%d vs 基线，p=%.3g）" % (name, em, x, y, pv))
    a3 = load("../results/fix_fp3.json", "auto_f@3")
    b3 = load("../results/fix_af3.json", "auto_f@3")
    print("  3%% 档：可行选择 %.3f vs 部署版 %.3f"
          % (sum(1 for k in a3.values() if k["correct"]) / len(a3),
             sum(1 for k in b3.values() if k["correct"]) / len(b3)))

    print("\n== test 5%：换打分器 ==")
    base = load("../results/x5_base.json", "auto_f@5")
    for name, p in (("部署版", "../results/fix_x5.json"), ("单格打分器", "../results/fix_free_test5.json")):
        R = load(p, "auto_f@5")
        ks = [k for k in R if k in base]
        print("%-10s EM %.3f" % (name, sum(1 for k in ks if R[k]["correct"]) / len(ks)))


if __name__ == "__main__":
    main()
