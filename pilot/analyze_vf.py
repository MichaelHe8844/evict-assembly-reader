"""重跑（先保数值）后的汇总：5.2 / 5.10 / 5.11。

旧口径文件保留，逐表对比，便于在论文里只改数字不改结构。
"""
import collections
import json
import math
import os

R = "../results/"


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    out = collections.defaultdict(dict)
    for r in d["records"]:
        out[r["id"]].setdefault(r["cond"], r)
    return out


def mcn(A, B, ids):
    x = sum(1 for k in ids if B[k] and not A[k])
    y = sum(1 for k in ids if A[k] and not B[k])
    m = x + y
    if m == 0:
        return x, y, 1.0
    kk = min(x, y)
    return x, y, min(1.0, 2 * sum(math.comb(m, i) for i in range(kk + 1)) / 2 ** m)


def em(recs):
    return sum(1 for r in recs if r["correct"]) / len(recs)


def anr(recs):
    return sum(1 for r in recs if r.get("anr")) / len(recs)


def sec1():
    print("== 5.2 主表（7B，20%）==")
    p = R + "vf_arith_main.json"
    if not os.path.exists(p):
        print("  (未完成)"); return
    N = load(p)
    O = load(R + "arith_big.json") if os.path.exists(R + "arith_big.json") else {}
    for c in ("full", "auto_f@20", "auto_n@20"):
        ids = [k for k in N if c in N[k]]
        if not ids:
            continue
        line = "  %-10s 新 %.3f (ANR %.3f)" % (c, em([N[k][c] for k in ids]),
                                              anr([N[k][c] for k in ids]))
        if O and all(c in O.get(k, {}) for k in ids):
            line += "   旧 %.3f (ANR %.3f)" % (em([O[k][c] for k in ids]),
                                              anr([O[k][c] for k in ids]))
        print(line + "   n=%d" % len(ids))


def sec2():
    print("\n== 5.10 dev 预算扫描（126 条，3B）==")
    print("  %-4s %-9s %-9s %-9s %-9s %-8s %s" %
          ("档", "基线新", "基线旧", "强制新", "强制旧", "整对装入", "配对"))
    for B in (3, 5, 10, 20):
        nb, nf = R + "vf_af%d.json" % B, R + "vf_af%d_sel.json" % B
        if not (os.path.exists(nb) and os.path.exists(nf)):
            print("  %d%% (未完成)" % B); continue
        Bn, Fn = load(nb), load(nf)
        c = "auto_f@%d" % B
        ids = [k for k in Fn if c in Fn[k] and k in Bn and c in Bn[k]]
        new_b = em([Bn[k][c] for k in ids]); new_f = em([Fn[k][c] for k in ids])
        fit = sum(1 for k in ids if Fn[k][c].get("pair_fit") == 0) / len(ids)
        x, y, p = mcn({k: Bn[k][c]["correct"] for k in ids},
                      {k: Fn[k][c]["correct"] for k in ids}, ids)
        ob = of = None
        if os.path.exists(R + "e2e_af%d_base.json" % B):
            O = load(R + "e2e_af%d_base.json" % B)
            ks = [k for k in ids if k in O and c in O[k]]
            if ks:
                ob = em([O[k][c] for k in ks])
        if os.path.exists(R + "fix_af%d.json" % B):
            O2 = load(R + "fix_af%d.json" % B)
            ks = [k for k in ids if k in O2 and c in O2[k]]
            if ks:
                of = em([O2[k][c] for k in ks])
        print("  %-4s %.3f     %-9s %.3f     %-9s %.1f%%     +%d/-%d p=%.2g" %
              ("%d%%" % B, new_b, ("%.3f" % ob) if ob is not None else "—",
               new_f, ("%.3f" % of) if of is not None else "—",
               100 * fit, x, y, p))


def sec3():
    print("\n== 5.10 5% 对照 ==")
    for tag, path, old in (("7% 纯预算", "vf_af7.json", "e2e_af7.json"),
                           ("随机两格", "vf_af5_rand.json", "fix_af5_rand.json"),
                           ("手写规则", "vf_af5_jr.json", "fix_af5_jr.json"),
                           ("种子 s1", "vf_af5_s1.json", "fix_af5_s1.json")):
        p = R + path
        if not os.path.exists(p):
            print("  %-10s (未完成)" % tag); continue
        N = load(p)
        c = [k for k in N[list(N)[0]] if k.startswith("auto_f")][0]
        ids = [k for k in N if c in N[k]]
        s = "  %-10s %.3f" % (tag, em([N[k][c] for k in ids]))
        if os.path.exists(R + old):
            O = load(R + old)
            ks = [k for k in ids if k in O and c in O[k]]
            if ks:
                s += "   旧 %.3f" % em([O[k][c] for k in ks])
        print(s + "   n=%d" % len(ids))


def sec4():
    print("\n== 5.11 test（182 条，3B）==")
    for B in (3, 5, 10):
        nb, nf = R + "vf_x%d.json" % B, R + "vf_x%d_sel.json" % B
        if not (os.path.exists(nb) and os.path.exists(nf)):
            print("  %d%% (未完成)" % B); continue
        Bn, Fn = load(nb), load(nf)
        c = "auto_f@%d" % B
        ids = [k for k in Fn if c in Fn[k] and k in Bn and c in Bn[k]]
        x, y, p = mcn({k: Bn[k][c]["correct"] for k in ids},
                      {k: Fn[k][c]["correct"] for k in ids}, ids)
        print("  %-4s 基线 %.3f → 强制 %.3f (%+.3f)  +%d/-%d p=%.2g  整对装入 %.1f%%" %
              ("%d%%" % B, em([Bn[k][c] for k in ids]), em([Fn[k][c] for k in ids]),
               em([Fn[k][c] for k in ids]) - em([Bn[k][c] for k in ids]), x, y, p,
               100 * sum(1 for k in ids if Fn[k][c].get("pair_fit") == 0) / len(ids)))


if __name__ == "__main__":
    sec1(); sec2(); sec3(); sec4()
