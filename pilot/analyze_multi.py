"""多步端到端结果分析。

核心问题：单步上「只保数字」与本文打平（0.410 vs 0.402），多步上是否拉开？
观察指标：
  1) 各条件 EM / ANR（全体 181 条，以及 2 步 / ≥3 步分层）
  2) 本文 vs 只保数字 的配对检验（记录级 + 文档级 cluster bootstrap）
  3) 本文与全上下文的差距（多步应比单步更依赖证据齐备）
"""
import collections
import json
import math
import random

RES = "results/multi_big.json"
KEYS = "results/multi_keys.json"
DOCS = "data/dev.json"


def load():
    d = json.load(open(RES, encoding="utf-8"))
    R = collections.defaultdict(dict)
    for r in d["records"]:
        R[r["id"]].setdefault(r["cond"], r)
    return R


def steps_map():
    import re
    raw = {e["id"]: e for e in json.load(open(DOCS, encoding="utf-8"))}
    out = {}
    for i, e in raw.items():
        p = ((e.get("qa") or {}).get("program") or "").strip()
        if p:
            out[i] = len(re.findall(r"[a-z_]+\(", p))
    return out


def mcn(R, a, b):
    ids = [k for k in R if a in R[k] and b in R[k]]
    x = sum(1 for k in ids if R[k][b]["correct"] and not R[k][a]["correct"])
    y = sum(1 for k in ids if R[k][a]["correct"] and not R[k][b]["correct"])
    m = x + y
    if m == 0:
        return x, y, 1.0
    k = min(x, y)
    return x, y, min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)


def cluster_boot(R, a, b, n=4000):
    """按文档重采样，返回 Δ 的 95% 区间（b 相对 a）与 p。"""
    doc = {k: k.rsplit("-", 1)[0] for k in R}
    by = collections.defaultdict(list)
    for k in R:
        if a in R[k] and b in R[k]:
            by[doc[k]].append(k)
    docs = list(by)
    if len(docs) < 3:
        return None
    diffs = []
    for _ in range(n):
        s = [by[random.choice(docs)] for _ in docs]
        ks = [k for g in s for k in g]
        diffs.append(sum(R[k][b]["correct"] - R[k][a]["correct"] for k in ks) / len(ks))
    diffs.sort()
    lo, hi = diffs[int(0.025 * n)], diffs[int(0.975 * n)]
    pos = sum(1 for x in diffs if x <= 0) / n
    p = 2 * min(pos, 1 - pos)
    ks = [k for g in by.values() for k in g]
    delta = sum(R[k][b]["correct"] - R[k][a]["correct"] for k in ks) / len(ks)
    return delta, lo, hi, max(p, 1.0 / n)


def main():
    R = load()
    st = steps_map()
    conds = [c for c in ("full", "snap@20", "h2o@20", "auto_f@20", "numb@20")
             if all(c in R[k] for k in R)]
    ids = [k for k in R if all(c in R[k] for c in conds)]
    print("样本 %d 条，条件 %s\n" % (len(ids), conds))

    print("%-10s %6s %6s %6s %6s %6s" % ("条件", "EM", "2步", "≥3步", "ANR", "数值全留"))
    for c in conds:
        e2 = [k for k in ids if st.get(k, 0) == 2]
        e3 = [k for k in ids if st.get(k, 0) >= 3]
        em = sum(R[k][c]["correct"] for k in ids) / len(ids)
        f2 = sum(R[k][c]["correct"] for k in e2) / max(1, len(e2))
        f3 = sum(R[k][c]["correct"] for k in e3) / max(1, len(e3))
        anr = sum(R[k][c].get("anr", False) for k in ids) / len(ids)
        num = sum(R[k][c].get("num", False) for k in ids) / len(ids)
        print("%-10s %6.3f %6.3f %6.3f %6.3f %8.3f  (2步 %d / ≥3步 %d)"
              % (c, em, f2, f3, anr, num, len(e2), len(e3)))

    print("\n配对检验（记录级 McNemar）")
    for a, b in (("full", "auto_f@20"), ("snap@20", "auto_f@20"), ("h2o@20", "auto_f@20"),
                 ("numb@20", "auto_f@20"), ("full", "snap@20"), ("auto_f@20", "numb@20")):
        if a in conds and b in conds:
            x, y, p = mcn(R, a, b)
            print("  %-9s → %-9s  %+d/-%+d  p=%.3g" % (a, b, x, y, p))

    print("\n文档级 cluster bootstrap（b 相对 a 的 Δ，95%% 区间）")
    for a, b in (("full", "auto_f@20"), ("snap@20", "auto_f@20"),
                 ("numb@20", "auto_f@20"), ("full", "numb@20")):
        if a in conds and b in conds:
            r = cluster_boot(R, a, b)
            if r:
                print("  %-9s → %-9s  Δ=%+.3f  CI[%+.3f, %+.3f]  p=%.3g"
                      % (a, b, r[0], r[1], r[2], r[3]))


if __name__ == "__main__":
    main()
