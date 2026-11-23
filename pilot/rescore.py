# -*- coding: utf-8 -*-
"""
重新评分：生成结果不变，只修正评分器。
1) 表达式里允许 $ 与千分位逗号（模型常写 '$116114 / $1244659 * 100'）
2) EM 允许百分比/比率两种约定（×100 / ÷100）

用法：python rescore.py ../results/arm_c_real.json
"""
import json
import re
import sys
import numpy as np
from collections import defaultdict

CLEAN = re.compile(r"[\$,\s]")
EXPR = re.compile(r"[-+*/()0-9.\s]{3,200}")


def parse_expr(text):
    if not text:
        return None
    t = text.replace("$", "").replace(",", "")
    m = EXPR.search(t)
    if not m:
        return None
    expr = m.group(0).strip()
    if not re.search(r"\d", expr):
        return None
    try:
        v = eval(expr, {"__builtins__": {}}, {})
    except Exception:
        return None
    if not isinstance(v, (int, float)):
        return None
    v = float(v)
    if v != v or abs(v) == float("inf"):
        return None
    return v


def close(a, g):
    return abs(a - g) <= max(0.01, 0.005 * abs(g))


def em_scale(pred, gold):
    if pred is None:
        return False
    return close(pred, gold) or close(pred / 100.0, gold) or close(pred * 100.0, gold)


def em_strict(pred, gold):
    return pred is not None and close(pred, gold)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "../results/arm_c_real.json"
    d = json.load(open(path, encoding="utf-8"))
    recs = d["records"]

    for r in recs:
        p = parse_expr(r.get("text", ""))
        r["pred2"] = p
        r["ok2"] = em_scale(p, r["gold"])
        r["ok2_strict"] = em_strict(p, r["gold"])

    conds = []
    for r in recs:
        if r["cond"] not in conds:
            conds.append(r["cond"])

    print("%-11s %5s %6s %8s %8s %8s %8s" % (
        "cond", "n", "parse", "EM", "EM_strict", "ANR", "P(c|完整)"))
    out = []
    for c in conds:
        rs = [r for r in recs if r["cond"] == c]
        n = len(rs)
        parse = np.mean([r["pred2"] is not None for r in rs])
        acc = np.mean([r["ok2"] for r in rs])
        accs = np.mean([r["ok2_strict"] for r in rs])
        anr = np.mean([r["anr"] for r in rs])
        a1 = [r["ok2"] for r in rs if r["anr"]]
        p1 = np.mean(a1) if a1 else float("nan")
        print("%-11s %5d %6.3f %8.3f %8.3f %8.3f %6.3f(%3d)" % (
            c, n, parse, acc, accs, anr, p1, len(a1)))
        out.append(dict(cond=c, n=n, parse=float(parse), em=float(acc),
                        em_strict=float(accs), anr=float(anr),
                        p_correct_intact=float(p1)))

    print("\n%-11s %6s %8s %8s" % ("cond", "AIR", "acc|AIR=1", "acc|AIR<1"))
    for c in conds:
        rs = [r for r in recs if r["cond"] == c and r.get("air") == r.get("air")]
        if not rs:
            continue
        a1 = [r["ok2"] for r in rs if r["air"] >= 0.999]
        a0 = [r["ok2"] for r in rs if r["air"] < 0.999]
        print("%-11s %6.3f %6.3f(%3d) %6.3f(%3d)" % (
            c, np.mean([r["air"] for r in rs]),
            np.mean(a1) if a1 else float("nan"), len(a1),
            np.mean(a0) if a0 else float("nan"), len(a0)))

    d["summary_rescored"] = out
    outp = path.replace(".json", "_rescored.json")
    json.dump(d, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n写入", outp)


if __name__ == "__main__":
    main()
