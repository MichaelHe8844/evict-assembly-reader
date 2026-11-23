# -*- coding: utf-8 -*-
"""Harness 校验（细化版）：按论文实际使用的单步子集，并区分‘舍入不符’与‘无法解析’。"""
import json
import re

OPS2 = {"add": lambda a, b: a + b, "subtract": lambda a, b: a - b,
        "multiply": lambda a, b: a * b, "divide": lambda a, b: a / b,
        "exp": lambda a, b: a ** b}
CALL = re.compile(r"^([a-z_]+)\((.+)\)$")
SINGLE = re.compile(r"^([a-z]+)\(([^()]*)\)$")
AGG = ("table_max", "table_min", "table_sum", "table_average")


def split_args(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def num(x):
    x = x.strip()
    pct = x.endswith("%")          # FinQA 里 "2.4%" 的语义是 0.024
    x = x.rstrip("%")
    if x.startswith("const_"):
        y = x[6:].replace("m", "-")   # const_m1 = -1
        try:
            return float(y)
        except ValueError:
            return None
    try:
        v = float(x.replace(",", ""))
    except ValueError:
        return None
    return v / 100.0 if pct else v


def ev(node):
    m = CALL.match(node.strip())
    if not m:
        return num(node)
    op, rest = m.group(1), m.group(2)
    args = [ev(a) for a in split_args(rest)]
    if op in OPS2 and all(a is not None for a in args):
        return OPS2[op](*args)
    if op == "greater" and all(a is not None for a in args):
        return 1.0 if args[0] > args[1] else 0.0
    return None


def match(v, g):
    """数据集把答案存成 4 位有效数字，按此比较。"""
    if v is None:
        return False
    if abs(v - g) <= 1e-9:
        return True
    for nd in (2, 3, 4):
        if abs(round(v, nd) - g) <= 1e-9 or abs(round(v, nd) - round(g, nd)) <= 1e-9:
            return True
    return abs(v - g) <= 0.005 * max(1e-9, abs(g))


ds = json.load(open("../data/dev.json", encoding="utf-8"))
stat = {}
for e in ds:
    qa = e["qa"]
    try:
        gold = float(qa["exe_ans"])
    except (TypeError, ValueError):
        continue
    prog = (qa.get("program_re") or qa.get("program") or "").strip()
    single = bool(SINGLE.match(prog))
    key = "single" if single else "multi"
    stat.setdefault(key, dict(n=0, ok=0, agg=0, unsup=0, bad=0))
    s = stat[key]
    s["n"] += 1
    if any(prog.startswith(a) for a in AGG):
        s["agg"] += 1
        continue
    v = ev(prog)
    if v is None:
        s["unsup"] += 1
    elif match(v, gold):
        s["ok"] += 1
    else:
        s["bad"] += 1

for k in ("single", "multi"):
    s = stat.get(k)
    if not s:
        continue
    print("%-6s n=%3d | 复现 %3d（%.1f%%）| 聚合算子 %2d | 无法解析 %2d | 数值不符 %2d" % (
        k, s["n"], s["ok"], 100 * s["ok"] / s["n"], s["agg"], s["unsup"], s["bad"]))
