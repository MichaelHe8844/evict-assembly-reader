# -*- coding: utf-8 -*-
"""错误归因（细化）：模型用的是"表里真实存在的另一个格子"还是"编出来的数"。

先把每个模型数字与文档里的表格单元格比对：
  · 命中某个真实单元格 → 选错单元格（行/列错位）
  · 不命中、但与 gold 操作数只差位数/十倍 → 数值转写
  · 只给了一个数 → 缺操作数
"""
import json
import re

NUM = re.compile(r"-?\d[\d,]*\.?\d*%?")
SINGLE = re.compile(r"^([a-z]+)\(([^()]*)\)$")
OPS = {"add", "subtract", "multiply", "divide", "exp"}


def to_float(tok):
    t = tok.strip()
    pct = t.endswith("%")
    t = t.rstrip("%").replace(",", "")
    try:
        v = float(t)
    except ValueError:
        return None
    return v / 100.0 if pct else v


def numbers(s):
    out = []
    for m in NUM.findall(s or ""):
        v = to_float(m)
        if v is not None:
            out.append(v)
    return out


def close(a, b, rel=0.005):
    return a is not None and b is not None and abs(a - b) <= rel * max(1e-9, abs(b))


def gold_prog(prog):
    m = SINGLE.match((prog or "").strip())
    if not m:
        return None
    args = [to_float(a) for a in m.group(2).split(",")]
    if len(args) != 2 or any(a is None for a in args):
        return None
    return m.group(1), args


def cell_values(entry):
    """文档里所有数值单元格的取值（只用于判断"这个数是不是表里真有"）。"""
    vals = []
    for row in entry.get("table", []) or []:
        for cell in row:
            for tok in re.findall(r"-?\d[\d,]*\.?\d*", str(cell)):
                v = to_float(tok)
                if v is not None:
                    vals.append(v)
    for txt in (entry.get("pre_text") or []) + (entry.get("post_text") or []):
        for tok in re.findall(r"-?\d[\d,]*\.?\d*", txt):
            v = to_float(tok)
            if v is not None:
                vals.append(v)
    return vals


d = json.load(open("../results/cand2_confirm_rescored.json"))
raw = {e["id"]: e for e in json.load(open("../data/dev.json", encoding="utf-8"))}
by = {}
for r in d["records"]:
    by.setdefault((r["id"], round(float(r["gold"]), 6)), {})[r["cond"]] = r

pairs = [(k, c["full"], c["auto_f@20"]) for k, c in by.items()
         if "full" in c and "auto_f@20" in c]


def classify(rec, key):
    entry = raw.get(key[0], {}) or {}
    prog = (entry.get("qa") or {}).get("program_re", "") or ""
    g = gold_prog(prog)
    ms = numbers(rec.get("text", ""))
    if not ms:
        return "输出为空/无数字"
    if g is None:
        return "标准程序非单步"
    _, gargs = g
    vals = cell_values(entry)
    # 常数（×100 之类）不参与"是否来自表格"的判断
    CONST = {1.0, 100.0, 1000.0, 0.01, 0.1, 0.001}
    ms_core = [m for m in ms if m not in CONST] or ms
    in_table = [any(close(m, v) for v in vals) for m in ms_core]
    if len(ms_core) == 1:
        return "缺操作数"
    if all(in_table):
        # 数字都来自表格，但和 gold 操作数不同 → 选错格
        if any(not any(close(m, a, 1e-3) for a in gargs) for m in ms_core):
            return "选错单元格"
    # 不在表里，且与 gold 操作数只是位数/十倍关系
    for m in ms:
        for a in gargs:
            if a and not close(m, a, 1e-3):
                for k in (10, 100, 1000, 0.1, 0.01, 0.001):
                    if close(m, a * k, 1e-3):
                        return "数值转写"
    # 顺序颠倒
    if len(ms) >= 2 and close(ms[0], gargs[1], 1e-3) and close(ms[1], gargs[0], 1e-3):
        return "操作数角色"
    # 结果差 100/1000 倍
    if rec.get("pred") and rec.get("gold"):
        for k in (100, 1000, 0.01, 0.001):
            if close(rec["pred"], rec["gold"] * k, 1e-2):
                return "单位与尺度"
    return "其它"


tax = {"全上下文对/压缩错": {}, "两边都错": {}}
groups = {
    "全上下文对/压缩错": [(k, f, c) for k, f, c in pairs if f["correct"] and not c["correct"]],
    "两边都错": [(k, f, c) for k, f, c in pairs if not f["correct"] and not c["correct"]],
}
for name, grp in groups.items():
    for k, f, c in grp:
        t = classify(c, k)
        tax[name][t] = tax[name].get(t, 0) + 1

print("配对 %d 题：两边对 %d | 全对对/压缩错 %d | 两边都错 %d | 全对错/压缩对 %d" % (
    len(pairs),
    sum(1 for _, f, c in pairs if f["correct"] and c["correct"]),
    len(groups["全上下文对/压缩错"]), len(groups["两边都错"]),
    sum(1 for _, f, c in pairs if not f["correct"] and c["correct"])))

print("\n%-14s %8s %8s" % ("错误类型", "全对/压缩错", "两边都错"))
keys = sorted(set(tax["全上下文对/压缩错"]) | set(tax["两边都错"]),
              key=lambda t: -(tax["全上下文对/压缩错"].get(t, 0) + tax["两边都错"].get(t, 0)))
for t in keys:
    print("%-14s %8d %8d" % (t, tax["全上下文对/压缩错"].get(t, 0), tax["两边都错"].get(t, 0)))
