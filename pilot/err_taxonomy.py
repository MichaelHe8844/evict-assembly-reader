# -*- coding: utf-8 -*-
"""配对 + 错误归因：先分‘全上下文对/压缩错’与‘两边都错’，再按类型统计。

分类只用模型自己写出的表达式（PoT 的 text 字段）与 gold 程序比对：
  运算符错误 / 操作数角色错误 / 数值转写错误 / 单位与尺度错误 / 输出为空 / 其它
"""
import json
import re
import numpy as np

NUM = re.compile(r"-?\d[\d,]*\.?\d*%?")
CALL = re.compile(r"^([a-z_]+)\((.+)\)$")
OPS = {"add", "subtract", "multiply", "divide", "exp"}
SINGLE = re.compile(r"^([a-z]+)\(([^()]*)\)$")


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


def parse_gold(prog):
    m = SINGLE.match(prog.strip())
    if not m:
        return None
    op, rest = m.group(1), m.group(2)
    args = [to_float(a) for a in rest.split(",")]
    if len(args) != 2 or any(a is None for a in args):
        return None
    return op, args


def close(a, b, rel=0.01):
    return a is not None and b is not None and abs(a - b) <= rel * max(1e-9, abs(b))


def digit_corrupt(model_nums, gold_args):
    """模型数字是否是 gold 某个数字的"位数级"改写（多一位、少一位、单位数错）。"""
    for m in model_nums:
        for g in gold_args:
            if g == 0:
                continue
            for k in (10, 100, 1000, 0.1, 0.01, 0.001):
                if close(m, g * k, 1e-3):
                    return True
            # 去掉一位数字后相同：1189 -> 11899 / 1189 -> 189
            sm, sg = str(m).replace(".", ""), str(g).replace(".", "")
            if len(sm) != len(sg) and (sg in sm or sm in sg):
                return True
    return False


d = json.load(open("../results/cand2_confirm_rescored.json"))
raw = {e["id"]: e for e in json.load(open("../data/dev.json", encoding="utf-8"))}
by = {}
for r in d["records"]:
    by.setdefault((r["id"], round(float(r["gold"]), 6)), {})[r["cond"]] = r

rows = []
for key, conds in by.items():
    if "full" not in conds or "auto_f@20" not in conds:
        continue
    rows.append((key, conds["full"], conds["auto_f@20"]))
print("配对数", len(rows))

both_ok = sum(1 for _, f, c in rows if f["correct"] and c["correct"])
full_ok_c_bad = [(k, f, c) for k, f, c in rows if f["correct"] and not c["correct"]]
both_bad = [(k, f, c) for k, f, c in rows if not f["correct"] and not c["correct"]]
c_ok_f_bad = sum(1 for _, f, c in rows if (not f["correct"]) and c["correct"])
print("两边都对 %d | 全上下文对/压缩错 %d | 两边都错 %d | 全上下文错/压缩对 %d" % (
    both_ok, len(full_ok_c_bad), len(both_bad), c_ok_f_bad))


def classify(rec, key):
    prog = ((raw.get(key[0], {}) or {}).get("qa") or {}).get("program_re", "") or ""
    g = parse_gold(prog)
    txt = rec.get("text", "") or ""
    ms = numbers(txt)
    if not ms:
        return "输出为空/无数字"
    if g is None:
        return "标准程序非单步"
    gop, gargs = g
    model_ops = [o for o in OPS if o in txt.lower()] or \
                (["divide"] if "/" in txt else []) + (["multiply"] if "*" in txt else []) + \
                (["subtract"] if "-" in txt else []) + (["add"] if "+" in txt else [])
    # 单位/尺度：结果差 100 倍且操作数正确
    if all(any(close(m, a) for m in ms) for a in gargs):
        if gop in model_ops or not model_ops:
            return "单位与尺度"
        # 操作数都在，但运算符不同
        if gop == "divide" and "multiply" in model_ops and any(close(m, 100) for m in ms):
            return "单位与尺度"
        return "运算符错误"
    # 操作数顺序/符号
    if len(ms) >= 2 and close(ms[0], gargs[1]) and close(ms[1], gargs[0]):
        return "操作数角色错误"
    if digit_corrupt(ms, gargs):
        return "数值转写错误"
    if len(ms) == 1:
        return "操作数缺失"
    return "其它数值错误"


tax = {"全上下文对/压缩错": {}, "两边都错": {}}
for name, group in (("全上下文对/压缩错", full_ok_c_bad), ("两边都错", both_bad)):
    for key, f, c in group:
        t = classify(c, key)
        tax[name][t] = tax[name].get(t, 0) + 1

print("\n%-16s %6s %6s" % ("错误类型", "全对/压缩错", "两边都错"))
allt = sorted(set(tax["全上下文对/压缩错"]) | set(tax["两边都错"]),
              key=lambda t: -(tax["全上下文对/压缩错"].get(t, 0) + tax["两边都错"].get(t, 0)))
for t in allt:
    print("%-16s %6d %6d" % (t, tax["全上下文对/压缩错"].get(t, 0), tax["两边都错"].get(t, 0)))

# “全对/压缩错”里，操作数是否仍完整（num=1 表示 V 完整）
v_ok = sum(1 for _, f, c in full_ok_c_bad if c.get("num"))
print("\n‘全上下文对/压缩错’的 %d 条里，V 单元格仍完整的 %d 条（%.0f%%）" % (
    len(full_ok_c_bad), v_ok, 100 * v_ok / max(1, len(full_ok_c_bad))))
