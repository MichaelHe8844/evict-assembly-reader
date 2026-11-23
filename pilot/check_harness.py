# -*- coding: utf-8 -*-
"""先验证测试工具（harness）：标准程序经我们的解释器能否复现标准答案。

若这一步就失败，那么"模型算错"里会混进解释器/评测口径的错误，归因不成立。
"""
import json
import re

OPS2 = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b,
    "exp": lambda a, b: a ** b,
}

CALL = re.compile(r"^([a-z_]+)\((.+)\)$")


def split_args(s):
    """按顶层逗号切参数。"""
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


def ev(node, steps, table_vals=None):
    node = node.strip()
    m = CALL.match(node)
    if m:
        op, rest = m.group(1), m.group(2)
        args = [ev(a, steps, table_vals) for a in split_args(rest)]
        if op in OPS2:
            if any(a is None for a in args):
                return None
            return OPS2[op](*args)
        if op == "greater":
            return 1.0 if args[0] > args[1] else 0.0
        return None
    if node.startswith("const_"):
        try:
            return float(node.split("_")[1])
        except ValueError:
            return None
    try:
        return float(node.replace(",", ""))
    except ValueError:
        return None


ds = json.load(open("../data/dev.json", encoding="utf-8"))
ok = bad_val = unsupported = 0
fails = []
for e in ds:
    qa = e["qa"]
    ans = qa.get("exe_ans")
    if ans in (None, "", "yes", "no"):
        continue
    try:
        gold = float(ans)
    except (TypeError, ValueError):
        continue
    prog = qa.get("program_re") or qa.get("program") or ""
    val = ev(prog, qa.get("steps") or [])
    if val is None:
        unsupported += 1
        fails.append((e["id"], prog[:60], "unsupported"))
        continue
    if abs(val - gold) <= max(1e-6, 1e-4 * abs(gold)):
        ok += 1
    else:
        bad_val += 1
        fails.append((e["id"], prog[:60], "%.4f vs %.4f" % (val, gold)))

tot = ok + bad_val + unsupported
print("可评测样本 %d：复现 %d（%.1f%%）| 数值不符 %d | 无法解析 %d" % (
    tot, ok, 100 * ok / tot, bad_val, unsupported))
for f in fails[:8]:
    print("  ", f)
