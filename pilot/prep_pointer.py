# -*- coding: utf-8 -*-
"""指针化可行性：把 gold 程序里的每个数映射成「证据中第 k 个数字」。

输出：
  - 各预算的可映射率（全部数字都能定位才算可映射）
  - 指针化后的程序样例
  - 缓存 results/ptr_cache_{split}.json：{qid: {budget: {"prog": "divide([3],[7])", "vals": [...]}}}
"""
import json
import re
import sys

NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def mentions(text):
    """按出现顺序返回证据中的数值列表（已去逗号）。"""
    out = []
    for m in NUM.finditer(text or ""):
        s = m.group(0).replace(",", "")
        try:
            out.append(round(float(s), 4))
        except ValueError:
            pass
    return out


def map_program(prog, nums):
    """把程序里的数字换成 [k]；返回 (pointer_prog, ok, vals)。"""
    seq, vals, ok = [], [], True
    pos = 0
    for m in NUM.finditer(prog):
        seq.append(prog[pos:m.start()])
        raw = m.group(0)
        try:
            v = round(float(raw.replace(",", "")), 4)
        except ValueError:
            ok = False
            seq.append(raw)
            pos = m.end()
            continue
        if v in nums:
            k = nums.index(v) + 1          # 1-based，取首次出现
            seq.append("[%d]" % k)
            vals.append(nums[k - 1])
        else:
            ok = False
            seq.append(raw)
        pos = m.end()
    seq.append(prog[pos:])
    return "".join(seq), ok, vals


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    cache = json.load(open("../results/ctx_cache_%s.json" % split, encoding="utf-8"))
    raw_path = "../data/%s.json" % ("train" if split == "train" else "dev")
    raw = {x["id"]: x for x in json.load(open(raw_path, encoding="utf-8"))}

    stats = {}
    out = {}
    shown = 0
    for qid, ent in cache.items():
        x = raw.get(qid)
        if not x:
            continue
        prog = ((x.get("qa") or {}).get("program") or "").strip()
        if not prog:
            continue
        out[qid] = {}
        for b, txt in ent.items():
            nums = mentions(txt)
            pp, ok, vals = map_program(prog, nums)
            out[qid][b] = dict(prog=pp, ok=ok, vals=vals, n_num=len(nums))
            t = stats.setdefault(b, [0, 0])
            t[0] += 1
            t[1] += ok
            if ok and shown < 3 and b == "20":
                shown += 1
                print("样例 %s [%s%%] %s  →  %s" % (qid, b, prog[:50], pp[:50]))

    print("\n%-6s %8s %10s" % ("预算", "题数", "可映射率"))
    for b in ("3", "5", "10", "20"):
        if b in stats:
            n, k = stats[b]
            print("%-6s %8d %10.3f" % (b + "%", n, k / n))

    json.dump(out, open("../results/ptr_cache_%s.json" % split, "w"),
              ensure_ascii=False)


if __name__ == "__main__":
    main()
