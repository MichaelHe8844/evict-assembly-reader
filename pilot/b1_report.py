# -*- coding: utf-8 -*-
"""B1 结果分析：同一 216 题、同一读者，只换「哪一个检查点选对」。

  python b1_report.py vf216_force_base vf216_force_b1

两文件均为 oracle_ceiling.py --cond force 的输出（216 题四档）。
另打印 182 子集（定位器有对可钉的那批）的切片，便于与原表对照。
"""
import json
import sys

BUD = ("3", "5", "10", "20")


def load(name):
    d = json.load(open("../results/%s.json" % name, encoding="utf-8"))["detail"]
    out = {}
    for b in BUD:
        out[b] = {r["id"]: bool(r["ok"]) for r in d[b] if r["cond"] == "force"}
    return out


def main():
    names = sys.argv[1:] or ["vf216_force_base", "vf216_force_b1"]
    pair = set(json.load(open("../results/ens_force_test.json", encoding="utf-8")))
    ms = {n: load(n) for n in names}

    print("%-6s" % "预算" + "".join("%14s" % n for n in names)
          + "   182 段（" + " / ".join(names) + "）")
    for b in BUD:
        row = "%-6s" % (b + "%")
        for n in names:
            v = ms[n][b]
            row += "%14.3f" % (sum(v.values()) / len(v))
        for n in names:
            v = {i: k for i, k in ms[n][b].items() if i in pair}
            row += "   %.3f" % (sum(v.values()) / max(1, len(v)))
        print(row)
    row = "%-6s" % "平均"
    for n in names:
        row += "%14.3f" % (sum(sum(ms[n][b].values()) / len(ms[n][b])
                               for b in BUD) / 4)
    print(row)

    if len(names) == 2:
        a, b = names
        print("\n逐题配对 %s -> %s" % (a, b))
        for bd in BUD:
            da, db = ms[a][bd], ms[b][bd]
            ids = [i for i in da if i in db]
            plus = sum(1 for i in ids if db[i] and not da[i])
            minus = sum(1 for i in ids if da[i] and not db[i])
            print("  %-4s %+.3f  +%d/-%d  (n=%d)"
                  % (bd + "%", (sum(db[i] for i in ids) - sum(da[i] for i in ids))
                     / len(ids), plus, minus, len(ids)))


if __name__ == "__main__":
    main()
