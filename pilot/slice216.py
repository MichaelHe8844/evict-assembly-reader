# -*- coding: utf-8 -*-
"""216 题评测的切片对照：拆成「182（定位器有对可钉）」与「34（无对可钉）」两段。

用途：判断新旧适配器的差异是否只是题集口径造成的。
  python slice216.py vf216_gold_vf [另一次运行名 ...]

对每次运行，额外打印与旧文件的 182 段参考值（lora_bc_test / lora_rft_test）的对照，
参考值来自 lora_budget.py 的 182 题四档评测（旧装配）。
"""
import json
import sys

BUD = ("3", "5", "10", "20")

# 旧装配下 182 题的四档 EM（lora_budget.py 口径，见 results/*_test.json）
REF = {
    "gold": {"3": 0.154, "5": 0.269, "10": 0.489, "20": 0.599},   # lora_bc
    "efst": {"3": 0.192, "5": 0.330, "10": 0.516, "20": 0.670},   # lora_rft
}


def main():
    names = sys.argv[1:]
    if not names:
        print("用法: python slice216.py <运行名> ...")
        return
    pair = set(json.load(open("../results/ens_force_test.json", encoding="utf-8")))
    for nm in names:
        d = json.load(open("../results/%s.json" % nm, encoding="utf-8"))
        print("=" * 62)
        print(nm)
        for cond, label in (("base", "旧装配"), ("force", "新装配")):
            row182, row34 = [], []
            for b in BUD:
                det = [r for r in d["detail"][b] if r["cond"] == cond]
                r182 = [r["ok"] for r in det if r["id"] in pair]
                r34 = [r["ok"] for r in det if r["id"] not in pair]
                row182.append((b, len(r182), sum(r182) / max(1, len(r182))))
                row34.append((b, len(r34), sum(r34) / max(1, len(r34))))
            allb = [(b, n1 + n2, (s1 * n1 + s2 * n2) / max(1, n1 + n2))
                    for (b, n1, s1), (b2, n2, s2) in zip(row182, row34)]
            print(" [%s]" % label)
            print("   182 段  " + "  ".join("%s%% %.3f" % (b, s) for b, n, s in row182))
            print("   34 段   " + "  ".join("%s%% %.3f" % (b, s) for b, n, s in row34))
            print("   216 合并 " + "  ".join("%s%% %.3f" % (b, s) for b, n, s in allb)
                  + "   平均 %.3f" % (sum(s for _, _, s in allb) / 4))

    if len(names) == 2:
        a, b = names
        da = json.load(open("../results/%s.json" % a, encoding="utf-8"))
        db = json.load(open("../results/%s.json" % b, encoding="utf-8"))
        print("=" * 62)
        print("逐题配对 %s -> %s（按题配对，不做文档聚类）" % (a, b))
        for cond in ("base", "force"):
            for bd in BUD:
                ka = {r["id"]: r["ok"] for r in da["detail"][bd] if r["cond"] == cond}
                kb = {r["id"]: r["ok"] for r in db["detail"][bd] if r["cond"] == cond}
                ids = [i for i in ka if i in kb]
                plus = sum(1 for i in ids if kb[i] and not ka[i])
                minus = sum(1 for i in ids if ka[i] and not kb[i])
                print("  %-6s %s%%  +%d/-%d  (n=%d)"
                      % ("旧" if cond == "base" else "新", bd, plus, minus, len(ids)))


if __name__ == "__main__":
    main()
