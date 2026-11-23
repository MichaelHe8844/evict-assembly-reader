"""诊断二：动作空间换成「预算」后还有没有可学空间。

已有结论：动作="在我自己的 top-5 里换一个"是退化的——
  r1 错时其余候选正确率仅 0.07，错误高度相关，oracle 的 +10.8pp 不可回收。

本脚本检验另一种动作：每题自己决定给多少预算（3% vs 5%），
状态是「整对代价 / 文档长度 / 可装性」，奖励是答对。
若「3% 失败但 5% 成功」的比例足够大，且能被 cost 预测，
那么这个动作空间里的策略有真实上界，RL 才有意义。
"""
import json
from collections import defaultdict


def load(B):
    d = {}
    for k in range(1, 6):
        p = "../results/stage0_r%d_b%d.json" % (k, B)
        try:
            j = json.load(open(p, encoding="utf-8"))
        except FileNotFoundError:
            continue
        for r in j["records"]:
            d.setdefault(r["id"], {})[k] = (bool(r["correct"]),
                                            float(r.get("pair_cost") or 0),
                                            float(r["n_doc"]))
    return d


def main():
    a, b = load(3), load(5)
    qs = [q for q in a if len(a[q]) == 5 and q in b and len(b[q]) == 5]
    n = len(qs)
    print("n=%d" % n)

    for k in range(1, 6):
        c3 = sum(a[q][k][0] for q in qs) / n
        c5 = sum(b[q][k][0] for q in qs) / n
        both = sum(a[q][k][0] and b[q][k][0] for q in qs) / n
        only3 = sum(a[q][k][0] and not b[q][k][0] for q in qs) / n
        only5 = sum(b[q][k][0] and not a[q][k][0] for q in qs) / n
        print("  r%d  3%%=%.3f  5%%=%.3f  都对=%.3f  只3%%=%.3f  只5%%=%.3f  净增益=%+.3f"
              % (k, c3, c5, both, only3, only5, c5 - c3))

    # r1 口径下：3% 错而 5% 对的题，能否被「3% 时整对装不下」预测
    print("\n[r1 口径] 3% 失败后换 5% 能救回多少，以及是否可预测")
    rescue, total_wrong = 0, 0
    tab = defaultdict(lambda: [0, 0])  # fit -> [rescued, n]
    for q in qs:
        c3, cost3, nd = a[q][1]
        c5 = b[q][1][0]
        if c3:
            continue
        total_wrong += 1
        fit = cost3 <= 0.03 * nd
        tab[fit][1] += 1
        if c5:
            rescue += 1
            tab[fit][0] += 1
    print("  3%% 答错 %d 题；其中 5%% 能答对 %d（%.1f%%）＝ 该动作的可回收上界"
          % (total_wrong, rescue, 100 * rescue / max(total_wrong, 1)))
    for fit in (True, False):
        r, m = tab[fit]
        print("    3%% 时整对%s：n=%3d，换 5%% 救回 %d（%.1f%%）"
              % ("装得下" if fit else "装不下", m, r, 100 * r / max(m, 1)))

    # 若规则=「装不下就升到 5%」，端到端收益
    gain = 0
    for q in qs:
        c3, cost3, nd = a[q][1]
        if c3:
            continue
        if cost3 > 0.03 * nd and b[q][1][0]:
            gain += 1
    print("  规则「r1 在 3%% 装不下→升到 5%%」净增 %d 题 = %+.1fpp（只升不降，无代价）"
          % (gain, 100.0 * gain / n))


if __name__ == "__main__":
    main()
