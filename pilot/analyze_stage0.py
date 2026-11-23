"""Stage 0 分析：候选对的奖励方差。

对每题 5 个候选对（按监督分数排序）各 rollout 一次，统计：
  - 有多少题的奖励随候选变化（= 有学习信号）
  - 各名次的 EM、每题最优/最差、oracle 上界
  - 奖励是否与整对代价相关（防"只会挑便宜的"）
"""
import json
from collections import defaultdict

BUDGETS = (3, 5)
TOPK = 5


def main():
    for b in BUDGETS:
        per_q = defaultdict(dict)
        cost = {}
        for k in range(1, TOPK + 1):
            p = "../results/stage0_r%d_b%d.json" % (k, b)
            try:
                d = json.load(open(p, encoding="utf-8"))
            except FileNotFoundError:
                continue
            for r in d["records"]:
                per_q[r["id"]][k] = bool(r["correct"])
                cost.setdefault(r["id"], {})[k] = r.get("pair_cost")
        qs = [q for q in per_q if len(per_q[q]) == TOPK]
        if not qs:
            print("%d%%: 无完整数据" % b)
            continue
        vary = sum(1 for q in qs if len(set(per_q[q].values())) > 1)
        ems = [sum(per_q[q][k] for q in qs) / len(qs) for k in range(1, TOPK + 1)]
        oracle = sum(1 for q in qs if any(per_q[q].values())) / len(qs)
        worst = sum(1 for q in qs if all(per_q[q].values())) / len(qs)
        print("== 预算 %d%%（%d 题）==" % (b, len(qs)))
        print("  奖励随候选变化的比例：%.1f%%" % (100 * vary / len(qs)))
        print("  各名次 EM：" + "  ".join("r%d=%.3f" % (k, ems[k - 1]) for k in range(1, TOPK + 1)))
        print("  该题最优(任一候选答对)：%.3f   全部答对：%.3f" % (oracle, worst))
        # 奖励 vs 代价
        ok_c = [cost[q][k] for q in qs for k in range(1, TOPK + 1)
                if per_q[q][k] and cost[q].get(k)]
        no_c = [cost[q][k] for q in qs for k in range(1, TOPK + 1)
                if not per_q[q][k] and cost[q].get(k)]
        if ok_c and no_c:
            print("  答对的整对代价 均值 %.1f；答错的 %.1f" %
                  (sum(ok_c) / len(ok_c), sum(no_c) / len(no_c)))


if __name__ == "__main__":
    main()
