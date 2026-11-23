"""诊断：奖励方差是不是「可学的方差」。

Stage 0 证明每题 5 个候选对的奖励有方差（46% 的题会变）。
stage1_policy 证明三参数线性策略学不到（最优 w=(0,0,0)）。
本脚本判定失败原因属于哪一类：

  H1 无观测信号：现有特征（交叉编码器分数、代价、可装性）对「r1 是否正确」
     没有判别力 → AUC≈0.5，那么任何以这些特征为状态的策略都学不到，
     换 RL 算法也没用。
  H2 有信号但线性不够：AUC 明显 >0.5，但线性加权重排不涨 → 换假设类有用。
  H3 天花板不在决策层：r1 错误时其余候选也很少正确 → oracle 的 +10.8pp
     是不可回收的（选择器本身的感知错误），策略这条路的上界就是 r1。

输出：AUC、分层表、阈值规则 CV、以及「r1 错时可回收比例」。
"""
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from r1_feat import QDS, M, collate

BASE = "E:/KV_models/bge-small-en-v1.5"
TOPK = 5
N_Q = 200


def auc(x, y):
    """Mann-Whitney AUC，无 sklearn 依赖。"""
    x = np.asarray(x, float)
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(x)
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(1, len(x) + 1)
    # 处理并列：用平均秩
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    n1 = y.sum()
    n0 = len(y) - n1
    return (ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    rows = json.load(open("../results/r1_pairs_train_full.json", encoding="utf-8"))
    random.seed(0)
    rows = random.sample(rows, N_Q)

    ck = torch.load("../results/r1c.pt", map_location="cpu", weights_only=False)
    model = M(BASE).cuda()
    model.load_state_dict(ck["state"])
    model.eval()

    ds = QDS(rows, tok, train=False)
    dl = DataLoader(ds, batch_size=16, shuffle=False,
                    collate_fn=lambda b: collate(b, pad))
    top = {}
    with torch.no_grad():
        idx = 0
        for ids, am, pm, y, F in dl:
            s = model(ids.cuda(), am.cuda(), F.cuda()).float().cpu()
            s = s.masked_fill(~pm, -1e4)
            order = s.argsort(1, descending=True)
            for i in range(s.shape[0]):
                row = rows[idx]
                idx += 1
                keep = [int(j) for j in order[i].tolist() if pm[i, j]][:TOPK]
                top[row["key"][0]] = [float(s[i, j]) for j in keep]

    for B in (3, 5):
        rew, cost, ndoc = {}, {}, {}
        for k in range(1, TOPK + 1):
            d = json.load(open("../results/stage0_r%d_b%d.json" % (k, B), encoding="utf-8"))
            for r in d["records"]:
                rew.setdefault(r["id"], {})[k - 1] = 1.0 if r["correct"] else 0.0
                cost.setdefault(r["id"], {})[k - 1] = float(r.get("pair_cost") or 0)
                ndoc[r["id"]] = float(r["n_doc"])
        qs = [q for q in rew if len(rew[q]) == TOPK and len(top.get(q, [])) == TOPK]
        n = len(qs)
        if not n:
            continue
        R = np.array([[rew[q][a] for a in range(TOPK)] for q in qs])
        C = np.array([[cost[q][a] for a in range(TOPK)] for q in qs])
        S = np.array([top[q] for q in qs])
        N = np.array([ndoc[q] for q in qs])
        z = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-6)

        print("=" * 62)
        print("预算 %d%%   n=%d   r1=%.3f   oracle=%.3f" %
              (B, n, R[:, 0].mean(), R.max(1).mean()))

        # H3：r1 错时，别的候选有多少能把题救回来
        wrong = R[:, 0] == 0
        recov = R[wrong, 1:].max(1).mean() if wrong.sum() else float("nan")
        print("  r1 错的比例 %.3f；其中「其余候选有一个对」= %.3f（可回收上界）"
              % (wrong.mean(), recov))
        # 条件正确率：给定 r1 错，各名次单独的正确率
        print("  给定 r1 错时各名次正确率：" +
              "  ".join("r%d=%.3f" % (k + 1, R[wrong, k].mean()) for k in range(1, TOPK)))

        # H1：现有特征能否预测 r1 正确
        margin = z[:, 0] - z[:, 1]
        print("  AUC 预测「r1 正确」：")
        print("    绝对分 z1        %.3f" % auc(z[:, 0], R[:, 0]))
        print("    分数间隔 z1-z2   %.3f" % auc(margin, R[:, 0]))
        print("    整对代价 cost1   %.3f" % auc(C[:, 0], R[:, 0]))
        print("    可装性 fit       %.3f" % auc((C[:, 0] <= B / 100.0 * N).astype(float),
                                                R[:, 0]))
        # 名次 2-5 的分数能否预测「它们自己对」
        print("    名次 2-5 自身分 → 自身对  %.3f"
              % auc(z[:, 1:].ravel(), R[:, 1:].ravel()))

        # H2：单阈值规则（低间隔就换 r2），5 折 CV
        rng = np.random.RandomState(0)
        folds = rng.permutation(n) % 5
        cand = np.unique(np.quantile(margin, np.linspace(0.05, 0.95, 19)))
        best = (R[:, 0].mean(), None)
        for t in cand:
            pick = np.where(margin < t, 1, 0)
            gain = R[np.arange(n), pick].mean() - R[:, 0].mean()
            if gain > best[0] - R[:, 0].mean():
                best = (R[np.arange(n), pick].mean(), t)
        # CV 版：阈值在训练折上选
        cv = []
        for f in range(5):
            tr, te = folds != f, folds == f
            bt, bg = None, -9
            for t in cand:
                g = (R[tr][np.arange(tr.sum()), np.where(margin[tr] < t, 1, 0)].mean())
                if g > bg:
                    bg, bt = g, t
            cv.append(R[te][np.arange(te.sum()), np.where(margin[te] < bt, 1, 0)].mean())
        print("  阈值规则「间隔<τ 换 r2」：全量最优 %.3f (τ=%.2f)，CV %.3f"
              % (best[0], best[1] if best[1] is not None else 0, float(np.mean(cv))))

        # 间隔五分位表：看有没有单调性
        print("  按间隔五分位：")
        edges = np.quantile(margin, [0, .2, .4, .6, .8, 1.0])
        for i in range(5):
            m = (margin >= edges[i]) & (margin <= edges[i + 1])
            if m.sum():
                print("    Q%d  n=%3d  r1=%.3f  r2=%.3f  最大候选=%.3f"
                      % (i + 1, m.sum(), R[m, 0].mean(), R[m, 1].mean(),
                         R[m].max(1).mean()))


if __name__ == "__main__":
    main()
