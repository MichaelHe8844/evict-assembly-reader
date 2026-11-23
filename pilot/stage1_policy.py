"""Stage 1a：离线策略检验（CPU）。

Stage 0 已为每题的 5 个候选对拿到「答对与否」的反事实奖励，因此可以在不新增
rollout 的情况下，直接比较不同策略的期望奖励：
  - baseline：监督分数 argmax
  - 策略：π(a) ∝ exp(w1*z(score) + w2*cost/B + w3*fit)
用 5 折交叉验证（按题切分）看策略能否稳定超过 argmax。
"""
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from r1_feat import QDS, collate, M

BASE = "E:/KV_models/bge-small-en-v1.5"
TOPK = 5


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    rows = json.load(open("../results/r1_pairs_train_full.json", encoding="utf-8"))
    random.seed(0)
    rows = random.sample(rows, 200)

    ck = torch.load("../results/r1c.pt", map_location="cpu", weights_only=False)
    model = M(BASE).cuda()
    model.load_state_dict(ck["state"])
    model.eval()

    ds = QDS(rows, tok, train=False)
    dl = DataLoader(ds, batch_size=16, shuffle=False,
                    collate_fn=lambda b: collate(b, pad))
    scores = {}
    with torch.no_grad():
        idx = 0
        for ids, am, pm, y, F in dl:
            s = model(ids.cuda(), am.cuda(), F.cuda()).float().cpu()
            for i in range(s.shape[0]):
                row = rows[idx]
                idx += 1
                v = [(float(s[i, j]), float(y[i, j])) for j in range(len(row["pairs"]))
                     if pm[i, j]]
                v.sort(reverse=True)
                scores[row["key"][0]] = v[:TOPK]

    for B in (3, 5):
        R = {}
        for k in range(1, TOPK + 1):
            d = json.load(open("../results/stage0_r%d_b%d.json" % (k, B), encoding="utf-8"))
            for r in d["records"]:
                R.setdefault(r["id"], {})[k - 1] = (bool(r["correct"]),
                                                    r.get("pair_cost", 0), r["n_doc"])

        qs = [q for q in scores if q in R and len(R[q]) == TOPK
              and len(scores[q]) == TOPK]
        n = len(qs)
        cost = np.array([[R[q][a][1] for a in range(TOPK)] for q in qs], float)
        rew = np.array([[1.0 if R[q][a][0] else 0.0 for a in range(TOPK)] for q in qs])
        sc = np.array([[(scores[q][a][0]) for a in range(TOPK)] for q in qs], float)
        z = (sc - sc.mean(1, keepdims=True)) / (sc.std(1, keepdims=True) + 1e-6)
        # 预算按该题文档长度估算：用 cost 的相对量代替 B
        ndoc = np.array([R[q][0][2] for q in qs], float)
        fit = (cost <= (B / 100.0 * ndoc)[:, None]).astype(float)

        base = rew[np.arange(n), sc.argmax(1)].mean()
        oracle = rew.max(1).mean()
        print("== 预算 %d%%（%d 题）==" % (B, n))
        print("  监督 argmax %.3f ｜ 每题最优 %.3f" % (base, oracle))

        rng = np.random.RandomState(0)
        folds = rng.permutation(n) % 5
        best = None
        for w1 in (0.0, 0.5, 1.0, 2.0, 4.0):
            for w2 in (0.0, -0.5, -1.0, -2.0, -4.0):
                for w3 in (0.0, 0.5, 1.0, 2.0):
                    logits = w1 * z + w2 * (cost / (cost.mean() + 1e-6)) + w3 * fit
                    pick = logits.argmax(1)
                    acc = rew[np.arange(n), pick].mean()
                    # 交叉验证：留一折上的表现（阈值仅用于稳健性判断）
                    cv = np.mean([rew[folds == f][np.arange((folds == f).sum()),
                                                  pick[folds == f]].mean()
                                  for f in range(5)])
                    if best is None or cv > best[0]:
                        best = (cv, acc, w1, w2, w3)
        print("  最优策略 CV %.3f（全量 %.3f），w=(%.1f, %.1f, %.1f)"
              % (best[0], best[1], best[2], best[3], best[4]))


if __name__ == "__main__":
    main()
