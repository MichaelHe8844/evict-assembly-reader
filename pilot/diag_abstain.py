"""诊断三：把动作从「换候选」换成「弃答」后，置信度值多少钱。

已知 z1 的 AUC=0.66（能预测 r1 是否正确），但换候选不涨（尾候选正确率 0.07）。
弃答不要求「知道哪个对」，只要求「知道自己可能错」，因此可能是唯一可兑现的动作。
输出：按 z1 排序后的 risk–coverage 曲线（部署视角，不需要 gold 选阈值）。
"""
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from r1_feat import QDS, M, collate

BASE = "E:/KV_models/bge-small-en-v1.5"
TOPK, N_Q = 5, 200


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
    z1 = {}
    with torch.no_grad():
        idx = 0
        for ids, am, pm, y, F in dl:
            s = model(ids.cuda(), am.cuda(), F.cuda()).float().cpu()
            s = s.masked_fill(~pm, -1e4)
            for i in range(s.shape[0]):
                row = rows[idx]
                idx += 1
                v = sorted([float(s[i, j]) for j in range(len(row["pairs"])) if pm[i, j]],
                           reverse=True)[:TOPK]
                z1[row["key"][0]] = (v[0], v[0] - v[1] if len(v) > 1 else 0.0)

    for B in (3, 5):
        rew = {}
        for k in range(1, TOPK + 1):
            d = json.load(open("../results/stage0_r%d_b%d.json" % (k, B), encoding="utf-8"))
            for r in d["records"]:
                rew.setdefault(r["id"], {})[k - 1] = 1.0 if r["correct"] else 0.0
        qs = [q for q in rew if len(rew[q]) == TOPK and q in z1]
        y = np.array([rew[q][0] for q in qs])
        for name, feat in (("绝对分 z1", 0), ("间隔 z1-z2", 1)):
            s = np.array([z1[q][feat] for q in qs])
            order = np.argsort(-s)
            ys = y[order]
            print("== %d%% / %s（n=%d，全量 acc=%.3f）==" % (B, name, len(qs), y.mean()))
            for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
                m = int(len(ys) * cov)
                print("   覆盖率 %3d%% → acc %.3f" %
                      (int(cov * 100), ys[:m].mean() if m else float("nan")))


if __name__ == "__main__":
    main()
