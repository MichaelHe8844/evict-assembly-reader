"""Stage 0：检验「选择不同候选对 → 奖励是否有方差」。

不做训练。做法：在 train 里取 200 题，用部署版打分器给 28 个候选对排序，
取 top-5 各生成一份 force 文件；随后每题 5 次 rollout，看答对与否是否随候选变化。
没有方差就没有学习信号，策略这条路直接放弃。
"""
import json
import random

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from r1_feat import QDS, collate, M

BASE = "E:/KV_models/bge-small-en-v1.5"
N_Q, TOPK = 200, 5


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
    ranks = [{} for _ in range(TOPK)]
    gold_in_top = 0
    with torch.no_grad():
        idx = 0
        for ids, am, pm, y, F in dl:
            s = model(ids.cuda(), am.cuda(), F.cuda()).float().cpu()
            s = s.masked_fill(~pm, -1e4)
            order = s.argsort(1, descending=True)
            for i in range(s.shape[0]):
                row = rows[idx]
                idx += 1
                top = [int(j) for j in order[i].tolist() if pm[i, j]][:TOPK]
                if any(y[i, j] > 0.5 for j in top):
                    gold_in_top += 1
                for k, j in enumerate(top):
                    a, b, _ = row["pairs"][j]
                    ca, cb = row["cands"][a], row["cands"][b]
                    ranks[k][row["key"][0]] = [[ca["r"], ca["c"]], [cb["r"], cb["c"]]]

    for k in range(TOPK):
        json.dump(ranks[k], open("../results/stage0_force_r%d.json" % (k + 1), "w",
                                 encoding="utf-8"), ensure_ascii=False)
    keys = [{"key": [r["key"][0], r["key"][1]]} for r in rows]
    json.dump(keys, open("../results/stage0_keys.json", "w", encoding="utf-8"),
              ensure_ascii=False)
    print("题数 %d；top-%d 内含金标对的题 %d（%.1f%%）"
          % (len(rows), TOPK, gold_in_top, 100 * gold_in_top / len(rows)))


if __name__ == "__main__":
    main()
