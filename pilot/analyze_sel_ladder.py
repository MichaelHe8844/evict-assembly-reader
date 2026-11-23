# -*- coding: utf-8 -*-
"""选择层收益拆解：双塔 / 单格交叉 / 成对交叉 / 成对+结构特征，逐样本比较。

只做评测，不训练。加载 r1_cell.py 训好的 .pt，按同一候选池重算逐题命中，
输出准确率与两两 McNemar（配对），供论文引用。
"""
import json
import math

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from r1_cell import DS, M, score_rows

BASE = "E:/KV_models/bge-small-en-v1.5"


def mcn(a, b):
    x = sum(1 for u, v in zip(a, b) if v and not u)
    y = sum(1 for u, v in zip(a, b) if u and not v)
    m = x + y
    if not m:
        return x, y, 1.0
    k = min(x, y)
    return x, y, min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    hidden = ck["state"]["head.0.weight"].shape[1]
    nf = hidden - AutoModel.from_pretrained(ck["base"]).config.hidden_size
    m = M(ck["base"], nf=nf).cuda()
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck["mode"]


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    dv = json.load(open("../results/r1_pairs_dev_full.json", encoding="utf-8"))
    te = json.load(open("../results/xds_pairs_test.json", encoding="utf-8"))
    models = {}
    for tag in ("ce_cell", "ce_pair", "ce_pairf"):
        models[tag] = load("../results/%s.pt" % tag)
    out = {}
    for name, rows in (("dev", dv), ("test", te)):
        H = {}
        for tag, (m, mode) in models.items():
            H[tag] = [bool(x) for x in score_rows(m, rows, tok, mode)]
            print("%s %-8s %.4f" % (name, tag, np.mean(H[tag])), flush=True)
        out[name] = H
        for a, b in (("ce_cell", "ce_pair"), ("ce_pair", "ce_pairf"), ("ce_cell", "ce_pairf")):
            x, y, p = mcn(H[a], H[b])
            print("   %s vs %s：+%d/-%d p=%.3g" % (b, a, x, y, p), flush=True)
    json.dump(out, open("../results/sel_ladder.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
