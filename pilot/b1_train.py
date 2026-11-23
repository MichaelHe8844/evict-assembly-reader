# -*- coding: utf-8 -*-
"""B1：用「读者是否答对」作标签，重训成对选择器（结构同 r1_feat.M）。

与 4.5 的定位器只差标签：原来学"这一对是否覆盖 gold 两个操作数"，
这里学"把这一对钉进证据后，读者会不会答对"。架构、特征、超参不变。

用法：
  python b1_train.py --rows ../results/b1_train_rows.json --out ../results/b1_rerank.pt
"""
import argparse
import json
import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from r1_feat import M, QDS, collate

BASE = "E:/KV_models/bge-small-en-v1.5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="../results/b1_train_rows.json")
    ap.add_argument("--out", default="../results/b1_rerank.pt")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    rows = json.load(open(args.rows, encoding="utf-8"))
    npos = sum(1 for r in rows for p in r["pairs"] if p[2])
    ntot = sum(len(r["pairs"]) for r in rows)
    print("题 %d，对 %d，正例 %d（%.1f%%）"
          % (len(rows), ntot, npos, 100 * npos / max(1, ntot)), flush=True)

    ds = QDS(rows, tok, train=False)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True,
                    collate_fn=lambda b: collate(b, pad))

    model = M(BASE).cuda()
    enc = [p for n, p in model.named_parameters() if n.startswith("enc")]
    head = [p for n, p in model.named_parameters() if not n.startswith("enc")]
    opt = torch.optim.AdamW([{"params": enc, "lr": args.lr},
                             {"params": head, "lr": args.head_lr}])
    steps = args.epochs * len(dl)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.head_lr],
                                              total_steps=steps)
    pw = torch.tensor(float(ntot - npos) / max(1, npos)).cuda()
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for ids, am, pm, y, F in dl:
            s = model(ids.cuda(), am.cuda(), F.cuda())
            loss = lossf(s[pm], y.cuda()[pm])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            tot += float(loss) * int(pm.sum())
            n += int(pm.sum())
        print("  epoch %d  loss %.4f  %.0fs" % (ep + 1, tot / max(1, n),
                                                time.time() - t0), flush=True)

    torch.save({"state": model.state_dict()}, args.out)
    print("写出", args.out)


if __name__ == "__main__":
    main()
