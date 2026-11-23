# -*- coding: utf-8 -*-
"""B1：用重排器给测试题的候选对打分，输出与 arm_c --force_json 兼容的文件。

用法：
  python b1_force.py --ckpt ../results/b1_rerank.pt \
      --pairs ../results/xds_pairs_test.json --out ../results/b1_force_test.json
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from r1_feat import M, QDS, collate

BASE = "E:/KV_models/bge-small-en-v1.5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../results/b1_rerank.pt")
    ap.add_argument("--pairs", default="../results/xds_pairs_test.json")
    ap.add_argument("--out", default="../results/b1_force_test.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = M(BASE).cuda()
    model.load_state_dict(ck["state"])
    model.eval()

    rows = json.load(open(args.pairs, encoding="utf-8"))
    ds = QDS(rows, tok, train=False)
    dl = DataLoader(ds, batch_size=16, shuffle=False,
                    collate_fn=lambda b: collate(b, pad))
    picks, idx, agree = {}, 0, 0
    with torch.no_grad():
        for ids, am, pm, y, F in dl:
            s = model(ids.cuda(), am.cuda(), F.cuda()).float().cpu()
            s = s.masked_fill(~pm, -1e4)
            for i in range(s.shape[0]):
                row = rows[idx + i]
                j = int(s[i].argmax())
                a, b, _ = row["pairs"][j]
                ca, cb = row["cands"][a], row["cands"][b]
                qid = row["qid"] if "qid" in row else row["key"][0]
                picks[qid] = [[ca["r"], ca["c"]], [cb["r"], cb["c"]]]
                agree += int(bool(row["pairs"][j][2]))
            idx += s.shape[0]

    json.dump(picks, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
    print("%d 条；argmax 落在原标签为正的比例 %.3f"
          % (len(picks), agree / max(1, len(picks))))


if __name__ == "__main__":
    main()
