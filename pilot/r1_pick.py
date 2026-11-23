# -*- coding: utf-8 -*-
"""用训好的检查点在评测集上出选择结果，并导出为 arm_c 的 --force_json 格式。

同时报选择层的三个基线（Ind / 手写规则 / 池内上界），便于逐次复现对比。
"""
import argparse
import json
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, ".")
from r1_feat import M, pair_features  # noqa: E402
from ground import tokens  # noqa: E402
from r1_feat import rule_and_ind, mcn  # noqa: E402


def encode(tok, q, a, b):
    qa = tok(q, add_special_tokens=False)["input_ids"][:64]
    ca = tok(a, add_special_tokens=False)["input_ids"][:32]
    cb = tok(b, add_special_tokens=False)["input_ids"][:32]
    return [tok.cls_token_id] + qa + [tok.sep_token_id] + ca + [tok.sep_token_id] + cb + [tok.sep_token_id]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="../results/r1_pairs_dev_full.json")
    ap.add_argument("--out", required=True, help="导出的 force json 路径")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rows = json.load(open(args.data, encoding="utf-8"))
    ck = torch.load(args.ckpt, map_location="cuda")
    tok = AutoTokenizer.from_pretrained(ck["base"])
    m = M(ck["base"]).cuda()
    m.load_state_dict(ck["state"])
    m.eval()

    picks, hit = {}, []
    for r in rows:
        gmax = max(c["gscore"] for c in r["cands"]) or 1.0
        qw = set(tokens(r["q"]))
        ps = r["pairs"]
        seqs = [encode(tok, r["q"], r["cands"][i]["txt"], r["cands"][j]["txt"]) for i, j, _ in ps]
        L = max(len(s) for s in seqs)
        ids = torch.zeros(1, len(ps), L, dtype=torch.long)
        am = torch.zeros(1, len(ps), L, dtype=torch.long)
        for k, s in enumerate(seqs):
            ids[0, k, :len(s)] = torch.tensor(s)
            am[0, k, :len(s)] = 1
        F = torch.tensor([[pair_features(r["cands"], i, j, r["q"], gmax, qw) for i, j, _ in ps]])
        with torch.no_grad():
            sc = m(ids.cuda(), am.cuda(), F.cuda()).float().cpu()[0]
        i, j, y = ps[int(sc.argmax())]
        hit.append(y)
        picks[r["qid"]] = [[r["cands"][i]["r"], r["cands"][i]["c"]],
                           [r["cands"][j]["r"], r["cands"][j]["c"]]]

    IND, JR = rule_and_ind(rows)
    ORC = [any(p[2] for p in r["pairs"]) for r in rows]
    print("%s 选择层（%d 题）：上界 %.3f | Ind %.3f | 规则 %.3f | R1 %.3f"
          % (args.tag or args.ckpt, len(rows), sum(ORC) / len(ORC),
             sum(IND) / len(IND), sum(JR) / len(JR), sum(hit) / len(hit)))
    x, y, p = mcn(hit, JR)
    print("  vs 规则：+%d/-%d p=%.4g" % (x, y, p))
    json.dump(picks, open(args.out, "w"), ensure_ascii=False)
    print("  已写 %s" % args.out)


if __name__ == "__main__":
    main()
