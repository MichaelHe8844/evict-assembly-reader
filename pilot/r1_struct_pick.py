# -*- coding: utf-8 -*-
"""用训练好的结构感知打分器出选择，并导出为 --force_json 格式。"""
import argparse
import json
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, ".")
from r1_struct import StructCE, PairDS, collate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cuda")
    tok = AutoTokenizer.from_pretrained(ck["base"])
    pad = tok.pad_token_id or 0
    a = ck["args"]
    model = StructCE(ck["base"], use_inter=not a["no_inter"],
                     use_struct=not a["no_struct"], cls_only=a["cls_only"]).cuda()
    model.load_state_dict(ck["state"])
    model.eval()

    rows = json.load(open(args.data, encoding="utf-8"))
    ds = PairDS(rows, tok)
    picks, hit = {}, []
    for k, (r, ps, fs) in enumerate(ds.items):
        batch = collate([ds[k]], pad)
        ids, am, pm, y, F, SP = batch
        with torch.no_grad():
            s = model(ids.cuda(), am.cuda(), F.cuda(), SP.cuda()).float().cpu()
        s = s.masked_fill(~pm, -1e4)
        pick = int(s.argmax(1))
        hit.append(float(y[0, pick]))
        i, j, _ = ps[pick]
        picks[r["qid"]] = [[r["cands"][i]["r"], r["cands"][i]["c"]],
                           [r["cands"][j]["r"], r["cands"][j]["c"]]]
    print("%s 选择层（%d 题）：整对命中 %.3f" % (args.tag or args.ckpt, len(hit), sum(hit) / len(hit)))
    json.dump(picks, open(args.out, "w"), ensure_ascii=False)
    print("已写", args.out)


if __name__ == "__main__":
    main()
