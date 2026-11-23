# -*- coding: utf-8 -*-
"""B1 试点：把定位器 top-k 候选对逐一钉进证据，用读者跑一遍，记录对错。

用途：先确认「换一对」会不会改变读者答案，再决定是否值得做重排训练。
输出：results/b1_pairs_label.json
  [{"qid","budget","pair":[[r,c],[r,c]],"loc":"top0..","score":s,"ok":bool,"prog":...}]
"""
import argparse
import json
import sys
import time

import torch

sys.path.insert(0, ".")
from common import build_dataset                     # noqa: E402
from oracle_ceiling import gold_literals, oracle_text  # noqa: E402
from lora_budget import clean, close, run_prog        # noqa: E402
from r1_feat import M, QDS, collate                   # noqa: E402

BASE = "E:/KV_models/bge-small-en-v1.5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="../results/r1_pairs_train_full.json")
    ap.add_argument("--data", default="../data/train.json")
    ap.add_argument("--ckpt", default="../results/r1c.pt")
    ap.add_argument("--adapter", default="../results/lora_vf_gold")
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--out", default="../results/b1_pairs_label.json")
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    tok = AutoTokenizer.from_pretrained(BASE)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    loc = M(BASE).cuda()
    loc.load_state_dict(ck["state"])
    loc.eval()

    rows = json.load(open(args.pairs, encoding="utf-8"))
    rows = rows[args.skip:args.skip + args.n]
    print("题数 %d，预算 %d%%，每题材取 top-%d" % (len(rows), args.budget, args.k),
          flush=True)

    # 1) 定位器给 28 对打分，取 top-k
    ds = QDS(rows, tok, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False,
                                     collate_fn=lambda b: collate(b, pad))
    tops, idx = [], 0
    with torch.no_grad():
        for ids, am, pm, y, F in dl:
            s = loc(ids.cuda(), am.cuda(), F.cuda()).float().cpu()
            s = s.masked_fill(~pm, -1e4)
            for i in range(s.shape[0]):
                valid = [(float(s[i, j]), j) for j in range(s.shape[1]) if pm[i, j]]
                valid.sort(reverse=True)
                tops.append([(j, sc) for sc, j in valid[:args.k]])
            idx += s.shape[0]
    print("打分完成", flush=True)

    # 2) 读者
    rbase = "E:/KV_models/Qwen2.5-3B-Instruct"
    rtok = AutoTokenizer.from_pretrained(rbase)
    if rtok.pad_token is None:
        rtok.pad_token = rtok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        rbase, quantization_config=bnb, device_map="cuda")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    recs = {d["id"]: d for d in build_dataset(args.data)}
    frac = args.budget / 100.0
    out, t0 = [], time.time()
    for i, (row, top) in enumerate(zip(rows, tops)):
        qid = row["qid"]
        rec = recs.get(qid) or recs.get(row["key"][0])
        if rec is None:
            continue
        ans = float(rec["exe_ans"])
        for rank, (pi, sc) in enumerate(top):
            a, b, _ = row["pairs"][pi]
            ca, cb = row["cands"][a], row["cands"][b]
            rc = [[ca["r"], ca["c"]], [cb["r"], cb["c"]]]
            ctx = oracle_text(rtok, rec, frac, [], cells_rc=rc)
            prompt = "文：\n" + ctx + "\n\n问：" + row["q"].strip() + "\n式："
            pi_t = rtok(prompt, return_tensors="pt", add_special_tokens=False,
                        truncation=True, max_length=1500).to("cuda")
            with torch.no_grad():
                g = model.generate(**pi_t, max_new_tokens=40, do_sample=False,
                                   pad_token_id=rtok.pad_token_id,
                                   stop_strings=["\n", "答"], tokenizer=rtok)
            txt = rtok.decode(g[0][pi_t["input_ids"].shape[1]:],
                              skip_special_tokens=True)
            v = run_prog(txt)
            out.append(dict(qid=qid, budget=args.budget, rank=rank, pair=rc,
                            score=sc, ok=bool(close(v, ans)),
                            prog=clean(txt)[:60]))
        if (i + 1) % 5 == 0:
            print("  %d/%d  已跑 %d 次  剩余约 %.0f 分钟"
                  % (i + 1, len(rows), len(out),
                     (time.time() - t0) / len(out) * (len(rows) - i - 1)
                     * args.k / 60), flush=True)

    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    by = {}
    for r in out:
        by.setdefault(r["qid"], []).append(r)
    hi = sum(1 for v in by.values() if any(x["ok"] for x in v))
    lo = sum(1 for v in by.values() if not any(x["ok"] for x in v))
    top1 = sum(1 for v in by.values() if v[0]["ok"])
    print("\n题数 %d | top-1 对答对 %d | 8 对中至少一对答对 %d | 全错 %d"
          % (len(by), top1, hi, lo))
    print("重排上界（改用最好的那对）= %d/%d = %.3f；现状 = %.3f"
          % (hi, len(by), hi / len(by), top1 / len(by)))


if __name__ == "__main__":
    main()
