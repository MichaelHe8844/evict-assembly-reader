"""RFT 采样：用当前读者自采样，只保留「可执行且答案正确」的程序。

产物是一份与 load_rows_budgeted 同格式的训练集（prompt / target / ans），
供 lora_budgeted.py --rows_json 直接训练。
"""
import argparse
import json
import random
import sys
import time

import torch

sys.path.insert(0, ".")

from lora_budget import clean, close, run_prog          # noqa: E402
from lora_budgeted import load_rows_budgeted            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="E:/KV_models/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="../results/lora_bc")
    ap.add_argument("--ctx_cache", default="../results/ctx_cache_train.json")
    ap.add_argument("--n_prompts", type=int, default=2000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="../results/rft_rows.json")
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    rows = load_rows_budgeted("../data/train.json", args.ctx_cache)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.n_prompts]
    print("采样 %d 条 prompt × K=%d" % (len(rows), args.k), flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map="cuda")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    kept, tried = [], 0
    t0 = time.time()
    for s in range(0, len(rows), args.bs):
        chunk = rows[s:s + args.bs]
        enc = tok([r["prompt"] for r in chunk], return_tensors="pt",
                  padding=True, truncation=True, max_length=1500).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc, do_sample=True, temperature=args.temp, top_p=args.top_p,
                num_return_sequences=args.k, max_new_tokens=args.max_new,
                pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        for j, r in enumerate(chunk):
            progs = []
            for kk in range(args.k):
                ids = gen[j * args.k + kk]
                txt = clean(tok.decode(ids, skip_special_tokens=True)).split("\n")[0]
                tried += 1
                if not txt:
                    continue
                val = run_prog(txt)
                if close(val, r["ans"]) and txt not in progs:
                    progs.append(txt)
            for pg in progs:
                kept.append(dict(id=r["id"], prompt=r["prompt"], target=pg,
                                 ans=r["ans"]))
        if (s // args.bs) % 40 == 0:
            print("  %d/%d  prompts，已保留 %d 条，%.0fs"
                  % (s, len(rows), len(kept), time.time() - t0), flush=True)

    print("采样 %d 次，保留 %d 条（接受率 %.3f）"
          % (tried, len(kept), len(kept) / max(1, tried)), flush=True)
    json.dump(kept, open(args.out, "w"), ensure_ascii=False)
    print("写出", args.out, flush=True)


if __name__ == "__main__":
    main()
