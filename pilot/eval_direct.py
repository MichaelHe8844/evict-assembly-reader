# -*- coding: utf-8 -*-
"""评测直接答题基线：同底座、同训练样本、同 LoRA 配置，只把目标换成数值。

输出：results/direct_test.json（四档 EM，与程序读者同一批题、同一判分）
"""
import argparse
import json
import math
import re
import sys
import time

import torch

sys.path.insert(0, ".")
from common import build_dataset                        # noqa: E402
from lora_budget import keep_text, close               # noqa: E402

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def parse_num(txt):
    m = NUM.search((txt or "").replace("%", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def mcnemar(a, b):
    o = sum(1 for x, y in zip(a, b) if y and not x)
    r = sum(1 for x, y in zip(a, b) if x and not y)
    n = o + r
    k = min(o, r)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    return o, r, p


def boot(pairs, B=2000, seed=0):
    import random
    by = {}
    for d, a, b in pairs:
        by.setdefault(d, []).append((a, b))
    docs = list(by)
    rng = random.Random(seed)
    obs = sum(b - a for _, a, b in pairs) / len(pairs)
    vals = []
    for _ in range(B):
        t = [docs[rng.randrange(len(docs))] for _ in docs]
        s = n = 0
        for d in t:
            for a, b in by[d]:
                s += b - a
                n += 1
        vals.append(s / n)
    vals.sort()
    return obs, vals[int(0.025 * B)], vals[int(0.975 * B)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="E:/KV_models/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="../results/lora_direct")
    ap.add_argument("--data", default="../data/fin_test.json")
    ap.add_argument("--ids", default="../results/vf_x10_sel.json")
    ap.add_argument("--budgets", default="3,5,10,20")
    ap.add_argument("--out", default="../results/direct_test.json")
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    keep_ids = {r["id"] for r in json.load(open(args.ids, encoding="utf-8"))["records"]}
    rows, seen = [], set()
    for d in build_dataset(args.data):
        if d["id"] in keep_ids and d["id"] not in seen:
            try:
                d["ans"] = float(d["exe_ans"])
            except (TypeError, ValueError):
                continue
            seen.add(d["id"])
            rows.append(d)
    print("评测 %d 题" % len(rows), flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map="cuda")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    prog = json.load(open("../results/lora_budget_test.json", encoding="utf-8"))["detail"]
    out, details = {}, {}
    for b in args.budgets.split(","):
        frac = float(b) / 100.0
        lm = {r["id"]: bool(r["correct"])
              for r in json.load(open("../results/vf_x%s.json" % b, encoding="utf-8"))["records"]}
        pk = {x["id"]: x["ok"] for x in prog[b]}
        n = ok = ok_num = 0
        t0 = time.time()
        rec = []
        for r in rows:
            ctx = keep_text(tok, r, frac)
            prompt = "预算：%s%%\n文：\n%s\n\n问：%s\n答：" % (b, ctx, r["question"].strip())
            pi = tok(prompt, return_tensors="pt", add_special_tokens=False,
                     truncation=True, max_length=1200).to("cuda")
            with torch.no_grad():
                g = model.generate(**pi, max_new_tokens=24, do_sample=False,
                                   pad_token_id=tok.pad_token_id,
                                   stop_strings=["\n"], tokenizer=tok)
            txt = tok.decode(g[0][pi["input_ids"].shape[1]:],
                             skip_special_tokens=True).strip()
            v = parse_num(txt)
            good = close(v, r["ans"])
            n += 1
            ok += good
            ok_num += v is not None
            rec.append((r["id"].rsplit("-", 1)[0], good, pk.get(r["id"], False)))
        em = ok / n
        gain = boot([(d, a, c) for d, a, c in rec])   # 程序 − 直接答题
        details[b] = [dict(id=i, doc=d, ok=g) for i, g, _ in rec]
        out[b] = dict(n=n, em_direct=em, parse_rate=ok_num / n,
                      em_prog=sum(1 for _, _, c in rec if c) / n,
                      lm=sum(lm[r["id"]] for r in rows if r["id"] in lm) / n,
                      prog_minus_direct=list(gain),
                      sec_per_q=(time.time() - t0) / n)
        d = out[b]
        print("预算 %-3s n=%d | 直接答题 %.3f（解析率 %.3f）| 程序读者 %.3f | 同底座 PoT %.3f "
              "| 程序−直接 %+.3f [%+.3f, %+.3f] | %.0f ms/题"
              % (b, n, d["em_direct"], d["parse_rate"], d["em_prog"], d["lm"],
                 gain[0], gain[1], gain[2], d["sec_per_q"] * 1000), flush=True)

    json.dump(dict(summary=out, detail=details), open(args.out, "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
