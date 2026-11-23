# -*- coding: utf-8 -*-
"""评测「带弃答的编译式读者」。

两种策略都报：
  A 仅弃答门：模型说 ABSTAIN → 转交语言模型
  B 弃答门 + 证书门：模型作答但程序里的数不在证据里 → 也转交
弃答/转交时用语言模型在该题该预算下的既有结果计分，故是真实混合 EM。
"""
import argparse
import json
import math
import re
import sys

import torch

sys.path.insert(0, ".")
from common import build_dataset                        # noqa: E402
from lora_budget import keep_text, run_prog, close     # noqa: E402

ABSTAIN = "ABSTAIN"
NUM = re.compile(r"\d+(?:\.\d+)?")


def numset(x):
    out = set()
    for t in NUM.findall(x or ""):
        try:
            out.add(round(float(t), 4))
        except ValueError:
            pass
    return out


def lm_of(path):
    d = json.load(open(path, encoding="utf-8"))
    rs = d["records"] if isinstance(d, dict) else d
    return {r["id"]: bool(r["correct"]) for r in rs}


def mcnemar(a, b):
    o = sum(1 for x, y in zip(a, b) if y and not x)
    r = sum(1 for x, y in zip(a, b) if x and not y)
    n = o + r
    k = min(o, r)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    return o, r, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="E:/KV_models/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="../results/lora_abs2")
    ap.add_argument("--data", default="../data/dev.json")
    ap.add_argument("--ids", default="../results/e2e_af20_base.json")
    ap.add_argument("--n", type=int, default=126)
    ap.add_argument("--budgets", default="3,5,10,20")
    ap.add_argument("--out", default="../results/lora_abs2_dev.json")
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

    res = {}
    for b in args.budgets.split(","):
        frac = float(b) / 100.0
        LM = lm_of("../results/vf_af%s.json" % b)
        n = ab = ok_a = ok_b = 0
        n_ans = ok_ans = 0
        vec_a, vec_b, vec_lm = [], [], []
        for r in rows:
            if r["id"] not in LM:
                continue
            ctx = keep_text(tok, r, frac)
            prompt = "预算：%s%%\n文：\n%s\n\n问：%s\n式：" % (b, ctx, r["question"].strip())
            pi = tok(prompt, return_tensors="pt", add_special_tokens=False,
                     truncation=True, max_length=1200).to("cuda")
            with torch.no_grad():
                g = model.generate(**pi, max_new_tokens=40, do_sample=False,
                                   pad_token_id=tok.pad_token_id,
                                   stop_strings=["\n", "答"], tokenizer=tok)
            out = tok.decode(g[0][pi["input_ids"].shape[1]:],
                             skip_special_tokens=True).strip()
            n += 1
            vec_lm.append(LM[r["id"]])

            if out.upper().startswith(ABSTAIN):
                ab += 1
                ok_a += LM[r["id"]]
                ok_b += LM[r["id"]]
                vec_a.append(LM[r["id"]])   # 弃答即采用 LM 的结果
                vec_b.append(LM[r["id"]])
                continue

            v = run_prog(out)
            good = close(v, float(r["ans"]))
            ok_a += good
            vec_a.append(good)
            n_ans += 1
            ok_ans += good

            gn, cn = numset(out.split("=")[0]), numset(ctx)
            if bool(gn) and gn <= cn:
                ok_b += good
                vec_b.append(good)
            else:
                ok_b += LM[r["id"]]
                vec_b.append(LM[r["id"]])

        oa, ra, pa = mcnemar(vec_a, vec_lm)
        ob, rb, pb = mcnemar(vec_b, vec_lm)
        res[b] = dict(n=n, abstain=ab / n, em_a=ok_a / n, em_b=ok_b / n,
                      ans_acc=ok_ans / max(1, n_ans), ans_cov=n_ans / n,
                      lm=sum(vec_lm) / n,
                      pair_a=[oa, ra, pa], pair_b=[ob, rb, pb])
        print("预算 %-3s 弃答 %.3f | 覆盖 %.3f 作答准 %.3f | A %.3f (+%d/-%d p=%.2g)"
              " | B %.3f (+%d/-%d p=%.2g) | LM %.3f"
              % (b, ab / n, n_ans / n, res[b]["ans_acc"], ok_a / n, oa, ra, pa,
                 ok_b / n, ob, rb, pb, res[b]["lm"]), flush=True)

    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    for k in ("em_a", "em_b"):
        print("%s 平均 %.3f" % (k, sum(v[k] for v in res.values()) / len(res)))


if __name__ == "__main__":
    main()
