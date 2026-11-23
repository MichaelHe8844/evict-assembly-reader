# -*- coding: utf-8 -*-
"""评测指针式读者：程序里的 [k] 解析为证据中第 k 个数值。

弃答 → 转交语言模型；指针越界/无法解析 → 同样转交。
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
from lora_budgeted import mention_list                 # noqa: E402

ABSTAIN = "ABSTAIN"
REF = re.compile(r"\[(\d+)\]")


def resolve(prog, nums):
    """把 [k] 换成实际数值；越界返回 None。"""
    def rep(m):
        k = int(m.group(1))
        return str(nums[k - 1]) if 1 <= k <= len(nums) else "@@"
    out = REF.sub(rep, prog)
    return None if "@@" in out else out


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
    ap.add_argument("--adapter", default="../results/lora_ptr")
    ap.add_argument("--data", default="../data/dev.json")
    ap.add_argument("--ids", default="../results/e2e_af20_base.json")
    ap.add_argument("--budgets", default="3,5,10,20")
    ap.add_argument("--out", default="../results/lora_ptr_dev.json")
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
        n = ab = ok = bad_ref = 0
        n_ans = ok_ans = 0
        vec, vec_lm = [], []
        for r in rows:
            if r["id"] not in LM:
                continue
            ctx = keep_text(tok, r, frac)
            nums = mention_list(ctx)
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

            ans_ok = False
            if out.upper().startswith(ABSTAIN):
                ab += 1
            else:
                resolved = resolve(out.split("=")[0], nums)
                if resolved is None:
                    bad_ref += 1
                else:
                    val = run_prog(resolved)
                    ans_ok = close(val, float(r["ans"]))
                    n_ans += 1
                    ok_ans += ans_ok

            ok += ans_ok if not out.upper().startswith(ABSTAIN) else LM[r["id"]]
            vec.append(ans_ok if not out.upper().startswith(ABSTAIN) else LM[r["id"]])

        o, rr, p = mcnemar(vec, vec_lm)
        res[b] = dict(n=n, abstain=ab / n, bad_ref=bad_ref / n,
                      coverage=n_ans / n, ans_acc=ok_ans / max(1, n_ans),
                      em=ok / n, lm=sum(vec_lm) / n, pair=[o, rr, p])
        print("预算 %-3s 弃答 %.3f 越界 %.3f | 覆盖 %.3f 作答准 %.3f | 混合 EM %.3f | LM %.3f | 我独对 %d / LM 独对 %d p=%.2g"
              % (b, ab / n, bad_ref / n, n_ans / n, res[b]["ans_acc"], ok / n,
                 res[b]["lm"], rr, o, p), flush=True)

    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("平均 EM %.3f" % (sum(v["em"] for v in res.values()) / len(res)))


if __name__ == "__main__":
    main()
