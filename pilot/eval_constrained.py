"""在约束解码下评测编译式读者：数值 token 只能拼出证据中出现过的数。"""
import argparse
import json
import re
import sys
import time

import torch

sys.path.insert(0, ".")

from common import build_dataset            # noqa: E402
from lora_budget import clean, close, keep_text, run_prog   # noqa: E402
from lora_budgeted import mention_list      # noqa: E402
from constrained_decode import NumberConstraint, allowed_numbers  # noqa: E402

NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="E:/KV_models/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="../results/lora_bc")
    ap.add_argument("--data", default="../data/fin_test.json")
    ap.add_argument("--ids", default="../results/vf_x10_sel.json")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--budgets", default="3,5,10,20")
    ap.add_argument("--consts", default="0,1,2,100,1000",
                    help="冻结的算术常数表（先声明后用，不做调参）")
    ap.add_argument("--use_question", action="store_true", default=True)
    ap.add_argument("--out", default="../results/constrained_test.json")
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, LogitsProcessorList)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    keep = {r["id"] for r in json.load(open(args.ids, encoding="utf-8"))["records"]}
    rows, seen = [], set()
    for d in build_dataset(args.data):
        if d["id"] not in keep or d["id"] in seen:
            continue
        seen.add(d["id"])
        try:
            ans = float(d["exe_ans"])
        except (TypeError, ValueError):
            continue
        rows.append(dict(id=d["id"], rec=d, q=d["question"], ans=ans))
        if len(rows) >= args.n:
            break
    print("评测 %d 题" % len(rows), flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map="cuda")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    out, details = {}, {}
    for b in args.budgets.split(","):
        frac = float(b) / 100.0
        t0 = time.time()
        ok = oob_q = oob_n = 0
        det = []
        for r in rows:
            ctx = keep_text(tok, r["rec"], frac)
            allowed = allowed_numbers(ctx, NUM)
            if args.use_question:                      # 题面里的数也是合法输入
                allowed += allowed_numbers(r["q"], NUM)
            allowed += [c for c in args.consts.split(",") if c]   # 冻结常数表
            prompt = "文：\n" + ctx + "\n\n问：" + r["q"].strip() + "\n式："
            pi = tok(prompt, return_tensors="pt", add_special_tokens=False,
                     truncation=True, max_length=1500).to("cuda")
            proc = NumberConstraint(tok, allowed)
            with torch.no_grad():
                g = model.generate(
                    **pi, max_new_tokens=40, do_sample=False,
                    pad_token_id=tok.pad_token_id,
                    logits_processor=LogitsProcessorList([proc]))
            text = tok.decode(g[0][pi["input_ids"].shape[1]:],
                              skip_special_tokens=True)
            val = run_prog(text)
            good = close(val, r["ans"])
            ok += good
            nums = [round(float(m.group(0).replace(",", "")), 4)
                    for m in NUM.finditer(clean(text).split("=")[0])]
            aset = {round(float(s.replace(",", "")), 4) for s in allowed}
            miss = [v for v in nums if v not in aset]
            oob_q += bool(miss)
            oob_n += len(miss)
            det.append(dict(id=r["id"], gen=clean(text)[:80], ok=good,
                            oob=len(miss), steps=proc.stats["steps"]))
        n = len(rows)
        out[b] = dict(n=n, em=ok / n, oob_question=oob_q / n, oob_numbers=oob_n,
                      sec=time.time() - t0, consts=args.consts,
                      use_question=bool(args.use_question))
        details[b] = det
        print("预算 %-3s 可执行 — EM %.3f  越界题 %.3f  越界数 %d  (%.0fs)"
              % (b, ok / n, oob_q / n, oob_n, time.time() - t0), flush=True)

    print("=== 约束解码 ===", flush=True)
    for b, v in out.items():
        print("  %-3s EM %.3f | 越界题 %.3f | 越界数 %d"
              % (b, v["em"], v["oob_question"], v["oob_numbers"]), flush=True)
    json.dump(dict(summary=out, detail=details),
              open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
