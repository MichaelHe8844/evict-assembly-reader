"""推理期执行验证搜索：采样 K 条程序，用执行器与证书筛选，再做一致性投票。

与贪心单次生成的区别只在推理：不改模型、不重训。
"""
import argparse
import json
import re
import sys
import time

import torch

sys.path.insert(0, ".")

from common import build_dataset                        # noqa: E402
from lora_budget import clean, close, keep_text, run_prog  # noqa: E402
from lora_budgeted import mention_list                  # noqa: E402

NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def cluster(vals):
    """按相对误差 1e-3 把数值聚成簇，返回 [(代表值, 成员下标)]。"""
    groups = []
    for idx, v in enumerate(vals):
        if v is None:
            continue
        placed = False
        for g in groups:
            if close(v, g[0]):
                g[1].append(idx)
                placed = True
                break
        if not placed:
            groups.append((v, [idx]))
    return groups


def vote(vals):
    gs = cluster(vals)
    if not gs:
        return None, 0
    gs.sort(key=lambda g: -len(g[1]))
    return gs[0][0], len(gs[0][1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="E:/KV_models/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="../results/lora_rft")
    ap.add_argument("--data", default="../data/fin_test.json")
    ap.add_argument("--ids", default="../results/vf_x10_sel.json")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--budgets", default="3,5,10,20")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new", type=int, default=40)
    ap.add_argument("--out", default="../results/tts_search.json")
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

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
    print("评测 %d 题，K=%d" % (len(rows), args.k), flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map="cuda")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    out, detail = {}, {}
    for b in args.budgets.split(","):
        frac = float(b) / 100.0
        t0 = time.time()
        res = {}
        for s in range(0, len(rows), args.bs):
            chunk = rows[s:s + args.bs]
            prompts, ctxs = [], []
            for r in chunk:
                ctx = keep_text(tok, r["rec"], frac)
                ctxs.append(ctx)
                prompts.append("文：\n" + ctx + "\n\n问：" + r["q"].strip() + "\n式：")
            enc = tok(prompts, return_tensors="pt", padding=True,
                      truncation=True, max_length=1500).to("cuda")
            with torch.no_grad():
                g = model.generate(
                    **enc, do_sample=True, temperature=args.temp, top_p=args.top_p,
                    num_return_sequences=args.k, max_new_tokens=args.max_new,
                    pad_token_id=tok.pad_token_id)
            gen = g[:, enc["input_ids"].shape[1]:]
            for j, r in enumerate(chunk):
                allow = {round(float(v), 4) for v in mention_list(ctxs[j])}
                samples = []
                for kk in range(args.k):
                    txt = clean(tok.decode(gen[j * args.k + kk],
                                           skip_special_tokens=True)).split("\n")[0]
                    val = run_prog(txt)
                    nums = {round(float(m.group(0).replace(",", "")), 4)
                            for m in NUM.finditer(txt.split("=")[0])}
                    samples.append(dict(prog=txt, val=val,
                                        cert=bool(nums) and nums <= allow))
                res[r["id"]] = dict(ans=r["ans"], samples=samples)
        n = len(rows)
        summary = {}
        for kk in (1, 2, 4, args.k):
            for mode in ("all", "cert"):
                ok = 0
                for r in rows:
                    rec = res[r["id"]]
                    smp = rec["samples"][:kk]
                    pick = smp if mode == "all" else [x for x in smp if x["cert"]]
                    if not pick:
                        continue
                    v, _ = vote([x["val"] for x in pick])
                    ok += close(v, rec["ans"])
                summary["%s@k%d" % (mode, kk)] = ok / n
        # 证书优先、失败回退全部
        ok = 0
        for r in rows:
            rec = res[r["id"]]
            cert = [x for x in rec["samples"] if x["cert"]]
            pick = cert if cert else rec["samples"]
            v, _ = vote([x["val"] for x in pick])
            ok += close(v, rec["ans"])
        summary["cert_first@k%d" % args.k] = ok / n
        out[b] = dict(n=n, **summary, sec=time.time() - t0)
        detail[b] = res
        print("预算 %-3s " % b + "  ".join("%s %.3f" % (k, v)
              for k, v in summary.items()) + "  (%.0fs)" % (time.time() - t0),
              flush=True)

    print("=== 推理期搜索 ===", flush=True)
    for b, v in out.items():
        print("  %-3s " % b + "  ".join("%s %.3f" % (k, x)
              for k, x in v.items() if isinstance(x, float) and k != "sec"),
              flush=True)
    json.dump(dict(summary=out, detail=detail), open(args.out, "w"),
              ensure_ascii=False)


if __name__ == "__main__":
    main()
