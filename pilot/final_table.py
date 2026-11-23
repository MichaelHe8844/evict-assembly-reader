# -*- coding: utf-8 -*-
"""决定主张的 test 表：五个对照 × 四档预算，同一批题、同一份证据、同一判分。

协议固定（均在 dev 上选定，test 未参与）：
  · 弃答标签：gold 程序中的数「全部」缺失才标 ABSTAIN
  · 核验规则：程序中的数值字面量须全部出现在压缩证据中（4 位小数归一）
  · 回退模型：同底座 Qwen2.5-3B-Instruct 的 PoT 结果（vf_x{k}.json）
  · 预算门槛：b >= 10%

输出：results/final_table_test.json
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time

import torch

sys.path.insert(0, ".")
from common import build_dataset                        # noqa: E402
from lora_budget import keep_text, run_prog, close     # noqa: E402
from lora_budgeted import mention_list                 # noqa: E402

NUM = re.compile(r"\d+(?:\.\d+)?")
ABSTAIN = "ABSTAIN"


def numset(x):
    return {round(float(t), 4) for t in NUM.findall(x or "")}


def doc_of(qid):
    return qid.rsplit("-", 1)[0]


def boot_ci(pairs, B=2000, seed=0):
    """按来源文档聚类的配对 bootstrap。pairs: [(doc, a, b)]，统计 mean(b)-mean(a)。"""
    by = {}
    for d, a, b in pairs:
        by.setdefault(d, []).append((a, b))
    docs = list(by)
    rng = random.Random(seed)
    obs = sum(b - a for _, a, b in pairs) / len(pairs)
    vals = []
    for _ in range(B):
        take = [docs[rng.randrange(len(docs))] for _ in docs]
        num = den = 0
        for d in take:
            for a, b in by[d]:
                num += (b - a)
                den += 1
        vals.append(num / den)
    vals.sort()
    return obs, vals[int(0.025 * B)], vals[int(0.975 * B)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="E:/KV_models/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="../results/lora_abs2")
    ap.add_argument("--data", default="../data/fin_test.json")
    ap.add_argument("--ids", default="../results/vf_x10_sel.json")
    ap.add_argument("--out", default="../results/final_table_test.json")
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
    print("题数 %d" % len(rows), flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map="cuda")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    prog_det = json.load(open("../results/lora_budget_test.json", encoding="utf-8"))["detail"]
    out = {}
    for b in ("3", "5", "10", "20"):
        frac = float(b) / 100.0
        lm = {r["id"]: bool(r["correct"])
              for r in json.load(open("../results/vf_x%s.json" % b, encoding="utf-8"))["records"]}
        prog = {x["id"]: x for x in prog_det[b]}
        n = 0
        rec = []
        t_gen = 0.0
        for r in rows:
            if r["id"] not in lm or r["id"] not in prog:
                continue
            ctx = keep_text(tok, r, frac)
            nums = mention_list(ctx)
            prompt = "预算：%s%%\n文：\n%s\n\n问：%s\n式：" % (b, ctx, r["question"].strip())
            pi = tok(prompt, return_tensors="pt", add_special_tokens=False,
                     truncation=True, max_length=1200).to("cuda")
            t0 = time.time()
            with torch.no_grad():
                g = model.generate(**pi, max_new_tokens=40, do_sample=False,
                                   pad_token_id=tok.pad_token_id,
                                   stop_strings=["\n", "答"], tokenizer=tok)
            t_gen += time.time() - t0
            txt = tok.decode(g[0][pi["input_ids"].shape[1]:],
                             skip_special_tokens=True).strip()
            abst = txt.upper().startswith(ABSTAIN)
            v = None if abst else run_prog(txt)
            ans_ok = (not abst) and close(v, r["ans"])
            gn = numset(txt.split("=")[0]) if not abst else set()
            cert = bool(gn) and gn <= numset(ctx)

            ok2 = bool(prog[r["id"]]["ok"])                 # 程序读者（已训模型，字面量版）
            ok3 = (ok2 if (int(b) >= 10 and cert) else lm[r["id"]])   # 手写规则版
            ok4 = (lm[r["id"]] if abst else ans_ok)         # 学习型弃答 + 回退
            ok5 = (ok4 if (not abst and cert) else lm[r["id"]])       # + 核验
            rec.append(dict(doc=doc_of(r["id"]), lm=lm[r["id"]], p2=ok2, p3=ok3,
                            p4=ok4, p5=ok5, abst=abst, cert=cert,
                            released=(not abst) and cert))
            n += 1
        if not n:
            continue
        em = {k: sum(x[k] for x in rec) / n for k in ("lm", "p2", "p3", "p4", "p5")}
        cov = sum(1 for x in rec if not x["abst"]) / n
        rel = [x for x in rec if not x["abst"]]
        err_rel = (1 - sum(x["p4"] for x in rel) / len(rel)) if rel else float("nan")
        # 放行 = 最终采用程序答案（4/5 各自）
        rel4 = [x for x in rec if not x["abst"]]
        rel5 = [x for x in rec if x["released"]]
        out[b] = dict(
            n=n,
            em_lm=em["lm"], em_prog=em["p2"], em_rule=em["p3"],
            em_abs=em["p4"], em_full=em["p5"],
            cov4=len(rel4) / n, err4=(1 - sum(x["p4"] for x in rel4) / len(rel4)) if rel4 else None,
            cov5=len(rel5) / n, err5=(1 - sum(x["p5"] for x in rel5) / len(rel5)) if rel5 else None,
            fallback5=1 - len(rel5) / n,
            sec_per_q=t_gen / n,
            gain=(sum(x["p5"] - x["p3"] for x in rec) / n,
                  *boot_ci([(x["doc"], x["p3"], x["p5"]) for x in rec])[1:]),
            gain_vs_lm=(sum(x["p5"] - x["lm"] for x in rec) / n,
                        *boot_ci([(x["doc"], x["lm"], x["p5"]) for x in rec])[1:]),
        )
        d = out[b]
        print("预算 %-3s n=%d | LM %.3f 程序 %.3f 规则 %.3f 弃答 %.3f 完整 %.3f "
              "| 放行 %.3f 放行错误率 %.3f 回退 %.3f | %.0f ms/题"
              % (b, n, d["em_lm"], d["em_prog"], d["em_rule"], d["em_abs"], d["em_full"],
                 d["cov5"], d["err5"] or 0, d["fallback5"], d["sec_per_q"] * 1000),
              flush=True)
        print("      完整-规则 %+.3f [%.3f, %.3f] | 完整-LM %+.3f [%.3f, %.3f]"
              % (d["gain"][0], d["gain"][1], d["gain"][2],
                 d["gain_vs_lm"][0], d["gain_vs_lm"][1], d["gain_vs_lm"][2]), flush=True)

    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("平均 EM：LM %.3f 程序 %.3f 规则 %.3f 弃答 %.3f 完整 %.3f"
          % tuple(sum(v[k] for v in out.values()) / len(out)
                  for k in ("em_lm", "em_prog", "em_rule", "em_abs", "em_full")))


if __name__ == "__main__":
    main()
