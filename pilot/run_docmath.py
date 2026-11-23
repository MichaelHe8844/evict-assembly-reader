# -*- coding: utf-8 -*-
"""
DocMath simplong 的下游验证（7B 8-bit，20% 预算）。

与 arm_c 的差别：
  1. 文档来自 DocMath 的 paragraphs（含 markdown 表格），用 docmath_segs 解析成行列网格；
  2. 没有 gold 单元格，因此不做 VRCU/ANR，只比 EM；
  3. **不能对长文档开 output_attentions**：28 层 × 28 头 × L² 会直接爆显存。
     改为「sdpa/eager 做无注意力 prefill → 只用最后 64 个 token 再前向一次取注意力」，
     此时注意力矩阵只有 64×L，代价可忽略。

用法：
  python run_docmath.py --limit 8 --conds full            # 校准：看模型能不能答
  python run_docmath.py --limit 40 --conds full,snap@20,auto_f@20
"""
import argparse
import copy
import json
import re

import numpy as np
import torch

from docmath_segs import parse_paragraphs
from anchor_auto import extract, table_index
from model_utils import load_model
from arm_c import trim_cache, greedy, em_scale, fit_budget
from transformers import AutoTokenizer

PLAIN = """Answer the question based on the document. Only output the final number.

Document:
{doc}
Question: {q}
Answer:"""

POT = """Solve the question with arithmetic. Write one line of expression, then the answer.
Only output the final number.

Example 1
Document:
item | 2019 | 2020
revenue | 120 | 150
Question: what is the percent increase in revenue from 2019 to 2020?
Expression: (150 - 120) / 120 * 100
Answer: 25

Example 2
Document:
segment | amount
north america | 400
europe | 250
Question: what is the total amount for all segments?
Expression: 400 + 250
Answer: 650

Document:
{doc}
Question: {q}
Answer:"""


def tokenize_segs(tok, segs):
    """带换行分隔的分词，返回 ids 与每段的 token 区间。"""
    ids, spans = [], []
    for k, s in enumerate(segs):
        t = tok(("\n" if k else "") + s["text"], add_special_tokens=False)["input_ids"]
        if not t:
            spans.append((-1, -1))
            continue
        spans.append((len(ids), len(ids) + len(t)))
        ids.extend(t)
    return ids, spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/docmath/simplong_testmini.jsonl")
    ap.add_argument("--model", default="E:/KV_models/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--conds", default="full")
    ap.add_argument("--prompt", default="plain", choices=["plain", "pot"])
    ap.add_argument("--chat", action="store_true", default=True)
    ap.add_argument("--no_chat", dest="chat", action="store_false")
    ap.add_argument("--n_new", type=int, default=48)
    ap.add_argument("--maxlen", type=int, default=4000)
    ap.add_argument("--out", default="../results/docmath_simplong.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, quant8=True)
    model.eval()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")][:args.limit]
    conds = args.conds.split(",")
    recs = []
    used = 0

    for row in rows:
        segs, diag = parse_paragraphs(row["paragraphs"])
        if diag["n_tables"] < 1:
            continue
        ids_doc, spans = tokenize_segs(tok, segs)
        n_doc = len(ids_doc)
        if n_doc < 200 or n_doc > args.maxlen:
            continue

        MARK = "%%DOC%%"
        if args.chat:
            if args.prompt == "pot":
                sys_p = ("You are a careful financial analyst. For each question, locate the "
                         "needed cells, write the arithmetic expression, then the final number.\n\n"
                         "Example 1\nitem | 2019 | 2020\nrevenue | 120 | 150\n"
                         "Question: what is the percent increase in revenue from 2019 to 2020?\n"
                         "Expression: (150 - 120) / 120 * 100\nAnswer: 25\n\n"
                         "Example 2\nsegment | amount\nnorth america | 400\neurope | 250\n"
                         "Question: what is the total amount for all segments?\n"
                         "Expression: 400 + 250\nAnswer: 650")
            else:
                sys_p = ("You are a careful financial analyst. Read the document and answer "
                         "the question with arithmetic. Output only the final number.")
            usr = "Document://n" + MARK + "\n\nQuestion: " + row["question"] + \
                  "\n\nAnswer with the final number only."
            msgs = [dict(role="system", content=sys_p), dict(role="user", content=usr)]
            full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            a, b = full.split(MARK)
            pre = tok(a, add_special_tokens=False)["input_ids"]
            suf = tok(b, add_special_tokens=False)["input_ids"]
        else:
            tpl = PLAIN if args.prompt == "plain" else POT
            prefix = tpl.split("{doc}")[0].replace("{q}", row["question"])
            prefix = prefix.replace("\nAnswer:", "").rstrip() + "\n"
            pre = tok(prefix, add_special_tokens=False)["input_ids"]
            suf = tok("\nQuestion: " + row["question"] + "\nAnswer:",
                      add_special_tokens=False)["input_ids"]

        ids = pre + ids_doc + suf
        off = len(pre)
        L = len(ids)
        W = min(64, len(suf))
        doc_pos = torch.arange(off, off + n_doc, device="cuda")

        # 1) 无注意力 prefill（长文档不开 output_attentions）
        with torch.no_grad():
            o1 = model(input_ids=torch.tensor([ids[:L - W]], device="cuda"),
                       use_cache=True)
        # 2) 仅最后 W 个 token 取注意力 → 注意力矩阵只有 W×L
        with torch.no_grad():
            o2 = model(input_ids=torch.tensor([ids[L - W:]], device="cuda"),
                       past_key_values=o1.past_key_values,
                       position_ids=torch.arange(L - W, L, device="cuda").unsqueeze(0),
                       output_attentions=True, use_cache=True)
        snap = torch.zeros(n_doc, device="cuda")
        for a in o2.attentions:                      # (1,H,W,K)
            snap += a[0].float()[:, :, off:off + n_doc].sum(dim=(0, 1))
        snap = snap.cpu().numpy()
        del o1
        base = copy.deepcopy(o2.past_key_values)
        torch.cuda.empty_cache()

        # 无监督抽取（问题 = DocMath 原问题）
        max_cells = max(1, int((0.85 * int(0.20 * n_doc)) / 6))
        try:
            auto_idx, _ = extract(segs, row["question"], max_cells=max_cells)
        except Exception:
            auto_idx = []
        protected = list(range(0, off)) + list(range(off + n_doc, L))

        for cond in conds:
            pol, _, bud = cond.partition("@")
            frac = float(bud) / 100.0 if bud else 1.0
            B = n_doc if pol == "full" else max(8, int(frac * n_doc))
            keep = np.zeros(n_doc, dtype=bool)
            if pol == "full":
                keep[:] = True
            elif pol == "snap":
                keep[np.argsort(-snap)[:B]] = True
            elif pol == "auto_f":
                g_spans = [(a + off, b + off) if a >= 0 else (-1, -1) for a, b in spans]
                toks = fit_budget([i for i in auto_idx if g_spans[i][0] >= off],
                                  segs, g_spans, B)
                for t in toks:
                    keep[t - off] = True
                # 与 arm_c 一致：锚点装完后，剩余预算按注意力分数填满，
                # 否则 auto_f 只占 3% 预算，与其他条件不可比。
                need = B - int(keep.sum())
                if need > 0:
                    cand = np.where(~keep)[0]
                    keep[cand[np.argsort(-snap[cand])[:need]]] = True
            else:
                raise ValueError(cond)

            idx = sorted(protected + (doc_pos[keep]).cpu().tolist())
            cut = [i for i in idx if i != L - 1]
            c = copy.deepcopy(base)
            trim_cache(c, cut)
            with torch.no_grad():
                o3 = model(input_ids=torch.tensor([[ids[L - 1]]], device="cuda"),
                           past_key_values=c,
                           position_ids=torch.tensor([[L - 1]], device="cuda"),
                           use_cache=True)
                out = greedy(model, o3.past_key_values, o3.logits, L, tok, args.n_new)
            text = tok.decode(out, skip_special_tokens=True).strip()
            nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
            pred = float(nums[-1]) if nums else None
            gold = float(row["ground_truth"])
            recs.append(dict(id=row["question_id"], cond=cond, pred=pred, gold=gold,
                             correct=em_scale(pred, gold), L=L, n_doc=n_doc,
                             keep_ratio=float(keep.sum()) / n_doc, text=text[:80]))
        used += 1
        del base, o2
        torch.cuda.empty_cache()
        print("  [%d/%d] L=%d n_doc=%d | %s" % (
            used, len(rows), L, n_doc, " ".join(
                "%s:%s" % (r["cond"], "✓" if r["correct"] else "✗")
                for r in recs[-len(conds):])), flush=True)

    print("\n%-10s %4s %8s %8s" % ("cond", "n", "EM", "mean keep"))
    for cond in conds:
        v = [r for r in recs if r["cond"] == cond]
        if v:
            print("%-10s %4d %8.3f %8.3f" % (
                cond, len(v), np.mean([r["correct"] for r in v]),
                np.mean([r["keep_ratio"] for r in v])))
    json.dump(dict(records=recs, prompt=args.prompt), open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False)
    print("写入", args.out)


if __name__ == "__main__":
    main()
