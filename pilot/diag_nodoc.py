# -*- coding: utf-8 -*-
"""无文档对照：只给问题、不给文档，看模型是否「本来就知道答案」。

起因：Llama-3.2-3B 在 gold 数值已从保留集消失的情况下仍有 54% 答对，
这不像检索，更像记忆。FinQA dev 是公开集，作对照必须排除这条捷径。
"""
import sys
import torch
from transformers import AutoTokenizer

from common import build_dataset, norm_num
from model_utils import load_model
from chat_utils import build_split, render_probe
from arm_c import greedy, parse_pred, em

M = sys.argv[1] if len(sys.argv) > 1 else "E:/KV_models/Llama-3.2-3B-Instruct"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 150
Q8 = len(sys.argv) > 3 and sys.argv[3] == "8bit"

tok = AutoTokenizer.from_pretrained(M)
model = load_model(M, quant8=Q8).eval()
sys_text = "You are a financial analyst. Answer with a single number only."

ds = build_dataset("../data/dev.json")
seen, used, hits, n = set(), 0, 0, 0
for d in ds:
    if used >= LIMIT:
        break
    ci = d["group"]["C"][0] if d["group"].get("C") else None
    ri = d["group"]["R"][0] if d["group"].get("R") else None
    if ci is None or ri is None:
        continue
    key = (d["id"], d["row"], d["col"])
    if key in seen:
        continue
    seen.add(key)
    g = norm_num(d["gold_value"])
    if g is None:
        continue
    used += 1
    q = "What is the value of %s for %s?" % (d["segs"][ci]["text"], d["segs"][ri]["text"])
    pre, suf = build_split(tok, sys_text, "", render_probe(sys_text, q, "Respond with only the final number."))
    ids = pre + suf
    with torch.no_grad():
        o = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=True)
    out = greedy(model, o.past_key_values, o.logits, len(ids), tok, 20)
    txt = tok.decode(out, skip_special_tokens=True).strip()
    pred = parse_pred(txt)
    ok = em(pred, float(g))
    hits += ok
    n += 1
    if ok:
        print("  HIT", d["id"][:24], pred, g, "|", txt[:20])

print("\n无文档答对 %d / %d = %.3f" % (hits, n, hits / max(1, n)))
