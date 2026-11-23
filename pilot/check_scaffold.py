# -*- coding: utf-8 -*-
"""校验：Qwen 下 template 脚手架与旧硬编码逐 token 相同（保证已有结果可复现）。"""
import json
from transformers import AutoTokenizer
from common import build_dataset
from pot_prompt import FEWSHOT, INSTR
from chat_utils import build_split, render_probe
from arm_a import tokenize_sample

M = "E:/KV_models/Qwen2.5-3B-Instruct"
tok = AutoTokenizer.from_pretrained(M)

sys_text = "You are a financial analyst. Answer with a single number only."
legacy_pre = tok("<|im_start|>system\n" + sys_text + "<|im_end|>\n<|im_start|>user\n",
                 add_special_tokens=False)["input_ids"]

ds = build_dataset("../data/dev.json")[:5]
bad = 0
for d in ds:
    q = "What is the value of A for B?"
    legacy_suf = tok("\n\nQuestion: %s\n%s<|im_end|>\n<|im_start|>assistant\n"
                     % (q, "Respond with only the final number."),
                     add_special_tokens=False)["input_ids"]
    p, s = build_split(tok, sys_text, "", render_probe(sys_text, q, "Respond with only the final number."))
    ok = (p == legacy_pre) and (s == legacy_suf)
    if not ok:
        bad += 1
        print("MISMATCH", d["id"], len(p), len(legacy_pre), len(s), len(legacy_suf))

# pot 分支
pre2 = tok("<|im_start|>system\nYou are a financial analyst.<|im_end|>\n<|im_start|>user\n",
           add_special_tokens=False)["input_ids"]
few = tok(FEWSHOT + "\nDocument:\n", add_special_tokens=False)["input_ids"]
p2, s2 = build_split(tok, "You are a financial analyst.", FEWSHOT + "\nDocument:\n",
                     "\n\nQuestion: Q\n" + INSTR)
print("pot pre ok:", (p2 == pre2 + few), len(p2), len(pre2 + few))
print("probe cells ok:", len(ds) - bad, "/", len(ds))
print(tok.apply_chat_template([{"role": "system", "content": "s"},
                               {"role": "user", "content": "u%%DOC%%v"}],
                              tokenize=False, add_generation_prompt=True).replace("\n", "\\n"))
