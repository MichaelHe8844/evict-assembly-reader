# -*- coding: utf-8 -*-
"""LoRA 训练：Qwen2.5-3B 学「问题 + 表 -> 可执行算式」，沿途评测。

与冒烟版的区别：
  - 用满单步训练集（3717 题），按 epoch 计步；
  - 每 eval_every 步在固定 dev 子集上评测，输出「步数-EM」曲线；
  - 保存验证最优的适配器。

用法：
  python lora_compile.py --epochs 3 --eval_every 200 --eval_n 100
"""
import argparse
import json
import math
import re
import time

import torch
from torch.utils.data import DataLoader, Dataset

OPS = {
    "add": lambda *a: sum(a),
    "subtract": lambda a, b: a - b,
    "multiply": lambda *a: math.prod(a),
    "divide": lambda a, b: a / b if b else None,
    "greater": lambda a, b: 1.0 if a > b else 0.0,
    "average": lambda *a: sum(a) / len(a),
    "exp": lambda a, b: a ** b,
}
SAFE = re.compile(r"^[0-9eE\.\,\-\+\(\)\sa-z_]+$")
CONST = re.compile(r"const_(\d+)")


def linearize(x):
    tab = x.get("table") or []
    rows = [" | ".join(str(c).strip() for c in r) for r in tab[:24]]
    pre = " ".join(x.get("pre_text") or [])
    post = " ".join(x.get("post_text") or [])
    body = (pre + " " + post).strip()
    return "表：\n" + "\n".join(rows) + "\n\n文：\n" + body[:1600]


def prompt_of(x):
    return linearize(x) + "\n\n问：" + x["qa"]["question"].strip() + "\n式："


def clean(prog):
    p = prog.strip().split("\n")[0].strip()
    p = re.split(r"答|式|问", p)[0].strip()
    p = p.split("=")[0].strip()          # 允许 "a - b = c" 的中缀写法
    return CONST.sub(r"\1", p)


def run_prog(prog):
    p = clean(prog).replace("%", "")
    if not SAFE.match(p):
        return None
    if "(" not in p:                     # 中缀：只允许四则
        if not re.match(r"^[0-9eE\.\,\-\+\*/\s]+$", p):
            return None
        try:
            return float(eval(p, {"__builtins__": {}}))
        except Exception:
            return None
    try:
        return float(eval(p, {"__builtins__": {}}, dict(OPS)))
    except Exception:
        return None


def close(a, b, tol=1e-3):
    if a is None or b is None:
        return False
    return abs(a - b) <= tol * max(1.0, abs(b))


def load_rows(path, single=True):
    d = json.load(open(path, encoding="utf-8"))
    out = []
    for x in d:
        qa = x.get("qa") or {}
        prog = (qa.get("program") or "").strip()
        if not prog or not prog.endswith(")"):
            continue
        if single and len(qa.get("steps") or []) != 1:
            continue
        try:
            ans = float(qa.get("exe_ans"))
        except (TypeError, ValueError):
            continue
        out.append(dict(id=x["id"], prompt=prompt_of(x),
                        target=CONST.sub(r"\1", prog), ans=ans))
    return out


class DS(Dataset):
    def __init__(self, rows, tok, max_len=896):
        self.items = []
        for r in rows:
            pi = tok(r["prompt"], add_special_tokens=False)["input_ids"]
            ti = tok(r["target"] + tok.eos_token, add_special_tokens=False)["input_ids"]
            if len(pi) + len(ti) > max_len:
                pi = pi[-(max_len - len(ti) - 8):]
            ids = pi + ti
            self.items.append((ids, [-100] * len(pi) + ti))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def coll(batch, pad):
    L = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), L), pad, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    am = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (a, b) in enumerate(batch):
        ids[i, :len(a)] = torch.tensor(a)
        lab[i, :len(b)] = torch.tensor(b)
        am[i, :len(a)] = 1
    return ids, am, lab


@torch.no_grad()
def evaluate(model, tok, rows, n, max_len):
    model.eval()
    ok_p = ok_e = 0
    samples = []
    for r in rows[:n]:
        pi = tok(r["prompt"], return_tensors="pt", add_special_tokens=False,
                 truncation=True, max_length=max_len - 64).to("cuda")
        g = model.generate(**pi, max_new_tokens=40, do_sample=False,
                           pad_token_id=tok.pad_token_id,
                           stop_strings=["\n", "答"], tokenizer=tok)
        out = tok.decode(g[0][pi["input_ids"].shape[1]:], skip_special_tokens=True)
        val = run_prog(out)
        ok_p += val is not None
        ok_e += close(val, r["ans"])
        if len(samples) < 4:
            samples.append((clean(out)[:56], r["target"][:44], val, r["ans"]))
    model.train()
    m = min(n, len(rows))
    return ok_p / m, ok_e / m, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="E:/KV_models/Qwen2.5-3B-Instruct")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=896)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_n", type=int, default=100)
    ap.add_argument("--out", default="../results/lora_full")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tr = load_rows("../data/train.json")
    dv = load_rows("../data/dev.json")
    print("[1/4] 训练 %d 题｜dev %d 题" % (len(tr), len(dv)), flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map="cuda")
    model = prepare_model_for_kbit_training(model)
    if args.resume:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True)
        print("[2/4] 续训：%s" % args.resume, flush=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.gradient_checkpointing_enable()
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[2/4] 可训练参数 %.2fM" % (n_p / 1e6), flush=True)

    ds = DS(tr, tok, args.max_len)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True,
                    collate_fn=lambda b: coll(b, tok.pad_token_id))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr)
    total = int(args.epochs * len(ds) / args.accum)
    print("[3/4] 目标 %d 步（%.1f epoch），每 %d 步评测"
          % (total, args.epochs, args.eval_every), flush=True)

    model.train()
    it, step, t0, run = 0, 0, time.time(), 0.0
    best = (-1.0, 0)
    curve = []
    while step < total:
        for ids, am, lab in dl:
            ids, am, lab = ids.cuda(), am.cuda(), lab.cuda()
            loss = model(input_ids=ids, attention_mask=am, labels=lab).loss
            (loss / args.accum).backward()
            run += float(loss.detach())
            if (it + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    print("   step %4d/%d  loss %.4f  %.0fs"
                          % (step, total, run / (it + 1), time.time() - t0),
                          flush=True)
                if step % args.eval_every == 0 or step >= total:
                    r_exec, r_em, smp = evaluate(model, tok, dv, args.eval_n,
                                                 args.max_len)
                    print("   >>> step %4d  可执行 %.3f  EM %.3f"
                          % (step, r_exec, r_em), flush=True)
                    for a, b, v, y in smp:
                        print("       %-40s | gold %-28s | %s vs %s"
                              % (a, b, v, y), flush=True)
                    curve.append(dict(step=step, exec=r_exec, em=r_em))
                    if r_em > best[0]:
                        best = (r_em, step)
                        model.save_pretrained(args.out)
                        print("       已保存（best EM %.3f）" % r_em, flush=True)
                if step >= total:
                    break
            it += 1

    print("\n=== 曲线 ===")
    for c in curve:
        print("step %4d  可执行 %.3f  EM %.3f" % (c["step"], c["exec"], c["em"]))
    print("最优 EM %.3f @ step %d，适配器 %s" % (best[0], best[1], args.out))
    json.dump(dict(curve=curve, best=best[0], best_step=best[1]),
              open(args.out + "_curve.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
