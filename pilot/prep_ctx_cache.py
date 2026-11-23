# -*- coding: utf-8 -*-
"""预计算：训练集每道单步题在四档预算下的压缩证据文本。

输出 results/ctx_cache_train.json：{qid: {"3": txt, "5": txt, "10": txt, "20": txt}}
"""
import json
import sys
import time

sys.path.insert(0, ".")
from common import build_dataset      # noqa: E402
from lora_budget import keep_text     # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

BUD = ("3", "5", "10", "20")


def main():
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    tok = AutoTokenizer.from_pretrained("E:/KV_models/Qwen2.5-3B-Instruct")
    raw = json.load(open("../data/train.json", encoding="utf-8"))
    single = set()
    for x in raw:
        qa = x.get("qa") or {}
        if len(qa.get("steps") or []) == 1 and (qa.get("program") or "").strip():
            single.add(x["id"])
    rows, seen = [], set()
    for d in build_dataset("../data/train.json"):
        if d["id"] not in single or d["id"] in seen:
            continue
        seen.add(d["id"])
        rows.append(d)
        if lim and len(rows) >= lim:
            break
    print("题目 %d" % len(rows), flush=True)

    out, t0 = {}, time.time()
    for i, d in enumerate(rows):
        out[d["id"]] = {b: keep_text(tok, d, float(b) / 100.0) for b in BUD}
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print("  %d/%d  %.1fs  预计 %.0f 分钟"
                  % (i + 1, len(rows), el, el / (i + 1) * (len(rows) - i - 1) / 60),
                  flush=True)
    json.dump(out, open("../results/ctx_cache_train.json", "w"), ensure_ascii=False)
    print("done %d 题，用时 %.0f 秒" % (len(out), time.time() - t0))


if __name__ == "__main__":
    main()
