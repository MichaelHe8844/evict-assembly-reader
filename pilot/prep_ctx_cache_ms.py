# -*- coding: utf-8 -*-
"""预计算多步题的压缩证据（四档），用于把读者扩到多步。

与 prep_ctx_cache_vf.py 的差别只有筛选：这里取 qa.steps 长度 >= 2 的题，
装配仍用 4.2 的先数值顺序。输出 results/ctx_cache_train_ms.json。
"""
import json
import sys
import time

sys.path.insert(0, ".")
import arm_c                              # noqa: E402
arm_c.VALUE_FIRST = True
from common import build_dataset          # noqa: E402
from lora_budget import keep_text         # noqa: E402
from transformers import AutoTokenizer    # noqa: E402

BUD = ("3", "5", "10", "20")


def main():
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    out_path = (sys.argv[2] if len(sys.argv) > 2
                else "../results/ctx_cache_train_ms.json")
    tok = AutoTokenizer.from_pretrained("E:/KV_models/Qwen2.5-3B-Instruct")
    raw = json.load(open("../data/train.json", encoding="utf-8"))
    multi = set()
    for x in raw:
        qa = x.get("qa") or {}
        if len(qa.get("steps") or []) >= 2 and (qa.get("program") or "").strip():
            multi.add(x["id"])
    rows, seen = [], set()
    for d in build_dataset("../data/train.json"):
        if d["id"] not in multi or d["id"] in seen:
            continue
        seen.add(d["id"])
        rows.append(d)
        if lim and len(rows) >= lim:
            break
    print("多步题 %d" % len(rows), flush=True)

    out, t0 = {}, time.time()
    for i, d in enumerate(rows):
        out[d["id"]] = {b: keep_text(tok, d, float(b) / 100.0) for b in BUD}
        if (i + 1) % 200 == 0:
            print("  %d/%d  %.1fs" % (i + 1, len(rows), time.time() - t0),
                  flush=True)
    json.dump(out, open(out_path, "w"), ensure_ascii=False)
    print("done %d 题，用时 %.0f 秒" % (len(out), time.time() - t0))


if __name__ == "__main__":
    main()
