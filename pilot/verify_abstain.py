# -*- coding: utf-8 -*-
"""可验证弃答：编译出的程序，其数字是否都能在证据里找到？

若不能，说明读者在"编"——此时应弃答、交给语言模型。
本脚本只做后验计算，不训练、不改模型：
  1) 对每条生成，抽出其中的数值；
  2) 检查这些数值是否都出现在该档的压缩证据中；
  3) 比较三种策略：纯编译、纯 LM、规则路由（可验证则用编译，否则用 LM）。
"""
import json
import re
import sys

sys.path.insert(0, ".")
from common import build_dataset            # noqa: E402
from lora_budget import keep_text           # noqa: E402
from transformers import AutoTokenizer      # noqa: E402

NUM = re.compile(r"\d+(?:\.\d+)?")


def nums(s):
    return [x for x in NUM.findall(s or "")]


def norm(x):
    try:
        return round(float(x), 4)
    except ValueError:
        return None


def load_lm(path, cond=None):
    d = json.load(open(path, encoding="utf-8"))
    rs = d["records"] if isinstance(d, dict) else d
    if cond:
        rs = [r for r in rs if r.get("cond") == cond]
    return {r["id"]: bool(r["correct"]) for r in rs}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if which == "dev":
        bpath, ids_path, data = "../results/lora_budget.json", "../results/e2e_af20_base.json", "../data/dev.json"
        LM = {"3": load_lm("../results/vf_af3.json"), "5": load_lm("../results/vf_af5.json"),
              "10": load_lm("../results/vf_af10.json"), "20": load_lm("../results/vf_af20.json")}
    else:
        bpath, ids_path, data = "../results/lora_budget_test.json", "../results/vf_x10_sel.json", "../data/fin_test.json"
        LM = {"3": load_lm("../results/vf_x3.json"), "5": load_lm("../results/vf_x5.json"),
              "10": load_lm("../results/vf_x10.json"), "20": load_lm("../results/vf_x20.json")}

    tok = AutoTokenizer.from_pretrained("E:/KV_models/Qwen2.5-3B-Instruct")
    det = json.load(open(bpath, encoding="utf-8"))["detail"]
    keep_ids = {r["id"] for r in json.load(open(ids_path, encoding="utf-8"))["records"]}
    rows, seen = [], set()
    for d in build_dataset(data):
        if d["id"] in keep_ids and d["id"] not in seen:
            seen.add(d["id"])
            rows.append(d)
    print("%s：%d 题" % (which, len(rows)), flush=True)

    ctx_cache = {}
    out = {}
    for b in ("3", "5", "10", "20"):
        gen = {x["id"]: x for x in det[b]}
        frac = float(b) / 100.0
        n = ok_c = ok_l = ok_h = cert = 0
        n_cert = ok_cert = 0
        for d in rows:
            qid = d["id"]
            if qid not in gen or qid not in LM.get(b, {}):
                continue
            key = (qid, b)
            if key not in ctx_cache:
                ctx_cache[key] = keep_text(tok, d, frac)
            ctx = ctx_cache[key]
            g = gen[qid]
            ns = [v for v in (norm(x) for x in nums(g["gen"])) if v is not None]
            # 可验证：程序里的每个数都能在证据文本中找到（按 4 位小数比较）
            cnums = {norm(x) for x in nums(ctx)}
            verifiable = bool(ns) and all(v in cnums for v in ns)
            n += 1
            ok_c += g["ok"]
            ok_l += LM[b][qid]
            cert += verifiable
            if verifiable:
                n_cert += 1
                ok_cert += g["ok"]
            ok_h += (g["ok"] if verifiable else LM[b][qid])
        out[b] = dict(n=n, cmp=ok_c / n, lm=ok_l / n, hybrid=ok_h / n, cert=cert / n)
        out[b]["acc_cert"] = ok_cert / max(1, n_cert)
        out[b]["acc_unc"] = (ok_c - ok_cert) / max(1, n - n_cert)
        print("预算 %-3s n=%d 可验证 %.3f | 编译 %.3f (可验证 %.3f / 不可验证 %.3f) LM %.3f 路由 %.3f"
              % (b, n, cert / n, ok_c / n, out[b]["acc_cert"], out[b]["acc_unc"],
                 ok_l / n, ok_h / n), flush=True)
    json.dump(out, open("../results/verifiable_%s.json" % which, "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
