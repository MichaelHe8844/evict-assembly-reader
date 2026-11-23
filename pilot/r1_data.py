# -*- coding: utf-8 -*-
"""R1 数据构建：把「问题 + 候选单元格对」整理成可训练/可评测的成对数据。

为什么不复用 pair_select.build()：那里带着 1200 token 的长度上限，
那是给模型前向用的；成对选择器只看表头文本，不需要长度过滤，
因此可以用满整个 FinQA train 集（几千题），把样本量从 111 提到几千。

输出结构：
  {qid, doc, q, gold: [[r,c]], cands: [{r,c,row,col,unit,v,txt}], pairs: [[i,j,label]]}
   label = 1 表示这一对覆盖了全部 gold 单元格
"""
import json
import re
import sys

sys.path.insert(0, ".")
from common import build_dataset  # noqa: E402
from anchor_auto import extract  # noqa: E402
from ground import cells_from_segs, rank, tokens  # noqa: E402
from operand_loc import gold_cells  # noqa: E402
from arm_a import tokenize_sample  # noqa: E402
from arm_c import fit_budget  # noqa: E402

SINGLE = re.compile(r"^[a-z]+\s*\([^()]*\)$")


def cell_text(c):
    """喂给编码器的单元格文本：行名 | 列名 | 单位 = 值。"""
    unit = (" (%s)" % c["unit"].strip()) if c.get("unit") else ""
    row = c.get("row", "") or ""
    col = c.get("col", "") or ""
    return ("%s | %s%s = %s" % (row, col, unit, c["raw"])).strip()


def build_pairs(path, raw_by_id, only_keys=None, exclude_keys=None, max_pairs=28,
                budget=0.20, topk=8, tok=None):
    ds = build_dataset(path)
    out, seen = [], set()
    for d in ds:
        try:
            key = (d["id"], round(float(d["exe_ans"]), 6))
        except (TypeError, ValueError):
            continue
        if key in seen:
            continue
        raw = raw_by_id.get(d["id"])
        if raw is None:
            continue
        prog = ((raw.get("qa") or {}).get("program") or "").strip()
        if not SINGLE.match(prog):
            continue
        if only_keys is not None and key not in only_keys:
            continue
        if exclude_keys is not None and key in exclude_keys:
            continue
        gc = gold_cells(raw)
        if not gc or any(x is None for x in gc):
            continue
        seen.add(key)
        # 候选池：训练阶段不做预算压缩，把整表的单元格交给 grounder 排序
        # （cells_from_segs 需要 token 区间，这里直接按 seg 结构构造，省掉分词）
        rh, cn, un = {}, {}, {}
        for s in d["segs"]:
            k = s.get("kind")
            if k == "row_header":
                rh[s.get("row")] = s["text"]
            elif k in ("col_name", "col_header"):
                cn[s.get("col")] = s["text"]
            elif k == "unit":
                un[s.get("col")] = s["text"]
        cells = []
        for s in d["segs"]:
            if s.get("kind") != "cell":
                continue
            from ground import to_float
            v = to_float(s["text"])
            if v is None:
                continue
            r, c = s.get("row"), s.get("col")
            cells.append(dict(r=r, c=c, v=v, raw=s["text"].strip(),
                              row=rh.get(r, ""), col=cn.get(c, ""), unit=un.get(c, ""),
                              row_t=tokens(rh.get(r, "")), col_t=tokens(cn.get(c, ""))))
        if budget is not None:
            # 部署设定：选择器只能在「压缩后仍保留」的格里挑。
            # 若用整表建池，选出来的格可能已被压缩掉，端到端会退回默认候选，
            # 等于把证据偷偷塞回上下文。
            doc, spans = tokenize_sample(tok, d["segs"])
            if len(doc) < 64:
                continue
            B = max(8, int(budget * len(doc)))
            mc = max(1, int((0.85 * B) / 6))
            idx, _ = extract(d["segs"], d["question"], max_cells=mc)
            keep = set(fit_budget([i for i in idx if spans[i][0] >= 0],
                                  d["segs"], spans, B))
            cells = cells_from_segs(d["segs"], spans, keep)
        cand = rank(cells, d["question"], topk=topk)
        if len(cand) < 2:
            continue
        gold = [(c["r"], c["c"]) for c in gc if c]
        cs = [c for _, c in cand]
        # 保存 grounder 原始分：Ind / Joint-R 基线要用它，否则基线会被削弱
        for s_, c in cand:
            c["gscore"] = float(s_)
        for c in cs:
            c["txt"] = cell_text(c)
        pairs = []
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                cov = set(gold) <= {(cs[i]["r"], cs[i]["c"]), (cs[j]["r"], cs[j]["c"])}
                pairs.append([i, j, 1 if cov else 0])
        if len(pairs) > max_pairs:
            # 保留全部正例 + 随机负例
            pos = [p for p in pairs if p[2]]
            neg = [p for p in pairs if not p[2]][: max_pairs - len(pos)]
            pairs = pos + neg
        out.append(dict(qid=d["id"], key=[d["id"], round(float(d["exe_ans"]), 6)],
                        doc=d["id"].rsplit("-", 1)[0], q=d["question"],
                        gold=gold, cands=cs, pairs=pairs))
    return out


def main():
    from transformers import AutoTokenizer
    which = sys.argv[1] if len(sys.argv) > 1 else "train"
    pool = sys.argv[2] if len(sys.argv) > 2 else "comp"
    if pool == "full":
        bud, suf = None, "_full"
    elif pool.startswith("b") and pool[1:].isdigit():
        bud, suf = float(pool[1:]) / 100.0, "_" + pool
    else:
        bud, suf = 0.20, ""
    tok = AutoTokenizer.from_pretrained("E:/KV_models/Qwen2.5-3B-Instruct")
    if which == "train":
        raw = {e["id"]: e for e in json.load(open("../data/train.json", encoding="utf-8"))}
        rows = build_pairs("../data/train.json", raw, tok=tok, budget=bud)
        json.dump(rows, open("../results/r1_pairs_train%s.json" % suf, "w"), ensure_ascii=False)
    else:
        raw = {e["id"]: e for e in json.load(open("../data/dev.json", encoding="utf-8"))}
        ev = {tuple(x["key"]) if "key" in x else (x[0], round(float(x[1]), 6))
              for x in json.load(open("../results/wide_keys.json", encoding="utf-8"))}
        rows = build_pairs("../data/dev.json", raw, only_keys=ev, tok=tok, budget=bud)
        json.dump(rows, open("../results/r1_pairs_dev%s.json" % suf, "w"), ensure_ascii=False)
    np_ = sum(len(r["pairs"]) for r in rows)
    npos = sum(p[2] for r in rows for p in r["pairs"])
    print("%s: %d 题 / %d 对 / 正例 %d (%.1f%%) / 覆盖文档 %d"
          % (which, len(rows), np_, npos, 100 * npos / max(1, np_),
             len({r["doc"] for r in rows})))


if __name__ == "__main__":
    main()
