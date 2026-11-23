# -*- coding: utf-8 -*-
"""锚点组的预算分配：把 4.2 的按类装箱换成一个有目标函数的组合优化问题。

问题（AGBA, anchor-group budget allocation）
    候选单元 U={u_i}，每个单元有其 token 集 T_i（单元格 + 行头 + 列名 + 单位）
    与价值 w_i ≥ 0；预算 B（文档区间可保留的 token 数）。
        max  Σ_i w_i x_i     s.t.  |∪_{i:x_i=1} T_i| ≤ B,  x_i ∈ {0,1}
    注意代价是**并集**大小：同一行的多个单元共享行头，因此不是线性背包。

性质
    · 单元两两不交时退化为 0-1 背包，DP 精确求解 O(nB)。
    · 一般情形含背包为特例，故 NP-hard；用密度贪心求近似。
    · 贪心按「边际价值 / 边际新 token」排序（不是按类优先级），
      取 max(贪心解, 单个最优单元) 时在不相交情形有 1/2 近似保证；
      相交情形无常数保证，故另给一个可证上界（见 dp_upper）。

上界（用于量化贪心差多少）
    把每个单元的代价换成它的**独占 token**（不被任何其他单元共享的部分），
    任何可行解在该代价下仍然可行，因此其最优值 ≥ 原问题最优值 —— 是个上界。
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np

from common import UNIT_TEXT_RE
from anchor_auto import table_index, UNIT_HINT


def token_spans(idxs, spans) -> List[int]:
    out = set()
    for i in idxs:
        a, b = spans[i]
        if a >= 0 and b > a:
            out.update(range(a, b))
    return sorted(out)


def build_units(segs, diag, spans, wmode="uniform", base_text=True, budget=None):
    """把抽取器的 (rows × cols) 叉乘变成候选单元。

    wmode: uniform=每个可审计事实等值；relevance=用行列的词面得分加权；
           log=对 relevance 取对数，压掉极端值。
    base_text: 把「单位句」等正文片段作为免费基底（代价极小，始终保留）。
    """
    cn, un, rh, cell, _, _ = table_index(segs)
    rsc = diag.get("rscore", {}) or {}
    csc = diag.get("cscore", {}) or {}
    units = []
    for i in diag.get("rows", []):
        for j in diag.get("cols", []):
            if (i, j) not in cell:
                continue
            T = {cell[(i, j)]}
            if i in rh:
                T.add(rh[i])
            if j in cn:
                T.add(cn[j])
            if j in un:
                T.add(un[j])
            w = 1.0
            if wmode == "relevance":
                # λ=0.3：相关性最多把一份事实抬到 1.6 倍。若系数取 1，一个超高分单元
                # 会在目标值上压过五个普通单元，使 max(贪心, 单单元最优) 退化为单选。
                w = 1.0 + 0.3 * (rsc.get(i, 0.0) + csc.get(j, 0.0))
            elif wmode == "log":
                w = 1.0 + np.log1p(10.0 * (rsc.get(i, 0.0) + csc.get(j, 0.0)))
            units.append(dict(segs=frozenset(T), w=float(w), row=i, col=j))
    # 附件（表级单位句）：它们**不计价值**，只计成本。
    # 若把它当成 w=1 的独立单元，目标函数会把「只保留一句 amounts in millions」
    # 也算成一份事实——这正是目标函数与"事实完整"脱节的地方。
    # 处理：作为零价值基底优先装入；只收"单位声明"型的短片段（≤40 token），
    # 否则正文里整段提到 millions 的句子会直接吃掉预算（实测可达 555 token）。
    base = set()
    if base_text:
        for i, seg in enumerate(segs):
            if seg["kind"] != "text":
                continue
            if not (UNIT_TEXT_RE.search(seg["text"]) or UNIT_HINT.search(seg["text"])):
                continue
            a, b = spans[i]
            if a < 0 or (b - a) > 40:
                continue
            base.add(i)
    return units, base


def greedy_density(units, spans, B, base=None) -> Tuple[List[int], Dict]:
    """密度贪心：每轮取「边际价值 / 边际新 token」最大的单元，装不下则跳过。

    复杂度 O(n² )（每次重算边际），对 n≤40 可忽略。
    """
    base = set(base or ())
    chosen = set(base)
    cov = set(token_spans(base, spans))
    rest = list(range(len(units)))
    val = 0.0
    while True:
        best, best_key = None, None
        for k in rest:
            u = units[k]
            new = set(token_spans(u["segs"], spans)) - cov
            c = len(new)
            if c == 0:                       # 已被别的单元覆盖，白拿
                best, best_key = k, (float("inf"), u["w"])
                break
            if len(cov) + c > B:
                continue
            key = (u["w"] / c, u["w"])
            if best_key is None or key > best_key:
                best, best_key = k, key
        if best is None:
            break
        u = units[best]
        chosen |= set(u["segs"])
        cov |= set(token_spans(u["segs"], spans))
        val += u["w"]
        rest.remove(best)
    return sorted(chosen), dict(value=val, tokens=len(cov), n_units=len(units))


def best_single(units, spans, B):
    """单单元最优解：贪心的经典补充项，保证 1/2 近似。"""
    best, bv = None, -1.0
    for u in units:
        t = len(token_spans(u["segs"], spans))
        if t <= B and u["w"] > bv:
            best, bv = u, u["w"]
    return (sorted(best["segs"]) if best else []), bv


def dp_upper(units, spans, B):
    """独占代价下的 0-1 背包（DP），给出原问题最优值的上界。

    代价取「不被其他任何单元共享的 token 数」，因此任何可行解在此代价下也可行，
    其最优值不会低于原问题最优值。
    """
    all_tok = [set(token_spans(u["segs"], spans)) for u in units]
    cnt = {}
    for t in all_tok:
        for x in t:
            cnt[x] = cnt.get(x, 0) + 1
    cost = [sum(1 for x in t if cnt[x] == 1) for t in all_tok]
    n = len(units)
    if n == 0:
        return 0.0, 0
    dp = np.zeros(B + 1)
    keep = np.zeros((n, B + 1), dtype=bool)
    for i in range(n):
        c, w = int(cost[i]), units[i]["w"]
        if c <= 0:
            dp = dp + w
            keep[i, :] = True
            continue
        prev = dp.copy()
        for b in range(B, c - 1, -1):
            if prev[b - c] + w > dp[b]:
                dp[b] = prev[b - c] + w
                keep[i, b] = True
    return float(dp[B]), int(sum(cost))


def greedy_static(units, spans, B, base=None):
    """静态密度：按 w_i / |T_i| 排序后装箱，**不考虑已覆盖 token**。

    与动态版（greedy_density）的唯一差别就是分母用静态代价还是边际代价；
    这组对照用来回答"共享成本建模到底贡献多少"。
    """
    base = set(base or ())
    chosen, cov = set(base), set(token_spans(base, spans))
    order = sorted(range(len(units)),
                   key=lambda k: -(units[k]["w"] / max(1, len(token_spans(units[k]["segs"], spans)))))
    val = 0.0
    for k in order:
        u = units[k]
        t = set(token_spans(u["segs"], spans))
        new = t - cov
        if len(cov) + len(new) > B:
            continue
        chosen |= set(u["segs"])
        cov |= t
        val += u["w"]
    return sorted(chosen), dict(value=val, tokens=len(cov), n_units=len(units))


def rank_only(units, spans, B, base=None):
    """只按价值排序：相关性高的先装，装不下就跳过。用来回答"排序是否已经够"。"""
    base = set(base or ())
    chosen, cov = set(base), set(token_spans(base, spans))
    order = sorted(range(len(units)), key=lambda k: -units[k]["w"])
    val = 0.0
    for k in order:
        u = units[k]
        t = set(token_spans(u["segs"], spans))
        new = t - cov
        if len(cov) + len(new) > B:
            continue
        chosen |= set(u["segs"])
        cov |= t
        val += u["w"]
    return sorted(chosen), dict(value=val, tokens=len(cov), n_units=len(units))
