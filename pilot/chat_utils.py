# -*- coding: utf-8 -*-
"""跨模型族的对话脚手架。

原来 arm_c 里把 Qwen 的 `<|im_start|>` 写死了，换族就崩。这里改成用
tokenizer 自己的 chat template 渲染，再把「文档占位符」处切成三段：

    pre_ids  +  doc_ids  +  suf_ids

这样文档区间下标仍可精确计算（预算只作用于 doc 段），而模板差异交给
tokenizer 自己处理。

对 Qwen2.5 来说渲染结果与旧的硬编码字符串逐 token 相同（有离线校验），
所以改这一层不影响已有结果的可复现性。
"""
DOC_SLOT = "%%DOC%%"


def build_split(tok, sys_text, user_prefix, user_suffix, add_generation_prompt=True):
    """返回 (pre_ids, suf_ids)。user_prefix 在文档之前，user_suffix 在文档之后。"""
    content = (user_prefix or "") + DOC_SLOT + (user_suffix or "")
    msgs = [{"role": "system", "content": sys_text},
            {"role": "user", "content": content}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=add_generation_prompt)
    if DOC_SLOT not in text:
        raise ValueError("chat template 吃掉了占位符：%r" % text[:200])
    a, b = text.split(DOC_SLOT)
    pre_ids = tok(a, add_special_tokens=False)["input_ids"]
    suf_ids = tok(b, add_special_tokens=False)["input_ids"]
    return pre_ids, suf_ids


def render_probe(sys_text, question, instruct):
    """探针任务的 user_suffix：换行 + 问题 + 指令。"""
    return "\n\nQuestion: %s\n%s" % (question, instruct)
