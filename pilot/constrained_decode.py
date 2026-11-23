"""解码层约束：**任何**数值字面量都只能由证据中出现过的数拼出。

只约束参数位置会留下缺口——模型可以在调用之外写任意数字。这里把规则
施加到所有位置：只要字符流末尾是一个数（或正要开始一个数），下一步就
只放行能让该前缀继续匹配某个允许数值的 token；当前缀本身已是允许数值
时，才放行非数字 token 来结束它。因此"写出一个证据里没有的数"在解码
层面不可表达，而不是事后检查。
"""
import re

import torch
from transformers import LogitsProcessor

# 纯数值 token：可选的句点开头、至少一位数字，之后只允许数字与句点
NUM_TOK = re.compile(r"^ ?\.?[0-9][0-9.]*$|^ ?\.$")
DIGIT = re.compile(r"[0-9]")


class NumberConstraint(LogitsProcessor):
    def __init__(self, tok, allowed_texts, device="cuda"):
        self.tok = tok
        self.device = device
        allowed = set()
        for s in allowed_texts:
            s = s.strip()
            if not s:
                continue
            allowed.add(s)
            if "," in s:
                allowed.add(s.replace(",", ""))
        self.allowed = allowed
        self.prefixes = {s[:i] for s in allowed for i in range(len(s) + 1)}

        self.num_ids, self.num_text = [], []
        self.digit_ids = []
        for i in range(len(tok)):
            t = tok.decode([i])
            if NUM_TOK.match(t):
                self.num_ids.append(i)
                self.num_text.append(t.strip())
            elif DIGIT.search(t):
                self.digit_ids.append(i)
        self.digit_mask = torch.zeros(len(tok), dtype=torch.bool)
        if self.digit_ids:
            self.digit_mask[self.digit_ids] = True
        self.num_mask = torch.zeros(len(tok), dtype=torch.bool)
        if self.num_ids:
            self.num_mask[self.num_ids] = True
        self.any_digit = self.digit_mask | self.num_mask

        self.prompt_len = None
        self._cache = {}
        self.stats = dict(steps=0, forced=0)

    # --- 状态 -----------------------------------------------------
    @staticmethod
    def _state(text):
        m = re.search(r"([0-9][0-9.]*)$", text)
        if m:
            return True, m.group(1)
        return False, ""

    # --- 掩码 -----------------------------------------------------
    def _mask(self, inside, prefix):
        key = (inside, prefix)
        if key in self._cache:
            return self._cache[key]
        allow = torch.zeros(len(self.tok), dtype=torch.bool)
        if not inside:
            # 边界：允许一切非数字 token；纯数值 token 只有能开始一个允许数值才放行
            allow[:] = True
            allow[self.any_digit] = False      # 含数字的 token 一律逐个校验
            for i, t in zip(self.num_ids, self.num_text):
                if t in self.prefixes:
                    allow[i] = True
        else:
            cont = 0
            for i, t in zip(self.num_ids, self.num_text):
                if (prefix + t) in self.prefixes:
                    allow[i] = True
                    cont += 1
            if prefix in self.allowed:
                allow[:] = allow | (~self.any_digit)   # 允许用非数字 token 收尾
            if cont == 0 and not bool(allow.any()):
                allow[:] = True          # 兜底，避免全 -inf
                self.stats["forced"] += 1
        self._cache[key] = allow
        return allow

    def __call__(self, input_ids, scores):
        if self.prompt_len is None:
            self.prompt_len = input_ids.shape[1]
        gen = input_ids[0, self.prompt_len:]
        text = self.tok.decode(gen, skip_special_tokens=True)
        inside, prefix = self._state(text)
        allow = self._mask(inside, prefix)
        if allow.all():
            return scores
        full = torch.zeros(scores.shape[-1], dtype=torch.bool)
        full[: allow.numel()] = allow
        mask = torch.full_like(scores, float("-inf"))
        mask[:, full.to(scores.device)] = 0.0
        self.stats["steps"] += 1
        return scores + mask


def allowed_numbers(ctx_text, num_re):
    """证据中出现过的数值字面量（原样保留，供前缀匹配）。"""
    return [m.group(0) for m in num_re.finditer(ctx_text or "")]
