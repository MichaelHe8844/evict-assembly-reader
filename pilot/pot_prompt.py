# -*- coding: utf-8 -*-
"""PoT 提示：few-shot 示例与指令。test_pot 与 arm_c 共用，避免两处漂移。"""

FEWSHOT = """Example 1
Document:
item | 2019 | 2020
revenue | 120 | 150
Question: what is the percent increase in revenue from 2019 to 2020?
Expression: (150 - 120) / 120 * 100

Example 2
Document:
segment | amount
north america | 400
europe | 250
Question: what is the total amount for all segments?
Expression: 400 + 250

Example 3
Document:
metric | 2021
operating income | 350
net revenue | 2450
Question: what is operating income as a percentage of net revenue?
Expression: 350 / 2450 * 100
"""

INSTR = ("Write ONE Python arithmetic expression that computes the answer, using only "
         "numbers that appear in the document. Output the expression only.")
