"""可行性判定：21pp 的残差是「选错数」还是「算错」。

result 里只存了求值后的 pred（没存原始表达式），所以不能查"字面量是否可溯源"，
但可以做一个更基础的二分：

  A 选错（copy）：pred 的数值出现在保留文本里，但不是 gold
     → 定位/选择问题，缓存侧或选择侧还有救
  B 算错（compute）：pred 的数值不在保留文本里
     → 表达式/算术问题，任何缓存方法都救不了
  C 空（none）：pred 不是数值

用法：python diag_ground_expr.py ../results/xxx.json ...
"""
import json
import re
import sys

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def norm(x):
    s = str(x).replace(",", "").replace("$", "").strip().rstrip("%")
    try:
        v = float(s)
    except ValueError:
        return None
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    return ("%.6f" % v).rstrip("0").rstrip(".")


def nums_of(s):
    out = set()
    for m in NUM.findall(s or ""):
        v = norm(m)
        if v is not None:
            out.add(v)
    return out


def main(path):
    d = json.load(open(path, encoding="utf-8"))
    recs = d["records"]
    n = len(recs)
    bad = [r for r in recs if not r["correct"]]
    ok = n - len(bad)
    a = b = c = 0
    for r in bad:
        pv = norm(r.get("pred"))
        if pv is None:
            c += 1
            continue
        pool = nums_of(r.get("text"))
        if pv in pool:
            a += 1
        else:
            b += 1
    print("%s  n=%d  EM=%.3f" % (path.split("/")[-1], n, ok / n))
    print("  错误 %d 题：A 选错 %d (%.1f%%) ｜ B 算错 %d (%.1f%%) ｜ C 非数值 %d (%.1f%%)"
          % (len(bad), a, 100.0 * a / max(len(bad), 1),
             b, 100.0 * b / max(len(bad), 1),
             c, 100.0 * c / max(len(bad), 1)))
    print("  ⇒ 若定位全对可救 %+.1fpp（A 全变对）；B 只能靠生成侧/模型" % (100.0 * a / n))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
