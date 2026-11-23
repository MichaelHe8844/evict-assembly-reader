# -*- coding: utf-8 -*-
"""参考文献 → GB/T 7714（《计算机科学》模板样式）。"""
import re

CONF = ["NeurIPS", "ICLR", "ICML", "ACL", "EMNLP", "NAACL", "AAAI", "KDD", "IJCAI",
        "CVPR", "Findings", "TRL", "Workshop", "MLSys", "SIGIR", "CIKM", "COLING",
        "EACL", "USENIX", "OSDI", "SOSP", "VLDB", "ICDE", "WWW", "WSDM", "TMLR"]
JOUR = ["IEEE Transactions", "IEEE Trans", "Transactions", "Journal", "TACL", "Nature",
        "Science", "学报", "研究与发展", "信息学报", "软件学报", "中文信息学报"]


def up_authors(auth):
    out = []
    for part in auth.split(","):
        part = part.strip().rstrip(".")
        if not part:
            continue
        if part.lower().startswith("et al"):
            out.append("et al")
            continue
        m = re.match(r"^([A-Z][A-Za-z\-']+)\s*(.*)$", part)
        out.append(m.group(1).upper() + ((" " + m.group(2)) if m.group(2) else "") if m else part)
    return ",".join(out)


def convert(entry):
    m = re.match(r"^\[(\d+)\]\s*(.*)$", entry.strip())
    if not m:
        return entry
    num, body = m.group(1), m.group(2).strip().rstrip(".")

    if re.search(r"[\u4e00-\u9fff]", body):          # 中文条目：保持，末尾补句点
        return "[%s] %s." % (num, body)

    # 年份（取最后一个）
    ys = re.findall(r"(?:19|20)\d{2}", body)
    year = ys[-1] if ys else ""
    if not year:
        m_ap = re.search(r"'\s?(\d{2})\b", body)
        if m_ap:
            year = "20" + m_ap.group(1)
    # arXiv 编号
    marx = re.search(r"arXiv:\s*(\d{4}\.\d{4,5})", body)

    # 作者与其余
    m1 = re.match(r"^(.*?)\bet al\.\s*(.*)$", body)
    if m1:
        auth = up_authors(m1.group(1)) + ",et al"
        rest = m1.group(2)
    else:
        m2 = re.match(r"^((?:[A-Z][A-Za-z\-']+\s+[A-Z]\.\s*,?\s*)+)(.*)$", body)
        if m2:
            auth = up_authors(m2.group(1))
            rest = m2.group(2)
        else:
            auth, rest = "", body

    rest = rest.strip().rstrip(".")
    # 去掉尾部年份/页码残留
    rest = re.sub(r"[,;]?\s*(?:19|20)\d{2}[a-z]?$", "", rest).rstrip(", ")

    # 预印本
    if marx and not any(k in rest for k in CONF):
        title = re.split(r"arXiv:", rest)[0].rstrip().rstrip(".").rstrip()
        return "[%s] %s.%s.[J].arXiv:%s,%s." % (num, auth, title, marx.group(1), year)

    conf = [k for k in CONF if k in rest]
    if conf:
        k0 = min(conf, key=lambda k: rest.index(k))
        idx = rest.index(k0)
        title, venue = rest[:idx].rstrip().rstrip("."), rest[idx:]
        venue = re.sub(r"\s*,\s*", ", ", venue).strip().rstrip(",")
        venue = re.sub(r"\b(?:19|20)\d{2}\b", "", venue)
        venue = re.sub(r"\s{2,}", " ", venue).replace(" ,", ",").strip().rstrip(",").strip()
        tail = (",%s" % year) if year else ""
        return "[%s] %s.%s.[C]//%s%s." % (num, auth, title, venue, tail)

    jr = [k for k in JOUR if k in rest]
    if jr:
        k0 = min(jr, key=lambda k: rest.index(k))
        idx = rest.index(k0)
        title, venue = rest[:idx].rstrip().rstrip("."), rest[idx:]
        venue = re.sub(r"\b(?:19|20)\d{2}\b", "", venue)
        venue = re.sub(r"\s{2,}", " ", venue).replace(" ,", ",").strip().rstrip(",").strip()
        tail = (",%s" % year) if year else ""
        return "[%s] %s.%s.[J].%s%s." % (num, auth, title, venue, tail)

    rest = re.sub(r"[,;]\s*$", "", rest)
    return ("[%s] %s.%s.[J].%s." % (num, auth, rest, year)) if year else ("[%s] %s.%s." % (num, auth, rest))


if __name__ == "__main__":
    import io, sys
    for line in io.open(sys.argv[1], encoding="utf-8").read().split("\n"):
        if line.startswith("["):
            print(convert(line))
