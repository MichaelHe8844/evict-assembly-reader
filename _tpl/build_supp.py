# -*- coding: utf-8 -*-
"""把 pandoc 产出的补充材料 docx 套上期刊排版（单栏、宋体小五）。"""
import copy

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

SRC = "_tpl/supp_raw.docx"
OUT = "论文_补充材料.docx"


def set_font(run, latin, cjk, size, bold=False):
    run.font.name = latin
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0)
    rPr = run._element.get_or_add_rPr()
    rf = rPr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rPr.append(rf)
    rf.set(qn("w:ascii"), latin)
    rf.set(qn("w:hAnsi"), latin)
    rf.set(qn("w:eastAsia"), cjk)


def main():
    doc = docx.Document(SRC)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.6)
    sec.top_margin = sec.bottom_margin = Cm(2.4)

    for p in doc.paragraphs:
        st = p.style.name
        txt = p.text.strip()
        if st == "Heading 1":
            for r in p.runs:
                set_font(r, "Times New Roman", "黑体", 14, bold=True)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(6)
        elif st == "Heading 2":
            for r in p.runs:
                set_font(r, "Times New Roman", "黑体", 10.5, bold=True)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if "论文：" in txt else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(2)
        elif txt.startswith("**表 S") or txt.startswith("表 S"):
            for r in p.runs:
                set_font(r, "Times New Roman", "宋体", 7.5)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Pt(0)
        else:
            for r in p.runs:
                sz = r.font.size.pt if r.font.size else None
                set_font(r, "Times New Roman", "宋体", sz if sz and sz < 9.5 else 9,
                         bold=bool(r.font.bold) and txt.startswith(("**", "说明")))
            p.paragraph_format.first_line_indent = Pt(0) if txt.startswith(("说明。", "本材料", "与正文")) else Pt(18)
            p.paragraph_format.line_spacing = 1.05
            p.paragraph_format.space_after = Pt(2)

    # 题注与表格同页（题注段落 keep_with_next）
    for p in doc.paragraphs:
        if p.text.strip().startswith("表 S"):
            p.paragraph_format.keep_with_next = True

    # 表格：定宽 + 7.5pt
    for t in doc.tables:
        t.alignment = 1
        try:
            ncol = len(t.rows[0].cells)
        except Exception:
            continue
        # 按各列最大内容长度分配（中文按 2 计）
        def wid(txt):
            return sum(2 if ord(ch) > 127 else 1 for ch in txt)

        maxlen = [1] * ncol
        for row in t.rows:
            for i2, cell in enumerate(row.cells):
                if i2 < ncol:
                    v = wid(cell.text.strip())
                    if cell._tc.xpath(".//m:oMath"):
                        v = max(v, 20)
                    maxlen[i2] = max(maxlen[i2], v)
        wts = [max(5, m + 2) for m in maxlen]
        ssum = sum(wts) or 1
        total = 15.0
        tbl = t._tbl
        grid = tbl.find(qn("w:tblGrid"))
        if grid is not None:
            tbl.remove(grid)
        wds = [total * x / ssum for x in wts]
        g = OxmlElement("w:tblGrid")
        for x in wds:
            gc = OxmlElement("w:gridCol")
            gc.set(qn("w:w"), str(int(round(x * 567))))
            g.append(gc)
        tbl.insert(list(tbl).index(tbl.tblPr) + 1, g)
        nrow = len(t.rows)
        for ri, row in enumerate(t.rows):
            for cell in row.cells:
                # 除最后一行外都与下段同页 → 整张表不分页
                if ri < nrow - 1:
                    for cp in cell.paragraphs:
                        cp.paragraph_format.keep_with_next = True
                for cp in cell.paragraphs:
                    cp.paragraph_format.first_line_indent = Pt(0)
                    cp.paragraph_format.line_spacing = 1.0
                    for r in cp.runs:
                        set_font(r, "Times New Roman", "宋体", 7.5)

    # 公式统一 9pt（pandoc 的 OMML 不带字号，会按默认 10.5 渲染）
    NS_M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
    for mr in doc.element.body.iter("{%s}r" % NS_M):
        rPr = mr.find(qn("w:rPr"))
        if rPr is None:
            rPr = OxmlElement("w:rPr")
            mrpr = mr.find("{%s}rPr" % NS_M)
            if mrpr is not None:
                mrpr.addnext(rPr)
            else:
                mr.insert(0, rPr)
        for tag in ("w:rFonts", "w:sz", "w:szCs"):
            for old in rPr.findall(qn(tag)):
                rPr.remove(old)
        f = OxmlElement("w:rFonts")
        f.set(qn("w:ascii"), "Cambria Math")
        f.set(qn("w:hAnsi"), "Cambria Math")
        f.set(qn("w:eastAsia"), "宋体")
        sz = OxmlElement("w:sz"); sz.set(qn("w:val"), "18")
        szc = OxmlElement("w:szCs"); szc.set(qn("w:val"), "18")
        rPr.append(f); rPr.append(sz); rPr.append(szc)

    # 页脚页码
    ftr = sec.footer.paragraphs[0]
    ftr.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = ftr.add_run("S-")
    set_font(run, "Times New Roman", "宋体", 9)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), " PAGE ")
    r = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = "1"
    r.append(t)
    fld.append(r)
    ftr._p.append(fld)

    doc.save(OUT)
    print("已生成", OUT)


if __name__ == "__main__":
    main()
