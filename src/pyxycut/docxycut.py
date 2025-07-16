from docx import Document
from docx.shared import Inches, Pt
from collections import defaultdict
import statistics, re, sys, json, math
from statistics import median, quantiles

EMU_PER_PT = 12700
BIN_WIDTH  = 1
COL_FACTOR = 1.0
ROW_FACTOR = 1.5
BULLET_RE  = re.compile(r"^[\u2022\u2023\u25E6\u2043\u2219\u00B7]+$")
TEXT_SCALE = 0.52
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

def _font_size(par):
    for run in par.runs:
        if run.font.size:
            return run.font.size.pt
    if par.style and par.style.font.size:
        return par.style.font.size.pt
    return 12.0

def _is_heading(par):
    if par.style and par.style.name.lower().startswith("heading"):
        return True
    return False

def _bullet_char(par):
    if par._p.pPr is None:
        return None
    numPr = par._p.pPr.numPr
    if numPr is not None:
        return "•"
    return None

from docx.oxml.ns import qn
def _column_layout(section):
    cols_el = getattr(section._sectPr, "cols", None)
    if cols_el is None:
        cols_el = section._sectPr.find(qn("w:cols"))

    if cols_el is not None:
        cnt_attr = cols_el.get(qn("w:num")) or cols_el.get("num")
        cnt = int(cnt_attr) if cnt_attr else 1
        spc_attr = cols_el.get(qn("w:space")) or cols_el.get("space")
        if spc_attr:
            spacing_pt = int(spc_attr) / 20.0
        else:
            spacing_pt = Pt(18).pt
        return cnt or 1, spacing_pt

    return 1, Pt(18).pt

from docx import Document

from docx.oxml.ns import qn
from docx.shared import Inches, Pt

def section_column_info(section, default_space=Inches(0.5)):
    sectPr = section._sectPr
    cols_el = sectPr.find(qn('w:cols'))

    if cols_el is None:
        return 1, default_space.pt

    num   = int(cols_el.get(qn('w:num'), 1))
    space = cols_el.get(qn('w:space'))
    spacing_pts = int(space) / 20.0 if space else default_space.pt

    return max(1, num), spacing_pts

def _table_to_rows(tbl):
    """Return (row_cnt, col_cnt, rows_as_2D_list_of_strings) for a python-docx Table"""
    grid = []
    for tr in tbl.findall('.//w:tr', NS):
        row = []
        for tc in tr.findall('.//w:tc', NS):
            # Collect *all* paragraphs in this cell
            cell_txt = []
            for p in tc.findall('.//w:p', NS):
                texts = [t.text for t in p.findall('.//w:t', NS) if t.text]
                if texts:
                    cell_txt.append(''.join(texts))
            row.append('\n'.join(cell_txt))
        grid.append(row)

    row_cnt = len(grid)
    col_cnt = max((len(r) for r in grid), default=0)
    for r in grid:
        r += [''] * (col_cnt - len(r))
    return row_cnt, col_cnt, grid

import base64
def extract_images(doc, page, column):
    for shape in doc.inline_shapes:
        rId  = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
        blob = doc.part.related_parts[rId].blob
        yield {
            "page":   page,
            "column": column,
            "type":   "image",
            "bbox":   [0, 0, 0, 0],      # need real coordinates → 0 for now
            "width":  shape.width.pt,
            "height": shape.height.pt,
            "text":   base64.b64encode(blob).decode(),
        }


from docx.enum.section import WD_SECTION_START

def extract_rows_from_docx(docx_path):
    doc    = Document(docx_path)
    rows   = []
    tables = []

    current_page   = 1
    current_column = 1

    # ── gather section/column info ─────────────────────────────────────
    section_cols = []                       # (start_page, num_cols, col_width, gutter)
    for i, section in enumerate(doc.sections):
        num_cols, gutter_pts = section_column_info(section)
        gutter   = Pt(gutter_pts)
        total_w  = section.page_width - section.left_margin - section.right_margin
        col_w    = (total_w - gutter*(num_cols-1)) / num_cols if num_cols > 1 else total_w
        start_pg = current_page
        if i > 0 and section.start_type == WD_SECTION_START.NEW_PAGE:
            start_pg += 1
        section_cols.append((start_pg, num_cols, col_w, gutter))

    # ❶ keep a vertical cursor for each page/column
    y_cursor = 72
    line_h   = 14

    for block in doc.element.body:

        if block.tag.endswith('sectPr'):
            if block.xpath('.//w:type[@w:val="nextPage"]'):
                current_page += 1
                y_cursor      = 72
            current_column = 1
            continue

        # ── PARAGRAPH ──────────────────────────────────────────────────
        if block.tag.endswith('p'):
            from docx.text.paragraph import Paragraph 
            para = Paragraph(block, doc)
            text = para.text.strip()

            text = ''.join(
                t.text or '' for t in block.findall('.//w:t', NS)
            ).strip()
            if not text:
                continue

            # synthetic geometry  ⤵︎
            # col_w & margins come from the *current* section:
            # find the tuple whose start_page ≤ current_page
            sec = max((s for s in section_cols if s[0] <= current_page),
                      key=lambda t: t[0])
            _, num_cols, col_w, _ = sec
            x0 = doc.sections[0].left_margin.pt + (current_column-1)*(col_w + sec[3].pt)
            x1 = x0 + col_w
            y0 = y_cursor
            y1 = y0 + line_h

            rows.append({
                "page":     current_page,
                "column":   current_column,
                "text":     text,
                "x0":       x0, "x1": x1,
                "y0":       y0, "y1": y1,
                "size":     _font_size(para),  # crude: default to body font
                "baseline": y1,
                "is_bullet_only": False,
            })

            y_cursor += line_h
            continue

        # ── TABLE ──────────────────────────────────────────────────────
        if block.tag.endswith('tbl'):
            row_cnt, col_cnt, grid = _table_to_rows(block)
            tables.append({
                "page":    current_page,
                "column":  current_column,
                "type":    "table",
                "bbox":    [0, 0, 0, 0],
                "row_cnt": row_cnt,
                "col_cnt": col_cnt,
                "rows":    grid,
                "text":    "",
            })
            y_cursor += line_h * row_cnt     # advance cursor roughly
            continue

    # ── images ─────────────────────────────────────────────────────────
    for img in extract_images(doc, current_page, current_column):
        tables.append(img)

    return rows, tables

def merge_bullets(rows, char_tol_factor=1.0):
    merged = []
    by_base= defaultdict(list)
    for r in rows:
        by_base[r["baseline"]].append(r)
    for base, seq in by_base.items():
        seq.sort(key=lambda r:r["x0"])
        pool=[(r["x1"]-r["x0"])/max(1,len(r["text"]))
              for r in seq if not r["is_bullet_only"]]
        cw = median(pool) if pool else 1.0
        keep,skip=[],set()
        for i,r in enumerate(seq):
            if r["is_bullet_only"] and i+1<len(seq):
                nxt = seq[i+1]
                if abs(nxt["x0"]-r["x0"])<=char_tol_factor*cw:
                    nxt["text"]=f"{r['text']} {nxt['text']}"
                    nxt["x0"]=min(r["x0"],nxt["x0"])
                    skip.add(id(r))
        merged.extend(r for r in seq if id(r) not in skip)
    merged.sort(key=lambda r:(r["y0"],r["x0"]))
    return merged

def make_block(seg, rows):
    x0=min(rows[i]["x0"] for i in seg)
    y0=min(rows[i]["y0"] for i in seg)
    x1=max(rows[i]["x1"] for i in seg)
    y1=max(rows[i]["y1"] for i in seg)
    font=statistics.median(rows[i]["size"] for i in seg)
    txt =" ".join(rows[i]["text"] for i in seg)
    return {"type":"text","font_size":round(font,2),"bbox":[x0,y0,x1,y1],"text":txt}

def compute_dominant_sizes(lines):
    cw, lh = [], []
    for ln in lines:
        n = sum(1 for c in ln["text"] if not c.isspace())
        if n:
            cw.append((ln["x1"] - ln["x0"]) / n)
        lh.append(ln["y1"] - ln["y0"])
    char = median(cw) if cw else 0
    if not lh:
        return char, 0
    q1 = quantiles(lh, n=4)[0]
    core = [h for h in lh if h >= q1]
    return char, median(core)

def find_vertical_gutter(lines, idx, page_w, char_w, tol=0.02):
    rx0 = min(lines[i]["x0"] for i in idx)
    rx1 = max(lines[i]["x1"] for i in idx)
    w   = rx1 - rx0
    if char_w == 0 or w <= 0:
        return None
    bins = int(w // BIN_WIDTH) + 1
    hist = [0]*bins
    for i in idx:
        x0 = max(lines[i]["x0"], rx0); x1 = min(lines[i]["x1"], rx1)
        b0 = int((x0-rx0)//BIN_WIDTH); b1 = int((x1-rx0)//BIN_WIDTH)
        for b in range(b0, b1+1):
            hist[b] += 1
    max_occ = max(1, int(tol*len(idx)))
    run_req = max(1, int((COL_FACTOR*char_w)//BIN_WIDTH))
    best = None; cur = None
    for j, occ in enumerate(hist+[max_occ+1]):
        if occ <= max_occ:
            cur = j if cur is None else cur
        elif cur is not None:
            run = j - cur
            if run >= run_req and (best is None or run > best[2]):
                best = (cur, j-1, run)
            cur = None
    if not best:
        return None
    g0, g1, _ = best
    return rx0 + g0*BIN_WIDTH, rx0 + g1*BIN_WIDTH + BIN_WIDTH

def find_horizontal_gap(lines, idx, page_h, line_h, *, tol=0.02):
    ry0 = min(lines[i]["y0"] for i in idx)
    ry1 = max(lines[i]["y1"] for i in idx)
    h   = ry1 - ry0
    if h <= 0 or line_h == 0:
        return None
    BIN_HEIGHT = 1
    bins = int(h // BIN_HEIGHT) + 1
    hist = [0]*bins
    for i in idx:
        y0 = max(lines[i]["y0"], ry0); y1 = min(lines[i]["y1"], ry1)
        b0 = int((y0-ry0)//BIN_HEIGHT); b1 = int((y1-ry0)//BIN_HEIGHT)
        for b in range(b0, b1+1):
            hist[b] += 1
    max_occ = max(1, int(tol*len(idx)))
    run_req = max(1, int((ROW_FACTOR*line_h)//BIN_HEIGHT))
    best = None; cur = None
    for j, occ in enumerate(hist+[max_occ+1]):
        if occ <= max_occ:
            cur = j if cur is None else cur
        elif cur is not None:
            run = j - cur
            if run >= run_req and (best is None or run > best[2]):
                best = (cur, j-1, run)
            cur = None
    if not best:
        return None
    g0, g1, _ = best
    return ry0 + g0*BIN_HEIGHT, ry0 + g1*BIN_HEIGHT + BIN_HEIGHT

def _gap(a, b):
    return max(0.0, b["y0"] - a["y1"])

def _looks_like_heading(row, median_font, size_ratio=1.15, short_len=60):
    txt = row["text"].strip()
    if not any(ch.isalpha() for ch in txt):
        return False
    big   = row["size"] >= median_font * size_ratio
    short = len(txt) <= short_len
    return big and short

def _is_large_gap(g, line_h, med_gap):
    return g >= max(ROW_FACTOR*line_h, 2.5*med_gap if med_gap else 0)

def xy_cut_region(idx, lines, page_w, page_h, tbl_boxes,
                  *, min_block_width=0.02, min_block_height=0.02):
    if not idx or len(idx) == 1:
        return [idx]

    # DOCX version: no table exclusion yet
    def _in_table(_): return False

    char_w, line_h = compute_dominant_sizes([lines[i] for i in idx if not _in_table(lines[i])])
    if char_w == 0 or line_h == 0:
        return [idx]

    # (1) vertical split
    gut = find_vertical_gutter(lines, idx, page_w, char_w)
    if gut:
        gx0, gx1 = gut
        left  = [i for i in idx if lines[i]["x1"] <= gx0]
        right = [i for i in idx if lines[i]["x0"] >= gx1]
        bridge= [i for i in idx if i not in left and i not in right and
                 lines[i]["x0"] < gx1 and lines[i]["x1"] > gx0]
        if left and right:
            min_w = min_block_width*page_w
            lw = max(lines[i]["x1"] for i in left ) - min(lines[i]["x0"] for i in left )
            rw = max(lines[i]["x1"] for i in right) - min(lines[i]["x0"] for i in right)
            if lw>=min_w and rw>=min_w:
                return (xy_cut_region(sorted(left ,key=lambda i:lines[i]["y0"]),
                                      lines,page_w,page_h,tbl_boxes,
                                      min_block_width=min_block_width,
                                      min_block_height=min_block_height)
                     + xy_cut_region(sorted(bridge,key=lambda i:lines[i]["y0"]),
                                      lines,page_w,page_h,tbl_boxes,
                                      min_block_width=min_block_width,
                                      min_block_height=min_block_height)
                     + xy_cut_region(sorted(right,key=lambda i:lines[i]["y0"]),
                                      lines,page_w,page_h,tbl_boxes,
                                      min_block_width=min_block_width,
                                      min_block_height=min_block_height))

    # (2) horizontal split
    hgap = find_horizontal_gap(lines, idx, page_h, line_h)
    if hgap:
        gy0, gy1 = hgap
        upper=[i for i in idx if lines[i]["y1"]<=gy0]
        lower=[i for i in idx if lines[i]["y0"]>=gy1]
        if upper and lower:
            min_h=min_block_height*page_h
            uh=max(lines[i]["y1"] for i in upper)-min(lines[i]["y0"] for i in upper)
            lh=max(lines[i]["y1"] for i in lower)-min(lines[i]["y0"] for i in lower)
            if uh>=min_h and lh>=min_h:
                return (xy_cut_region(sorted(upper,key=lambda i:lines[i]["y0"]),
                                      lines,page_w,page_h,tbl_boxes,
                                      min_block_width=min_block_width,
                                      min_block_height=min_block_height)
                     + xy_cut_region(sorted(lower,key=lambda i:lines[i]["y0"]),
                                      lines,page_w,page_h,tbl_boxes,
                                      min_block_width=min_block_width,
                                      min_block_height=min_block_height))

    # (3) heading‑boosted gap scan
    by_top = sorted(idx, key=lambda i: lines[i]["y0"])
    med_font = statistics.median(lines[i]["size"] for i in idx)
    gaps_white, gaps_base = [], []
    for a,b in zip(by_top, by_top[1:]):
        white = _gap(lines[a], lines[b])
        base  = lines[b]["y0"] - lines[a]["y0"]
        if _looks_like_heading(lines[a], med_font) or _looks_like_heading(lines[b], med_font):
            base = max(base, ROW_FACTOR*line_h + 1)
        gaps_white.append(white)
        gaps_base .append(base)
    med_gap = statistics.median(gaps_white) if gaps_white else 0
    big = [k for k,g in enumerate(gaps_base) if _is_large_gap(g, line_h, med_gap)]
    if big:
        cut = max(big, key=lambda k:gaps_base[k]) + 1
        up,lo = by_top[:cut], by_top[cut:]
        return (xy_cut_region(up, lines,page_w,page_h,tbl_boxes,
                              min_block_width=min_block_width,
                              min_block_height=min_block_height)
             + xy_cut_region(lo, lines,page_w,page_h,tbl_boxes,
                              min_block_width=min_block_width,
                              min_block_height=min_block_height))

    return [idx]

def iterate_chunks_docx(path):
    rows, tables = extract_rows_from_docx(path)
    rows = merge_bullets(rows)

    pages = defaultdict(lambda: {"rows": [], "tables": []})
    for r in rows:   pages[r["page"]]["rows"  ].append(r)
    for t in tables: pages[t["page"]]["tables"].append(t)

    out = []
    page_w, page_h = 612, 792
    for pg in sorted(pages):
        rows_pg   = pages[pg]["rows"]
        tbls_pg   = pages[pg]["tables"]
        idx_rows  = list(range(len(rows_pg)))

        segs = xy_cut_region(idx_rows, rows_pg, page_w, page_h, tbl_boxes=[])
        text_blocks = [make_block(seg, rows_pg) for seg in segs if seg]

        out.append({"page": pg, "blocks": text_blocks + tbls_pg})
    return out


if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: docx_xycut.py <file.docx>")
        sys.exit(1)
    result = iterate_chunks_docx(sys.argv[1])
    print(json.dumps(result,indent=2,ensure_ascii=False))
