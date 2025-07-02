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

def _px(value):
    """convert python‑docx Length (EMU) or int EMU to *points*"""
    return value / EMU_PER_PT

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

def extract_rows_from_docx(docx_path: str):
    """
    Parse paragraphs + tables into row‑dicts that XY‑cut understands.

    Key improvements vs. previous attempt
    -------------------------------------
    • Honors *space_before* / *space_after* so large gaps create separate
      blocks (exactly your PDF behaviour).
    • Uses real column positions (Word Columns feature) instead of the
      naive stripe distribution, so the gutter on page 3 is empty.
    • Bbox width = full column width => vertical‑gutter detector sees the
      white stripe even when the paragraph text is short.
    """
    doc = Document(docx_path)
    rows, tables = [], []

    # -- initial section settings -----------------------------------------
    section_iter = iter(doc.sections)
    section      = next(section_iter)
    col_cnt, col_spc = _column_layout(section)
    pg_w_pt = section.page_width / EMU_PER_PT
    left_mar= section.left_margin / EMU_PER_PT
    col_w   = (pg_w_pt - left_mar*2 - col_spc*(col_cnt-1)) / col_cnt
    y_cursor= section.top_margin / EMU_PER_PT

    def start_new_section(sec):
        nonlocal section, col_cnt, col_spc, col_w, pg_w_pt, left_mar, y_cursor
        section   = sec
        col_cnt, col_spc = _column_layout(section)
        pg_w_pt   = section.page_width / EMU_PER_PT
        left_mar  = section.left_margin / EMU_PER_PT
        col_w     = (pg_w_pt - left_mar*2 - col_spc*(col_cnt-1)) / col_cnt
        y_cursor  = max(y_cursor, section.top_margin / EMU_PER_PT)

    # --------------------------------------------------------------------
    para_idx = 0
    for xml_block in doc.element.body:
        tag = xml_block.tag.split('}')[-1]

        # ----------------- PARAGRAPH ------------------------------------
        if tag == "p":
            par = doc.paragraphs[para_idx]; para_idx += 1
            raw_text = par.text.replace('\xa0',' ').strip()
            bullet   = _bullet_char(par)
            if bullet and not raw_text.startswith(bullet):
                raw_text = f"{bullet} {raw_text}"
            if not raw_text:
                # empty line: still respect Word spacing_after
                space_after = par.paragraph_format.space_after
                y_cursor += (space_after.pt if space_after else _font_size(par))
                continue

            size      = _font_size(par)
            heading   = _is_heading(par)
            # Word paragraph spacing (points)
            space_before = par.paragraph_format.space_before
            space_after  = par.paragraph_format.space_after
            before_pt = space_before.pt if space_before else 0
            after_pt  = space_after .pt if space_after  else 0

            # apply BEFORE spacing *once* before drawing this paragraph
            y_cursor += before_pt

            # assign to correct column band ------------------------------
            # Word flows paragraphs round‑robin through columns **within
            # each section page**.  We approximate by cycling modulo col_cnt.
            col_idx = (para_idx-1) % col_cnt
            x0      = left_mar + col_idx*(col_w + col_spc)
            x1      = x0 + col_w                        # full column width

            line_h  = size * 1.20
            y0, y1  = y_cursor, y_cursor + line_h
            y_cursor = y1 + after_pt                    # AFTER spacing

            rows.append({
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "size": size,
                "text": raw_text,
                "baseline": y0,
                "is_bullet_only": BULLET_RE.fullmatch(raw_text) is not None,
                "heading": heading,
            })

        # ----------------- TABLE ---------------------------------------
        elif tag == "tbl":
            tbl = doc.tables[len(tables)]
            n_rows, n_cols = len(tbl.rows), len(tbl.columns)
            mat = [[tbl.cell(r,c).text.strip() for c in range(n_cols)]
                   for r in range(n_rows)]

            x0 = left_mar
            x1 = left_mar + col_cnt*col_w + (col_cnt-1)*col_spc
            y0 = y_cursor
            y1 = y_cursor + n_rows*12                    # coarse guess
            y_cursor = y1 + 12

            tables.append({"type":"table","bbox":[x0,y0,x1,y1],"rows":mat})

        # ----------------- SECTION BREAK --------------------------------
        elif tag == "sectPr":
            start_new_section(next(section_iter, section))

    rows.sort(key=lambda r:(r["y0"],r["x0"]))
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
    idx  = list(range(len(rows)))

    # approximate page size in points
    page_w = 612; page_h = 792      # default Letter
    segments = xy_cut_region(idx, rows, page_w, page_h, tbl_boxes=[])

    text_blocks=[make_block(seg, rows) for seg in segments if seg]
    return text_blocks+tables

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: docx_xycut.py <file.docx>")
        sys.exit(1)
    result = [{"page":1,"blocks": iterate_chunks_docx(sys.argv[1])}]
    print(json.dumps(result,indent=2,ensure_ascii=False))
