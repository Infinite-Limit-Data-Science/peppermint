from docx import Document
from docx.shared import Inches, Pt
from collections import defaultdict
import statistics, re, sys, json, math
from statistics import median, quantiles

from math import ceil
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.table import Table
import base64, itertools, re
from docx.oxml.ns import qn
from docx import Document
from docx.shared import Inches

EMU_TO_PT   = 1.0 / 12700                      # 1 EMU  = 1/12700 pt
TWIP_TO_PT  = 1.0 / 20                        # 1 twip = 1/20 pt
AVG_CHAR_W  = 0.50                            # ~½ × font‑size
DEFAULT_LH  = 1.20                            # 1.2 × font‑size
DEFAULT_GAP = 720                             # 720 tw = 0.5 in

EMU_PER_PT = 12700
BIN_WIDTH  = 1
COL_FACTOR = 1.0
ROW_FACTOR = 1.5
BULLET_RE  = re.compile(r"^[\u2022\u2023\u25E6\u2043\u2219\u00B7]+$")
TEXT_SCALE = 0.52
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def font_size(par):
    for run in par.runs:
        if run.font.size:
            return run.font.size.pt
    if par.style and par.style.font.size:
        return par.style.font.size.pt
    return 12.0

def section_column_info(section, default_space=Inches(0.5)):
    sectPr = section._sectPr
    cols_el = sectPr.find(qn('w:cols'))

    if cols_el is None:
        return 1, default_space.pt

    num   = int(cols_el.get(qn('w:num'), 1))
    space = cols_el.get(qn('w:space'))
    spacing_pts = int(space) / 20.0 if space else default_space.pt

    return max(1, num), spacing_pts

def extract_images(doc, page, column):
    for shape in doc.inline_shapes:
        rId  = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
        blob = doc.part.related_parts[rId].blob
        yield {
            "page":   page,
            "column": column,
            "type":   "image",
            "bbox":   [0, 0, 0, 0],
            "width":  shape.width.pt,
            "height": shape.height.pt,
            "text":   base64.b64encode(blob).decode(),
        }

def _iter_blocks(doc):
    body = doc.element.body
    for elm in body.iterchildren():
        if elm.tag.endswith('}p'):
            yield Paragraph(elm, doc)
        elif elm.tag.endswith('}tbl'):
            yield Table(elm, doc)

def _section_props(section):
    pgW   = section.page_width
    pgH   = section.page_height
    top   = section.top_margin
    bot   = section.bottom_margin
    left  = section.left_margin
    right = section.right_margin

    cols_el = section._sectPr.find(qn('w:cols'))
    numCols = int(cols_el.get(qn('w:num'), 1)) if cols_el is not None else 1
    if cols_el is not None and cols_el.get(qn('w:space')):
        gapTwip = int(cols_el.get(qn('w:space')))
    else:
        gapTwip = DEFAULT_GAP if numCols > 1 else 0

    body_twip   = (pgW - left - right)
    total_gap   = gapTwip * (numCols - 1)
    colW_twip   = int((body_twip - total_gap) / max(1, numCols))

    to_pt = lambda L: L * EMU_TO_PT if hasattr(L, 'emu') else L.pt
    return (
        to_pt(pgW), to_pt(pgH),
        to_pt(top), to_pt(bot), to_pt(left), to_pt(right),
        numCols, gapTwip, colW_twip
    )

def extract_rows_from_docx(path):
    doc = Document(path)

    sec_iter     = itertools.chain(doc.sections, [doc.sections[-1]])  # sentinel
    cur_section  = next(sec_iter)
    (pgW, pgH, top, bot, left, right,
     numCols, gapTw, colW_tw) = _section_props(cur_section)

    pages        = []
    page_no      = 1
    col_idx      = 1
    y_cursor     = 0.0

    def new_page():
        nonlocal page_no, col_idx, y_cursor
        pages.append({"page": page_no, "blocks": []})
        page_no += 1
        col_idx  = 1
        y_cursor = 0.0
    new_page()

    def _advance(block_h_pt):
        nonlocal col_idx, y_cursor
        usable_h = pgH - top - bot
        if y_cursor + block_h_pt <= usable_h:
            return
        if col_idx < numCols:
            col_idx += 1
            y_cursor = 0.0
        else:
            pages[-1]["blocks"] and None
            new_page()

    for blk in _iter_blocks(doc):

        if isinstance(blk, Paragraph):
            sectPr = blk._p.find('.//w:sectPr', namespaces={"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"})
            if sectPr is not None:
                typ = sectPr.find(qn('w:type'))
                if not (typ is not None and typ.get(qn('w:val')) == 'continuous'):
                    new_page()
                cur_section._sectPr = sectPr
                (pgW, pgH, top, bot, left, right,
                 numCols, gapTw, colW_tw) = _section_props(cur_section)

        if isinstance(blk, Table):
            rows = [
                ["\n".join(p.text for p in cell.paragraphs).strip()
                 for cell in row.cells]
                for row in blk.rows
            ]
            row_cnt = len(rows)
            col_cnt = len(rows[0]) if row_cnt else 0

            font_pt = blk.style.font.size.pt if blk.style and blk.style.font.size else 11
            row_h   = font_pt * DEFAULT_LH
            tbl_h   = row_cnt * row_h

            _advance(tbl_h)

            x0 = left + (col_idx-1)*(colW_tw*TWIP_TO_PT + gapTw*TWIP_TO_PT)
            y0 = y_cursor + top
            x1 = x0 + colW_tw*TWIP_TO_PT
            y1 = y0 + tbl_h

            pages[-1]["blocks"].append({
                "page":    pages[-1]["page"],
                "column":  col_idx,
                "type":    "table",
                "row_cnt": row_cnt,
                "col_cnt": col_cnt,
                "rows":    rows,
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            })
            y_cursor += tbl_h
            continue

        if not isinstance(blk, Paragraph):
            continue

        pPr          = blk.paragraph_format
        space_before = (pPr.space_before.pt if pPr.space_before else 0.0)
        space_after  = (pPr.space_after.pt  if pPr.space_after  else 0.0)
        indent_left  = (pPr.left_indent.pt  if pPr.left_indent else 0.0)
        indent_right = (pPr.right_indent.pt if pPr.right_indent else 0.0)
        font_pt      = font_size(blk)

        bullet_only  = (
            (blk._p.pPr is not None and blk._p.pPr.numPr is not None)  # list para
            and not blk.text.strip()
        ) or (re.fullmatch(r"^[\u2022\u2023\u25E6\u2043\u2219\u00B7]+$", blk.text))

        segments = []
        buff     = ""

        for run in blk.runs:
            drawing = run._r.xpath('.//w:drawing')
            pict    = run._r.xpath('.//w:pict')
            if drawing or pict:
                if buff:
                    segments.append(("text", buff))
                    buff = ""
                blip = run._r.xpath('.//a:blip')
                rel  = blip[0].get(qn('r:embed')) if blip else None
                width_pt = height_pt = 0.0
                ext = run._r.xpath('.//wp:extent')
                if ext:
                    width_pt  = int(ext[0].get('cx')) * EMU_TO_PT
                    height_pt = int(ext[0].get('cy')) * EMU_TO_PT
                if rel and rel in doc.part.related_parts:
                    img_bytes = doc.part.related_parts[rel].blob
                    img_b64   = base64.b64encode(img_bytes).decode()
                else:
                    img_b64 = ""
                segments.append(("image", (img_b64, width_pt, height_pt)))
            else:
                buff += run.text
        if buff or not segments:
            segments.append(("text", buff))

        for s_type, payload in segments:
            if s_type == "text":
                txt = payload
                if not txt and bullet_only:
                    txt = "•"
                colW_pt   = colW_tw * TWIP_TO_PT - indent_left - indent_right
                colW_pt   = max(colW_pt, 1)
                n_chars   = max(1, len(txt.replace("\n", "")))
                est_lines = ceil((n_chars * AVG_CHAR_W * font_pt) / colW_pt)
                est_lines += txt.count("\n")
                line_h    = font_pt * DEFAULT_LH
                blk_h     = est_lines * line_h + space_before + space_after

                _advance(blk_h)

                x0 = left + indent_left + (col_idx-1)*(colW_tw*TWIP_TO_PT + gapTw*TWIP_TO_PT)
                y0 = y_cursor + top + space_before
                x1 = left + (col_idx-1)*(colW_tw*TWIP_TO_PT + gapTw*TWIP_TO_PT) + colW_tw*TWIP_TO_PT - indent_right
                y1 = y0 + (est_lines * line_h)

                pages[-1]["blocks"].append({
                    "page":    pages[-1]["page"],
                    "column":  col_idx,
                    "type":    "text",
                    "text":    txt,
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "font_size": font_pt,
                    "baseline": y1,
                    "is_bullet_only": bool(bullet_only and not payload)
                })
                y_cursor = (y1 - top) + space_after

            elif s_type == "image":
                img_b64, w_pt, h_pt = payload
                if w_pt == 0 or h_pt == 0:
                    w_pt = colW_tw * TWIP_TO_PT * 0.8
                    h_pt = w_pt * 0.75

                _advance(h_pt)

                x0 = left + indent_left + (col_idx-1)*(colW_tw*TWIP_TO_PT + gapTw*TWIP_TO_PT)
                y0 = y_cursor + top
                x1 = x0 + w_pt
                y1 = y0 + h_pt

                pages[-1]["blocks"].append({
                    "page":   pages[-1]["page"],
                    "column": col_idx,
                    "type":   "image",
                    "text":   img_b64,
                    "width":  w_pt,
                    "height": h_pt,
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1
                })
                y_cursor += h_pt

    return pages

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

def extract_blocks(path):
    return extract_rows_from_docx(path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: docx_xycut.py <file.docx>")
        sys.exit(1)

    result = extract_blocks(sys.argv[1])
    print(json.dumps(result, indent=2, ensure_ascii=False))