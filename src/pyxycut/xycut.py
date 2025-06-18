#!/usr/bin/env python3
# Extract text, headings and ruled tables from scientific PDFs
# ------------------------------------------------------------
# – page layout is analysed with a classic recursive XY‑cut
# – paragraphs are detected by vertical whitespace
# – headings are recognised by font size *or* outline numbers
#   *or* (new) almost‑all‑caps strings such as “A B S T R A C T”
# – simple ruled tables are returned as a separate JSON block
#
# This version is tested on
#   “2024‑Artificial‑empathy‑in‑healthcare‑chatbots.pdf”
# and fixes the two long‑standing issues:
#   1. footer “2” being merged with header “L. Seitz”
#   2. “ARTICLE INFO” disappearing after code refactors
# ------------------------------------------------------------
import fitz, statistics, json, re
from   typing import List, Tuple

# ──────── layout constants ───────────────────────────────────
COL_FACTOR   = 1.0          # min. column gap  = 1.0 × char‑width
ROW_FACTOR   = 1.5          # min. paragraph gap = 1.5 × line‑height
BIN_WIDTH    = 1            # projection histogram bin in points
FONT_TOL     = 0.5          # two spans are same font if ≤ 0.5 pt apart
SIZE_FACTOR  = 1.15         # heading if ≥ 1.15 × region median font
HDR_RE = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+[A-Z]")   # “1.” / “2.1 …”

BBox = Tuple[float, float, float, float]               # helper alias

print(fitz.__doc__.split()[1])  

# ───────────────── table helpers ─────────────────────────────
def find_tables_fast(page: fitz.Page) -> List["fitz.Table"]:
    print(">>> scanning tables on page", page.number+1)
    for strategy, tol in (("lines", 14), ("structure", None)):
        try:
            if strategy == "lines":
                tf = page.find_tables(strategy=strategy,
                                      intersection_tolerance=tol)
            else:                       # structure: no intersection tol.
                tf = page.find_tables(strategy=strategy)
            print("   strategy", strategy, "→", len(tf.tables) if tf else 0)
            if tf and tf.tables:
                return tf.tables
        except Exception:
            pass
    return []

def rows_from_table(page: fitz.Page, tbl: "fitz.Table") -> List[List[str]]:
    try:                               # MuPDF built‑in extractor
        return tbl.extract()
    except Exception:
        pass
    
    x0, y0, x1, y1 = tbl.bbox
    words = [w for w in page.get_text("words")
             if x0 <= 0.5*(w[0]+w[2]) <= x1 and y0 <= 0.5*(w[1]+w[3]) <= y1]
    if not words:
        return []
    words.sort(key=lambda w: (round(w[1], 1), w[0]))
    rows, cur, cur_y = [], [], None
    for w in words:
        yy = round(w[1], 1)
        if cur_y is None or abs(yy-cur_y) > 2:
            if cur:
                rows.append([" ".join(cur)])
            cur, cur_y = [w[4]], yy
        else:
            cur.append(w[4])
    if cur:
        rows.append([" ".join(cur)])
    return rows

def table_feature_dict(page: fitz.Page, tbl: "fitz.Table", pno: int) -> dict:
    rows = rows_from_table(page, tbl)
    return {
        "type":    "table",
        "page":    pno,
        "bbox":    [round(v, 2) for v in tbl.bbox],
        "row_cnt": len(rows),
        "col_cnt": max((len(r) for r in rows), default=0),
        "rows":    rows,
        "text":    ""
    }

def in_any_table(x0, y0, x1, y1, tbl_boxes):
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    return any(x0b <= cx <= x1b and y0b <= cy <= y1b for
               x0b, y0b, x1b, y1b in tbl_boxes)

def get_lines(page: fitz.Page, tbl_boxes=()) -> List[dict]:
    out: List[dict] = []

    for blk in page.get_text("dict")["blocks"]:
        if blk["type"] != 0:          # skip images etc.
            continue

        for l in blk["lines"]:
            if not l["spans"]:        # empty line
                continue

            x0 = min(s["bbox"][0] for s in l["spans"])
            y0 = min(s["bbox"][1] for s in l["spans"])
            x1 = max(s["bbox"][2] for s in l["spans"])
            y1 = max(s["bbox"][3] for s in l["spans"])

            if in_any_table(x0, y0, x1, y1, tbl_boxes):
                continue

            text = "".join(s["text"] for s in l["spans"]).strip("\n")
            if not text:              # all‑whitespace line
                continue

            size = statistics.median(s["size"] for s in l["spans"])
            out.append({
                "idx":  len(out),
                "x0":   x0,  "y0": y0,
                "x1":   x1,  "y1": y1,
                "text": text,
                "size": size,
            })

    return out

# ───────────── text helpers ──────────────────────────────────
def compute_dominant_sizes(lines: List[dict]) -> Tuple[float, float]:
    cw, lh = [], []
    for ln in lines:
        n = sum(1 for c in ln["text"] if not c.isspace())
        if n:
            cw.append((ln["x1"]-ln["x0"])/n)
        lh.append(ln["y1"]-ln["y0"])
    char = statistics.median(cw) if cw else 0
    if not lh:
        return char, 0
    q1 = statistics.quantiles(lh, n=4)[0]       # throw away tiny superscripts
    core = [h for h in lh if h >= q1]
    return char, statistics.median(core)

def is_all_caps(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 4:                # ignore tiny words
        return False
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters) > 0.8

def looks_like_heading(ln: dict, median_font: float) -> bool:
    txt = ln["text"]
    spaced_caps = txt.replace(" ", "").isupper() and len(txt) <= 30
    bigger      = ln["size"] >= 1.10 * median_font
    numeric     = bool(HDR_RE.match(txt))
    return bigger or numeric or spaced_caps

def gap(a: dict, b: dict) -> float:
    return b["y0"] - a["y1"]

def is_large_gap(g: float, line_h: float, med_gap: float) -> bool:
    abs_floor = 0.6 * ROW_FACTOR * line_h
    rel_floor = 1.2 * med_gap if med_gap else abs_floor
    return g >= abs_floor and g >= rel_floor

def compute_gap(a,b): return b["y0"]-a["y1"]

def split_on_big_internal_gap(lines,row_factor,line_h):
    out,cur=[],[lines[0]]
    for p,n in zip(lines,lines[1:]):
        if compute_gap(p,n)>=row_factor*line_h:
            out.append(cur); cur=[n]
        else: cur.append(n)
    out.append(cur); return out

# ───────────── recursive XY‑cut ───────────────────────────────
def find_vertical_gutter(lines, idx, page_w, char_w, tol=0.02):
    rx0 = min(lines[i]["x0"] for i in idx)
    rx1 = max(lines[i]["x1"] for i in idx)
    w   = rx1 - rx0
    if char_w == 0 or w <= 0:
        return None
    bins  = int(w // BIN_WIDTH) + 1
    hist  = [0]*bins
    for i in idx:
        x0 = max(lines[i]["x0"], rx0); x1 = min(lines[i]["x1"], rx1)
        b0 = int((x0-rx0)//BIN_WIDTH); b1 = int((x1-rx0)//BIN_WIDTH)
        for b in range(b0, b1+1):
            hist[b] += 1
    max_occ = max(1, int(tol*len(idx)))
    run_req = max(1, int((COL_FACTOR*char_w)//BIN_WIDTH))
    best = None; cur = None
    for j, occ in enumerate(hist+[max_occ+1]):          # sentinel
        if occ <= max_occ:
            cur = j if cur is None else cur
        elif cur is not None:
            run = j-cur
            if run >= run_req and (best is None or run > best[2]):
                best = (cur, j-1, run)
            cur = None
    if not best:
        return None
    g0, g1, _ = best
    return rx0 + g0*BIN_WIDTH, rx0 + g1*BIN_WIDTH + BIN_WIDTH

def xy_cut_region(idx, lines, page_w, page_h, tbl_boxes,
                  min_block_width=0.0):
    if len(idx) <= 1:
        return [idx]

    def in_table(ln):
        cx = (ln["x0"] + ln["x1"]) / 2
        cy = (ln["y0"] + ln["y1"]) / 2
        return any(x0 <= cx <= x1 and y0 <= cy <= y1
                   for x0, y0, x1, y1 in tbl_boxes)

    char_w, line_h = compute_dominant_sizes(
        [lines[i] for i in idx if not in_table(lines[i])]
    )
    if char_w == 0 or line_h == 0:
        return [idx]

    # ───── try vertical split (columns) ───────────────
    gutter = find_vertical_gutter(lines, idx, page_w, char_w)
    if gutter:
        gx0, gx1 = gutter
        left  = [i for i in idx if lines[i]["x1"] <= gx0]
        right = [i for i in idx if lines[i]["x0"] >= gx1]
        bridge = [i for i in idx if i not in left and i not in right
                    and lines[i]["x0"] < gx1 and lines[i]["x1"] > gx0]
        if left and right:
            min_w = min_block_width * page_w
            if min_block_width and (
                (max(lines[i]["x1"] for i in left) -
                 min(lines[i]["x0"] for i in left) < min_w) or
                (max(lines[i]["x1"] for i in right) -
                 min(lines[i]["x0"] for i in right) < min_w)):
                gutter = None
        else:
            gutter = None

        if gutter:
            return (
                xy_cut_region(sorted(left,  key=lambda i: lines[i]["y0"]),
                              lines, page_w, page_h, tbl_boxes, min_block_width)
              + xy_cut_region(sorted(bridge,key=lambda i: lines[i]["y0"]),
                              lines, page_w, page_h, tbl_boxes, min_block_width)
              + xy_cut_region(sorted(right, key=lambda i: lines[i]["y0"]),
                              lines, page_w, page_h, tbl_boxes, min_block_width)
            )

    # ───── horizontal split (paragraphs) ──────────────
    by_top   = sorted(idx, key=lambda i: lines[i]["y0"])
    med_font = statistics.median(lines[i]["size"] for i in idx)
    gaps, mids = [], []
    for a, b in zip(by_top, by_top[1:]):
        g = gap(lines[a], lines[b])
        if looks_like_heading(lines[a], med_font) or looks_like_heading(lines[b], med_font):
            g = max(g, ROW_FACTOR * line_h + 1)
        gaps.append(g)
    med_gap = statistics.median(gaps) if gaps else 0
    big = [k for k, g in enumerate(gaps) if is_large_gap(g, line_h, med_gap)]
    if big:
        s = max(big, key=lambda k: gaps[k]) + 1
        upper = by_top[:s]; lower = by_top[s:]
        return (xy_cut_region(upper, lines, page_w, page_h, tbl_boxes, min_block_width) +
                xy_cut_region(lower, lines, page_w, page_h, tbl_boxes, min_block_width))
    return [idx]

# ───────── main page routine ─────────────────────────
def segment_page(page, min_block_width_frac: float = 0.0) -> list[dict]:
    """
    Return a list[{type,font_size,bbox,text}] for *one* MuPDF page.
    – honours: multi‑column layout, headings, very wide gaps, page header/footer,
               and masks ruling‑line tables when computing statistics.
    """
    tbls            = find_tables_fast(page)
    tbl_boxes       = [t.bbox for t in tbls]
    lines           = get_lines(page, tbl_boxes)

    if not lines and not tbls:
        return []

    # ----------------- basic layout information -----------------
    page_w, page_h = page.rect.width, page.rect.height
    line_h = statistics.median(ln["y1"] - ln["y0"] for ln in lines)
    first_text_y = min(ln["y0"] for ln in lines)
    top_band     = first_text_y + 2.0 * line_h

    last_text_y  = max(ln["y1"] for ln in lines)
    bot_band     = last_text_y - 2.0 * line_h

    idx            = list(range(len(lines)))          # <‑‑ was missing
    head_idx       = [i for i in idx if lines[i]["y1"] <= top_band]
    foot_idx       = [i for i in idx if lines[i]["y0"] >= bot_band]
    body_idx       = [i for i in idx if i not in head_idx + foot_idx]

    # ----------------- tables (only for *masking* stats) --------
 
    # ----------------- XY‑cut on each zone ----------------------
    blocks_idx  : list[list[int]] = []

    if head_idx:
        blocks_idx += xy_cut_region(sorted(head_idx, key=lambda i: lines[i]["y0"]),
                                    lines, page_w, page_h,
                                    tbl_boxes, min_block_width_frac)

    if body_idx:
        blocks_idx += xy_cut_region(sorted(body_idx, key=lambda i: lines[i]["y0"]),
                                    lines, page_w, page_h,
                                    tbl_boxes, min_block_width_frac)

    if foot_idx:
        blocks_idx += xy_cut_region(sorted(foot_idx, key=lambda i: lines[i]["y0"]),
                                    lines, page_w, page_h,
                                    tbl_boxes, min_block_width_frac)

    # ----------------- build JSON blocks ------------------------
    out      : list[dict] = []
    for idx_list in blocks_idx:
        blk_lines = [lines[i] for i in idx_list]
        if not blk_lines:                       # <‑‑ guards the median()
            continue

        # ❶ split on very wide vertical gaps inside *this* block
        line_h  = statistics.median(ln["y1"] - ln["y0"] for ln in blk_lines)
        parts   = split_on_big_internal_gap(blk_lines, ROW_FACTOR, line_h)

        for part in parts:
            # ❷ sub‑split on font changes
            segments, cur = [], [part[0]]
            for prev, nxt in zip(part, part[1:]):
                same_font = abs(nxt["size"] - prev["size"]) <= FONT_TOL
                big_gap   = (nxt["y0"] - prev["y1"]) >= ROW_FACTOR * line_h
                if same_font and not big_gap:
                    cur.append(nxt)
                else:
                    segments.append(cur); cur = [nxt]
            segments.append(cur)

            # ❸ emit JSON
            for seg in segments:
                xs = [ln["x0"] for ln in seg] + [ln["x1"] for ln in seg]
                ys = [ln["y0"] for ln in seg] + [ln["y1"] for ln in seg]

                text, prev = [], ""
                for ln in seg:
                    t = ln["text"].strip()
                    if text and prev.endswith("-"):
                        text[-1] = text[-1][:-1] + t     # soft hyphen
                    else:
                        text.append(t)
                    prev = t

                out.append({
                    "type":      "text",
                    "order":     seg[0]["idx"],
                    "font_size": round(statistics.median(ln["size"] for ln in seg), 2),
                    "bbox":      [round(min(xs),2), round(min(ys),2),
                                  round(max(xs),2), round(max(ys),2)],
                    "text":      " ".join(text)
                })

    # natural reading order
    out.sort(key=lambda b: b["order"])
    for b in out:
        del b["order"]
    
    for t in sorted(tbls, key=lambda T: T.bbox[1]):
        out.insert(0, table_feature_dict(page, t, page.number))

    return out


# ──────────────── CLI wrapper ────────────────────────
if __name__ == "__main__":
    import sys, pathlib

    def parse_pages(expr, max_p):
        if expr is None:
            return range(max_p)
        pages = set()
        for part in expr.split(","):
            if "-" in part:
                a, b = map(int, part.split("-", 1)); pages.update(range(a-1, b))
            else:
                pages.add(int(part)-1)
        return sorted(p for p in pages if 0 <= p < max_p)

    arg_pdf, arg_pages = None, None
    if len(sys.argv) >= 2:
        if re.fullmatch(r"[\d, -]+", sys.argv[1]):
            arg_pages = sys.argv[1]
        else:
            arg_pdf   = sys.argv[1]
            if len(sys.argv) >= 3:
                arg_pages = sys.argv[2]

    pdf_path = pathlib.Path(arg_pdf or
                            "2024-Artificial-empathy-in-healthcare-chatbots.pdf")
    doc   = fitz.open(pdf_path)
    pages = parse_pages(arg_pages, doc.page_count)

    result = [{"page": p+1, "blocks": segment_page(doc.load_page(p))}
              for p in pages]
    print(json.dumps(result, ensure_ascii=False, indent=2))
