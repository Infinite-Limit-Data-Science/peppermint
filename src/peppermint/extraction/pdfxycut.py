import re, base64
import fitz
from pathlib import Path
import statistics
from statistics import median
from collections import defaultdict
from typing import List, Tuple, Dict, Iterable, Any

COL_FACTOR   = 1.0
BIN_WIDTH    = 1
BULLET_RE = re.compile(r"^[\u2022\u2023\u25E6\u2043\u2219\u00B7]+$")

xy_cut_region_depth = 0

DEBUG = True

def find_tables_fast(page) -> list[Any]:
    DEBUG = True

    def _horiz_segments(min_len=80, y_tol=1.0):
        segs = []

        for path in page.get_drawings():
            r = path["rect"]
            if r.height <= y_tol and r.width >= min_len:
                y = r.y0
                segs.append((r.x0, y, r.x1, y))

        if not segs:
            tdict = page.get_text("dict")
            for blk in tdict.get("blocks", []):
                if blk.get("type") != 4:
                    continue
                for item in blk.get("items", []):
                    cmd = item[0]
                    if cmd == "l":
                        (x0, y0), (x1, y1) = item[1]
                        if abs(y1 - y0) <= y_tol and (x1 - x0) >= min_len:
                            segs.append((x0, y0, x1, y1))
                    elif cmd == "re":
                        x0, y0, x1, y1 = item[1]
                        if (x1 - x0) >= min_len:
                            segs.append((x0, y0, x1, y0))
                            segs.append((x0, y1, x1, y1))

        if DEBUG:
            import sys
            print(f"   horiz segments ≥{min_len}px: {len(segs)}",
                  file=sys.stderr)
            for s in segs[:10]:
                print(f"      y={s[1]:.1f}  x0={s[0]:.1f} x1={s[2]:.1f}",
                      file=sys.stderr)

        return segs

    native = page.find_tables(strategy="lines", intersection_tolerance=14)
    if native and native.tables:
        if DEBUG:
            import sys
            print(f"[page {page.number+1}] native tables: {len(native.tables)}",
                  file=sys.stderr)
        return native.tables                    # ← best quality

    if DEBUG:
        import sys
        print(f"[page {page.number+1}] native detector found nothing",
              file=sys.stderr)

    hlines = _horiz_segments()
    if len(hlines) < 2:
        if DEBUG:
            import sys
            print("   not enough horizontals → abort", file=sys.stderr)
        return []

    # longest horizontal at top & bottom
    top = min(hlines, key=lambda s: (s[1], -(s[2] - s[0])))
    bot = max(hlines, key=lambda s: (s[1],  (s[2] - s[0])))
    x0, y0, x1, y1 = top[0], top[1], top[2], bot[1]

    if DEBUG:
        import sys
        print(f"   frame y0={y0:.1f} y1={y1:.1f} h={y1-y0:.1f} "
              f"w={x1-x0:.1f}", file=sys.stderr)

    if (y1 - y0) < 40 or (x1 - x0) < 250:
        if DEBUG:
            import sys
            print("   frame too small → reject", file=sys.stderr)
        return []

    # gather words inside frame
    words = [w for w in page.get_text("words")
             if x0 <= (w[0] + w[2]) * 0.5 <= x1
             and y0 <= (w[1] + w[3]) * 0.5 <= y1]
    if DEBUG:
        import sys
        print(f"   words inside frame: {len(words)}", file=sys.stderr)
    if not words:
        return []

    # infer columns
    xs = sorted(set(round(w[0], 1) for w in words))
    col_starts = [xs[0]]
    for a, b in zip(xs, xs[1:]):
        if b - a >= 40:
            col_starts.append(b)
    col_starts.append(x1)
    if DEBUG:
        import sys
        print(f"   column starts ({len(col_starts)-1}): {col_starts[:-1]}",
              file=sys.stderr)

    # ─── reject frames that are almost certainly NOT tables ────────────
    # 1) only one real column detected   → looks like plain text
    if len(col_starts) <= 2:            # note: sentinel makes it 2
        if DEBUG:
            print("   only one inferred column → reject", file=sys.stderr)
        return []

    # 2) top & bottom horizontals must span (almost) the same width
    EPS = 5.0
    same_span = abs(top[0] - bot[0]) <= EPS and abs(top[2] - bot[2]) <= EPS
    if not same_span:
        if DEBUG:
            print("   frame edges have different spans → reject", file=sys.stderr)
        return []
    # -------------------------------------------------------------------
    
    # build rows
    rows_map = {}
    for w in words:
        mid_y = round(w[1], 1)
        for c in range(len(col_starts) - 1):
            if col_starts[c] <= w[0] < col_starts[c + 1]:
                rows_map.setdefault(mid_y, [""] * (len(col_starts) - 1))
                cell = rows_map[mid_y][c]
                rows_map[mid_y][c] = (cell + " " + w[4]).strip()
                break

    rows = [rows_map[k] for k in sorted(rows_map)]
    if DEBUG:
        import sys
        print(f"   built {len(rows)} rows", file=sys.stderr)

    # fabricate pseudo‑Table
    class _PseudoTable:
        def __init__(self, bbox, rows):
            self.bbox = bbox
            self.row_count = len(rows)
            self.col_count = max(len(r) for r in rows)
            self._rows = rows
        def extract(self):
            return self._rows

    return [_PseudoTable((x0, y0, x1, y1), rows)]

def tight_bbox_from_cells(t: "fitz.Table") -> tuple[float, float, float, float]:
    xs0, ys0, xs1, ys1 = [], [], [], []
    for row in (t.cells or []):
        for cell in (row or []):
            if cell is None:
                continue
            x0, y0, x1, y1 = cell.bbox
            xs0.append(x0); ys0.append(y0); xs1.append(x1); ys1.append(y1)
    if not xs0:
        return t.bbox
    return min(xs0), min(ys0), max(xs1), max(ys1)

def rows_from_table(page: fitz.Page, tbl: "fitz.Table") -> list[list[str]]:
    """
    1. Try the native PyMuPDF extractor (works for *true* fitz.Table).
    2. If that fails we rebuild the rows from the words that fall
       inside tbl.bbox - this also works for the pseudo-tables the
       fallback detector creates.
    """
    try:
        return tbl.extract()
    except Exception:
        pass

    x0, y0, x1, y1 = tbl.bbox
    words = [w for w in page.get_text("words")
             if x0 <= 0.5*(w[0]+w[2]) <= x1 and
                y0 <= 0.5*(w[1]+w[3]) <= y1]

    if not words:
        return []

    words.sort(key=lambda w: (round(w[1], 1), w[0]))

    rows, cur_row, cur_y = [], [], None
    for w in words:
        y = round(w[1], 1)
        if cur_y is None or abs(y - cur_y) > 2:
            if cur_row:
                rows.append(cur_row)
            cur_row, cur_y = [w[4]], y
        else:
            cur_row.append(w[4])

    if cur_row:
        rows.append(cur_row)

    split_on = re.compile(r'\s{2,}')
    return [split_on.split(" ".join(r).strip()) for r in rows]

def _compress_columns(rows, min_populated=2):
    """
    Remove columns that have fewer than `min_populated` non-empty cells.
    Returns a new list of rows.
    """
    if not rows:
        return rows

    max_cols = max(len(r) for r in rows)
    counts   = [0] * max_cols
    for r in rows:
        for j, cell in enumerate(r):
            if j < max_cols and cell and str(cell).strip():
                counts[j] += 1

    keep = [j for j, c in enumerate(counts) if c >= min_populated]
    if len(keep) == max_cols:
        return rows

    new_rows = [[ (r[j] if j < len(r) else None) for j in keep ]
                for r in rows]

    new_rows = [r for r in new_rows if any(cell and str(cell).strip() for cell in r)]
    return new_rows

def table_feature_dict(page: fitz.Page, tbl: "fitz.Table", pno: int) -> dict:
    rows = rows_from_table(page, tbl)
    rows = _compress_columns(rows, min_populated=2)
    
    return {
        "type":    "table",
        "page":    pno + 1,
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


def table_feature_dict(page: fitz.Page, tbl: "fitz.Table", pno: int) -> dict:
    rows = rows_from_table(page, tbl)
    rows = _compress_columns(rows, min_populated=2)
    
    return {
        "type":    "table",
        "page":    pno + 1,
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


def add_page_bbox(block: dict, page_rect) -> dict:
    block["page_x0"] = float(page_rect.x0)
    block["page_y0"] = float(page_rect.y0)
    block["page_x1"] = float(page_rect.x1)
    block["page_y1"] = float(page_rect.y1)
    return block

def _median_plus_mad(values: List[float], k: float = 1.5) -> float:
    if not values:
        return 0.0
    med = statistics.median(values)
    mad = statistics.median(abs(v - med) for v in values)
    return med + k * mad

def compute_page_gap_stats(rows: List[Dict]) -> tuple[float, float]:
    if len(rows) <= 1:
        return 9999.0, 0.0

    by_top   = sorted(rows, key=lambda r: r["y0"])
    gaps     = [by_top[i+1]["y0"] - by_top[i]["y0"]
                for i in range(len(by_top) - 1)]
    gaps     = [g for g in gaps if g > 0]
    gaps.sort()

    if not gaps:
        return 9999.0, 0.0

    jump_idx, jump_ratio = None, 1.0
    for i in range(1, len(gaps)):
        if gaps[i-1] == 0:
            continue
        ratio = gaps[i] / gaps[i-1]
        if ratio > jump_ratio:
            jump_ratio, jump_idx = ratio, i

    page_abs_thr = _median_plus_mad(gaps, k=1.5)

    # 75‑th percentile for the relative rule
    k75 = int(0.75 * (len(gaps) - 1))
    page_p75_gap = gaps[k75]

    return page_abs_thr, page_p75_gap

def region_threshold(idx: List[int],
                     lines: List[dict],
                     *,
                     page_abs_thr: float,
                     page_p75_gap: float,
                     jump_ratio_req: float = 1.35,
                     cap_mult: float = 3.0
                     ) -> Tuple[float, float]:

    if len(idx) < 6:
        return page_abs_thr, page_p75_gap

    by_top = sorted(idx, key=lambda i: lines[i]["y0"])
    gaps   = [lines[b]["y0"] - lines[a]["y0"]
              for a, b in zip(by_top, by_top[1:])
              if lines[b]["y0"] > lines[a]["y0"]]

    if not gaps:
        return page_abs_thr, page_p75_gap

    gaps.sort()
 

    local_abs = _median_plus_mad(gaps, k=1.5)

    # 2.  safety‑cap against ridiculous values
    med_gap = statistics.median(gaps)
    local_abs = min(local_abs, med_gap * cap_mult)

    # 3.  local 75‑th percentile
    k75      = int(0.75 * (len(gaps) - 1))
    local_p75 = gaps[k75]

    return local_abs, local_p75

def merge_adjacent_blocks(
        blocks: List[Dict],
        *,
        page_abs_thr: float,
        page_p75_gap: float,
        cap_mult: float   = 3.0,
        min_horiz_overlap: float = 0.50
) -> List[Dict]:
    if len(blocks) <= 1:
        return blocks

    merged: List[Dict] = [blocks[0]]

    merge_abs_thr = min(page_abs_thr, page_p75_gap * cap_mult)

    for blk in blocks[1:]:
        cur = merged[-1]

        v_gap = blk["bbox"][1] - cur["last_baseline"]
        overlap = min(cur["bbox"][2], blk["bbox"][2]) - max(cur["bbox"][0],
                                                            blk["bbox"][0])
        min_width = min(cur["bbox"][2] - cur["bbox"][0],
                        blk["bbox"][2] - blk["bbox"][0])
        h_ratio = overlap / min_width if min_width > 0 else 0.0

        if v_gap < merge_abs_thr and h_ratio >= min_horiz_overlap:
            cur["text"] = cur["text"].rstrip() + " " + blk["text"].lstrip()
            cur["bbox"][0] = min(cur["bbox"][0], blk["bbox"][0])
            cur["bbox"][2] = max(cur["bbox"][2], blk["bbox"][2])
            cur["bbox"][3] = blk["bbox"][3]
            cur["last_baseline"] = blk["last_baseline"]
        else:
            merged.append(blk)

    return merged



def extract_baseline_lines(page: fitz.Page) -> List[Dict]:
    blocks = page.get_text("dict")["blocks"]
    lines = []
    for blk in blocks:
        for ln in blk.get("lines", []):
            sizes = [sp["size"] for sp in ln["spans"] if "size" in sp]
            if not sizes:
                continue
            dominant = max(set(sizes), key=sizes.count)
            lines.append(
                {
                    "bbox": ln["bbox"],
                    "spans": ln["spans"],
                    "fontsize": dominant,
                }
            )
    return lines


def bucket_words_by_baseline(page: fitz.Page, y_tol: float = 0.5) -> Dict[float, List[Dict]]:
    words = page.get_text("words")
    buckets: Dict[float, List[Dict]] = defaultdict(list)
    for w in words:
        x0, y0, x1, y1, text, *_ = w
        key = round(y0 / y_tol) * y_tol
        buckets[key].append(
            {
                "bbox": (x0, y0, x1, y1),
                "text": text,
                "x0": x0,
                "x1": x1,
                "height": y1 - y0,
            }
        )
    for seq in buckets.values():
        seq.sort(key=lambda w: w["x0"])
    return buckets


def char_widths(word_dicts: Iterable[Dict]) -> List[float]:
    widths = []
    for w in word_dicts:
        text = w["text"]
        if not text:
            continue
        width = (w["x1"] - w["x0"]) / len(text)
        widths.append(width)
    return widths


def split_baseline_into_chunks(
    line: Dict,
    line_words: List[Dict],
    gap_factor: float = 2.5,
) -> List[Dict]:
    if not line_words:
        if DEBUG:
            print(f"[DROP] baseline {line['bbox'][1]:.2f} – "
                f"0 words found, line lost")
        return []
    
    width_vals = char_widths(line_words)
    med = median(width_vals) if width_vals else 1.0
    x_threshold = gap_factor * med

    chunks = []
    current_chunk: List[Dict] = [line_words[0]]
    for prev, cur in zip(line_words, line_words[1:]):
        gap = cur["x0"] - prev["x1"]
        if gap >= x_threshold:
            chunks.append(current_chunk)
            current_chunk = []
        current_chunk.append(cur)
    chunks.append(current_chunk)

    logical = []
    for words in chunks:
        x0 = min(w["x0"] for w in words)
        x1 = max(w["x1"] for w in words)
        y0 = min(w["bbox"][1] for w in words)
        y1 = max(w["bbox"][3] for w in words)
        text = " ".join(w["text"] for w in words)
        logical.append(
            {
                "bbox": (x0, y0, x1, y1),
                "text": text,
                "fontsize": line["fontsize"],
                "baseline": y0,
            }
        )
    return logical

def split_and_tag_lines(page: fitz.Page) -> List[Dict]:
    lines = extract_baseline_lines(page)
    word_buckets = bucket_words_by_baseline(page)

    logical_rows = []
    for ln in lines:
        y0 = ln["bbox"][1]
        if DEBUG:
            print(f"[BASE] y0={y0:7.2f}  text="
                f"{' '.join(sp.get('text', '') for sp in ln['spans'])[:50]}")
        key = round(y0 / 0.5) * 0.5
        words_same_baseline = word_buckets.get(key, [])
        chunks = split_baseline_into_chunks(ln, words_same_baseline)
        for ck in chunks:
            ck["is_bullet_only"] = bool(BULLET_RE.fullmatch(ck["text"].strip()))
            logical_rows.append(ck)

    return logical_rows


def merge_bullets(rows: List[Dict], char_tol_factor: float = 1.0) -> List[Dict]:
    merged = []
    by_baseline: Dict[float, List[Dict]] = defaultdict(list)
    for r in rows:
        by_baseline[r["baseline"]].append(r)

    for baseline, seq in by_baseline.items():
        seq.sort(key=lambda r: r["bbox"][0])
        if len(seq) == 1:
            merged.extend(seq)
            continue

        pool = [
            (r["bbox"][2] - r["bbox"][0]) / max(1, len(r["text"]))
            for r in seq if not r["is_bullet_only"]
        ]
        cw = median(pool) if pool else 1.0

        keep = []
        skip_ids = set()
        for i, r in enumerate(seq):
            if r["is_bullet_only"] and i + 1 < len(seq):
                nxt = seq[i + 1]
                if (
                    abs(nxt["baseline"] - r["baseline"]) <= 3.0
                    and abs(nxt["bbox"][0] - r["bbox"][0]) <= char_tol_factor * cw
                ):
                    nxt["text"] = f"{r['text']} {nxt['text']}"
                    x0 = min(r["bbox"][0], nxt["bbox"][0])
                    nxt["bbox"] = (x0, *nxt["bbox"][1:])
                    skip_ids.add(id(r))
        for r in seq:
            if id(r) not in skip_ids:
                keep.append(r)
        merged.extend(keep)

    merged.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    return merged

def flatten_rows(rows: List[Dict]) -> List[Dict]:
    flat = []
    for r in rows:
        x0, y0, x1, y1 = r["bbox"]
        body = {k: v for k, v in r.items() if k not in {"bbox", "fontsize"}}
        flat.append(
            {"x0": x0, "y0": y0, "x1": x1, "y1": y1,
             "size": r["fontsize"], **body}
        )
    return flat

def preprocess_page(page: fitz.Page, tbl_boxes=()) -> List[Dict]:
    logical_rows = split_and_tag_lines(page)
    cleaned_rows = merge_bullets(logical_rows)

    if tbl_boxes:
        logical_rows = [
            r for r in logical_rows
            if not in_any_table(r["bbox"][0], r["bbox"][1],
                                r["bbox"][2], r["bbox"][3],
                                tbl_boxes)
        ]

    if DEBUG:
        for i, r in enumerate(cleaned_rows):
            if "Conceptual background" in r["text"]:
                x0, y0, x1, y1 = r["bbox"]
                print(f"[ROW]  idx={i:3d}  survives preprocess  "
                    f"x0={x0:.1f}  y0={y0:.1f}  width={x1-x0:.1f}")   

    if DEBUG:
        print(f"[STAT] rows after merge_bullets: {len(cleaned_rows)}")
    return flatten_rows(cleaned_rows)

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
    q1 = statistics.quantiles(lh, n=4)[0]
    core = [h for h in lh if h >= q1]
    return char, statistics.median(core)

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
    for j, occ in enumerate(hist+[max_occ+1]):
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

def gap(a: Dict, b: Dict) -> float:
    return max(0.0, b["y0"] - a["y1"])

def looks_like_heading(row: Dict,
                       median_font: float,
                       *,
                       size_ratio: float = 1.15,
                       short_len: int = 60) -> bool:
    text = row["text"].strip()

    if not any(ch.isalpha() for ch in text):
        return False

    big_enough = row["size"] >= median_font * size_ratio
    short_text = len(text) <= short_len
    return big_enough and short_text

def is_large_gap(g: float, *, abs_thr: float, p75_gap: float, rel_mult: float = 1.10) -> bool:
    if p75_gap == 0:
        return g >= abs_thr
    return g >= abs_thr and g >= rel_mult * p75_gap

def find_horizontal_gap(lines, idx, *, run_req_pt: float, tol: float = 0.02):
    ry0 = min(lines[i]["y0"] for i in idx)
    ry1 = max(lines[i]["y1"] for i in idx)
    h   = ry1 - ry0
    if h <= 0:
        return None

    BIN_HEIGHT = 1
    bins = int(h // BIN_HEIGHT) + 1
    hist = [0] * bins

    for i in idx:
        y0 = max(lines[i]["y0"], ry0); y1 = min(lines[i]["y1"], ry1)
        b0 = int((y0 - ry0) // BIN_HEIGHT)
        b1 = int((y1 - ry0) // BIN_HEIGHT)
        for b in range(b0, b1 + 1):
            hist[b] += 1

    max_occ = max(1, int(tol * len(idx)))
    run_req = max(1, int(run_req_pt // BIN_HEIGHT))

    best = None; cur = None
    for j, occ in enumerate(hist + [max_occ + 1]):
        if occ <= max_occ:
            cur = j if cur is None else cur
        elif cur is not None:
            run = j - cur
            if run >= run_req and (best is None or run > best[2]):
                best = (cur, j - 1, run)
            cur = None

    if not best:
        return None
    g0, g1, _ = best
    return ry0 + g0 * BIN_HEIGHT, ry0 + g1 * BIN_HEIGHT + BIN_HEIGHT


# ──────────────────────────────────────────────────────────────────────────────
#  recursive XY‑cut algorithm
# ──────────────────────────────────────────────────────────────────────────────
def xy_cut_region(idx, lines, page_w, page_h, tbl_boxes,
                  *, page_abs_thr: float, page_p75_gap: float,
                  min_block_width=0.02, min_block_height=0.02):
    global xy_cut_region_depth
    
    if not idx or len(idx) == 1:
        return [idx]
    
    if DEBUG:
        first = min(lines[i]["y0"] for i in idx)
        last  = max(lines[i]["y1"] for i in idx)
        print(f"[XY] depth={xy_cut_region_depth:2d}  "
              f"rows={len(idx):3d}  span=({first:.1f} … {last:.1f})")
    xy_cut_region_depth += 1

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
    
    region_abs, region_p75 = region_threshold(
        idx, lines,
        page_abs_thr=page_abs_thr,
        page_p75_gap=page_p75_gap
    )

    use_region = len(idx) >= 6 and region_abs < page_abs_thr * 0.95
    abs_thr    = region_abs   if use_region else page_abs_thr
    p75_gap    = region_p75   if use_region else page_p75_gap

    if DEBUG:
        print(f"[XY] depth={xy_cut_region_depth:2d}  "
            f"abs_thr={abs_thr:4.1f}  p75={p75_gap:4.1f}  "
            f"rows={len(idx):3d}")

    # ── 1) vertical split ──────────────────────────────
    gutter = find_vertical_gutter(lines, idx, page_w, char_w)
    if gutter:
        gx0, gx1 = gutter
        left  = [i for i in idx if lines[i]["x1"] <= gx0]
        right = [i for i in idx if lines[i]["x0"] >= gx1]
        bridge = [i for i in idx if i not in left and i not in right
                  and lines[i]["x0"] < gx1 and lines[i]["x1"] > gx0]

        if left and right:                                   # sane split?
            min_w = min_block_width * page_w
            l_w   = max(lines[i]["x1"] for i in left) - min(lines[i]["x0"] for i in left)
            r_w   = max(lines[i]["x1"] for i in right) - min(lines[i]["x0"] for i in right)
            if l_w >= min_w and r_w >= min_w:
                return (xy_cut_region(sorted(left,  key=lambda i: lines[i]["y0"]),
                                      lines, page_w, page_h, tbl_boxes,
                                      page_abs_thr=abs_thr,
                                      page_p75_gap=p75_gap,
                                      min_block_width=min_block_width,
                                      min_block_height=min_block_height)
                     + xy_cut_region(sorted(bridge,key=lambda i: lines[i]["y0"]),
                                      lines, page_w, page_h, tbl_boxes,
                                      page_abs_thr=abs_thr,
                                      page_p75_gap=p75_gap,
                                      min_block_width=min_block_width,
                                      min_block_height=min_block_height)
                     + xy_cut_region(sorted(right, key=lambda i: lines[i]["y0"]),
                                      lines, page_w, page_h, tbl_boxes,
                                      page_abs_thr=abs_thr,
                                      page_p75_gap=p75_gap,
                                      min_block_width=min_block_width,
                                      min_block_height=min_block_height))

    # ── 2) *horizontal* split ──────────────────────────────
    hgap = find_horizontal_gap(lines, idx,
                           run_req_pt = abs_thr,
                           tol = 0.02)
    if hgap:
        gy0, gy1 = hgap

        upper  = [i for i in idx if lines[i]["y1"] <= gy0]
        lower  = [i for i in idx if lines[i]["y0"] >= gy1]
        bridge = [i for i in idx
                  if i not in upper and i not in lower
                  and lines[i]["y0"] < gy1
                  and lines[i]["y1"] > gy0]

        if upper and lower:
            min_h = min_block_height * page_h
            u_h   = max(lines[i]["y1"] for i in upper) - min(lines[i]["y0"] for i in upper)
            l_h   = max(lines[i]["y1"] for i in lower) - min(lines[i]["y0"] for i in lower)

            if u_h >= min_h and l_h >= min_h:
                return (
                    # recurse on the *upper* chunk
                    xy_cut_region(sorted(upper,  key=lambda i: lines[i]["y0"]),
                                   lines, page_w, page_h, tbl_boxes,
                                   page_abs_thr=abs_thr,
                                   page_p75_gap=p75_gap,
                                   min_block_width=min_block_width,
                                   min_block_height=min_block_height)
                  + # recurse on rows that straddled the gap
                    xy_cut_region(sorted(bridge, key=lambda i: lines[i]["y0"]),
                                   lines, page_w, page_h, tbl_boxes,
                                   page_abs_thr=abs_thr,
                                   page_p75_gap=p75_gap,
                                   min_block_width=min_block_width,
                                   min_block_height=min_block_height)
                  + # recurse on the *lower* chunk
                    xy_cut_region(sorted(lower,  key=lambda i: lines[i]["y0"]),
                                   lines, page_w, page_h, tbl_boxes,
                                   page_abs_thr=abs_thr,
                                   page_p75_gap=p75_gap,
                                   min_block_width=min_block_width,
                                   min_block_height=min_block_height)
                )

    by_top   = sorted(idx, key=lambda i: lines[i]["y0"])

    gaps_base = [lines[b]["y0"] - lines[a]["y0"]
                 for a, b in zip(by_top, by_top[1:])]

    rel_mult = 1.10
    if DEBUG:
        for k, g in enumerate(gaps_base):
            flag = is_large_gap(g, abs_thr=abs_thr, p75_gap=p75_gap)
            print(f"[GAP] k={k:3d}  g={g:5.1f}  "
                  f"abs_thr={abs_thr:5.1f}  "
                  f"rel_thr={rel_mult*p75_gap:5.1f}  "
                  f"large? {flag}")

    big = [k for k, g in enumerate(gaps_base)
           if is_large_gap(g,
                           abs_thr=abs_thr,
                           p75_gap=p75_gap,
                           rel_mult=rel_mult)]

    if big:
        split_pos = max(big, key=lambda k: gaps_base[k]) + 1
        upper = by_top[:split_pos]
        lower = by_top[split_pos:]
        return (xy_cut_region(upper, lines, page_w, page_h, tbl_boxes,
                              page_abs_thr=abs_thr,
                              page_p75_gap=p75_gap,
                              min_block_width=min_block_width,
                              min_block_height=min_block_height)
             + xy_cut_region(lower, lines, page_w, page_h, tbl_boxes,
                              page_abs_thr=abs_thr,
                              page_p75_gap=p75_gap,
                              min_block_width=min_block_width,
                              min_block_height=min_block_height))

    return [idx]

def make_output_chunk(seg: List[int], rows: List[Dict]) -> Dict:
    x0 = min(rows[i]["x0"] for i in seg)
    y0 = min(rows[i]["y0"] for i in seg)
    x1 = max(rows[i]["x1"] for i in seg)
    y1 = max(rows[i]["y1"] for i in seg)

    font = statistics.median(rows[i]["size"] for i in seg)

    text = " ".join(rows[i]["text"] for i in seg)

    return {
        "type": "text",
        "font_size": round(font, 2),
        "bbox": [x0, y0, x1, y1],
        "last_baseline": max(rows[i]["y0"] for i in seg),
        "text": text,
    }

def pixmap_base64(doc: fitz.Document, xref: int) -> str:
    try:
        pm = fitz.Pixmap(doc, xref)
        if pm.n > 4:  # convert CMYK / GRAY+ALPHA → RGB
            pm = fitz.Pixmap(fitz.csRGB, pm)
        return base64.b64encode(pm.tobytes("png")).decode()
    except Exception:
        return ""

def extract_images_from_page(page: fitz.Page) -> list[dict]:
    doc = page.parent
    img_blocks = []
    seen_rects = set()

    for img in page.get_images(full=True):
        xref, smask, width, height, bpc, cs, alt_cs, name, *_ = img

        for rect in page.get_image_rects(xref):
            key = (
                xref,
                round(rect.x0, 3),
                round(rect.y0, 3),
                round(rect.x1, 3),
                round(rect.y1, 3),
            )
            if key in seen_rects:
                continue
            seen_rects.add(key)

            img_blocks.append({
                "type":   "image",
                "page":   page.number + 1,
                "bbox":   [
                    float(rect.x0), float(rect.y0),
                    float(rect.x1), float(rect.y1)
                ],
                "width":  width,
                "height": height,
                "text":   pixmap_base64(doc, xref),
            })

    return img_blocks


def iterate_chunks(page):
    tbls       = find_tables_fast(page)
    tbl_boxes  = [t.bbox for t in tbls]

    rows   = preprocess_page(page, tbl_boxes)
    rows = [r for r in rows
            if not in_any_table(r["x0"], r["y0"],
                                r["x1"], r["y1"], tbl_boxes)]
    
    page_abs_thr, page_p75_gap = compute_page_gap_stats(rows)
    if DEBUG:
        print(f"[PAGE] abs_thr={page_abs_thr:.2f}  "
              f"p75_gap={page_p75_gap:.2f}")

    idx    = list(range(len(rows)))
    page_w, page_h = page.rect.width, page.rect.height

    segs = xy_cut_region(idx, rows, page_w, page_h, tbl_boxes,
                         page_abs_thr=page_abs_thr,
                         page_p75_gap=page_p75_gap,
                         min_block_width=0.02)

    text_blocks = [make_output_chunk(seg, rows) for seg in segs if seg]
    text_blocks = merge_adjacent_blocks(text_blocks,
                                        page_abs_thr=page_abs_thr,
                                        page_p75_gap=page_p75_gap)

    table_blocks = [
        table_feature_dict(page, t, page.number) for t in tbls
    ]

    image_blocks = extract_images_from_page(page)

    page_rect = page.rect
    table_blocks = [add_page_bbox(b, page_rect) for b in table_blocks]
    image_blocks = [add_page_bbox(b, page_rect) for b in image_blocks]
    text_blocks  = [add_page_bbox(b, page_rect) for b in text_blocks]
    
    return table_blocks + image_blocks + text_blocks

def extract_blocks(pdf_path: str | Path):
    with fitz.open(pdf_path) as pdf:
        for page_no in range(pdf.page_count):
            yield from iterate_chunks(pdf.load_page(page_no), page_no)

if __name__ == "__main__":
    import sys, pathlib, json

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

    pdf_path = pathlib.Path(arg_pdf)
    doc   = fitz.open(pdf_path)
    pages = parse_pages(arg_pages, doc.page_count)

    result = [{"page": p+1, "blocks": iterate_chunks(doc.load_page(p))}
              for p in pages]
    print(json.dumps(result, ensure_ascii=False, indent=2))