import re
import fitz
import statistics
from statistics import median
from collections import defaultdict
from typing import List, Tuple, Dict, Iterable

COL_FACTOR   = 1.0
ROW_FACTOR   = 1.5
BIN_WIDTH    = 1
BULLET_RE = re.compile(r"^[\u2022\u2023\u25E6\u2043\u2219\u00B7]+$")

def extract_baseline_lines(page: fitz.Page) -> List[Dict]:
    """
    Return one dictionary **per physical baseline** on the page.

    ── PDF text hierarchy handled here ─────────────────────────────
    • Block  - PyMuPDF “paragraph-level” unit, inferred from layout.
    • Line   - all glyphs that share the *same baseline* inside a block.
               A single baseline can mix styles, so PyMuPDF breaks it
               into multiple *spans* while keeping one common bbox.
    • Span   - a consecutive run of glyphs that share identical font
               family, size, style, colour, writing direction, etc.

      ┌─ Example:  “The  quick  bold  brown  fox”
      │            └────┬────────────────────────── same line/baseline
      │                 │
      │        regular-face span(s)      bold-face span
      └─────────────────────────────────────────────────────────────

    Because one line may contain several spans, we compute the line’s
    *dominant* font size as the statistical mode of its span sizes
    (most common size across all spans).  This baseline-level record
    is later combined with word-level data for gap detection, bullets,
    and XY-cut segmentation.

    Returns
    -------
    List[Dict]
        Each dict contains:
        • bbox      - (x0, y0, x1, y1) of the line
        • spans     - raw span list for debugging
        • fontsize  - dominant font size (float, points)
    """
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
    """
    Group every *word* on the page into buckets keyed by a quantized baseline
    (Y₀ coordinate).
        - Look at every word on the page.
        - PyMuPDF hands each word to you with its position: (left x, top y, right x, bottom y, text, …)
        - Give each word a “line-ID”.

    ── Source granularity ─────────────────────────────────────────────
    Uses ``page.get_text("words")`` which yields one tuple **per word**:
        (x0, y0, x1, y1, text, block_no, line_no, word_no)

    ── What “baseline bucketing” means ───────────────────────────────
    • Words whose top-left Y₀ differs by ≤ *y_tol* points are considered to
      lie on the *same* physical baseline.   
    • The key is ``round(y0 / y_tol) * y_tol`` – a small grid that absorbs
      minor anti-aliasing / hinting noise.

    ── Output structure ──────────────────────────────────────────────
    {
        baseline_y : [                                 # sorted left→right
            {
                "bbox"  : (x0, y0, x1, y1),            # word rectangle
                "text"  : "The",
                "x0"    : x0,                          # cached for speed
                "x1"    : x1,
                "height": y1 - y0
            },
            …
        ],
        …
    }

    This fast lookup table is later consumed by ``split_physical_line`` and
    column / gutter logic that require precise word-level gaps on a given
    baseline.

    Parameters
    ----------
    page : fitz.Page
        Page from which to extract word tuples.
    y_tol : float, default 0.5 pt
        Quantisation quantum for baseline grouping.

    Returns
    -------
    Dict[float, List[Dict]]
        Maps each quantised baseline to its left-to-right list of words.
    """
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
    """
    Break one **printed baseline** into smaller “logical rows” whenever a
    *visually large* horizontal blank separates consecutive words.

    Parameters
    ----------
    baseline_meta : Dict
        The record produced by ``extract_baseline_lines`` for this baseline.
        It already contains:
            • bbox      – bounding-box for the *whole* baseline
            • fontsize  – dominant font size on that baseline
    words_on_baseline : List[Dict]
        All word-level dicts that share this baseline, supplied by
        ``bucket_words_by_baseline`` (each has x0, x1, bbox, text, …).  The
        list **must be pre-sorted left→right**.
    gap_factor : float, default 2.5
        A horizontal gap is considered “large” if it is at least
        ``gap_factor × median_character_width``.  Increase to split less,
        decrease to split more aggressively.

    Returns
    -------
    List[Dict]
        Zero or more chunk dicts, each representing **one logical row**
        printed on the same baseline but separated by a large gap.  Every
        chunk contains:
            • bbox      – union of its words
            • text      – concatenation of its words with single spaces
            • fontsize  – copied from *baseline_meta*
            • baseline  – y-coordinate of this baseline (top of bbox)

    Why this matters
    ----------------
    PDFs often place two semantically distinct items on the same baseline:

        ▪ “• Apples       • Oranges”
        ▪ “Name     Price”
        ▪ “Premium:        $400”

    Treating the entire baseline as one string would lose that structure.
    By detecting unusually wide gaps (scaled by character width) we preserve
    bullet lists, key-value pairs, and pseudo-columns for later layout
    analysis.

    Algorithm
    ---------
    1. Compute the median **character width** on this baseline:
           w_char_med = median( (x1−x0)/len(text) for each word )
    2. Define a threshold:
           gap_threshold = gap_factor × w_char_med
    3. Scan words left→right; whenever the white-space between *prev* and
       *curr* (``curr.x0 − prev.x1``) ≥ gap_threshold, start a new chunk.
    4. For every chunk, build the bounding box and join the text.
    """
    if not line_words:
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
    """
    Runs the full baseline → words mapping + sub‑line splitting.
    Adds field 'is_bullet_only' to each logical chunk.
    """
    lines = extract_baseline_lines(page)
    word_buckets = bucket_words_by_baseline(page)

    logical_rows = []
    for ln in lines:
        y0 = ln["bbox"][1]
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

def preprocess_page(page: fitz.Page) -> List[Dict]:
    logical_rows = split_and_tag_lines(page)
    cleaned_rows = merge_bullets(logical_rows)
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
    """
    Detect a **mostly empty vertical stripe** (a “gutter”) inside the region
    spanned by the rows indexed in *idx*.  Such a gutter is strong evidence
    of a multi-column layout, and `xy_cut_region` will split on it before
    attempting any horizontal cuts.

    Parameters
    ----------
    lines : List[Dict]
        All pre-processed row dictionaries for the current page (output of
        `preprocess_page`).  Only geometry fields (`x0, x1`) are used here.
    idx : List[int]
        Indices of the rows that form the *current* rectangular region under
        consideration.  We try to find a gutter **only inside this subset**.
    page_w : float
        Full page width, needed for a sanity check that prevents splitting
        off unreasonably narrow columns.
    char_w : float
        Median character width for this region (obtained from
        `compute_dominant_sizes`).  The minimum gutter width is scaled by
        this value so “one character wide” on a small font is still accepted.
    tol : float, default 0.02
        Maximum fraction of rows that may *touch* the candidate gutter
        (default 2 %).  Allows a few overhanging headlines or images without
        cancelling the split.

    Returns
    -------
    Optional[Tuple[float, float]]
        `(gx0, gx1)` – left and right x-coordinates of the gutter in page
        space – if a suitable stripe is found; otherwise `None`.

    How the algorithm works
    -----------------------
    1. **Restrict the search** to the horizontal span that encloses all rows
       in *idx* (`rx0 … rx1`).  Work entirely within that mini-rectangle.

    2. **Discretise** that span into 1 pt-wide vertical bins
       (`BIN_WIDTH = 1`).  For every row, mark all bins it overlaps.  The
       result is a histogram `hist[b]` counting “how many rows touch this
       bin?”

    3. **Thresholds**  
       • *max_occ* = `max(1, tol × len(idx))`  
         → bins touched by no more than ~ 2 % of rows are “sufficiently
         empty”.  
       • *run_req* = `max(1, int(COL_FACTOR × char_w / BIN_WIDTH))`  
         → the gutter must span at least one **character width** of empty
         bins.

    4. **Scan the histogram** looking for the *longest* consecutive run of
       low-occupancy bins (≤ *max_occ*).  Store the best run.

    5. **Return** the page-x coordinates of that run if it meets
       *run_req*; else return `None`.

    6. **Safety check in `xy_cut_region`**  
       Even when a gutter is reported, the caller confirms both resulting
       sub-regions are wider than `min_block_width × page_w` to avoid bogus
       splits on accidental white-space.

    Practical outcome
    -----------------
    For two-column documents, the gutter is the white channel between
    columns; for key-value layouts it can be the wide gap after the keys.
    Detecting it early lets XY-Cut recurse **left → middle-bridge → right**
    instead of slicing horizontally first.
    """
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
    """
    Measure the **vertical white-space** between two printed rows.

    Parameters
    ----------
    a, b : Dict
        Row dictionaries that already carry absolute page coordinates:
        • ``y0`` – top edge of the row (baseline-aligned in PyMuPDF)
        • ``y1`` – bottom edge of the row

    Returns
    -------
    float
        • **> 0.0**  → the rows are separated by that many points of blank paper  
        • **0.0**    → the rows touch or overlap (negative distances are clamped)

    Notes
    -----
    ``b["y0"] - a["y1"]`` gives the raw distance from the bottom of *row a*
    to the top of *row b*.  Wrapping the result in ``max(0.0, …)`` normalises
    all *touching / overlapping* cases to **“no gap” = 0.0** so later logic
    can focus on *large enough* gaps without worrying about negative values.

    This helper is the primitive that higher-level heuristics
    (median-gap estimates, heading boosts, XY-cut recursion) build upon.
    """
    return max(0.0, b["y0"] - a["y1"])

def looks_like_heading(row: Dict,
                       median_font: float,
                       *,
                       size_ratio: float = 1.15,
                       short_len: int = 60) -> bool:
    """Return True for rows that look like a heading or sub‑heading."""
    text = row["text"].strip()

    # NEW: must contain at least one letter → excludes "• •" lines
    if not any(ch.isalpha() for ch in text):
        return False

    big_enough = row["size"] >= median_font * size_ratio
    short_text = len(text) <= short_len
    return big_enough and short_text

def is_large_gap(g: float, line_h: float, med_gap: float) -> bool:
    thresh_abs = ROW_FACTOR * line_h
    thresh_rel = 2.5 * med_gap if med_gap else 0
    return g >= max(thresh_abs, thresh_rel)

def find_horizontal_gap(lines, idx, page_h, line_h, *, tol=0.02):
    """
    Look for a mostly empty horizontal stripe between the rows in *idx*.

    Returns
    -------
    Optional[Tuple[float, float]]
        (gy0, gy1) – top & bottom Y of the gap in page coordinates –  
        or None if no sufficiently wide / clean stripe exists.
    """
    ry0 = min(lines[i]["y0"] for i in idx)
    ry1 = max(lines[i]["y1"] for i in idx)
    h   = ry1 - ry0
    if h <= 0 or line_h == 0:
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
    run_req = max(1, int((ROW_FACTOR * line_h) // BIN_HEIGHT))

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
#  Main recursive XY‑cut
# ──────────────────────────────────────────────────────────────────────────────
def xy_cut_region(idx, lines, page_w, page_h, tbl_boxes,
                  *, min_block_width=0.02, min_block_height=0.02):
    """
    Split the region that consists of rows *idx* until no significant
    vertical (column) **or** horizontal (paragraph) whitespace remains.

    Returns
    -------
    List[List[int]]
        A list of “leaf” segments, each being the list of row indices
        that belong to one output text block.
    """
    if not idx or len(idx) == 1:
        return [idx]

    # ── exclude rows that sit inside user‑supplied table boxes ──────────────
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

    # ── 1) try vertical split (classic XY‑cut) ──────────────────────────────
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
                                      min_block_width=min_block_width,
                                      min_block_height=min_block_height)
                     + xy_cut_region(sorted(bridge,key=lambda i: lines[i]["y0"]),
                                      lines, page_w, page_h, tbl_boxes,
                                      min_block_width=min_block_width,
                                      min_block_height=min_block_height)
                     + xy_cut_region(sorted(right, key=lambda i: lines[i]["y0"]),
                                      lines, page_w, page_h, tbl_boxes,
                                      min_block_width=min_block_width,
                                      min_block_height=min_block_height))

    # ── 2) try *horizontal* split (new branch) ──────────────────────────────
    hgap = find_horizontal_gap(lines, idx, page_h, line_h)
    if hgap:
        gy0, gy1 = hgap
        upper = [i for i in idx if lines[i]["y1"] <= gy0]
        lower = [i for i in idx if lines[i]["y0"] >= gy1]

        if upper and lower:                                 # sane split?
            min_h = min_block_height * page_h
            u_h   = max(lines[i]["y1"] for i in upper) - min(lines[i]["y0"] for i in upper)
            l_h   = max(lines[i]["y1"] for i in lower) - min(lines[i]["y0"] for i in lower)
            if u_h >= min_h and l_h >= min_h:
                return (xy_cut_region(sorted(upper, key=lambda i: lines[i]["y0"]),
                                      lines, page_w, page_h, tbl_boxes,
                                      min_block_width=min_block_width,
                                      min_block_height=min_block_height)
                     + xy_cut_region(sorted(lower, key=lambda i: lines[i]["y0"]),
                                      lines, page_w, page_h, tbl_boxes,
                                      min_block_width=min_block_width,
                                      min_block_height=min_block_height))

    by_top   = sorted(idx, key=lambda i: lines[i]["y0"])
    med_font = statistics.median(lines[i]["size"] for i in idx)

    gaps_white  = [] 
    gaps_base   = [] 
    gaps = []
    for a, b in zip(by_top, by_top[1:]):
        white_gap = gap(lines[a], lines[b])
        base_gap  = lines[b]["y0"] - lines[a]["y0"]

        # heading → boost gap artificially
        if looks_like_heading(lines[a], med_font) or looks_like_heading(lines[b], med_font):
            base_gap = max(base_gap, ROW_FACTOR * line_h + 1)
        gaps_white.append(white_gap)
        gaps_base.append(base_gap)

    med_gap = statistics.median(gaps_white) if gaps_white else 0
    gaps    = gaps_base

    big = [k for k, g in enumerate(gaps_base)
           if is_large_gap(g, line_h, med_gap)]

    if big:
        split_pos = max(big, key=lambda k: gaps[k]) + 1
        upper = by_top[:split_pos]
        lower = by_top[split_pos:]
        return (xy_cut_region(upper, lines, page_w, page_h, tbl_boxes,
                              min_block_width=min_block_width,
                              min_block_height=min_block_height)
             + xy_cut_region(lower, lines, page_w, page_h, tbl_boxes,
                              min_block_width=min_block_width,
                              min_block_height=min_block_height))

    # ── 4) nothing to cut – return the current region ──────────────────────
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
        "text": text,
    }

def iterate_chunks(page):
    rows = preprocess_page(page)
    idx  = list(range(len(rows)))
    tbl_boxes = []
    page_w, page_h = page.rect.width, page.rect.height

    segs = xy_cut_region(idx, rows, page_w, page_h,
                            tbl_boxes, min_block_width=0.02)
    return [make_output_chunk(seg, rows) for seg in segs if seg]

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

    pdf_path = pathlib.Path(arg_pdf or
                            "25M06-02C.pdf") # 2024-Artificial-empathy-in-healthcare-chatbots.pdf 2025Centene.pdf 64654-genesys.pdf 25M06-02C.pdf
    doc   = fitz.open(pdf_path)
    pages = parse_pages(arg_pages, doc.page_count)

    result = [{"page": p+1, "blocks": iterate_chunks(doc.load_page(p))}
              for p in pages]
    print(json.dumps(result, ensure_ascii=False, indent=2))