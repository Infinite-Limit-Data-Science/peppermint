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

def extract_physical_lines(page: fitz.Page) -> List[Dict]:
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


def group_words_by_baseline(page: fitz.Page, y_tol: float = 0.5) -> Dict[float, List[Dict]]:
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


def split_physical_line(
    line: Dict,
    line_words: List[Dict],
    gap_factor: float = 2.5,
) -> List[Dict]:
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
    lines = extract_physical_lines(page)
    word_buckets = group_words_by_baseline(page)

    logical_rows = []
    for ln in lines:
        y0 = ln["bbox"][1]
        key = round(y0 / 0.5) * 0.5
        words_same_baseline = word_buckets.get(key, [])
        chunks = split_physical_line(ln, words_same_baseline)
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

def looks_like_heading(row: Dict, median_font: float, *, size_ratio: float = 1.15,
                       short_len: int = 60) -> bool:
    big_enough = row["size"] >= median_font * size_ratio
    short_text = len(row["text"]) <= short_len
    return big_enough and short_text

def is_large_gap(g: float, line_h: float, med_gap: float) -> bool:
    thresh_abs = ROW_FACTOR * line_h
    thresh_rel = 2.5 * med_gap if med_gap else 0
    return g >= max(thresh_abs, thresh_rel)

def xy_cut_region(idx, lines, page_w, page_h, tbl_boxes,
                  min_block_width=0.02):
    
    if not idx:
        return []

    if len(idx) == 1:
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