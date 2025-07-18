import argparse, json, pandas as pd, sys, pathlib, itertools as it

def iter_blocks(files):
    for f in files:
        with open(f, "r", encoding="utf‑8") as handle:
            try:
                pages = json.load(handle)
            except json.JSONDecodeError as e:
                sys.exit(f"{f}: cannot parse JSON – {e}")
        if not isinstance(pages, list):
            sys.exit(f"{f}: top level must be a list of pages")
        for page in pages:
            for blk in page.get("blocks", []):
                yield blk

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="one or more *.json files")
    ap.add_argument("-o", "--out", default="segments.csv",
                    help="output CSV (default: segments.csv)")
    args = ap.parse_args(argv)

    rows = []
    required_bbox_keys = {"bbox", "page_x0", "page_y0", "page_x1", "page_y1"}

    for blk in iter_blocks(args.inputs):
        if not required_bbox_keys.issubset(blk):
            print(f"warning: block missing keys, skipped: {blk.get('type','?')}")
            continue
        try:
            x0, y0, x1, y1 = blk["bbox"]
        except (ValueError, TypeError):
            print(f"warning: bbox not 4-tuple, skipped: {blk.get('bbox')}")
            continue

        rows.append(
            {
                "x0": x0, "y0": y0,
                "x1": x1, "y1": y1,
                "page_x0": blk["page_x0"],
                "page_y0": blk["page_y0"],
                "page_x1": blk["page_x1"],
                "page_y1": blk["page_y1"],
                "Segment": ""
            }
        )

    if not rows:
        sys.exit("No valid blocks found - nothing written.")

    df = pd.DataFrame(rows, columns=[
        "x0","y0","x1","y1","page_x0","page_y0","page_x1","page_y1","Segment"
    ])
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} observations to {args.out}")

if __name__ == "__main__":
    main()
