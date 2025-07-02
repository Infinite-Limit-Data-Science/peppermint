To segment PDF text into reading-order chunks separated by whitespace, we implement a recursive X–Y cut algorithm.  The method follows the classic top-down approach:
- Vertical cuts (X-cuts): Find a significant vertical white gap (column gutter) and split the region into left/right sub-regions.
- Horizontal cuts (Y-cuts): In each sub-region, find a significant horizontal gap (extra space between lines/paragraphs) and split into top/bottom sub-regions.
This ensures columns are split rather than the fatal error of reading left to right spilling from one column into another, causing the content to be nonsensical. 

The XY cut algorithm steps repeat recursively until no large whitespace gap remains.

We use data-driven thresholds based on the dominant font size in each region: 
- roughly one average character width for vertical gaps, 
- and ~1.5× the average line height for horizontal gaps.

This adaptive thresholding (inspired by PdfPig’s implementation) ensures that only substantial whitespace (e.g. inter-column gutters or paragraph breaks) triggers a cut. We also provide an optional minimum block-width filter (default off) to avoid cutting off extremely narrow text fragments (e.g. an isolated bullet or page number) as separate chunks

The output is assembled in reading order (top-to-bottom, left-to-right). Each text chunk is returned with its bounding box and merged text content. We preserve paragraph integrity unless a clear blank line or extra spacing indicates a new paragraph. This way, multi-column layouts (even with misaligned columns) and separated sections (abstract, headings, etc.) are correctly chunked without merging across columns.

Note once segments are fully divided, the below algorithm then looks at font sizes between lines and separates text chunks by font size so we can clearly capture headers within a reduced segment.



Why “spans” are convenient for text extraction but treacherous for page‑layout analysis

| PyMuPDF term | What the object really is                                                                                                | How the library groups it                               | Why the grouping is *arbitrary* from a layout point of view                                                                                      |
| ------------ | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **word**     | A run of non‑white‑space chars whose **baselines are contiguous**.                                                       | Returned by `page.get_text("words")`.                   | Closest to what humans call a “word”; gaps between words are preserved.                                                                          |
| **span**     | *All* consecutive text‑show operators that share **exactly the same graphics state** (font, size, colour, matrix, etc.). | Returned inside each `line` by `page.get_text("dict")`. | Two physically remote fragments can live in one span if the PDF author issues several `Tj` commands in a row before changing the graphics state. |
| **line**     | All the spans that share the same **text origin** (baseline) according to the PDF stream.                                | Returned inside each `block`.                           | If the author prints a full paragraph with one text matrix the line covers the entire paragraph, even across columns.                            |
| **block**    | All objects drawn until the graphics state says *“a new paragraph starts here”*.                                         | Top‑level items in `page.get_text("dict")`.             | A single block may cover half a page if the source stream never closes the current text object.                                                  |

Why a single span can straddle two columns
d
PDF text positioning works like a typewriter:

```text
Tm               # set text matrix
Tj "What applies to the out‑of‑pocket maximum?"
Td 18 0          # move 18 points right  (still same line)
Tj "•"
Td 6 0
Tj "All Copayments …"
```

As long as the font, size, colour and matrix are unchanged, every Tj call is appended to the same span by PyMuPDF.

Result: the span’s bbox stretches from the left margin through the gutter into the right column, so your XY‑cut sees no vertical gap.

Why XY‑cut fails when it only sees spans
- Vertical cuts look for long runs where no bbox overlaps the hypothetical gutter. If one span covers both columns, its wide bbox masks the gap.
- Horizontal cuts look for tall gaps between bboxes on the Y axis. A paragraph printed in one graphics state becomes one big line → one big bbox → no internal vertical whitespace.

Why bullet rows are a special headache
- Publishers often print the bullet glyph (•) in a separate font (symbol font, different colour) → bullet gets its own span with an extremely narrow bbox (just the dot).
- If you naïvely split on x‑gaps you detach the dot from the text, ending with a useless “• •” span and a separate text span.

Practical consequences:
- Spans are great when you only want the text string, because you get proper words in reading order. EXACTLY.
- Spans are unreliable for geometry‑based algorithms (XY‑cut, column detection, table detection) because bounding boxes can be much larger than the visible glyph clusters.

| Step                                                                                                         | What to use                                                                                                                                | Why                                                                                                                     |
| ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| 1 Split physical lines into logical chunks                                                                   | Iterate through `page.get_text("words")`. For every physical line (same rounded `y0`) look at gaps between **word** `x1`/`x0`.             | Words preserve the original inter‑word whitespace. A single rogue span cannot hide the gutter if you work word‑by‑word. |
| 2 Detect bullets                                                                                             | After you formed a *logical chunk*, test whether it contains only “bullet glyphs” (regex on the resulting text).                           | Works regardless of how many separate spans the bullet pieces came from.                                                |
| 3 Merge bullet‑only chunks with the chunk that starts on the **same baseline** within a small `x` tolerance. | Re‑attaches the dot to its text while still keeping the chunk’s true width.                                                                |                                                                                                                         |
| 4 Feed those logical chunks to the recursive XY‑cut.                                                         | Now each chunk’s bbox covers exactly the visible ink; real gutters and paragraph gaps re‑appear, so vertical & horizontal cuts work again. |                                                                                                                         |

Why “spans / lines” and “words” each solve only half of the layout puzzle

| MuPDF extractor                                    | What you really get                                                                                             | Typical strong point                                                                                                                                            | Typical weak point                                                                                                                                                                                     |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `page.get_text("dict")` → *blocks → lines → spans* | **Physical** PDF text objects exactly as the file stores them (one baseline per “line”, one span per font run). | *Vertical* reasoning is excellent: every span already carries the exact baseline `y0`, so detecting paragraph breaks, headers/footers, etc. is straightforward. | For *horizontal* reasoning the objects are **too coarse**. A single span can run across the whole page, so left/right columns or bulleted lists that sit on the **same baseline** cannot be separated. |
| `page.get_text("words")`                           | MuPDF tokenises the same text into individual **words** (one‑word bbox for every token).                        | *Horizontal* reasoning is now easy: you see the real gaps between adjacent words and can split at “large” gaps to discover columns or list items.               | You lose the grouping into physical baselines. Every word has its own `y0`, so paragraph detection based on line‐to‐line gaps (or header/footer detection) becomes noisier.                            |


A common issue:

In the plan schedule PDF the whole bullet zone is drawn like this (simplified):

```css
(<span>• •</span>)      (<span>All Copayments … Non‑covered charges</span>)
(<span>• •</span>)      (<span>Coinsurance … Benefit penalties</span>) (<span>•</span>)
(<span>DED (including Pharmacy)</span>)
```
- The first two spans go across both logical columns; inside each of them all bullet glyphs (or all list text) are concatenated with just ordinary spaces.
- For MuPDF that is one span → one “line” at y ≈ 522 pt. As far as the extractor is concerned there is no gap in that line, so xy_cut_region() never finds a vertical gutter.
- When you switch to get_text("words") the extractor sees the bullets and the text tokens (because words must be cut at spaces). That does reveal a 200 pt wide gap between the last bullet in the left column and the first word in the right column, but you lose the neat line‑by‑line structure that helps all your vertical logic.

How people usually handle this
Keep the line structure from "dict"
Use it for everything that cares about baselines: paragraph / header / footer detection, vertical whitespace analysis, average line height, etc.

Within each physical line look at words to decide whether that line must be split horizontally
A robust recipe is:

collect all words with the same baseline (round(y0, 0.5 pt) is usually good enough);

compute the median character width for that line (ignore spaces);

if the gap between two consecutive words is ≥ N × char_width (empirically N ≈ 2.5…3.0) start a new logical line.

Re‑assemble bullets
After you have split a physical line into logical sub‑lines you may discover some sub‑lines that contain only bullet glyphs (•, ‣, ◦, etc.).
Merge such a bullet row with the text row that starts at (almost) the same x0 and shares the same baseline within a tolerance of 3–4 points. This gives you one logical line that reads “• Coinsurance …” instead of two separate objects.


| extractor                | what it gives you                                                                              | typical strengths                                                                                                                                      | typical weaknesses                                                                                                                                                     |
| ------------------------ | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `page.get_text("dict")`  | hierarchical list of **blocks → lines → spans**<br>(every span keeps the original glyph order) | • Very robust vertical positioning (each physical baseline comes back as a separate *line*).<br>• Font size is already attached to every span.         | • When two logical columns are drawn with one text‑run the whole thing remains **one span** – there is **no information about the wide horizontal gap** in the middle. |
| `page.get_text("words")` | a flat list of words with individual bboxes                                                    | • Every real word has its own box – wide spaces between words show up as large X‑gaps, **even if the PDF creator placed several columns in one span**. | • Words on the same baseline are **not grouped**, so you first have to put them back together to know where one physical line ends and the next begins.                |

For bullet lists in your 25M06‑02C.pdf page both problems occur:
- all bullets+texts are emitted as one span ("dict") – so the XY‑cut sees no vertical gap after “• DED …”;
- if you look only at "words" you lose the guarantee that words belonging to the same baseline are adjacent in the list, so heading/footer detection etc. becomes less stable.

The reliable way is to combine the two:
- Use "dict" to build physical lines (one entry per baseline, font size intact).
-  For every physical line, look up all words whose vertical centre falls onto that baseline; split the line into logical sub‑lines wherever the gap between consecutive words is large.


{
  "type": "text",
  "text": "What applies to the out‑of‑pocket maximum?",
  ...
},
{
  "type": "text",
  "text": "• All Copayments (including Pharmacy)\n• Coinsurance (including Pharmacy)\n• DED (including Pharmacy)",
  ...
},
{
  "type": "text",
  "text": "What does not apply to out‑of‑pocket maximums?",
  ...
},
{
  "type": "text",
  "text": "• Non‑covered charges\n• Benefit penalties",
  ...
}
