"""PDF -> per-page text.

Reads every PDF in data/ and writes data/pages.jsonl, one record per page:
    {"source_doc": "Cotton.pdf", "page": 3, "text": "..."}

Page is the unit here, not the chunk. Chunking happens separately in chunk.py so
that re-chunking never means re-parsing all 532 pages.
"""

import glob
import json
import os
import re
from collections import Counter

import pymupdf

DATA_DIR = "data"
OUT_PATH = os.path.join(DATA_DIR, "pages.jsonl")

# A block repeated on more than this fraction of a document's pages is a running
# header or footer, not content. Measured: "ITC (HS), 2018 / SCHEDULE 2 - EXPORT
# POLICY" appears on 125 of 126 pages of the export policy.
BOILERPLATE_RATIO = 0.5

# Fraction of page height at the top and bottom treated as margin. Only blocks
# landing there are candidates for being a running header or footer.
EDGE_BAND = 0.12

# Pages below this are covers and disclaimer pages. They carry no retrievable text.
MIN_PAGE_CHARS = 20

# Symbol fonts that extract as garbage. U+00ED is what the dash markers in the
# tariff hierarchy ("--- Other") become; U+F0B7 is a Wingdings bullet landing in
# the private use area. Neither is real text.
JUNK_CHARS = "í"

# Real words that are a single letter, so they must keep their space when a
# column wrap splits them off. Any other lone lowercase letter is the tail of a
# word broken across lines ("Prohibite" + "d") and joins without one.
SINGLE_LETTER_WORDS = {"a", "i"}


def join_wrapped(lines):
    """Rejoin the lines inside one block into a single string.

    Narrow table columns wrap descriptions one or two words per line. Most
    fragments are whole words and want a space. A lone lowercase letter is a
    broken word and does not.
    """
    text = ""
    for line in lines:
        if not text:
            text = line
        elif len(line) == 1 and line.islower() and line not in SINGLE_LETTER_WORDS:
            text += line
        else:
            text += " " + line
    return text


def page_blocks(page):
    """Text blocks of a page in reading order, wrapping repaired.

    Block mode keeps table cells apart, which plain get_text() does not: it
    would interleave a row's description with the next row's duty rate.

    Returns (text, is_edge) pairs. is_edge marks blocks in the top or bottom
    margin, the only place a running header or footer can live.
    """
    height = page.rect.height
    blocks = []
    for b in sorted(page.get_text("blocks"), key=lambda b: (round(b[1]), b[0])):
        raw = b[4]
        for ch in JUNK_CHARS:
            raw = raw.replace(ch, "")
        lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
        if lines:
            is_edge = b[1] < EDGE_BAND * height or b[3] > (1 - EDGE_BAND) * height
            blocks.append((join_wrapped(lines), is_edge))
    return blocks


def header_key(text):
    """Collapse digits so a header varies only by its page number.

    "128 SECTION-II CHAPTER-9" and "129 SECTION-II CHAPTER-9" are the same
    header. Without this they look like two distinct blocks and survive.
    """
    return re.sub(r"\d+", "#", text)


def find_boilerplate(pages_blocks):
    """Header/footer keys appearing on over half the pages of one document.

    Restricted to edge blocks on purpose. Applied to the whole page, the digit
    collapsing above would also fold genuine tariff rows together — every
    "#### ## ## --- Other kg. ###%" row would look like the same repeated block
    and real content would be deleted.
    """
    counts = Counter(
        header_key(t) for blocks in pages_blocks for t, edge in set(blocks) if edge
    )
    threshold = BOILERPLATE_RATIO * len(pages_blocks)
    return {k for k, n in counts.items() if n > threshold and len(k) < 120}


def extract(path):
    """One PDF -> list of page records, boilerplate and blank pages removed."""
    doc = pymupdf.open(path)
    pages_blocks = [page_blocks(p) for p in doc]
    doc.close()

    boilerplate = find_boilerplate(pages_blocks)
    records = []
    for i, blocks in enumerate(pages_blocks):
        kept = [
            t
            for t, edge in blocks
            if not (edge and header_key(t) in boilerplate) and not t.strip().isdigit()
        ]
        text = "\n".join(kept).strip()
        if len(text) >= MIN_PAGE_CHARS:
            records.append(
                {"source_doc": os.path.basename(path), "page": i + 1, "text": text}
            )
    return records, len(pages_blocks), len(boilerplate)


def main():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    all_records = []
    print(f"{'file':46} {'pages':>6} {'kept':>5} {'boiler':>7} {'chars':>8}")
    for path in paths:
        records, n_pages, n_boiler = extract(path)
        chars = sum(len(r["text"]) for r in records)
        all_records.extend(records)
        name = os.path.basename(path)
        print(f"{name[:46]:46} {n_pages:6} {len(records):5} {n_boiler:7} {chars:8}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    chars = sum(len(r["text"]) for r in all_records)
    print(f"\n{len(paths)} files -> {len(all_records)} pages, {chars} chars")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
