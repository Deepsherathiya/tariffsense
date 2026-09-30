"""Page text -> overlapping chunks.

Reads data/pages.jsonl and writes data/chunks.jsonl, one record per chunk:
    {"chunk_id": ..., "text": ..., "source_doc": ..., "page": ..., "char_start": ...}

Fixed-size windows are the deliberate baseline, not the assumed best answer.
Structure-aware splitting (on HS code boundaries, on numbered paragraphs) is the
variant to measure against this in Phase 5 — so the choice gets decided by
recall@5 rather than by argument.

Chunks never cross a page boundary. The data contract requires a page number and
the UI displays it, so a chunk spanning two pages would have no honest one. The
cost is that a clause broken by a page break is split; that is a real trade.
"""

import argparse
import json
import os
import re

PAGES_PATH = os.path.join("data", "pages.jsonl")
OUT_PATH = os.path.join("data", "chunks.jsonl")

CHUNK_SIZE = 1000
OVERLAP = 150

# --- structure-aware strategy ---------------------------------------------
# Where a new structural unit begins. Three patterns cover the corpus:
#   "0902 10 10"  spaced HS code, the tariff chapters
#   "07019000"    contiguous 8-digit code, the DGFT export policy schedule
#   "2.31 "       numbered paragraph at line start, FTP / HBP / general notes
STRUCTURE_BOUNDARY = re.compile(
    r"(?=\b\d{4} \d{2} \d{2}\b)|(?=\b\d{8}\b)|(?=^\s*\d+\.\d+\s)", re.M
)

# Adjacent units are merged until a chunk reaches this size. One tariff row on
# its own is about 88 characters and carries no context -- not even which
# chapter it belongs to. Measured over the corpus: at 250 the median tariff
# chunk holds 3 distinct HS codes against 10 under fixed-size chunking, which is
# the dilution this strategy exists to reduce, while keeping ~314 characters of
# surrounding context. Lowering it further trades context for precision.
MIN_UNIT = 250

# Below this a chunk is a stray fragment, not a passage.
MIN_CHUNK_CHARS = 25

# How far back from a window edge to look for whitespace before giving up and
# cutting mid-token. Splitting an 8-digit HS code in half would put a code in the
# index that exists in neither the document nor any query.
SNAP_WINDOW = 200


def split_point(text, end):
    """Nearest whitespace at or before `end`, if one is close enough."""
    if end >= len(text):
        return len(text)
    window = text.rfind("\n", end - SNAP_WINDOW, end)
    if window == -1:
        window = text.rfind(" ", end - SNAP_WINDOW, end)
    return window if window != -1 else end


def chunk_page(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """Yield (char_start, chunk_text) covering one page with overlap."""
    start = 0
    while start < len(text):
        end = split_point(text, start + size)
        piece = text[start:end].strip()
        if piece:
            yield start, piece
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)


def structural_units(text):
    """Split a page at structural boundaries. Returns (start, end) offsets."""
    starts = sorted({0} | {m.start() for m in STRUCTURE_BOUNDARY.finditer(text) if m.start() > 0})
    return list(zip(starts, starts[1:] + [len(text)]))


def emit(text, start, end):
    """Yield (char_start, chunk_text), splitting anything over CHUNK_SIZE."""
    while start < end:
        if end - start <= CHUNK_SIZE:
            stop = end
        else:
            stop = split_point(text, start + CHUNK_SIZE)
            if stop <= start:
                stop = min(end, start + CHUNK_SIZE)
        piece = text[start:stop].strip()
        if len(piece) >= MIN_CHUNK_CHARS:
            yield start, piece
        start = stop


def chunk_page_structured(text):
    """Yield (char_start, chunk_text) following the document's own structure.

    No overlap here, unlike the fixed strategy. Overlap exists to stop an
    arbitrary character boundary from severing a passage; a structural boundary
    is not arbitrary, so repeating text across chunks would only inflate the
    index and let one passage match twice.
    """
    span = None
    for start, end in structural_units(text):
        if span is None:
            span = [start, end]
        elif end - span[0] <= CHUNK_SIZE and span[1] - span[0] < MIN_UNIT:
            span[1] = end
        else:
            yield from emit(text, span[0], span[1])
            span = [start, end]
    if span is not None:
        yield from emit(text, span[0], span[1])


def make_id(source_doc, page, char_start):
    """Stable and unique: same inputs always produce the same id."""
    stem = os.path.splitext(source_doc)[0]
    return f"{stem}|p{page}|c{char_start}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--strategy",
        choices=["fixed", "structure"],
        default="fixed",
        help="fixed = character windows (baseline); structure = split on HS codes "
        "and numbered paragraphs",
    )
    args = ap.parse_args()
    splitter = chunk_page if args.strategy == "fixed" else chunk_page_structured

    with open(PAGES_PATH, encoding="utf-8") as f:
        pages = [json.loads(line) for line in f]

    chunks = []
    for page in pages:
        for char_start, text in splitter(page["text"]):
            chunks.append(
                {
                    "chunk_id": make_id(page["source_doc"], page["page"], char_start),
                    "text": text,
                    "source_doc": page["source_doc"],
                    "page": page["page"],
                    "char_start": char_start,
                }
            )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    sizes = sorted(len(c["text"]) for c in chunks)
    ids = {c["chunk_id"] for c in chunks}
    print(f"strategy: {args.strategy}")
    print(f"{len(pages)} pages -> {len(chunks)} chunks")
    print(f"unique ids: {len(ids)} (duplicates: {len(chunks) - len(ids)})")
    print(f"chunk chars: min={sizes[0]} median={sizes[len(sizes) // 2]} max={sizes[-1]}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
