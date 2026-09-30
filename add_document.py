"""Fold a new PDF into the existing indexes without rebuilding everything.

    python add_document.py                  # scan data/ for anything new
    python add_document.py Cotton.pdf       # one file

Nothing here trains anything. The embedding model is frozen and downloaded; adding
a document means extract text -> chunk -> embed the new chunks -> append. Calling
this "training" would be wrong, and it is the kind of wrong an interviewer notices.

Re-embedding all 1,287 existing chunks to add one file takes about 80 seconds and
grows with the corpus, which is the whole reason this exists.
"""

import argparse
import glob
import hashlib
import json
import os

import numpy as np

import chunk as chunker
import ingest

DATA_DIR = "data"
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")
PAGES_PATH = ingest.OUT_PATH
CHUNKS_PATH = chunker.OUT_PATH


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def append_jsonl(path, records):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def add(filename, strategy="fixed"):
    """Process one PDF and append it to pages, chunks and both indexes."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    pages, n_pages, _ = ingest.extract(path)
    if not pages:
        raise ValueError(f"{filename}: no extractable text (scanned PDF?)")

    splitter = chunker.chunk_page if strategy == "fixed" else chunker.chunk_page_structured
    new_chunks = []
    for page in pages:
        for char_start, text in splitter(page["text"]):
            new_chunks.append(
                {
                    "chunk_id": chunker.make_id(page["source_doc"], page["page"], char_start),
                    "text": text,
                    "source_doc": page["source_doc"],
                    "page": page["page"],
                    "char_start": char_start,
                }
            )

    existing = set()
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            existing = {json.loads(line)["chunk_id"] for line in f}
    clashes = [c["chunk_id"] for c in new_chunks if c["chunk_id"] in existing]
    if clashes:
        raise ValueError(f"{len(clashes)} chunk_ids already exist, e.g. {clashes[0]}")

    append_jsonl(PAGES_PATH, pages)
    append_jsonl(CHUNKS_PATH, new_chunks)

    # Dense: embed only the new chunks and stack them onto the saved matrix.
    from index_dense import IDS_PATH, VECTORS_PATH, get_model

    model = get_model()
    vectors = model.encode(
        [c["text"] for c in new_chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    if os.path.exists(VECTORS_PATH):
        vectors = np.vstack([np.load(VECTORS_PATH), vectors])
    np.save(VECTORS_PATH, vectors)

    ids = []
    if os.path.exists(IDS_PATH):
        with open(IDS_PATH, encoding="utf-8") as f:
            ids = json.load(f)
    ids.extend(c["chunk_id"] for c in new_chunks)
    with open(IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(ids, f)

    if len(ids) != vectors.shape[0]:
        raise RuntimeError(f"index desync: {len(ids)} ids vs {vectors.shape[0]} vectors")

    # BM25 is rebuilt whole. IDF depends on the entire corpus, so a partial
    # update would leave every existing term weighted against a stale document
    # count -- wrong in a way that produces plausible rankings and hides itself.
    import index_sparse

    index_sparse.build()

    manifest = load_manifest()
    manifest[filename] = {
        "sha256": file_hash(path),
        "pages": len(pages),
        "chunks": len(new_chunks),
    }
    save_manifest(manifest)

    return {"pages": len(pages), "chunks": len(new_chunks), "total_chunks": len(ids)}


def seed():
    """Record the PDFs already present in pages.jsonl without reprocessing them.

    Needed once, because the corpus was built by full runs of ingest/chunk before
    this script existed. Without it the first scan would treat all 21 documents as
    new and try to add them a second time.
    """
    counts = {}
    with open(PAGES_PATH, encoding="utf-8") as f:
        for line in f:
            counts.setdefault(json.loads(line)["source_doc"], 0)
            counts[json.loads(line)["source_doc"]] += 1

    chunk_counts = {}
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)["source_doc"]
            chunk_counts[doc] = chunk_counts.get(doc, 0) + 1

    manifest = load_manifest()
    for name, pages in counts.items():
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            continue
        manifest[name] = {
            "sha256": file_hash(path),
            "pages": pages,
            "chunks": chunk_counts.get(name, 0),
        }
    save_manifest(manifest)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filename", nargs="?", help="PDF in data/; omit to scan for new ones")
    ap.add_argument("--strategy", choices=["fixed", "structure"], default="fixed")
    ap.add_argument("--seed", action="store_true",
                    help="record already-indexed PDFs without reprocessing")
    args = ap.parse_args()

    if args.seed:
        manifest = seed()
        print(f"manifest seeded with {len(manifest)} documents")
        return

    manifest = load_manifest()
    if args.filename:
        targets = [args.filename]
    else:
        targets = [
            os.path.basename(p)
            for p in sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
            if os.path.basename(p) not in manifest
            or manifest[os.path.basename(p)]["sha256"] != file_hash(p)
        ]

    if not targets:
        print("nothing new in data/")
        return

    for name in targets:
        stats = add(name, args.strategy)
        print(f"{name}: +{stats['pages']} pages, +{stats['chunks']} chunks "
              f"(corpus now {stats['total_chunks']})")


if __name__ == "__main__":
    main()
