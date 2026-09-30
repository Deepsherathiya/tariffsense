"""Run the labelled query set against all three systems. Report recall@5 and MRR.

    python evaluate.py
    python evaluate.py --k 10
    python evaluate.py --disagreements

Labels are anchored to (source_doc, page, snippet), not to chunk_id, so they are
resolved to whatever chunks currently contain the snippet. Re-chunking therefore
costs nothing: change CHUNK_SIZE, rebuild, and the same labels still apply.
"""

import argparse
import json
import os
import re

QUERIES_PATH = os.path.join("data", "queries.jsonl")
CHUNKS_PATH = os.path.join("data", "chunks.jsonl")

DEFAULT_K = 5


def normalise(text):
    """Whitespace-insensitive comparison.

    Chunk text keeps the newlines that separated table cells, so a snippet
    spanning a cell boundary would never match literally.
    """
    return re.sub(r"\s+", " ", text).strip().lower()


def load_chunks():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_queries():
    with open(QUERIES_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


MIN_SNIPPET_COVERAGE = 0.5


def load_pages():
    with open(os.path.join("data", "pages.jsonl"), encoding="utf-8") as f:
        return {(p["source_doc"], p["page"]): p["text"] for p in (json.loads(l) for l in f)}


def snippet_span(page_text, snippet):
    """Locate a snippet in the page and return its (start, end) offsets.

    Matching is whitespace-flexible because chunk and page text keep the
    newlines that separated table cells.
    """
    pattern = r"\s+".join(re.escape(w) for w in snippet.split())
    m = re.search(pattern, page_text, re.I)
    return (m.start(), m.end()) if m else None


def resolve_labels(queries, chunks, pages=None):
    """Map each query to the set of chunk_ids that count as correct.

    Resolution is by character-range overlap, not substring containment. Under
    fixed-size chunking a snippet always sat inside one chunk, so containment
    was enough; structure-aware chunks carry no overlap, so a snippet crossing a
    boundary lives in no single chunk and containment silently drops the label.

    A chunk counts when it covers at least half the snippet, which keeps a chunk
    holding two trailing words from being scored as the answer. A snippet can
    still resolve to more than one chunk when chunks overlap, and both are
    correct: the passage really is in both.
    """
    pages = pages if pages is not None else load_pages()
    by_page = {}
    for c in chunks:
        by_page.setdefault((c["source_doc"], c["page"]), []).append(c)

    resolved = {}
    for q in queries:
        correct = set()
        for a in q["answers"]:
            key = (a["source_doc"], a["page"])
            page_text = pages.get(key, "")
            span = snippet_span(page_text, a["snippet"])
            snippet = normalise(a["snippet"])
            for c in by_page.get(key, []):
                if snippet in normalise(c["text"]):
                    correct.add(c["chunk_id"])
                elif span:
                    start = c["char_start"]
                    end = start + len(c["text"])
                    overlap = min(end, span[1]) - max(start, span[0])
                    if overlap >= MIN_SNIPPET_COVERAGE * (span[1] - span[0]):
                        correct.add(c["chunk_id"])
        resolved[q["query_id"]] = correct
    return resolved


def rank_of_first_hit(ranked_ids, correct):
    for rank, cid in enumerate(ranked_ids, 1):
        if cid in correct:
            return rank
    return None


def evaluate(search, queries, resolved, k=DEFAULT_K):
    """Return per-system metrics plus the per-query ranks behind them."""
    modes = {
        "dense": search.dense_search,
        "bm25": search.sparse_search,
        "hybrid": search.hybrid_search,
        "rerank": search.rerank_search,
    }
    results = {name: {"hits": 0, "rr": 0.0, "ranks": {}} for name in modes}

    for q in queries:
        correct = resolved[q["query_id"]]
        if not correct:
            continue
        for name, fn in modes.items():
            ranked = [cid for cid, _ in fn(q["question"], k)]
            rank = rank_of_first_hit(ranked, correct)
            results[name]["ranks"][q["query_id"]] = rank
            if rank is not None:
                results[name]["hits"] += 1
                results[name]["rr"] += 1.0 / rank

    n = sum(1 for q in queries if resolved[q["query_id"]])
    for r in results.values():
        r["recall"] = r["hits"] / n
        r["mrr"] = r["rr"] / n
    return results, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--disagreements", action="store_true",
                    help="list queries where the systems differ")
    args = ap.parse_args()

    from search import Search

    chunks = load_chunks()
    queries = load_queries()
    resolved = resolve_labels(queries, chunks)

    unresolved = [qid for qid, c in resolved.items() if not c]
    if unresolved:
        print(f"WARNING: {len(unresolved)} labels resolve to no chunk: {unresolved}")

    results, n = evaluate(Search(), queries, resolved, args.k)

    print(f"\n{n} queries, top-{args.k}\n")
    print(f"{'system':10} {f'recall@{args.k}':>10} {'MRR':>8}")
    print("-" * 30)
    for name in ("dense", "bm25", "hybrid", "rerank"):
        r = results[name]
        print(f"{name:10} {r['recall']:>10.3f} {r['mrr']:>8.3f}")

    if args.disagreements:
        byq = {q["query_id"]: q["question"] for q in queries}
        rows = []
        for qid in results["dense"]["ranks"]:
            d = results["dense"]["ranks"][qid]
            b = results["bm25"]["ranks"][qid]
            h = results["hybrid"]["ranks"][qid]
            if len({d, b, h}) > 1:
                rows.append((qid, d, b, h, byq[qid]))
        print(f"\n{len(rows)} queries where the systems disagree:\n")
        print(f"{'query':6} {'dense':>6} {'bm25':>6} {'hybrid':>7}  question")
        for qid, d, b, h, question in rows:
            f = lambda v: "-" if v is None else str(v)
            print(f"{qid:6} {f(d):>6} {f(b):>6} {f(h):>7}  {question[:58]}")


if __name__ == "__main__":
    main()
