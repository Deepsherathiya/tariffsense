"""Query -> three ranked lists: dense, sparse, fused.

    python search.py "is cashew export restricted?"

Reciprocal rank fusion lives here. All three modes are first-class paths because
the evaluation compares them against each other on the same queries.
"""

import argparse
import json
import os

CHUNKS_PATH = os.path.join("data", "chunks.jsonl")

# RRF constant. Controls how much a top-1 result from a single system dominates.
# At k=1 rank 1 scores 0.5 and rank 2 scores 0.33 -- a 50% gap, so whichever
# system is most confident wins outright. At k=60 the gap is about 2%, so a chunk
# both systems rank moderately well beats one that a single system ranks first.
# 60 is the value from Cormack et al. (2009); it is tuned, not derived.
RRF_K = 60

# How deep to read each ranked list before fusing. A chunk sitting at rank 40 in
# one system and rank 3 in the other can finish top-5 after fusion, so fusing
# only the top 5 of each would hide exactly the cases fusion exists to fix.
CANDIDATE_DEPTH = 50


def load_chunks():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return {c["chunk_id"]: c for c in (json.loads(line) for line in f)}


def reciprocal_rank_fusion(ranked_lists, k=RRF_K):
    """Merge ranked lists of chunk_ids into one ranking.

    Only positions are used. The underlying scores are deliberately discarded:
    cosine similarity lives in [-1, 1] while BM25 is unbounded and query
    dependent, so the two are not on a common scale and adding them would let
    whichever system happens to produce bigger numbers dominate.

    Ties are real and common: a chunk found by only one retriever scores exactly
    1/(k + rank), so two chunks each seen by a different retriever at the same
    rank come out equal. Sorting on score alone leaves Python's stable sort to
    resolve those by insertion order, which would hand every tie to whichever
    list was passed first -- a silent advantage for one of the two systems the
    evaluation is meant to compare. Breaking on chunk_id is arbitrary, but it is
    arbitrary in a way that correlates with neither retriever.
    """
    points = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, 1):
            points[chunk_id] = points.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(points.items(), key=lambda kv: (-kv[1], kv[0]))


class Search:
    def __init__(self, with_reranker=True):
        from index_dense import DenseIndex
        from index_sparse import SparseIndex

        self.dense = DenseIndex()
        self.sparse = SparseIndex()
        self.chunks = load_chunks()
        self._reranker = None
        self._want_reranker = with_reranker

    @property
    def reranker(self):
        """Loaded on first use so dense/BM25-only runs never pay for it."""
        if self._reranker is None:
            import rerank

            if not rerank.ENABLED:
                return None
            self._reranker = rerank.Reranker()
        return self._reranker

    def dense_search(self, query, k=5):
        return self.dense.search(query, k)

    def sparse_search(self, query, k=5):
        return self.sparse.search(query, k)

    def hybrid_search(self, query, k=5, depth=CANDIDATE_DEPTH):
        d = [cid for cid, _ in self.dense.search(query, depth)]
        s = [cid for cid, _ in self.sparse.search(query, depth)]
        return reciprocal_rank_fusion([d, s])[:k]

    def rerank_search(self, query, k=5):
        """Hybrid candidates, reordered by the cross-encoder.

        Fusion is used to build the candidate pool rather than dense alone
        because recall into the pool is what matters here: the reranker can only
        promote a passage that retrieval already surfaced somewhere in the top
        RERANK_DEPTH.
        """
        import rerank

        model = self.reranker
        if model is None:
            return self.hybrid_search(query, k)
        pool = self.hybrid_search(query, rerank.RERANK_DEPTH)
        candidates = [(cid, self.chunks[cid]["text"]) for cid, _ in pool]
        return model.rerank(query, candidates, k)


def wrap(text, width, indent="     "):
    """Wrap to `width` columns so long passages stay readable in a terminal."""
    words, lines, line = text.split(), [], ""
    for w in words:
        if line and len(line) + 1 + len(w) > width:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        lines.append(line)
    return "\n".join(indent + l for l in lines)


def show(title, results, chunks, chars, width):
    print(f"\n{title}")
    for rank, (cid, score) in enumerate(results, 1):
        c = chunks[cid]
        text = " ".join(c["text"].split())
        if chars:
            text = text[:chars]
        print(f"  {rank}. {score:.4f}  {c['source_doc']}  p{c['page']}")
        print(wrap(text, width))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument(
        "--mode",
        choices=["dense", "sparse", "hybrid", "rerank", "all"],
        default="all",
    )
    ap.add_argument("--chars", type=int, default=400,
                    help="characters of each passage to show; 0 = the whole chunk")
    ap.add_argument("--width", type=int, default=100, help="wrap column")
    args = ap.parse_args()

    chunks = load_chunks()
    s = Search()
    opts = (chunks, args.chars, args.width)
    if args.mode in ("dense", "all"):
        show("=== DENSE ===", s.dense_search(args.query, args.k), *opts)
    if args.mode in ("sparse", "all"):
        show("=== BM25 ===", s.sparse_search(args.query, args.k), *opts)
    if args.mode in ("hybrid", "all"):
        show("=== HYBRID (RRF) ===", s.hybrid_search(args.query, args.k), *opts)
    if args.mode in ("rerank", "all"):
        show("=== RERANKED ===", s.rerank_search(args.query, args.k), *opts)


if __name__ == "__main__":
    main()
