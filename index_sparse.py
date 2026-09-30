"""Sparse retrieval: BM25 over the same chunks the dense index uses.

Build:
    python index_sparse.py --build

Query:
    python index_sparse.py "what is the duty on tea?"

Same interface as index_dense.py on purpose. The evaluation compares dense-only,
BM25-only and fused, so all three have to be first-class callable paths.
"""

import argparse
import json
import os
import pickle
import re

CHUNKS_PATH = os.path.join("data", "chunks.jsonl")
INDEX_PATH = os.path.join("data", "bm25_index.pkl")

# BM25Okapi defaults. k1 controls how fast repeated terms stop adding score;
# b controls how hard long chunks are penalised.
K1 = 1.5
B = 0.75


STEMMING = True


def stem(token):
    """Strip English plural suffixes.

    BM25 matches strings, not meanings: "cashews" appears in 0 chunks while
    "cashew" appears in 4, so a query written in the plural scores nothing
    against the passage that answers it. The same split runs through the corpus
    (vehicle 7 / vehicles 29, duty 218 / duties 22).

    Deliberately crude -- a dozen rules, no dependency. NLTK is not on the PRD's
    permitted list, and the point here is to measure whether closing the
    singular/plural gap is worth anything at all, not to build a linguist's
    stemmer.
    """
    if not token.isalpha() or len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text):
    """Lowercase alphanumeric runs, optionally stemmed.

    Splitting on non-alphanumerics turns "0902 10 10" into ["0902", "10", "10"],
    so an HS code survives as a searchable token. This is the whole reason BM25
    can answer code lookups that the dense index cannot: strip the digits here
    and the sparse side loses its only advantage. stem() guards on isalpha() for
    the same reason -- numeric tokens are never touched.
    """
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [stem(t) for t in tokens] if STEMMING else tokens


def load_chunks():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build():
    from rank_bm25 import BM25Okapi

    chunks = load_chunks()
    corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus, k1=K1, b=B)
    ids = [c["chunk_id"] for c in chunks]

    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids}, f)

    lengths = [len(d) for d in corpus]
    print(f"indexed {len(corpus)} chunks, {sum(lengths)} tokens")
    print(f"avg chunk length: {sum(lengths) / len(lengths):.0f} tokens")
    print(f"vocabulary: {len({t for d in corpus for t in d})} unique terms")
    print(f"wrote {INDEX_PATH}")


class SparseIndex:
    def __init__(self):
        with open(INDEX_PATH, "rb") as f:
            state = pickle.load(f)
        self.bm25 = state["bm25"]
        self.ids = state["ids"]

    def search(self, query, k=5):
        """Return the k highest-scoring chunks as (chunk_id, score) pairs.

        Unlike cosine similarity, BM25 scores have no fixed range — they depend
        on the query's IDF weights. A score of 12 for one query and 3 for another
        says nothing about which matched better. Only the ranking is comparable,
        which is exactly why fusion in Phase 4 uses ranks and not scores.
        """
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.ids[i], float(scores[i])) for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--chars", type=int, default=400,
                    help="characters of each passage to show; 0 = the whole chunk")
    ap.add_argument("--width", type=int, default=100)
    args = ap.parse_args()

    if args.build:
        build()
        return
    if not args.query:
        ap.error("give a query, or --build")

    from search import wrap

    chunks = {c["chunk_id"]: c for c in load_chunks()}
    for rank, (cid, score) in enumerate(SparseIndex().search(args.query, args.k), 1):
        c = chunks[cid]
        text = " ".join(c["text"].split())
        if args.chars:
            text = text[: args.chars]
        print(f"\n{rank}. {score:.3f}  {c['source_doc']}  p{c['page']}")
        print(wrap(text, args.width, indent="   "))


if __name__ == "__main__":
    main()
