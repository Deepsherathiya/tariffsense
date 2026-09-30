"""Dense retrieval: embed chunks, persist vectors, search by cosine similarity.

Build the index (one-off, slow):
    python index_dense.py --build

Query it:
    python index_dense.py "how much tax on importing seafood?"

Vectors are stored normalised to unit length, so cosine similarity reduces to a
plain dot product and the whole search is one matrix multiply.
"""

import argparse
import json
import os

import numpy as np

CHUNKS_PATH = os.path.join("data", "chunks.jsonl")
VECTORS_PATH = os.path.join("data", "dense_vectors.npy")
IDS_PATH = os.path.join("data", "dense_ids.json")

# PRD names all-MiniLM-L6-v2 as the baseline and bge-small-en-v1.5 as the
# alternative to try. Both output 384 dimensions, so nothing downstream changes.
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE models are trained asymmetrically: passages are embedded bare, queries get
# an instruction prefix. Omitting it costs accuracy silently -- the vectors still
# come out the right shape and the search still runs. MiniLM wants no prefix, so
# this is blank when MODEL_NAME points back at it.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

BATCH_SIZE = 64


def load_chunks():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def get_model():
    """Imported lazily so that loading the index costs nothing until needed."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def build():
    """Embed every chunk and write the vectors to disk.

    normalize_embeddings=True divides each vector by its own length. After that
    every vector sits on the unit sphere, so the dot product of any two is
    exactly their cosine similarity — no per-query division needed.
    """
    chunks = load_chunks()
    model = get_model()
    texts = [c["text"] for c in chunks]

    print(f"embedding {len(texts)} chunks with {MODEL_NAME} ...")
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    np.save(VECTORS_PATH, vectors)
    with open(IDS_PATH, "w", encoding="utf-8") as f:
        json.dump([c["chunk_id"] for c in chunks], f)

    print(f"vectors: {vectors.shape}  ->  {VECTORS_PATH}")
    print(f"ids:     {len(chunks)}     ->  {IDS_PATH}")


class DenseIndex:
    """Loaded vectors plus the model needed to embed incoming queries."""

    def __init__(self):
        self.vectors = np.load(VECTORS_PATH)
        with open(IDS_PATH, encoding="utf-8") as f:
            self.ids = json.load(f)
        self.model = get_model()

    def search(self, query, k=5):
        """Return the k most similar chunks as (chunk_id, score) pairs.

        The query is embedded and normalised the same way the chunks were, so
        `vectors @ q` is a cosine similarity against every chunk at once.
        """
        q = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        scores = self.vectors @ q
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", help="question to search for")
    ap.add_argument("--build", action="store_true", help="build the index")
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
    for rank, (cid, score) in enumerate(DenseIndex().search(args.query, args.k), 1):
        c = chunks[cid]
        text = " ".join(c["text"].split())
        if args.chars:
            text = text[: args.chars]
        print(f"\n{rank}. {score:.3f}  {c['source_doc']}  p{c['page']}")
        print(wrap(text, args.width, indent="   "))


if __name__ == "__main__":
    main()
