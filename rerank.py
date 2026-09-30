"""Cross-encoder reranking over retrieved candidates.

The bi-encoder in index_dense.py embeds the query and the passage separately and
compares the two vectors. Nothing in that comparison ever sees the query and the
passage at the same time, so a single qualifier -- "non-", "other than",
"excluding" -- barely moves the passage vector. That is why asking for the HS code
of basmati rice returns the *Non*-Basmati entry first.

A cross-encoder takes (query, passage) as one input and scores the pair directly,
so it can attend to that qualifier. The cost is that it cannot be precomputed:
every candidate has to be run through the model at query time. Hence the two-stage
design -- cheap retrieval narrows 1287 chunks to RERANK_DEPTH, the expensive model
only orders those.
"""

import os

# ~22M parameters, roughly 90 MB. BAAI/bge-reranker-base scores better but is
# 278M parameters and cannot fit beside torch and bge-small inside the 512 MB
# Render free tier. Accuracy traded for a deployable footprint.
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many candidates to rerank. Latency is linear in this number.
RERANK_DEPTH = 50

# Deployment escape hatch: the reranker is the first thing to drop if the
# container runs out of memory.
ENABLED = os.environ.get("ENABLE_RERANKER", "true").lower() != "false"


class Reranker:
    def __init__(self):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(MODEL_NAME)

    def rerank(self, query, candidates, k=5):
        """Reorder (chunk_id, text) candidates by cross-encoder score.

        Scores are raw logits, not probabilities, and are not comparable across
        queries -- only the ordering within one query means anything.
        """
        if not candidates:
            return []
        pairs = [(query, text) for _, text in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip((cid for cid, _ in candidates), scores),
            key=lambda cs: (-cs[1], cs[0]),
        )
        return [(cid, float(score)) for cid, score in ranked[:k]]
