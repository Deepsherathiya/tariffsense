# PRD — Hybrid Search over Indian Trade Documentation

**Status:** Not started. Empty folder.
**Owner:** Deep (final-year student, portfolio project #2 of 3)
**Audience for this document:** Claude Code, as the working spec for the build.

---

## 1. What this is

A retrieval system that answers export-related questions by finding the exact passages in Indian trade documentation that contain the answer.

Example queries:
- "what is the HS code for cotton bedsheets?"
- "is cashew export restricted?"
- "duty on tea exported to the EU"

The system returns ranked source passages with document name and page number. Nothing else.

## 2. Hard scope boundary — retrieval only

**There is no LLM in this system. There is no generated answer.**

The output is a ranked list of source passages, not prose. This is a deliberate design decision, not a simplification to be corrected later:

- For regulatory and legal text, the actual clause is more useful than a paraphrase of it.
- It removes hallucination from the system entirely.
- The engineering interest is in the retrieval mechanics, which is what the project is meant to demonstrate.

Do not add a generation step. Do not suggest one. If a phase seems to need one, it doesn't.

## 3. The thesis being tested

Two retrieval methods that fail in opposite ways:

| Method | Strong at | Fails at |
|---|---|---|
| Dense (embeddings) | Meaning. Matches "tea export duty" to "tax on sending tea abroad" | Exact identifiers. An HS code like `0902` is just digits — no semantic content |
| Sparse (BM25) | Exact strings — codes, clause numbers, statutory phrasing | Vocabulary mismatch. Fails when question and document share no words |

Trade documentation contains both dense legal prose *and* numeric tariff codes, so neither is sufficient alone. Combining them with reciprocal rank fusion **should** beat either individually.

**Proving or disproving this with measured numbers is the point of the project.** If hybrid loses to dense-only on this corpus, that is the finding and it gets reported as such.

## 4. Corpus

Indian trade documents as PDFs — HS tariff schedules and DGFT policy circulars. Target roughly a few thousand pages.

The hard part: regulatory text is full of near-identical clauses differing in a single qualifier. Chunking decisions therefore have real consequences, and this is precisely where naive retrieval breaks down. Chunk size and overlap are design decisions to be reasoned about and, where practical, tested — not defaults to accept.

## 5. Constraints — these are binding

**5.1 No RAG frameworks.** No LangChain, no LlamaIndex, no wrapper libraries that own the pipeline. They hide exactly the mechanics that need to be explainable in an interview. The goal is being able to say "I implemented RRF with k=60 and here's why that constant matters" rather than "the framework handled it."

Permitted libraries: PyMuPDF, sentence-transformers, rank_bm25, numpy, faiss-cpu, Flask, standard scientific Python.

**5.2 No invented numbers.** Do not write target metrics, expected results, aspirational benchmarks, or placeholder values anywhere — not in code comments, not in the README, not in docstrings, not as example output. If a number has not been measured, it does not appear in the repo. Empty result tables are correct until the evaluation fills them.

**5.3 Explain as we go.** One phase at a time. Explain each piece before building it, and stop for confirmation before moving to the next phase. Assume no prior knowledge of embeddings, chunking strategy, BM25 internals, or retrieval metrics — explain these in plain language when they come up, using the actual project data as the example.

The previous project was built too fast and ended up not understood, which is useless in an interview. Slower is correct here. If a design choice has alternatives, say what they were and why this one was picked.

**5.4 Flat structure.** A handful of Python files at the repo root. No nested package, no `src/`, no `__init__.py` tree. This is a few hundred lines of code and should not be organised like a framework.

**5.5 Total code budget.** The whole pipeline should be a few hundred lines that can be read line by line. If a file is growing past a couple hundred lines, that's a signal to simplify, not to split into more files.

## 6. Architecture

```
PDFs ──► extract text ──► chunks (+ metadata)
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
              dense index        BM25 index
            (embeddings)         (rank_bm25)
                    │                 │
                    └────────┬────────┘
                             ▼
                    reciprocal rank fusion
                             ▼
                     top-5 passages ──► Flask UI
```

### Proposed file layout

| File | Responsibility |
|---|---|
| `ingest.py` | PDF → raw text per page, with source + page retained |
| `chunk.py` | Text → overlapping chunks with metadata |
| `index_dense.py` | Embed chunks, build and persist vector index |
| `index_sparse.py` | Build and persist BM25 index over the same chunks |
| `search.py` | Query → three ranked lists (dense, sparse, fused). RRF lives here |
| `evaluate.py` | Run the labelled query set, compute recall@5 and MRR for all three systems |
| `app.py` | Flask interface |
| `data/` | Raw PDFs, `chunks.jsonl`, `queries.jsonl`, index artifacts |

All three retrieval modes must be callable independently — the evaluation depends on comparing them, so dense-only and BM25-only are first-class paths, not debug flags.

### Data contracts

Fix these early; everything downstream depends on them.

**Chunk record** (`chunks.jsonl`, one JSON object per line):
```
chunk_id      stable unique id
text          the chunk text
source_doc    filename of the source PDF
page          page number in that PDF
char_start    offset in the page/document text
```

**Labelled query record** (`queries.jsonl`):
```
query_id
question          natural-language question
relevant_chunks   list of chunk_ids that actually answer it
notes             optional — why this one is interesting/hard
```

Note the coupling: labels reference `chunk_id`, so re-chunking invalidates the label set. Either chunk before labelling and freeze it, or anchor labels to page + text span and re-resolve to chunk ids after each re-chunk. Decide this in Phase 1 — it is the single most annoying thing to get wrong.

## 7. Evaluation — the centrepiece

A hand-labelled test set of ~100 questions, each mapped to the passage(s) that actually answer it.

**Metrics:**
- **recall@5** — how often a correct passage appears in the top 5 results
- **MRR** — how highly the correct passage ranks on average

**Reported as a three-way comparison:** dense-only vs BM25-only vs hybrid, on the same queries, same chunks, same run.

The labelled query set is what turns this from a demo into something measurable, so it is built early rather than bolted on at the end.

The question mix should deliberately include both failure modes named in §3 — questions that hinge on an exact code or clause number, and questions phrased in everyday language that shares no vocabulary with the source text. A test set of only one kind cannot distinguish the three systems.

## 8. Phases

Work through these in order. Stop at each checkpoint.

**Phase 1 — Corpus and labelled query set** ← start here
Source the PDFs. Extract text. Decide and justify chunking strategy. Build `chunks.jsonl`. Then hand-build the query set against those chunks.
*Done when:* chunks exist with correct source/page metadata, and the query set is populated and frozen.

**Phase 2 — Dense retrieval**
Embed chunks with `all-MiniLM-L6-v2` (`bge-small-en-v1.5` as the alternative to try). Cosine similarity search over stored vectors.
*Done when:* a question returns a ranked list of chunks, and the mechanics of embeddings and cosine similarity have been explained.

**Phase 3 — Sparse retrieval**
BM25 over the identical chunk set via `rank_bm25`.
*Done when:* same interface as Phase 2, plus an explanation of what BM25 is actually scoring.

**Phase 4 — Fusion**
Reciprocal rank fusion merging the two ranked lists. Implemented directly, with the role of the `k` constant understood.
*Done when:* all three modes are callable and return comparable output.

**Phase 5 — Evaluation**
Run all three against the query set. Record whatever comes out.
*Done when:* the three-way comparison table is filled with measured numbers, and disagreements between systems have been inspected on individual queries.

**Phase 6 — Serving**
Flask interface: type a question, see top-5 passages with source document and page. Deploy to Render free tier.
*Done when:* it runs and the deployed URL works.

## 9. Stack

Python · PyMuPDF · sentence-transformers · rank_bm25 · numpy or faiss-cpu · Flask · Render (free tier)

## 10. Background on the owner

Full-stack Python — Flask, MySQL, SQLAlchemy. Recent work with pandas, scikit-learn, XGBoost. Previous project: a shipment delay classifier (XGBoost vs logistic regression, temporal split, PR-AUC, Flask UI).

**New to embeddings and retrieval.** That is the gap this project exists to close, so the explanations in §5.3 are the deliverable as much as the code is.

## 11. First task

Sourcing and preparing the corpus, then building the labelled query set — everything else is measured against it.
