# TariffSense

Hybrid retrieval over Indian trade documentation. Ask an export question, get the
exact clause back — with its source document and page number.

**Retrieval-first, generation optional.** The core system returns ranked source
passages and nothing else. An LLM answer layer sits on top, is off by default, and
when it runs it may only speak from the retrieved passages and must cite them.

This README documents every design decision, every measurement, and the findings
that contradicted the original hypothesis.

---

## Contents

- [What it does](#what-it-does)
- [The thesis being tested](#the-thesis-being-tested)
- [Corpus](#corpus)
- [Pipeline](#pipeline)
- [1. Ingestion](#1-ingestion)
- [2. Chunking](#2-chunking)
- [3. Dense retrieval](#3-dense-retrieval)
- [4. Sparse retrieval (BM25)](#4-sparse-retrieval-bm25)
- [5. Fusion (RRF)](#5-fusion-rrf)
- [6. Cross-encoder reranking](#6-cross-encoder-reranking)
- [Evaluation](#evaluation)
- [Results](#results)
- [Findings](#findings)
- [LLM answer layer](#llm-answer-layer)
- [Web application](#web-application)
- [Incremental ingestion](#incremental-ingestion)
- [Running it](#running-it)
- [Repository layout](#repository-layout)
- [Known limitations](#known-limitations)
- [What I would do next](#what-i-would-do-next)

---

## What it does

Example queries the system is built for:

| Query | Type |
|---|---|
| `is cashew export restricted?` | policy prose |
| `0902` | exact HS code |
| `duty on basmati rice` | near-identical clauses (Basmati vs Non-Basmati) |
| `what counts as handmade goods?` | no vocabulary shared with the source |

Four retrieval modes are first-class and independently callable — the evaluation
depends on comparing them, so none of them is a debug flag:

```bash
python search.py "is cashew export restricted?" --mode dense
python search.py "is cashew export restricted?" --mode sparse
python search.py "is cashew export restricted?" --mode hybrid
python search.py "is cashew export restricted?" --mode rerank
```

---

## The thesis being tested

Two retrieval methods that fail in opposite directions:

| Method | Strong at | Fails at |
|---|---|---|
| **Dense** (embeddings) | Meaning. Matches "tea export duty" to "tax on sending tea abroad" | Exact identifiers. An HS code like `0902` is four digits with no semantic content to embed |
| **Sparse** (BM25) | Exact strings — codes, clause numbers, statutory phrasing | Vocabulary mismatch. Scores nothing when question and document share no words |

Trade documentation contains both dense legal prose *and* numeric tariff codes, so
the hypothesis was that fusing the two would beat either alone.

**It did not.** That result, and the reason for it, is the most interesting part of
this project. See [Findings](#findings).

---

## Corpus

Indian trade documents as PDFs, sourced from official government portals.

| | |
|---|---|
| Documents | **21 PDFs** (10.8 MB) |
| Pages extracted | **518** (14 cover/disclaimer pages dropped) |
| Page text | **921,491 characters** |
| Chunks | **1,287** |
| Chunk text | 1,036,693 characters — ~12.5% larger than the page text, which is the overlap doing its job |

**Sources:**

| Source | Content |
|---|---|
| CBIC First Schedule | Tariff chapters 3, 8, 9, 10, 29, 30, 52, 60–63, 71, 87 — HS codes and duty rates |
| CBIC Second Schedule | Export tariff, general notes, appendices |
| DGFT ITC(HS) Schedule 2 | Export policy: Free / Restricted / Prohibited, with conditions |
| DGFT FTP 2023 | Foreign Trade Policy, chapters 1–2 |
| DGFT Handbook of Procedures | Chapter 2 |

The CBIC chapter PDFs sit four levels deep behind a JavaScript portal and are
served only to a real browser session — `curl` returns a 404 or the Angular shell.
They were downloaded by hand. The DGFT files are directly fetchable.

This is a deliberate subset, not all of Indian trade law.

---

## Pipeline

```
PDFs ──► ingest.py ──► pages.jsonl ──► chunk.py ──► chunks.jsonl
                                                         │
                                           ┌─────────────┴─────────────┐
                                           ▼                           ▼
                                   index_dense.py              index_sparse.py
                                   bge-small-en-v1.5             BM25Okapi
                                   1287 × 384 float32           169,202 tokens
                                           │                           │
                                           └─────────────┬─────────────┘
                                                         ▼
                                               search.py — RRF (k=60)
                                                         ▼
                                            rerank.py — cross-encoder
                                                         ▼
                                       top-5 passages ──► app.py / answer.py
```

No LangChain, no LlamaIndex, no wrapper library. Every mechanic is implemented
directly so that it can be explained. Libraries used: PyMuPDF,
sentence-transformers, rank_bm25, numpy, Flask.

---

## 1. Ingestion

`ingest.py` — PDF → one record per page in `data/pages.jsonl`, retaining
`source_doc` and `page`.

**Block-mode extraction.** PyMuPDF's `get_text("blocks")` is used rather than plain
`get_text()`. Plain extraction interleaves table columns, mixing a row's
description with the next row's duty rate. Block mode keeps cells apart; blocks are
sorted by `(round(y0), x0)` for reading order.

**Three real defects in the raw text, each measured before being fixed:**

| Defect | Scale | Fix |
|---|---|---|
| Running headers (`ITC (HS), 2018`, `SECTION-II / CHAPTER-9`) | 125 of 126 pages in one document | Drop blocks in the top/bottom 12% of page height whose digit-normalised text repeats on >50% of pages |
| Mid-word splits from narrow table columns (`Prohibite` + `d`) | 696 candidates in Schedule 2, 165 in HBP, 85 in tariff chapters | Join lines with a space, except a lone lowercase letter that is not `a` or `i` |
| Symbol-font garbage (U+00ED from tariff dash markers, U+F0B7 Wingdings bullet) | 29 on a single tariff page; 20 PUA glyphs corpus-wide | Strip |

The header fix took two attempts. The first version compared whole block text,
which failed on tariff pages because the header embeds the page number
(`128 SECTION-II CHAPTER-9`) and therefore differed on every page. Digit-normalising
the comparison key fixed it — but normalising *every* block would have folded
genuine tariff rows together (`#### ## ## --- Other kg. ###%`) and deleted real
content, so the rule is restricted to blocks sitting in the page margin.

Verified afterwards: 0 remaining header occurrences, 0 word splits (`Prohibited`
now appears 57 times as a single token), 0 junk glyphs, and 637 genuine
`--- Other` tariff rows still present.

One remnant is left in deliberately: the column-number row `(1) (2) (3) (4) (5)`
survives 129 times. It sits below the header rather than in the margin, and
loosening the rule to catch it would risk deleting tariff rows — the more expensive
failure.

---

## 2. Chunking

`chunk.py` — writes `data/chunks.jsonl`, one record per chunk:

```json
{"chunk_id": "Cotton|p7|c1680", "text": "...", "source_doc": "Cotton.pdf",
 "page": 7, "char_start": 1680}
```

**Active strategy: fixed-size windows.** 1,000 characters, 150 overlap, snapping to
whitespace within 200 characters of the boundary so that an 8-digit HS code is
never cut in half.

Chunk sizes: min 25, median 933, max 999.

**Chunks never cross a page boundary.** The data contract requires a page number
and the UI displays it, so a chunk spanning two pages would have no honest one. The
cost is that a clause broken by a page break is split.

**A structure-aware strategy was also built and measured** (`--strategy structure`):
split on HS-code boundaries and numbered paragraph markers, merging units up to 250
characters. It cut the median distinct HS codes per tariff chunk from **10 to 3** —
and made retrieval *worse*. It is kept in the code because a measured regression is
still a finding. See [Findings](#findings).

---

## 3. Dense retrieval

`index_dense.py`

| | |
|---|---|
| Model | **`BAAI/bge-small-en-v1.5`** — 384 dimensions, ~33M parameters |
| Baseline model | `all-MiniLM-L6-v2` — 384 dimensions, swapped after measurement |
| Index | `1287 × 384` float32, **1.89 MB** |
| Similarity | Cosine |
| Build time | ~80 seconds on CPU |

**Vectors are normalised at build time** (`normalize_embeddings=True`). Every vector
then sits on the unit sphere, so the cosine denominator `|A| × |B|` equals 1 and
similarity reduces to a plain dot product. The whole search is one matrix multiply:
`vectors @ q`.

**BGE needs an instruction prefix on the query side only** —
`"Represent this sentence for searching relevant passages: "` — while passages are
embedded bare. Omitting it costs accuracy silently: the vectors still come out the
right shape and the search still runs.

---

## 4. Sparse retrieval (BM25)

`index_sparse.py` — `rank_bm25.BM25Okapi`, `k1 = 1.5`, `b = 0.75`.

| | |
|---|---|
| Tokens indexed | **169,202** (avg 131 per chunk) |
| Vocabulary | 10,431 raw → **9,639 after stemming** |

**Tokenisation is the decision that matters.** `re.findall(r"[a-z0-9]+", text.lower())`
turns `0902 10 10` into `["0902", "10", "10"]`, so an HS code survives as a
searchable token. Strip the digits and BM25 loses its only advantage over dense
retrieval.

**What BM25 is actually scoring**, with real IDF values from this corpus:

| Term | Chunks containing it | IDF |
|---|---|---|
| `the` | 836 / 1287 | 0.43 |
| `export` | 391 | 1.19 |
| `duty` | 218 | 1.77 |
| `tea` | 15 | 4.42 |
| `0902` | 5 | **5.46** |
| `cashew` | 4 | **5.66** |

`cashew` carries 13× the weight of `the`. And `0902` — worthless to the dense index —
is among the most valuable signals in the corpus here.

The other two components: `k1 = 1.5` saturates term frequency so a term repeated 9
times scores only slightly above one appearing once; `b = 0.75` penalises chunks
longer than the 131-token average.

A worked example from the corpus — `is cashew export restricted?`:

| Rank | Length | `cashew` | `restricted` | `export` | `is` | Score |
|---|---|---|---|---|---|---|
| 1 | **65** | 1 | 1 | 1 | 1 | **12.42** |
| 2 | 159 | 1 | **9** | 0 | 0 | 10.73 |
| 5 | 131 | **0** | 3 | 3 | 3 | 6.58 |

Rank 1 wins on length normalisation (65 tokens, half the average, with all four
query terms). Rank 2 loses despite nine occurrences of `restricted`, because of TF
saturation. Rank 5 loses despite matching three terms three times each, because it
lacks the one rare term — `cashew`'s IDF of 5.66 outweighs the other three combined.

**Stemming was added after measurement.** `cashews` appears in **0** chunks while
`cashew` appears in 4, so a plural query scored nothing against the passage that
answered it. The same split runs throughout the corpus: `vehicle` 7 / `vehicles` 29,
`duty` 218 / `duties` 22. The fix is a dozen suffix rules (`ies→y`,
`ses/xes/zes/ches/shes`, trailing `s`), guarded by `isalpha()` so numeric tokens are
never touched. No NLTK dependency.

Effect: `can I sell cashews abroad?` moved from BM25 rank **988 → 3**, while `0902`
stayed at rank 1.

---

## 5. Fusion (RRF)

`search.py` — reciprocal rank fusion.

```
score(d) = Σ  1 / (k + rank_i(d))        k = 60, candidate depth = 50
```

**Only positions are used; the underlying scores are discarded on purpose.** Cosine
similarity lives in [-1, 1] while BM25 is unbounded and query-dependent (scores of
12.4 are normal). Adding them directly would let whichever system happens to produce
bigger numbers dominate.

**What `k` controls:** how much a single system's top result dominates.

| k | rank 1 | rank 2 | rank 10 |
|---|---|---|---|
| 1 | 0.5000 | 0.3333 | 0.0909 |
| **60** | 0.0164 | 0.0161 | 0.0143 |

At `k=1`, rank 1 scores 50% more than rank 2, so whichever retriever is most
confident wins outright. At `k=60` the gap is about 2%, so a chunk that both systems
rank moderately beats one that a single system ranks first. Consensus wins over
confidence. 60 is the value from Cormack et al. (2009) — tuned, not derived.

**Candidate depth is 50, not 5.** A chunk at rank 40 in one system and rank 3 in the
other can finish top-5 after fusion; fusing only each top-5 would hide exactly the
cases fusion exists to fix.

**A tie-breaking bias was found and fixed.** A chunk appearing in only one list
scores exactly `1/(k + rank)`, so two chunks seen by different retrievers at the same
rank tie exactly. Sorting on score alone left Python's stable sort to resolve those
by insertion order — and the dense list was passed first, so **every tie resolved in
favour of dense**. Measured: **19 of 100** queries had ties inside the hybrid top-5,
and **24 of 100** returned different results depending on which list was passed
first. Adding `chunk_id` as a secondary sort key dropped order-dependence to
**0 of 100**.

---

## 6. Cross-encoder reranking

`rerank.py`

| | |
|---|---|
| Model | **`cross-encoder/ms-marco-MiniLM-L-6-v2`** — ~22M parameters, ~90 MB |
| Depth | Top 50 hybrid candidates rescored |
| Latency | ~1–2 s per query on CPU |

A bi-encoder embeds query and passage *separately*; nothing in that comparison ever
sees both at once, so a single qualifier like `non-` barely moves the passage vector.
A cross-encoder takes `(query, passage)` as one input and scores the pair directly,
so it can attend to `non-`, `other than`, `excluding` — the qualifiers that separate
near-identical regulatory clauses.

The cost is that it cannot be precomputed, hence the two-stage design: cheap
retrieval narrows 1,287 chunks to 50, and the expensive model only orders those.

`BAAI/bge-reranker-base` scores better but is 278M parameters (~1.1 GB) and cannot
fit alongside torch and bge-small inside a 512 MB free tier. Accuracy traded for a
deployable footprint.

---

## Evaluation

**100 hand-labelled questions.** This is what turns the project from a demo into
something measurable, and it was built before any retrieval code.

### Label design: span-anchored, not chunk-anchored

```json
{"query_id": "q001",
 "question": "what is the HS code for ephedrine?",
 "answers": [{"source_doc": "Pharmaceutical products.pdf", "page": 7,
              "snippet": "Containing ephedrine or its salts"}],
 "notes": "exact-code lookup"}
```

`chunk_id` appears nowhere. Labels point at `(source_doc, page, snippet)` and are
resolved to whichever chunks currently contain that snippet, at evaluation time.

**This is the single most important structural decision in the project.** If labels
referenced chunk ids, changing `CHUNK_SIZE` would invalidate all 100 hand-made
labels. Because they don't, re-chunking costs nothing — which is what made the
structure-aware chunking experiment possible at all.

Resolution is by **character-range overlap**, not substring containment. Under
fixed-size chunking a snippet always sat inside one chunk, so containment was
sufficient. Structure-aware chunks carry no overlap, so a snippet crossing a boundary
lives inside no single chunk — containment silently dropped 3 labels before this was
caught. A chunk now counts when it covers at least 50% of the snippet.

### Query set design

The mix matters as much as the count. Vocabulary overlap between each question and
its answer page was measured, and the set was deliberately built across three cells
so that question length and vocabulary overlap could be told apart:

| Group | n | Mean overlap | Median length | Who should win |
|---|---|---|---|---|
| Low overlap (<30%) | **36** | 17% | 12 words | Dense |
| Mid (30–70%) | **52** | 46% | 8 words | — |
| High overlap (>70%) | **12** | 82% | 7 words | BM25 |

Mean overlap across the whole set: **40%**. Median question length: **8 words**.

Two rounds of labels were rejected and rewritten during construction:

1. The first 14 questions averaged **64%** word overlap with their answer page —
   BM25 would have won them by construction. Rewritten and supplemented.
2. Four questions had no subject anchor at all (`"what does this framework mean?"`),
   because the banned-word list used during labelling had removed the only
   identifying nouns. No retriever can answer those, so they measure nothing.
   Rewritten.

### Coverage

| | |
|---|---|
| Queries | **100** (100 unique ids) |
| Labels resolving to a chunk | **100 / 100** |
| Documents covered | **21 / 21** |
| Distinct answer pages | **77** |

### Metrics

- **recall@5** — the fraction of queries where a correct passage appears in the top
  5. Position-blind: rank 1 and rank 5 score the same, rank 6 scores zero.
- **MRR** — the mean of `1 / rank_of_first_correct_passage`. Rewards ranking the
  answer higher.

Both are reported because two systems can share a recall@5 while one consistently
ranks the answer first and the other fifth.

```bash
python evaluate.py                  # the table
python evaluate.py --disagreements  # per-query ranks where systems differ
```

---

## Results

### Final system

100 queries, top-5. Chunking: fixed. Dense: bge-small-en-v1.5. BM25: stemmed.

| System | recall@5 | MRR |
|---|---|---|
| Dense | 0.610 | 0.422 |
| BM25 | 0.420 | 0.283 |
| Hybrid (RRF) | 0.550 | 0.402 |
| **Reranked** | **0.680** | **0.490** |

### How it got there

Every row is a measured run, in the order the changes were made.

| Configuration | Dense | BM25 | Hybrid | Rerank |
|---|---|---|---|---|
| Baseline — fixed chunks, MiniLM, no stemming | 0.520 | 0.400 | 0.530 | — |
| Structure-aware chunking | 0.460 | 0.390 | 0.500 | — |
| ↩ reverted to fixed, + BM25 stemming | 0.520 | **0.420** | **0.550** | — |
| + bge-small-en-v1.5 | **0.610** | 0.420 | 0.550 | — |
| + cross-encoder reranking | 0.610 | 0.420 | 0.550 | **0.680** |

Dense recall@5 went **0.520 → 0.610**; the full system reached **0.680**.

### By vocabulary overlap

Where the thesis actually lives. The headline average hides all of this.

| Group | n | Dense | BM25 | Hybrid | Reranked |
|---|---|---|---|---|---|
| Low overlap (<30%) | 36 | 0.472 | 0.083 | 0.278 | **0.500** |
| Mid (30–70%) | 52 | 0.635 | 0.558 | 0.654 | **0.731** |
| High overlap (>70%) | 12 | 0.917 | 0.833 | 0.917 | **1.000** |

Reranking wins every band, and answers **12 of 12** high-overlap questions.

- On low-overlap questions, dense beats BM25 **0.472 to 0.083** — 5.7×. The first
  half of the thesis holds.
- On high-overlap questions BM25 closes to within one query of dense. The second
  half holds directionally, and earlier in the project (with MiniLM) BM25 led this
  group outright at 0.833 vs 0.750.
- Hybrid only wins in the middle band, where both retrievers are competent.

### Score separation

Not a metric, but visible and diagnostic. For `is cashew export restricted?`:

Top-5 score spread on `is cashew export restricted?`:

| Configuration | Spread |
|---|---|
| MiniLM dense | 0.034 |
| bge-small dense | 0.056 |
| Cross-encoder | **6.54** |

MiniLM was effectively saying "these five are equally relevant". When scores are
that close, ranking is noise — a small phrasing change reshuffles the order. The
same target chunk ranked 7, 10, 30 and 59 across four phrasings of the same
question.

---

## Findings

### 1. Hybrid fusion lost to dense-only

**0.550 vs 0.610.** The central hypothesis was wrong on this corpus.

The mechanism is visible in the per-query data: hybrid **rescued 4** queries that
neither dense nor BM25 found in their own top-5, and **lost 19** that one of them
had found. Seven of those losses were queries where dense had ranked the correct
passage **first**.

RRF assumes both inputs are worth listening to. When one retriever fails outright —
BM25 scores 0.083 on low-overlap questions — it does not fall silent; it submits 50
confidently wrong candidates, and their rank-1 and rank-2 positions outscore the
correct answer sitting at rank 7 in the good list.

An oracle that picked the better retriever per query would score **0.70**. RRF got
0.550. The reranker recovered most of that gap at 0.680, and its trade is the
mirror image of hybrid's: measured against dense-only it **rescued 18** queries and
**lost 11**, a net gain of 7. Hybrid's own trade was +4 / −19.

### 2. Structure-aware chunking made things worse

The diagnosis said chunking was the bottleneck. An **oracle test** — replacing each
question with the exact text of its own answer snippet — showed that with fixed
chunks, only **63 of 100** labels were retrievable even when the query *was* the
answer. Tariff chunks held a median of 10 distinct HS codes, so each product's
signal was one-tenth of the chunk's meaning.

Splitting on HS-code boundaries cut that to 3 codes per chunk. The oracle score went
**63 → 59** and recall fell across the board.

Reducing dilution increased **confusability**. 1,924 shorter chunks, ~1,099 of them
containing HS codes, look nearly identical to each other — `--- Other kg. 30% -`
repeated endlessly. The near-identical-clause problem got worse, not better.

Reverted. Kept behind a flag with the number recorded.

### 3. The biggest single win was the embedding model, not the pipeline

Swapping `all-MiniLM-L6-v2` for `bge-small-en-v1.5` moved dense recall
**0.520 → 0.610** and the oracle test **63 → 72**. Same dimensions, same index size,
one line plus a query prefix. This was ranked third out of three planned
improvements beforehand; it turned out to be first.

### 4. The retrievers failed in opposite directions — until they were improved

Measured on the **baseline** system (MiniLM, unstemmed BM25), the two retrievers
were almost perfect mirrors — rank of the correct chunk out of 1,287:

| Query | Dense | BM25 |
|---|---|---|
| `0902` | 851 | **1** |
| `duty on tea` | 59 | **4** |
| `can I sell cashews abroad?` | **7** | 988 |
| `is cashew export restricted?` | 16 | **1** |

The same queries on the **final** system (bge-small, stemmed BM25):

| Query | Dense | BM25 |
|---|---|---|
| `0902` | 12 | **1** |
| `duty on tea` | 208 | **4** |
| `can I sell cashews abroad?` | **1** | 3 |
| `is cashew export restricted?` | **1** | **1** |

Both fixes attacked the other side's weakness. BGE narrowed dense's blind spot for
codes (`0902`: 851 → 12), and stemming removed BM25's plural failure
(`cashews`: 988 → 3). The two retrievers are less complementary than they were —
which is part of why fusing them pays off less than the thesis predicted.

`duty on tea` went the other way, 59 → 208. BGE is better on average and worse on
that particular query; single queries move in both directions, which is the whole
reason the 100-query set decides anything.

### 5. A single-query demo will mislead you

Early in the project a demo appeared to show dense retrieval ranking the tea tariff
table at rank 3. It was wrong: the test matched any chunk containing the string
`0902`, and the chunk it found was the Second Schedule export tariff, not the import
duty table that answered the question. The real rank was 59.

This is the whole reason the 100-query evaluation exists.

### 6. Near-identical clauses are still the hard part

Asking `what is the HS code for basmati rice?` returns the **Non-Basmati** entry
first; the actual `1006 30 20` row ranks second. Reranking did not fix this
particular case. A single `non-` is the entire difference between the two passages,
and it is the failure mode the corpus is full of.

---

## LLM answer layer

`answer.py` — optional, off by default.

| | |
|---|---|
| Provider | Groq (OpenAI-compatible endpoint) |
| Model | `llama-3.3-70b-versatile` |
| Temperature | 0 |
| Key | `GROQ_API_KEY` environment variable only — never in source |

The prompt supplies numbered passages with their `source_doc` and page, and requires
the model to:

1. use only the passages, never outside knowledge;
2. cite a passage number after every factual claim;
3. reply *"The answer is not in the retrieved passages."* when the answer is absent;
4. say so explicitly when passages cover a near-identical but different item
   (Basmati vs Non-Basmati);
5. quote exact figures, HS codes and clause wording rather than paraphrasing.

Rule 3 is doing real work rather than decorating the prompt: at recall@5 of 0.68,
roughly one query in three hands the model passages that do not contain the answer.

**Every failure path degrades to passages-only.** Missing key, network error, rate
limit — all return `answer: None` and leave the passages untouched. Retrieval is the
product; the answer is a convenience on top of it.

---

## Web application

`app.py` (Flask JSON API) + `frontend/` (React + Vite + Tailwind).

One service, one origin: Vite builds to static assets, Flask serves them alongside
the API, so there is no CORS and no second deployment.

| Endpoint | Purpose |
|---|---|
| `POST /api/search` | `{query, mode, k, answer}` → `{answer, citations, passages}` |
| `POST /api/upload` | PDF → incremental indexing |
| `GET /api/status` | chunk count, reranker enabled, LLM configured |

Indexes load **once at startup**, not per request — `DenseIndex.__init__` loads a
model, and doing that per request would put seconds on every search.

**UI design.** A dark instrument shell wrapping light "paper" passage cards: the
chrome is the tool, each passage is a page out of a government schedule. Long
regulatory prose stays readable on light paper while the application around it reads
as a dark technical tool.

The results list carries a **rank spine** — rank, a score bar, and the raw score in
monospace — because score falloff is this project's actual finding and a retrieval
interface should show it rather than flatten it into an ordering. HS codes render as
monospace chips and `Free` / `Restricted` / `Prohibited` as bordered stamps, reusing
the corpus's own vocabulary.

Measured on the built page: accent contrast **8.48:1**, button text **7.78:1**, no
horizontal overflow at 375px or 1440px, 44px touch targets, passage measure capped
at 78 characters.

---

## Incremental ingestion

`add_document.py` — adding a PDF does not re-embed the corpus.

**Nothing is trained.** The models are frozen and pre-trained; adding a document
means extract → chunk → embed the new chunks → append. Re-embedding all 1,287
existing chunks to add one file takes ~80 seconds and grows with the corpus, which is
why this exists.

- `data/manifest.json` maps each PDF to its SHA-256 hash plus its page and chunk
  counts.
- New chunks are appended to `chunks.jsonl`; their vectors are `np.vstack`-ed onto
  the saved matrix; `dense_ids.json` is extended. A desync check compares id count to
  vector count.
- Chunk-id collisions are refused before anything is written.
- **BM25 is rebuilt in full.** IDF depends on the whole corpus, so a partial update
  would leave every existing term weighted against a stale document count — wrong in
  a way that still produces plausible-looking rankings.

```bash
python add_document.py --seed        # record already-indexed PDFs, once
python add_document.py               # scan data/ for anything new
python add_document.py Cotton.pdf    # one file
```

---

## Running it

```bash
pip install -r requirements.txt

# Build the corpus and indexes from the PDFs in data/
python ingest.py
python chunk.py                  # or: --strategy structure
python index_dense.py --build    # ~80s on CPU
python index_sparse.py --build

# Search
python search.py "is cashew export restricted?"
python search.py "0902" --mode sparse --chars 0

# Evaluate
python evaluate.py
python evaluate.py --disagreements

# Optional: grounded answers
setx GROQ_API_KEY "your-key"     # Windows; export on Unix
python answer.py "what is the duty on tea?"

# Web app
cd frontend && npm install && npm run build && cd ..
python app.py                    # http://localhost:5000
```

`data/dense_vectors.npy`, `data/dense_ids.json` and `data/bm25_index.pkl` are
gitignored — they are derived artifacts and rebuild from the committed PDFs.

Environment flags: `ENABLE_RERANKER=false` drops the reranker (memory escape hatch
for constrained deployments); `GROQ_API_KEY` enables the answer layer.

---

## Repository layout

Flat — a handful of files at the root. This is a few hundred lines of retrieval code
and should not be organised like a framework.

| File | Responsibility |
|---|---|
| `ingest.py` | PDF → per-page text, with source and page retained |
| `chunk.py` | Page text → overlapping chunks (`fixed` or `structure`) |
| `index_dense.py` | Embed chunks, persist vectors, cosine search |
| `index_sparse.py` | BM25 index over the same chunks, with stemming |
| `search.py` | Query → three ranked lists; RRF lives here |
| `rerank.py` | Cross-encoder reranking over retrieved candidates |
| `evaluate.py` | Labelled query set → recall@5 and MRR for all four modes |
| `answer.py` | Grounded answer generation over retrieved passages |
| `app.py` | Flask JSON API + static host |
| `add_document.py` | Incremental ingestion of a new PDF |
| `frontend/` | React + Vite + Tailwind UI |
| `data/` | Source PDFs, `pages.jsonl`, `chunks.jsonl`, `queries.jsonl`, `manifest.json` |

---

## Known limitations

- **recall@5 is 0.68.** Roughly one query in three does not surface a correct
  passage in the top five.
- **Near-identical clauses.** Basmati vs Non-Basmati, Free vs Restricted variants of
  the same commodity — the reranker helps but does not solve it.
- **Block-ordering artifact.** Column text positioned lower on a page can land after
  the policy column within the same row, so a qualifier occasionally sits outside its
  own clause.
- **`(1) (2) (3) (4) (5)`** column headers survive 129 times, left in deliberately.
- **Query set concentration.** 100 queries cover 77 distinct answer pages; 40 of them
  were written two-per-passage over 20 passages.
- **High-overlap group is small** — 12 queries, so BM25's advantage there is measured
  on thin evidence.
- **English only.** Both the corpus and the embedding models are English; a Hinglish
  or Hindi query fails on all four modes.
- **Corpus is a subset**, chosen to exercise both failure modes rather than for
  coverage.

---

## What I would do next

1. **Weighted fusion.** RRF weights both retrievers equally, which is what costs
   hybrid its 19 lost queries. Weighting dense higher — or dropping a retriever when
   its top score is below a threshold — should close part of the 0.550 → 0.70 gap.
   Built but not yet measured.
2. **Chunk context injection.** Prepend the chapter heading and HS code prefix to
   each tariff chunk. Structure-aware splitting failed because short chunks became
   indistinguishable; adding context rather than removing it attacks the same
   dilution problem from the other side.
3. **Tune `RRF_K` and candidate depth.** Both currently sit at published defaults, 60
   and 50, and neither has been swept.
4. **Expand the high-overlap query group** past 12 so that BM25's advantage rests on
   more than a handful of queries.
5. **A larger reranker** (`bge-reranker-base`) once memory allows, since the 22M-param
   model was a deployment compromise.
