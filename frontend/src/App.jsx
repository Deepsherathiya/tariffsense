import { useEffect, useRef, useState } from "react";

const CORPUS = { documents: 21, pages: 518, passages: 1287 };

const MODES = [
  {
    id: "dense",
    label: "Dense",
    short: "Meaning",
    does: "Matches meaning, so a question can share no words with the clause and still find it.",
    fails: "An HS code like 0902 is four digits with no meaning to embed.",
    recall: "0.610",
  },
  {
    id: "bm25",
    label: "BM25",
    short: "Exact terms",
    does: "Matches exact strings, and weights rare words far above common ones.",
    fails: "Fails completely when the question and the clause share no vocabulary.",
    recall: "0.420",
  },
  {
    id: "hybrid",
    label: "Hybrid",
    short: "Both, fused",
    does: "Fuses both rankings by position, so neither score scale dominates.",
    fails: "When one retriever fails outright, its wrong results still get a vote.",
    recall: "0.550",
  },
  {
    id: "rerank",
    label: "Reranked",
    short: "Re-read",
    does: "Re-reads each query and passage together, so qualifiers like ‘non-’ register.",
    fails: "Slower — roughly one to two seconds per query.",
    recall: "0.680",
    best: true,
  },
];

const EXAMPLES = [
  { q: "is cashew export restricted?", note: "policy prose" },
  { q: "0902", note: "exact code" },
  { q: "duty on basmati rice", note: "near-identical clauses" },
  { q: "what counts as handmade goods?", note: "no shared words" },
];

const RESULTS = [
  { system: "Dense", recall: "0.610", mrr: "0.422" },
  { system: "BM25", recall: "0.420", mrr: "0.283" },
  { system: "Hybrid", recall: "0.550", mrr: "0.402" },
  { system: "Reranked", recall: "0.680", mrr: "0.490", best: true },
];

const SOURCES = [
  "CBIC First Schedule — tariff chapters 3, 8, 9, 10, 29, 30, 52, 60–63, 71, 87",
  "DGFT ITC(HS) Schedule 2 — export policy and conditions",
  "Foreign Trade Policy 2023 — chapters 1 and 2",
  "Handbook of Procedures — chapter 2",
];

const MARKUP = /(\b\d{4} \d{2} \d{2}\b|\b\d{8}\b|\bFree\b|\bRestricted\b|\bProhibited\b)/g;
const STAMP = { Free: "text-free", Restricted: "text-restricted", Prohibited: "text-prohibited" };

function Marked({ text }) {
  return (
    <>
      {text.split(MARKUP).map((part, i) => {
        if (STAMP[part]) return <span key={i} className={`stamp ${STAMP[part]}`}>{part}</span>;
        if (/^\d{4} \d{2} \d{2}$|^\d{8}$/.test(part))
          return <span key={i} className="code-chip">{part}</span>;
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}

function Wordmark({ className = "" }) {
  return (
    <span className={`flex items-center gap-2.5 ${className}`}>
      <svg viewBox="0 0 32 32" className="h-6 w-6 shrink-0" aria-hidden="true">
        <rect x="2" y="4" width="4" height="24" rx="2" fill="var(--color-accent)" />
        <rect x="11" y="10" width="4" height="18" rx="2" fill="var(--color-edge)" />
        <rect x="20" y="17" width="4" height="11" rx="2" fill="var(--color-edge)" />
      </svg>
      <span className="font-semibold tracking-[-0.02em]">TariffSense</span>
    </span>
  );
}

function Header({ status, compact }) {
  if (compact) return null;
  return (
    <header className="relative z-20 mx-auto flex max-w-6xl items-center justify-between px-6 py-5">
      <Wordmark />
      <nav className="hidden items-center gap-7 text-sm text-fg-soft md:flex">
        <a className="hover:text-fg" href="#how">How it works</a>
        <a className="hover:text-fg" href="#corpus">What's indexed</a>
        <a className="hover:text-fg" href="#results">Results</a>
      </nav>
      <span className="font-mono text-xs text-fg-soft">
        {(status?.chunks ?? CORPUS.passages).toLocaleString()} passages
      </span>
    </header>
  );
}

function SearchField({ query, setQuery, onSubmit, loading, big }) {
  return (
    <form onSubmit={onSubmit} className={big ? "" : "flex-1"}>
      <div
        className={`flex items-center gap-2 rounded-xl border border-edge bg-shell-2 ${
          big ? "px-4 py-3" : "px-3 py-2"
        } focus-within:border-accent`}
      >
        <svg viewBox="0 0 24 24" className="h-5 w-5 shrink-0 text-fg-soft" aria-hidden="true">
          <circle cx="11" cy="11" r="7" fill="none" stroke="currentColor" strokeWidth="2" />
          <path d="m20 20-4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        </svg>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask about an export rule, or paste an HS code"
          aria-label="Search the corpus"
          className={`min-w-0 flex-1 bg-transparent outline-none placeholder:text-fg-soft/60 ${
            big ? "text-lg" : "text-sm"
          }`}
        />
        <button
          type="submit"
          disabled={loading || !query.trim()}
          // min-h-11 keeps the tap target at 44px on touch screens; the padding
          // alone left it at 36px.
          className={`min-h-11 shrink-0 cursor-pointer rounded-lg bg-accent px-5 font-semibold text-[#241a02] transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-35 ${
            big ? "text-sm" : "text-xs"
          }`}
        >
          {loading ? "Searching…" : "Search"}
        </button>
      </div>
    </form>
  );
}

function Hero({ query, setQuery, onSubmit, onExample, loading }) {
  return (
    <section className="relative overflow-hidden px-6 pb-20 pt-14 sm:pt-20">
      <div className="glow drift -left-32 top-0 h-96 w-96 bg-accent/12" aria-hidden="true" />
      <div
        className="glow -right-24 top-24 h-80 w-80 bg-indigo-500/12"
        style={{ animation: "drift 26s ease-in-out infinite reverse" }}
        aria-hidden="true"
      />
      <div className="relative mx-auto max-w-3xl text-center">
        <p className="label fade">Indian customs tariff &amp; DGFT export policy</p>
        <h1 className="fade mt-4 text-4xl font-bold leading-[1.08] tracking-[-0.03em] sm:text-6xl">
          Find the exact clause,
          <br />
          <span className="text-accent">not a summary of it.</span>
        </h1>
        <p className="fade mx-auto mt-5 max-w-xl font-doc text-lg leading-relaxed text-fg-soft">
          Search {CORPUS.pages} pages of Indian trade documentation. Every result is the
          source passage, with its document and page number.
        </p>
        <div className="rise mx-auto mt-8 max-w-2xl" style={{ animationDelay: "120ms" }}>
          <SearchField
            big
            query={query}
            setQuery={setQuery}
            onSubmit={onSubmit}
            loading={loading}
          />
        </div>
        <div className="rise mt-5 flex flex-wrap justify-center gap-2" style={{ animationDelay: "200ms" }}>
          {EXAMPLES.map((e) => (
            <button
              key={e.q}
              onClick={() => onExample(e.q)}
              className="inline-flex min-h-11 cursor-pointer items-center rounded-full border border-edge px-4 text-xs text-fg-soft transition-colors hover:border-accent hover:text-fg"
            >
              {e.q}
              <span className="ml-2 font-mono text-[0.65rem] text-fg-soft/60">{e.note}</span>
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}

function Landing() {
  return (
    <>
      <section id="how" className="border-t border-edge px-6 py-16">
        <div className="mx-auto max-w-6xl">
          <p className="label">How the search works</p>
          <h2 className="mt-3 max-w-2xl text-2xl font-semibold tracking-[-0.02em] sm:text-3xl">
            Four retrievers. Each one fails at something the next one handles.
          </h2>
          <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {MODES.map((m) => (
              <article
                key={m.id}
                className={`rounded-xl border p-5 ${
                  m.best ? "border-accent/50 bg-accent/5" : "border-edge bg-shell"
                }`}
              >
                <div className="flex items-baseline justify-between">
                  <h3 className="font-semibold">{m.label}</h3>
                  <span className="font-mono text-sm text-accent">{m.recall}</span>
                </div>
                <p className="mt-3 text-sm leading-relaxed text-fg-soft">{m.does}</p>
                <p className="mt-3 border-t border-edge pt-3 text-sm leading-relaxed text-fg-soft">
                  <span className="label">Fails</span>
                  <br />
                  {m.fails}
                </p>
              </article>
            ))}
          </div>
          <p className="mt-4 font-mono text-xs text-fg-soft">
            recall@5 on 100 hand-labelled questions
          </p>
        </div>
      </section>

      <section id="corpus" className="border-t border-edge px-6 py-16">
        <div className="mx-auto grid max-w-6xl gap-10 lg:grid-cols-[1fr_1.1fr]">
          <div>
            <p className="label">What's indexed</p>
            <h2 className="mt-3 text-2xl font-semibold tracking-[-0.02em] sm:text-3xl">
              A working subset, not all of Indian trade law.
            </h2>
            <dl className="mt-7 flex gap-8">
              {[
                ["Documents", CORPUS.documents],
                ["Pages", CORPUS.pages],
                ["Passages", CORPUS.passages.toLocaleString()],
              ].map(([k, v]) => (
                <div key={k}>
                  <dt className="label">{k}</dt>
                  <dd className="mt-1 font-mono text-2xl text-accent">{v}</dd>
                </div>
              ))}
            </dl>
          </div>
          <ul className="space-y-3">
            {SOURCES.map((s) => (
              <li
                key={s}
                className="rounded-lg border border-edge bg-shell px-4 py-3 font-doc text-[0.98rem] leading-relaxed text-fg-soft"
              >
                {s}
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section id="results" className="border-t border-edge px-6 py-16">
        <div className="mx-auto max-w-6xl">
          <p className="label">Measured, not claimed</p>
          <h2 className="mt-3 max-w-2xl text-2xl font-semibold tracking-[-0.02em] sm:text-3xl">
            100 hand-labelled questions. Every number below came out of a run.
          </h2>
          <div className="mt-8 grid gap-8 lg:grid-cols-[1fr_1fr]">
            <div className="overflow-hidden rounded-xl border border-edge">
              <table className="w-full text-left">
                <thead className="bg-shell">
                  <tr className="label">
                    <th className="px-4 py-3 font-medium">System</th>
                    <th className="px-4 py-3 text-right font-medium">recall@5</th>
                    <th className="px-4 py-3 text-right font-medium">MRR</th>
                  </tr>
                </thead>
                <tbody className="font-mono text-sm">
                  {RESULTS.map((r) => (
                    <tr
                      key={r.system}
                      className={`border-t border-edge ${r.best ? "bg-accent/5 text-accent" : ""}`}
                    >
                      <td className="px-4 py-3 font-sans">{r.system}</td>
                      <td className="px-4 py-3 text-right">{r.recall}</td>
                      <td className="px-4 py-3 text-right">{r.mrr}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="font-doc text-[1.02rem] leading-relaxed text-fg-soft">
              <p>
                Hybrid fusion was supposed to be the answer. It wasn't — dense retrieval
                alone beat it, 0.610 to 0.550, because reciprocal rank fusion gives an
                equal vote to a retriever that has failed outright.
              </p>
              <p className="mt-4">
                Reranking fixed it. Reading each question and passage together, rather
                than comparing two separately-built vectors, lifted recall to 0.680 — close
                to the 0.70 ceiling of always picking the better retriever per query.
              </p>
              <p className="mt-4">
                Structure-aware chunking was tried and made things worse. That number is
                in the repository too.
              </p>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}

function Toolbar({ query, setQuery, onSubmit, loading, mode, setMode, wantAnswer, setWantAnswer, onClear }) {
  return (
    <div className="glass sticky top-0 z-30 border-b border-edge">
      <div className="mx-auto max-w-6xl px-6 py-3">
        <div className="flex items-center gap-4">
          <button onClick={onClear} className="cursor-pointer" aria-label="Back to home">
            <Wordmark className="hidden text-sm sm:flex" />
          </button>
          <SearchField query={query} setQuery={setQuery} onSubmit={onSubmit} loading={loading} />
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-4">
          <div role="radiogroup" aria-label="Retrieval mode" className="flex rounded-lg border border-edge p-0.5">
            {MODES.map((m) => (
              <button
                key={m.id}
                role="radio"
                aria-checked={mode === m.id}
                onClick={() => setMode(m.id)}
                title={m.does}
                className={`inline-flex min-h-10 cursor-pointer items-center rounded-md px-3 text-xs font-medium transition-colors ${
                  mode === m.id ? "bg-accent text-[#241a02]" : "text-fg-soft hover:text-fg"
                }`}
              >
                {m.label}
                <span className="ml-1.5 hidden font-normal opacity-70 lg:inline">{m.short}</span>
              </button>
            ))}
          </div>
          <label className="flex cursor-pointer items-center gap-2 text-xs text-fg-soft">
            <input
              type="checkbox"
              checked={wantAnswer}
              onChange={(e) => setWantAnswer(e.target.checked)}
              className="h-4 w-4 cursor-pointer accent-[#e0a42b]"
            />
            Write an answer from the passages
          </label>
        </div>
      </div>
    </div>
  );
}

function AnswerCard({ answer, citations, error, onJump }) {
  if (!answer && !error) return null;
  if (!answer) {
    return (
      <div className="rise mb-8 rounded-xl border border-edge bg-shell px-5 py-4">
        <p className="label">Answer unavailable</p>
        <p className="mt-2 text-sm text-fg-soft">{error}. The passages below are unaffected.</p>
      </div>
    );
  }
  const parts = answer.split(/(\[\d+\])/g);
  return (
    <div className="rise mb-8 rounded-xl border border-accent/40 bg-shell px-5 py-5">
      <p className="label text-accent">Written from the passages below</p>
      <p className="mt-3 font-doc text-lg leading-relaxed">
        {parts.map((part, i) => {
          const m = part.match(/^\[(\d+)\]$/);
          if (!m) return <span key={i}>{part}</span>;
          return (
            <button
              key={i}
              onClick={() => onJump(Number(m[1]))}
              className="mx-0.5 cursor-pointer rounded bg-accent/15 px-1.5 align-super font-mono text-[0.65rem] font-semibold text-accent hover:bg-accent/25"
            >
              {m[1]}
            </button>
          );
        })}
      </p>
      {citations.length > 0 && (
        <p className="mt-4 font-mono text-xs text-fg-soft">
          cites passage{citations.length > 1 ? "s" : ""} {citations.join(", ")}
        </p>
      )}
    </div>
  );
}

function PassageCard({ p, rank, width, refEl }) {
  return (
    <li ref={refEl} className="rise flex gap-4" style={{ animationDelay: `${rank * 60}ms` }}>
      {/* Rank spine. Score falloff is the finding this project is about, so it is
          shown rather than flattened into an ordering. */}
      <div className="w-14 shrink-0 pt-4 text-right sm:w-16">
        <div className="font-mono text-sm font-semibold">{rank}</div>
        <div className="mt-1.5 h-1 rounded-full bg-accent-dim/40">
          <div
            className="h-1 rounded-full bg-accent"
            style={{ width: `${Math.max(width * 100, 5)}%` }}
          />
        </div>
        <div className="mt-1.5 font-mono text-[0.65rem] text-fg-soft">{p.score.toFixed(3)}</div>
      </div>
      <article className="min-w-0 flex-1 rounded-xl border border-paper-edge bg-paper px-5 py-4 shadow-lg shadow-black/20">
        <div className="flex flex-wrap items-baseline gap-x-3 border-b border-paper-edge pb-2">
          <h3 className="text-sm font-semibold text-paper-ink">{p.source_doc}</h3>
          <span className="font-mono text-xs text-paper-soft">page {p.page}</span>
        </div>
        {/* Capped at 78ch. The card is allowed to run the full column width so the
            metadata row can breathe, but at 1440 the text alone would set ~115
            characters to the line, which is punishing for legal prose. */}
        <p className="mt-3 max-w-[78ch] font-doc text-[1.02rem] leading-relaxed text-paper-ink">
          <Marked text={p.text} />
        </p>
      </article>
    </li>
  );
}

function Skeleton() {
  return (
    <ul className="space-y-6">
      {[0, 1, 2].map((i) => (
        <li key={i} className="flex gap-4">
          <div className="w-14 shrink-0 sm:w-16" />
          <div className="h-36 flex-1 animate-pulse rounded-xl border border-edge bg-shell" />
        </li>
      ))}
    </ul>
  );
}

function NoResults({ query, onExample }) {
  return (
    <div className="rounded-xl border border-dashed border-edge px-6 py-14 text-center">
      <p className="text-lg font-semibold">Nothing matched “{query}”.</p>
      <p className="mx-auto mt-2 max-w-md font-doc text-fg-soft">
        Try a different retrieval mode — BM25 finds exact codes that dense retrieval
        misses, and the reverse is true for paraphrased questions. Or start from one of
        these:
      </p>
      <div className="mt-5 flex flex-wrap justify-center gap-2">
        {EXAMPLES.map((e) => (
          <button
            key={e.q}
            onClick={() => onExample(e.q)}
            className="inline-flex min-h-11 cursor-pointer items-center rounded-full border border-edge px-4 text-xs text-fg-soft hover:border-accent hover:text-fg"
          >
            {e.q}
          </button>
        ))}
      </div>
    </div>
  );
}

function Upload({ upload, onUpload }) {
  return (
    <div className="mt-14 border-t border-edge pt-6">
      <p className="label">Add a document</p>
      <label className="mt-3 flex cursor-pointer items-center justify-center rounded-xl border border-dashed border-edge px-4 py-5 text-sm text-fg-soft transition-colors hover:border-accent hover:text-fg">
        <input type="file" accept=".pdf" onChange={onUpload} className="sr-only" />
        Choose a PDF to index. Only the new passages are embedded — nothing is retrained.
      </label>
      {upload?.state === "working" && (
        <p className="mt-2 text-xs text-fg-soft">Indexing {upload.name}…</p>
      )}
      {upload?.state === "done" && (
        <p className="mt-2 text-xs text-free">
          Added {upload.name}: {upload.pages} pages, {upload.chunks} passages.
        </p>
      )}
      {upload?.state === "error" && (
        <p className="mt-2 text-xs text-prohibited">{upload.message}</p>
      )}
    </div>
  );
}

function Footer() {
  return (
    <footer className="border-t border-edge px-6 py-10">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 sm:flex-row sm:items-start sm:justify-between">
        <div className="max-w-md">
          <Wordmark className="text-sm" />
          <p className="mt-3 text-sm leading-relaxed text-fg-soft">
            TariffSense returns source passages for reference. It is not legal or customs
            advice — verify against the official document before acting on anything here.
          </p>
        </div>
        <p className="font-mono text-xs leading-relaxed text-fg-soft">
          bge-small-en-v1.5 · rank_bm25 · ms-marco cross-encoder
          <br />
          Flask · React · no retrieval framework
        </p>
      </div>
    </footer>
  );
}

export default function App() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("rerank");
  const [wantAnswer, setWantAnswer] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState(null);
  const [upload, setUpload] = useState(null);
  const passageRefs = useRef([]);

  useEffect(() => {
    fetch("/api/status").then((r) => r.json()).then(setStatus).catch(() => {});
  }, []);

  async function runSearch(e, override) {
    e?.preventDefault();
    const q = (override ?? query).trim();
    if (!q) return;
    if (override) setQuery(override);
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, mode, k: 5, answer: wantAnswer }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `request failed (${res.status})`);
      setResult(data);
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  async function handleUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUpload({ state: "working", name: file.name });
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch("/api/upload", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "upload failed");
      setUpload({ state: "done", name: data.added, chunks: data.chunks, pages: data.pages });
      fetch("/api/status").then((r) => r.json()).then(setStatus);
    } catch (err) {
      setUpload({ state: "error", message: err.message });
    } finally {
      e.target.value = "";
    }
  }

  function clearAll() {
    setResult(null);
    setError(null);
    setQuery("");
  }

  const searching = loading || result || error;
  const passages = result?.passages ?? [];
  const scores = passages.map((p) => p.score);
  const lo = Math.min(...scores);
  const hi = Math.max(...scores);
  const spread = hi - lo;

  return (
    <div className="flex min-h-screen flex-col">
      {searching ? (
        <Toolbar
          query={query}
          setQuery={setQuery}
          onSubmit={runSearch}
          loading={loading}
          mode={mode}
          setMode={setMode}
          wantAnswer={wantAnswer}
          setWantAnswer={setWantAnswer}
          onClear={clearAll}
        />
      ) : (
        <Header status={status} />
      )}

      <main className="flex-1">
        {!searching && (
          <>
            <Hero
              query={query}
              setQuery={setQuery}
              onSubmit={runSearch}
              onExample={(q) => runSearch(null, q)}
              loading={loading}
            />
            <Landing />
          </>
        )}

        {searching && (
          <div className="mx-auto max-w-6xl px-6 py-8">
            {error && (
              <div className="rounded-xl border border-prohibited/40 bg-shell px-4 py-3 text-sm text-prohibited">
                {error}
              </div>
            )}
            {loading && <Skeleton />}
            {!loading && result && passages.length === 0 && (
              <NoResults query={result.query} onExample={(q) => runSearch(null, q)} />
            )}
            {!loading && result && passages.length > 0 && (
              <>
                <AnswerCard
                  answer={result.answer}
                  citations={result.citations}
                  error={result.answer_error}
                  onJump={(n) =>
                    passageRefs.current[n - 1]?.scrollIntoView({
                      behavior: "smooth",
                      block: "center",
                    })
                  }
                />
                <div className="mb-5 flex items-baseline justify-between border-b border-edge pb-2">
                  <h2 className="text-sm font-semibold">{passages.length} passages</h2>
                  <span className="font-mono text-xs text-fg-soft">
                    {result.mode} · score spread {spread.toFixed(3)}
                  </span>
                </div>
                <ol className="space-y-6">
                  {passages.map((p, i) => (
                    <PassageCard
                      key={p.chunk_id}
                      p={p}
                      rank={i + 1}
                      width={spread > 0 ? (p.score - lo) / spread : 1}
                      refEl={(el) => (passageRefs.current[i] = el)}
                    />
                  ))}
                </ol>
                <Upload upload={upload} onUpload={handleUpload} />
              </>
            )}
          </div>
        )}
      </main>

      <Footer />
    </div>
  );
}
