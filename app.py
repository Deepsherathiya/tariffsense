"""Flask API + static host for the React build.

    python app.py                      # dev, http://localhost:5000
    gunicorn -w 1 app:app              # production; one worker, see below

One worker is deliberate. Each worker holds its own copy of the embedding model
and the reranker, so a second worker doubles memory on a tier that has none to
spare.
"""

import os

from flask import Flask, jsonify, request, send_from_directory

DIST_DIR = os.path.join("frontend", "dist")
DATA_DIR = "data"
MAX_UPLOAD_MB = 25

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

_search = None


def get_search():
    """Indexes load once, on first request rather than at import.

    Loading at import would make gunicorn's boot timeout the thing that decides
    whether the app starts, and the models take a while on a cold container.
    """
    global _search
    if _search is None:
        from search import Search

        _search = Search()
    return _search


@app.post("/api/search")
def api_search():
    body = request.get_json(silent=True) or {}
    question = (body.get("query") or "").strip()
    mode = body.get("mode", "rerank")
    k = min(int(body.get("k", 5)), 20)
    want_answer = bool(body.get("answer", False))

    if not question:
        return jsonify({"error": "query is required"}), 400

    s = get_search()
    fn = {
        "dense": s.dense_search,
        "bm25": s.sparse_search,
        "hybrid": s.hybrid_search,
        "rerank": s.rerank_search,
    }.get(mode)
    if fn is None:
        return jsonify({"error": f"unknown mode {mode!r}"}), 400

    passages = [
        {
            "chunk_id": cid,
            "text": s.chunks[cid]["text"],
            "source_doc": s.chunks[cid]["source_doc"],
            "page": s.chunks[cid]["page"],
            "score": round(float(score), 4),
        }
        for cid, score in fn(question, k)
    ]

    result = {"query": question, "mode": mode, "passages": passages,
              "answer": None, "citations": [], "answer_error": None}

    if want_answer:
        import answer

        generated = answer.generate(question, passages)
        result["answer"] = generated["answer"]
        result["citations"] = generated["citations"]
        result["answer_error"] = generated["error"]

    return jsonify(result)


@app.post("/api/upload")
def api_upload():
    """Accept a PDF and fold it into the existing indexes.

    Nothing is trained. The models are frozen; this extracts text, chunks it,
    embeds the new chunks and appends them.
    """
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "a .pdf file is required"}), 400

    name = os.path.basename(file.filename)
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        return jsonify({"error": f"{name} already exists"}), 409
    file.save(path)

    try:
        import add_document

        stats = add_document.add(name)
    except Exception as exc:
        os.remove(path)
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    global _search
    _search = None  # force a reload so the new chunks are searchable
    return jsonify({"added": name, **stats})


@app.get("/api/status")
def api_status():
    import rerank

    return jsonify(
        {
            "reranker_enabled": rerank.ENABLED,
            "llm_configured": bool(os.environ.get("GROQ_API_KEY")),
            "chunks": len(get_search().chunks),
        }
    )


@app.get("/", defaults={"path": ""})
@app.get("/<path:path>")
def serve_frontend(path):
    """Serve the Vite build. Unknown paths fall through to index.html."""
    full = os.path.join(DIST_DIR, path)
    if path and os.path.isfile(full):
        return send_from_directory(DIST_DIR, path)
    if os.path.isfile(os.path.join(DIST_DIR, "index.html")):
        return send_from_directory(DIST_DIR, "index.html")
    return (
        "<h1>Frontend not built</h1><p>Run <code>npm install &amp;&amp; npm run build</code> "
        "inside <code>frontend/</code>.</p>",
        200,
    )


if __name__ == "__main__":
    # use_reloader=False: the reloader watches the working tree, and data/ holds a
    # multi-megabyte vector file plus the BM25 pickle. Watching those wedges the
    # dev server on Windows. Restart by hand after editing.
    app.run(debug=True, use_reloader=False, port=5000)
