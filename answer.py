"""Grounded answer generation over retrieved passages (Groq).

This layer never searches. It is handed whatever the retriever found and may only
speak from that. Retrieval recall@5 is around 0.6, which means roughly four times
in ten the passages below do not contain the answer at all -- so the instruction
to refuse is doing real work, not decorating the prompt.

The answer is always returned alongside the passages it came from. On regulatory
text a paraphrase is not a substitute for the clause; it is a pointer to it.
"""

import os
import re

API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"
TIMEOUT = 30
MAX_PASSAGE_CHARS = 1200

SYSTEM_PROMPT = """You answer questions about Indian trade regulation using ONLY \
the numbered passages provided.

Rules:
1. Use only the passages. Never use outside knowledge, even if you are confident.
2. Cite the passage number in square brackets after every factual claim, like [2].
3. If the passages do not contain the answer, say exactly: "The answer is not in \
the retrieved passages." Do not guess, and do not answer from general knowledge.
4. If passages disagree or cover a near-identical but different item (for example \
Basmati versus Non-Basmati rice, or a restricted versus a free variant), say so \
explicitly instead of picking one.
5. Quote exact figures, HS codes and clause wording rather than paraphrasing them.
6. Be brief. Two or three sentences unless the question needs more."""


def build_prompt(question, passages):
    """Number the passages and attach their source, then ask the question."""
    blocks = []
    for i, p in enumerate(passages, 1):
        text = " ".join(p["text"].split())[:MAX_PASSAGE_CHARS]
        blocks.append(f"[{i}] ({p['source_doc']}, page {p['page']})\n{text}")
    return "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def extract_citations(text, n_passages):
    """Passage numbers the model actually cited, in order of appearance."""
    seen = []
    for match in re.findall(r"\[(\d+)\]", text):
        i = int(match)
        if 1 <= i <= n_passages and i not in seen:
            seen.append(i)
    return seen


def generate(question, passages):
    """Return {answer, citations, error}. Never raises.

    A failure here must not take search down with it: the passages are the
    product, the answer is a convenience on top. Every error path returns
    answer=None so the caller can render results unchanged.
    """
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return {"answer": None, "citations": [], "error": "GROQ_API_KEY not set"}
    if not passages:
        return {"answer": None, "citations": [], "error": "no passages retrieved"}

    try:
        import requests

        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": MODEL,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(question, passages)},
                ],
            },
            timeout=TIMEOUT,
        )
        if response.status_code != 200:
            return {
                "answer": None,
                "citations": [],
                "error": f"groq returned {response.status_code}",
            }
        text = response.json()["choices"][0]["message"]["content"].strip()
        return {
            "answer": text,
            "citations": extract_citations(text, len(passages)),
            "error": None,
        }
    except Exception as exc:
        return {"answer": None, "citations": [], "error": f"{type(exc).__name__}: {exc}"}


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--mode", default="rerank",
                    choices=["dense", "sparse", "hybrid", "rerank"])
    args = ap.parse_args()

    from search import Search

    s = Search()
    fn = {
        "dense": s.dense_search,
        "sparse": s.sparse_search,
        "hybrid": s.hybrid_search,
        "rerank": s.rerank_search,
    }[args.mode]

    passages = [dict(s.chunks[cid], score=score) for cid, score in fn(args.question, args.k)]
    result = generate(args.question, passages)

    print(f"\nQ: {args.question}\n")
    if result["answer"]:
        print(result["answer"])
        print(f"\ncited passages: {result['citations'] or 'none'}")
    else:
        print(f"[no answer generated: {result['error']}]")
    print("\n--- passages ---")
    for i, p in enumerate(passages, 1):
        print(f"\n[{i}] {p['source_doc']} p{p['page']}  (score {p['score']:.3f})")
        print("    " + " ".join(p["text"].split())[:200])


if __name__ == "__main__":
    main()
