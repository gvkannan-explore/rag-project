"""Run a local CPU model trial with synthetic text and no paid requests."""

import json
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from pubmed_rag.rankings import fuse_rankings
from pubmed_rag.reranking import MiniLMReranker, rerank_candidates
from pubmed_rag.search import bm25_search


def main() -> None:
    query = "Which planet is known as the Red Planet?"
    texts = [
        "Mars is known as the Red Planet.",
        "Venus is the second planet from the Sun.",
        "Irrelevant filler text. " * 600,
    ] + [f"A synthetic document about topic {i}." for i in range(47)]
    chunks = [
        {
            "id": str(i),
            "title": f"Synthetic paper {i}",
            "chunk_text": text,
            "meta": {"hash": "synthetic-cpu-trial"},
        }
        for i, text in enumerate(texts)
    ]
    candidates = fuse_rankings([], bm25_search(chunks, query, 50))
    reranker = MiniLMReranker()
    pairs = [(query, row["source_text"]) for row in candidates]
    tokenizer = reranker.model.tokenizer
    before = tokenizer(pairs, truncation=False, padding=False)["input_ids"]
    after = tokenizer(pairs, truncation="longest_first", max_length=512, padding=False)[
        "input_ids"
    ]
    assert max(map(len, after)) <= 512
    assert max(map(len, before)) > 512
    started = perf_counter()
    results = rerank_candidates(query, candidates, reranker)
    reranker.metadata.rerank_seconds = perf_counter() - started
    reranker.metadata.candidate_count = len(candidates)
    assert {row["pmid"] for row in results} == {row["pmid"] for row in candidates}
    assert results[0]["pmid"] == "0", "The explicit Mars passage should rank first"
    payload = {
        **asdict(reranker.metadata),
        "trial": "50 synthetic candidates, including one input longer than 512 tokens",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("sentence-transformers", "torch", "transformers")
        },
        "maximum_tokens_before_truncation": max(map(len, before)),
        "maximum_tokens_after_truncation": max(map(len, after)),
        "top_five_synthetic_ids": [row["pmid"] for row in results[:5]],
        "paid_requests": 0,
    }
    output = Path("data/reranker_trial.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
