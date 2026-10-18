"""Retrieval configurations shared by the CLI and future evaluation runner."""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from pubmed_rag.rankings import CANDIDATE_LIMIT, fuse_rankings
from pubmed_rag.reranking import MiniLMReranker, RerankingMetadata, rerank_candidates
from pubmed_rag.search import (
    RetrievalResult,
    bm25_search,
    dense_search,
    validate_paid_query,
)

METHODS = {"A": "dense", "B": "bm25", "C": "hybrid", "D": "reranked"}


@dataclass
class RetrievalRun:
    configuration: str
    results: list[RetrievalResult]
    reranking: RerankingMetadata | None = None

    @property
    def context(self) -> list[RetrievalResult]:
        """The five papers to supply to generation, without reducing evaluation depth."""
        return self.results[:5]


def retrieve(
    query: str,
    method: str,
    chunks: list[dict[str, Any]],
    selection: dict[str, Any],
    database_url: str = "",
    allow_paid_query: bool = False,
    query_budget_usd: float | None = None,
    depth: int = CANDIDATE_LIMIT,
) -> RetrievalRun:
    """Retain deep rankings independently of CLI display and generation limits."""
    configuration = next(
        (code for code, name in METHODS.items() if method in (code, name)), None
    )
    if configuration is None:
        raise ValueError(f"Unknown retrieval method: {method}")
    if depth <= 0:
        raise ValueError("limit must be greater than zero")
    if configuration in ("C", "D") and depth > CANDIDATE_LIMIT:
        raise ValueError("Hybrid configurations retain at most 50 fused candidates")
    if not query.strip():
        raise ValueError("Question must not be blank")
    limit = max(CANDIDATE_LIMIT, depth)
    if configuration == "B":
        return RetrievalRun(configuration, bm25_search(chunks, query, limit))
    reranker = None
    if configuration == "D":
        validate_paid_query(query, allow_paid_query, query_budget_usd)
        reranker = MiniLMReranker()
    dense = dense_search(
        query,
        limit,
        database_url,
        selection["hash"],
        allow_paid_query,
        query_budget_usd=query_budget_usd,
        expected_count=selection["count"],
    )
    if configuration == "A":
        return RetrievalRun(configuration, dense)
    lexical = bm25_search(chunks, query, CANDIDATE_LIMIT)
    candidates = fuse_rankings(dense, lexical)
    if reranker is not None:
        started = perf_counter()
        results = rerank_candidates(query, candidates, reranker)
        reranker.metadata.rerank_seconds = perf_counter() - started
        reranker.metadata.candidate_count = len(candidates)
        return RetrievalRun(configuration, results, reranker.metadata)
    return RetrievalRun(configuration, candidates)
