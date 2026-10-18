"""CPU MiniLM scoring and deterministic reranking of the shared hybrid pool."""

import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, cast

from pubmed_rag.rankings import CANDIDATE_LIMIT
from pubmed_rag.search import RetrievalResult

# Verified against official Sentence Transformers docs and the model repository.
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
MAX_LENGTH = 512
BATCH_SIZE = 16


class PairScorer(Protocol):
    def score(self, pairs: list[tuple[str, str]]) -> list[float]: ...


@dataclass
class RerankingMetadata:
    model_id: str = RERANKER_MODEL_ID
    revision: str = RERANKER_REVISION
    device: str = "cpu"
    max_length: int = MAX_LENGTH
    batch_size: int = BATCH_SIZE
    truncation: str = "longest_first"
    truncation_side: str = "right"
    score_type: str = "raw_logit"
    memory_mapped_weights: bool = False
    load_seconds: float = 0.0
    rerank_seconds: float = 0.0
    candidate_count: int = 0


class MiniLMReranker:
    """Load one pinned local model; keep loading outside the scoring timer."""

    def __init__(self, cache_folder: Path = Path("data/models")) -> None:
        started = perf_counter()
        try:
            from sentence_transformers import CrossEncoder
            from torch.nn import Identity

            self.model: Any = cast(Any, CrossEncoder)(
                RERANKER_MODEL_ID,
                revision=RERANKER_REVISION,
                device="cpu",
                max_length=MAX_LENGTH,
                cache_folder=str(cache_folder),
                activation_fn=Identity(),
                trust_remote_code=False,
                # Avoid unaligned mapped weights producing NaNs on the CPU.
                model_kwargs={"disable_mmap": True},
            )
            self.model.tokenizer.truncation_side = "right"
        except Exception as exc:
            raise ValueError(
                "Cannot load the pinned CPU reranker. Install the reranking dependency "
                f"group and check model download/cache access: {exc}"
            ) from exc
        self.metadata = RerankingMetadata(load_seconds=perf_counter() - started)

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = self.model.predict(
            pairs,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(score) for score in scores]


def rerank_candidates(
    query: str, candidates: list[RetrievalResult], scorer: PairScorer
) -> list[RetrievalResult]:
    """Change ordering only; retrieval scores and fused ranks remain intact."""
    if not candidates:
        return []
    if len(candidates) > CANDIDATE_LIMIT or len(
        {row["pmid"] for row in candidates}
    ) != len(candidates):
        raise ValueError("Reranking accepts at most 50 unique candidates")
    try:
        scores = scorer.score([(query, row["source_text"]) for row in candidates])
    except Exception as exc:
        raise ValueError(f"Reranking failed: {exc}") from exc
    if len(scores) != len(candidates) or any(
        not math.isfinite(score) for score in scores
    ):
        raise ValueError("Reranking requires one finite score per candidate")
    scored: list[RetrievalResult] = [
        {**row, "method": "reranked", "reranker_score": score}
        for row, score in zip(candidates, scores, strict=True)
    ]
    scored.sort(key=lambda row: (-row.get("reranker_score", 0.0), row["pmid"]))
    return [{**row, "rank": rank} for rank, row in enumerate(scored, 1)]
