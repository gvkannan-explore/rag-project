from copy import deepcopy

import pytest
from test_rankings import ranked

from pubmed_rag.rankings import fuse_rankings
from pubmed_rag.reranking import rerank_candidates


class FixedScorer:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.pairs: list[tuple[str, str]] = []

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.pairs = pairs
        return self.scores


def test_reranking_uses_scores_preserves_identity_and_breaks_ties_by_pmid() -> None:
    candidates = fuse_rankings(ranked(["3", "2", "1"]), [])
    original = deepcopy(candidates)
    scorer = FixedScorer([-1.0, 7.0, 7.0])

    results = rerank_candidates("question", candidates, scorer)

    assert [row["pmid"] for row in results] == ["1", "2", "3"]
    assert [row.get("reranker_score") for row in results] == [7.0, 7.0, -1.0]
    assert [row["rank"] for row in results] == [1, 2, 3]
    assert scorer.pairs == [("question", row["source_text"]) for row in original]
    for result in results:
        source = next(row for row in original if row["pmid"] == result["pmid"])
        assert result == {
            **source,
            "method": "reranked",
            "rank": result["rank"],
            "reranker_score": result.get("reranker_score"),
        }
    assert candidates == original


@pytest.mark.parametrize("scores", [[], [float("nan")], [float("inf")], [1.0, 2.0]])
def test_invalid_reranker_output_fails_instead_of_returning_hybrid(
    scores: list[float],
) -> None:
    candidates = fuse_rankings(ranked(["1"]), [])
    with pytest.raises(ValueError, match="finite score"):
        rerank_candidates("question", candidates, FixedScorer(scores))


def test_empty_pool_skips_scoring_and_oversized_pool_is_rejected() -> None:
    scorer = FixedScorer([])
    assert rerank_candidates("question", [], scorer) == []
    assert scorer.pairs == []
    candidates = fuse_rankings(ranked([str(i) for i in range(50)]), [])
    with pytest.raises(ValueError, match="50 unique"):
        rerank_candidates("question", candidates + candidates[:1], scorer)


def test_scoring_error_is_reported_as_reranking_failure() -> None:
    class BrokenScorer:
        def score(self, pairs: list[tuple[str, str]]) -> list[float]:
            raise RuntimeError("model unavailable")

    with pytest.raises(ValueError, match="Reranking failed: model unavailable"):
        rerank_candidates("question", fuse_rankings(ranked(["1"]), []), BrokenScorer())
