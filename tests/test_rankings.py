from copy import deepcopy

import pytest

from pubmed_rag.rankings import fuse_rankings
from pubmed_rag.search import MODEL_ID, RetrievalResult


def ranked(pmids: list[str], method: str = "dense") -> list[RetrievalResult]:
    return [
        {
            "pmid": pmid,
            "title": f"Paper {pmid}",
            "source_text": f"Paper {pmid}\n\nAbstract {pmid}",
            "corpus_hash": "selected",
            "model_id": MODEL_ID if method == "dense" else None,
            "method": method,  # type: ignore[typeddict-item]
            "score": 100.0 - rank,
            "rank": rank,
        }
        for rank, pmid in enumerate(pmids, 1)
    ]


def test_fusion_adds_overlap_and_keeps_single_source_candidates() -> None:
    dense = ranked(["1", "2", "3"])
    lexical = ranked(["2", "4"], "bm25")
    original = deepcopy((dense, lexical))

    results = fuse_rankings(dense, lexical)

    assert [row["pmid"] for row in results] == ["2", "1", "4", "3"]
    # Worked RRF example, k=60: 1/62 + 1/61, 1/61, 1/62, 1/63.
    assert [row["score"] for row in results] == pytest.approx(
        [
            0.03252247488101534,
            0.01639344262295082,
            0.016129032258064516,
            0.015873015873015872,
        ]
    )
    assert results[0].get("sources") == {
        "dense": {"rank": 2, "score": 98.0, "model_id": MODEL_ID},
        "bm25": {"rank": 1, "score": 99.0, "model_id": None},
    }
    assert results[0].get("fused_score") == results[0]["score"]
    assert results[0].get("fused_rank") == results[0]["rank"] == 1
    assert (dense, lexical) == original


def test_fusion_ties_and_top_50_use_source_ranks_not_input_order() -> None:
    dense = ranked([f"{i:03}" for i in range(60)])
    lexical = ranked(["001", "000"], "bm25")
    results = fuse_rankings(dense, lexical)
    assert [row["pmid"] for row in results[:2]] == ["000", "001"]
    assert len(results) == 50
    assert results == fuse_rankings(dense[::-1], lexical[::-1])
    assert fuse_rankings([], []) == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("corpus_hash", "other"),
        ("source_text", "Changed text"),
    ],
)
def test_fusion_rejects_incompatible_sources(field: str, value: str) -> None:
    dense = ranked(["1"])
    lexical = ranked(["1"], "bm25")
    lexical[0][field] = value  # type: ignore[literal-required]
    with pytest.raises(ValueError, match="identity"):
        fuse_rankings(dense, lexical)
