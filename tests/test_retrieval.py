import sys
from types import ModuleType, SimpleNamespace

import pytest
from test_search import FakeConnection, install_fake_postgres

from pubmed_rag.retrieval import retrieve


def test_bm25_mode_retains_evaluation_depth_and_selects_five_context_papers() -> None:
    chunks = [
        {
            "id": f"{i:03}",
            "title": f"Paper {i}",
            "chunk_text": f"Topic {i}",
            "meta": {"hash": "selected"},
        }
        for i in range(25)
    ]
    run = retrieve("topic", "B", chunks, {"hash": "selected", "count": 25})
    assert len(run.results) == 25
    assert run.configuration == "B"
    assert [row["pmid"] for row in run.context] == ["000", "001", "002", "003", "004"]


@pytest.mark.parametrize("method", ["A", "C", "D", "dense", "hybrid", "reranked"])
def test_dense_modes_preserve_paid_query_consent(method: str) -> None:
    with pytest.raises(ValueError, match="paid question embedding"):
        retrieve("topic", method, [], {"hash": "selected", "count": 0})


def test_c_and_d_share_top_50_candidates_and_d_retains_twenty_for_labeling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        {
            "id": f"{i:03}",
            "title": f"Paper {i}",
            "chunk_text": f"Topic {i}",
            "meta": {"hash": "selected"},
        }
        for i in range(75)
    ]
    rows = [
        (
            row["id"],
            row["title"],
            row["chunk_text"],
            "selected",
            "text-embedding-3-small_1536",
            1.0 - i / 100,
        )
        for i, row in enumerate(chunks[::-1][:50])
    ]
    conn = FakeConnection(rows)
    conn.ready_count = 75
    install_fake_postgres(monkeypatch, conn)
    import openai

    requests: list[object] = []

    def embed(**kwargs: object) -> SimpleNamespace:
        requests.append(kwargs)
        return SimpleNamespace(
            model="text-embedding-3-small",
            data=[SimpleNamespace(embedding=[1.0] + [0.0] * 1535)],
        )

    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kwargs: SimpleNamespace(embeddings=SimpleNamespace(create=embed)),
    )
    pairs: list[tuple[str, str]] = []

    class FixedCrossEncoder:
        def __init__(self, *args: object, **kwargs: object) -> None:
            assert kwargs["model_kwargs"] == {"disable_mmap": True}
            self.tokenizer = SimpleNamespace(truncation_side="right")

        def predict(
            self, inputs: list[tuple[str, str]], **kwargs: object
        ) -> list[float]:
            pairs.extend(inputs)
            return [float(i) for i in range(len(inputs))]

    module = ModuleType("sentence_transformers")
    module.CrossEncoder = FixedCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    args = ("topic", chunks, {"hash": "selected", "count": 75})
    c = retrieve(args[0], "C", args[1], args[2], "postgresql://fake", True, 0.001)
    d = retrieve(args[0], "D", args[1], args[2], "postgresql://fake", True, 0.001)
    assert len(c.results) == len(d.results) == 50
    assert [text for _, text in pairs] == [row["source_text"] for row in c.results]
    assert [row["pmid"] for row in d.results] == [
        row["pmid"] for row in c.results[::-1]
    ]
    assert len(d.results[:20]) == 20
    assert len(d.context) == 5
    assert len(requests) == 2
    assert d.reranking is not None
    assert d.reranking.candidate_count == 50
    assert d.reranking.device == "cpu"
    for result in d.results:
        source = next(row for row in c.results if row["pmid"] == result["pmid"])
        assert result.get("sources") == source.get("sources")
        assert result.get("fused_rank") == source["rank"]


def test_unavailable_reranker_fails_before_any_paid_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ValueError, match="Cannot load the pinned CPU reranker"):
        retrieve(
            "topic",
            "D",
            [],
            {"hash": "selected", "count": 0},
            "postgresql://must-not-connect",
            True,
            0.001,
        )


@pytest.mark.parametrize("budget", [None, 0.0, -1.0, float("nan"), 1e-12])
def test_d_checks_budget_before_loading_model(
    monkeypatch: pytest.MonkeyPatch,
    budget: float | None,
) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ValueError, match="budget"):
        retrieve(
            "topic",
            "D",
            [],
            {"hash": "selected", "count": 0},
            "postgresql://must-not-connect",
            True,
            budget,
        )
