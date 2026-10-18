from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Self

import numpy as np
import pytest

from pubmed_rag.search import (
    MODEL_ID,
    bm25_search,
    dense_search,
    index_vectors,
    load_selection,
    tokenize,
    validate_cached_vectors,
)


class FakeConnection:
    def __init__(self, query_rows: Sequence[tuple[object, ...]] | None = None) -> None:
        self.indexed: dict[str, tuple[object, ...]] = {}
        self.query_rows = list(query_rows or [])
        self.statements: list[str] = []
        self.ready_count = 2

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, sql: str, params: object = None) -> FakeConnection:
        self.statements.append(sql)
        return self

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def fetchall(self) -> list[tuple[object, ...]]:
        return self.query_rows

    def fetchone(self) -> tuple[int]:
        return (self.ready_count,)


class FakeCursor:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def executemany(
        self, sql: str, rows: Iterator[tuple[object, ...]] | list[tuple[object, ...]]
    ) -> None:
        self.connection.statements.append(sql)
        for row in rows:
            self.connection.indexed[str(row[0])] = row


def install_fake_postgres(
    monkeypatch: pytest.MonkeyPatch, conn: FakeConnection
) -> None:
    psycopg = ModuleType("psycopg")
    psycopg.connect = lambda database_url: conn  # type: ignore[attr-defined]
    pgvector_psycopg = ModuleType("pgvector.psycopg")
    pgvector_psycopg.register_vector = lambda connection: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycopg", psycopg)
    monkeypatch.setitem(sys.modules, "pgvector.psycopg", pgvector_psycopg)


def test_tokenize_preserves_biomedical_identifiers() -> None:
    assert tokenize("HER2, BRCA1-positive; Stage-II") == [
        "her2",
        "brca1",
        "positive",
        "stage",
        "ii",
    ]


def test_bm25_ranking_and_stable_ties() -> None:
    chunks: list[dict[str, Any]] = [
        {"id": "2", "title": "BRCA1 study", "chunk_text": "brca1 mutation"},
        {"id": "1", "title": "BRCA1 overview", "chunk_text": "brca1 mutation"},
        {"id": "3", "title": "HER2 study", "chunk_text": "her2 expression"},
    ]
    for chunk in chunks:
        chunk["meta"] = {"hash": "selected"}
    results = bm25_search(chunks, "BRCA1", 5)
    assert [result["pmid"] for result in results] == ["1", "2", "3"]
    assert len(results) == 3


def test_selection_reuse_fails_when_paper_text_changes(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    selection_path = tmp_path / "selection.json"
    rows = [
        {"pmid": str(i), "title": f"Title {i}", "abstract": f"Abstract {i}"}
        for i in range(1, 201)
    ]
    corpus.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    load_selection(selection_path, corpus)

    rows[0]["abstract"] = "Changed abstract"
    corpus.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="selection changed"):
        load_selection(selection_path, corpus)


def test_bm25_uses_okapi_scores_and_returns_shared_results() -> None:
    chunks = [
        {
            "id": str(i),
            "title": f"Paper {i}",
            "chunk_text": text,
            "meta": {"hash": "selected"},
        }
        for i, text in enumerate(
            [
                "brca1 mutation",
                "brca1 mutation",
                "her2 expression",
                "egfr study",
                "tp53 study",
            ],
            start=1,
        )
    ]
    results = bm25_search(chunks, "BRCA1", 10)
    assert [result["pmid"] for result in results] == ["1", "2", "3", "4", "5"]
    assert [result["score"] for result in results] == pytest.approx(
        [0.3364722366212129, 0.3364722366212129, 0, 0, 0]
    )
    assert results[0] == {
        "pmid": "1",
        "title": "Paper 1",
        "source_text": "brca1 mutation",
        "corpus_hash": "selected",
        "method": "bm25",
        "model_id": None,
        "score": pytest.approx(0.3364722366212129),
        "rank": 1,
    }


def test_selection_reuse_fails_when_member_missing(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    selection_path = tmp_path / "selection.json"
    rows = [
        {"pmid": str(i), "title": f"Title {i}", "abstract": f"Abstract {i}"}
        for i in range(1, 201)
    ]
    corpus.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    load_selection(selection_path, corpus)
    corpus.write_text(
        "".join(json.dumps(row) + "\n" for row in rows[1:]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing or invalid PMIDs"):
        load_selection(selection_path, corpus)


def test_selection_records_rejected_records_and_reasons(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    selection_path = tmp_path / "selection.json"
    rows = [
        {"pmid": str(i), "title": f"Title {i}", "abstract": f"Abstract {i}"}
        for i in range(1, 201)
    ]
    content = "not json\n"
    content += json.dumps({"pmid": "missing-abstract", "title": "No abstract"}) + "\n"
    content += "".join(json.dumps(row) + "\n" for row in rows)
    content += json.dumps(rows[0]) + "\n"
    corpus.write_text(content, encoding="utf-8")

    chunks, selection = load_selection(selection_path, corpus)

    assert len(chunks) == selection["count"] == 200
    assert selection["rejected_count"] == 3
    assert selection["rejection_reasons"]["duplicate PMID"] == 1
    assert any(
        reason.startswith("missing or blank abstract")
        for reason in selection["rejection_reasons"]
    )


def test_cached_vectors_validate_and_keep_parent_corpus_identity(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "vectors.npz"
    parent_hash = "parent-corpus-hash"
    chunks = [{"id": "1", "title": "A", "chunk_text": "Title A\n\nText A"}]
    records = [
        {
            "id": pmid,
            "chunk_text": ("Title A\n\nText A" if pmid == "1" else "Title 2\n\nText 2"),
            "embedding": np.eye(1, 1536, 0, dtype=np.float32).ravel(),
            "meta": {"hash": parent_hash, "embedding_model_id": MODEL_ID},
        }
        for pmid in ("1", "2")
    ]
    np.savez(archive, d=np.array(records, dtype=object))
    archive.with_suffix(".manifest.json").write_text(
        json.dumps({"pmids": ["1", "2"], "count": 2, "hash": parent_hash}),
        encoding="utf-8",
    )

    vectors, actual_parent_hash = validate_cached_vectors(chunks, archive)

    assert list(vectors) == ["1"]
    assert vectors["1"][0]["meta"]["hash"] == parent_hash
    assert actual_parent_hash == parent_hash


def test_cached_vectors_reject_selected_text_mismatch(tmp_path: Path) -> None:
    archive = tmp_path / "vectors.npz"
    record = {
        "id": "1",
        "chunk_text": "Old text",
        "embedding": np.eye(1, 1536, 0, dtype=np.float32).ravel(),
        "meta": {"hash": "parent", "embedding_model_id": MODEL_ID},
    }
    np.savez(archive, d=np.array([record], dtype=object))
    archive.with_suffix(".manifest.json").write_text(
        json.dumps({"pmids": ["1"], "count": 1, "hash": "parent"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="text mismatch"):
        validate_cached_vectors(
            [{"id": "1", "title": "A", "chunk_text": "New text"}], archive
        )


def test_indexing_upserts_by_stable_chunk_id_without_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "vectors.npz"
    vector = np.eye(1, 1536, 0, dtype=np.float32).ravel()
    record = {
        "id": "1",
        "chunk_text": "Title A\n\nText A",
        "embedding": vector,
        "meta": {"hash": "parent", "embedding_model_id": MODEL_ID},
    }
    np.savez(archive, d=np.array([record], dtype=object))
    archive.with_suffix(".manifest.json").write_text(
        json.dumps({"pmids": ["1"], "count": 1, "hash": "parent"}),
        encoding="utf-8",
    )
    conn = FakeConnection()
    install_fake_postgres(monkeypatch, conn)
    chunks = [{"id": "1", "title": "Title A", "chunk_text": "Title A\n\nText A"}]
    selection = {"hash": "selected"}

    assert index_vectors(chunks, selection, archive, "postgresql://fake") == 1
    assert index_vectors(chunks, selection, archive, "postgresql://fake") == 1

    assert len(conn.indexed) == 1
    assert "ON CONFLICT (chunk_id) DO UPDATE" in "\n".join(conn.statements)


def test_dense_query_uses_cosine_order_and_selected_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_rows = [
        ("1", "First", "First source", "selected", MODEL_ID, 0.99),
        ("2", "Second", "Second source", "selected", MODEL_ID, 0.75),
    ]
    conn = FakeConnection(response_rows)
    install_fake_postgres(monkeypatch, conn)
    import openai

    question_vector = np.eye(1, 1536, 0, dtype=np.float32).ravel().tolist()
    embeddings = SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(
            model=kwargs["model"],
            data=[SimpleNamespace(embedding=question_vector)],
        )
    )
    monkeypatch.setattr(
        openai, "OpenAI", lambda **kwargs: SimpleNamespace(embeddings=embeddings)
    )

    results = dense_search(
        "question",
        10,
        "postgresql://fake",
        "selected",
        True,
        query_budget_usd=0.001,
        expected_count=2,
    )

    assert [row["pmid"] for row in results] == ["1", "2"]
    sql = " ".join(conn.statements)
    assert "embedding <=> %s" in sql
    assert "WHERE corpus_hash = %s AND model_id = %s" in sql


@pytest.mark.parametrize("budget", [None, 0.0, -1.0, float("inf"), float("nan")])
def test_dense_search_requires_a_finite_positive_budget_before_paid_calls(
    monkeypatch: pytest.MonkeyPatch, budget: float | None
) -> None:
    import openai

    def reject_paid_client(**kwargs: Any) -> None:
        pytest.fail("A rejected budget must not create a paid client")

    monkeypatch.setattr(openai, "OpenAI", reject_paid_client)
    with pytest.raises(ValueError, match="budget"):
        dense_search(
            "question",
            5,
            "postgresql://fake",
            "selected",
            True,
            query_budget_usd=budget,
        )


@pytest.mark.parametrize("ready_count", [0, 1])
def test_dense_search_rejects_incomplete_index_without_paid_calls(
    monkeypatch: pytest.MonkeyPatch, ready_count: int
) -> None:
    conn = FakeConnection()
    conn.ready_count = ready_count
    install_fake_postgres(monkeypatch, conn)
    import openai

    def reject_paid_client(**kwargs: Any) -> None:
        pytest.fail("An incomplete index must not create a paid client")

    monkeypatch.setattr(openai, "OpenAI", reject_paid_client)
    with pytest.raises(ValueError, match="index"):
        dense_search(
            "question",
            5,
            "postgresql://fake",
            "selected",
            True,
            query_budget_usd=0.001,
            expected_count=2,
        )


@pytest.mark.parametrize("query,budget", [("question", 1e-12), ("   ", 0.001)])
def test_dense_search_rejects_unusable_query_without_paid_calls(
    monkeypatch: pytest.MonkeyPatch, query: str, budget: float
) -> None:
    install_fake_postgres(monkeypatch, FakeConnection())
    import openai

    def reject_paid_client(**kwargs: Any) -> None:
        pytest.fail("An unusable or over-budget query must not create a paid client")

    monkeypatch.setattr(openai, "OpenAI", reject_paid_client)
    with pytest.raises(ValueError, match="budget|blank"):
        dense_search(
            query, 5, "postgresql://fake", "selected", True, query_budget_usd=budget
        )
