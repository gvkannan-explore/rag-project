"""Opt-in Postgres checks. All embeddings are synthetic; no model calls occur."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from pubmed_rag.search import MODEL_ID, RetrievalResult, dense_search, index_vectors

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("RAG_TEST_DATABASE_URL"),
        reason="Set RAG_TEST_DATABASE_URL for live Postgres checks",
    ),
]


@pytest.fixture
def vector_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    import openai
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    base_url = os.environ["RAG_TEST_DATABASE_URL"]
    schema = "test_vectors_" + uuid4().hex
    with psycopg.connect(base_url, autocommit=True) as conn:
        version_row = conn.execute("SHOW server_version_num").fetchone()
        assert version_row is not None
        version = int(version_row[0])
        assert 160000 <= version < 170000
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    url = make_conninfo(base_url, options=f"-csearch_path={schema},public")

    def embedding_client(**kwargs: Any) -> SimpleNamespace:
        assert kwargs["max_retries"] == 0

        def create(**request: Any) -> SimpleNamespace:
            assert request["model"] == "text-embedding-3-small"
            assert request["dimensions"] == 1536
            vector = [1.0] + [0.0] * 1535
            return SimpleNamespace(
                model=request["model"], data=[SimpleNamespace(embedding=vector)]
            )

        return SimpleNamespace(embeddings=SimpleNamespace(create=create))

    monkeypatch.setattr(openai, "OpenAI", embedding_client)
    chunks: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for i in range(12, 0, -1):
        chunk = {
            "id": f"{i:02}",
            "title": f"Synthetic {i}",
            "chunk_text": f"Synthetic source {i}",
        }
        chunks.append(chunk)
        vector = np.zeros(1536, dtype=np.float32)
        vector[:2] = (
            (1, 0)
            if i in (1, 2)
            else (0.8, 0.6)
            if i == 3
            else (-1, 0)
            if i == 4
            else (0, 1)
        )
        records.append(
            {
                **chunk,
                "embedding": vector,
                "meta": {"hash": "parent", "embedding_model_id": MODEL_ID},
            }
        )
    archive = tmp_path / "vectors.npz"
    np.savez(archive, d=np.array(records, dtype=object))
    archive.with_suffix(".manifest.json").write_text(
        json.dumps({"pmids": [c["id"] for c in chunks], "count": 12, "hash": "parent"}),
        encoding="utf-8",
    )
    try:
        assert index_vectors(chunks, {"hash": "selected"}, archive, url) == 12
        assert index_vectors(chunks, {"hash": "selected"}, archive, url) == 12
        with psycopg.connect(url) as conn:
            conn.execute("""INSERT INTO paper_chunks
                SELECT '90', '90', title, source_text, embedding,
                       'other-corpus', model_id, source_corpus_hash
                FROM paper_chunks WHERE pmid = '01'""")
            conn.execute("""INSERT INTO paper_chunks
                SELECT '91', '91', title, source_text, embedding,
                       corpus_hash, 'other-model', source_corpus_hash
                FROM paper_chunks WHERE pmid = '01'""")
        yield url
    finally:
        with psycopg.connect(base_url, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
            )


def retrieve(url: str, limit: int = 10) -> list[RetrievalResult]:
    return list(
        dense_search(
            "synthetic question",
            limit,
            url,
            "selected",
            True,
            query_budget_usd=0.001,
            expected_count=12,
        )
    )


def test_known_vectors_scores_ties_filtering_and_repeat_indexing(
    vector_store: str,
) -> None:
    results = retrieve(vector_store)
    assert [row["pmid"] for row in results] == [
        "01",
        "02",
        "03",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
    ]
    assert [row["score"] for row in results] == pytest.approx(
        [1, 1, 0.8, 0, 0, 0, 0, 0, 0, 0]
    )
    full_results = retrieve(vector_store, 50)
    assert len(full_results) == 12
    assert full_results[-1]["pmid"] == "04"
    assert full_results[-1]["score"] == pytest.approx(-1)
    assert all(
        row["corpus_hash"] == "selected" and row["model_id"] == MODEL_ID
        for row in results
    )


@pytest.mark.skipif(
    os.environ.get("RAG_TEST_RESTART") != "1",
    reason="Set RAG_TEST_RESTART=1 to restart the Compose database",
)
def test_vectors_survive_database_restart(vector_store: str) -> None:
    import psycopg

    before = retrieve(vector_store, 50)
    subprocess.run(
        ["docker", "compose", "restart", "db"],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        timeout=60,
    )
    deadline = time.monotonic() + 30
    while True:
        try:
            with psycopg.connect(vector_store, connect_timeout=2):
                break
        except psycopg.OperationalError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.2)
    assert retrieve(vector_store, 50) == before
