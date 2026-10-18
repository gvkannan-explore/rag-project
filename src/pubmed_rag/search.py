"""Frozen-corpus lexical and persistent-vector retrieval."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from pubmed_rag.chunk import load_chunks, subset_manifest

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
SELECTION_SIZE = 200
MODEL_ID = "text-embedding-3-small_1536"


class RetrievalResult(TypedDict):
    pmid: str
    title: str
    source_text: str
    corpus_hash: str
    model_id: str | None
    method: Literal["bm25", "dense"]
    score: float
    rank: int


def _unique_strings(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    items: list[str] = []
    for item in cast(list[Any], value):
        if not isinstance(item, str):
            return None
        items.append(item)
    if not items or len(set(items)) != len(items):
        return None
    return items


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens; retain biomedical forms such as HER2."""
    return TOKEN_RE.findall(text.lower())


def load_selection(
    path: Path, corpus_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and validate a fixed selection, creating the initial 200-paper view."""
    if path.exists():
        selection: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        pmids = _unique_strings(selection.get("pmids"))
        if pmids is None:
            raise ValueError(f"Invalid frozen selection in {path}")
        chunks = list(load_chunks(corpus_path, selected_pmids=pmids))
        manifest = subset_manifest(chunks)
        if manifest["pmids"] != pmids or any(
            manifest[key] != selection.get(key) for key in ("count", "hash")
        ):
            raise ValueError("Frozen corpus selection changed; refusing replacement")
        return chunks, selection

    rejections: list[dict[str, str]] = []
    all_chunks = list(
        load_chunks(corpus_path, limit_abstracts=-1, rejections=rejections)
    )
    chunks = all_chunks[:SELECTION_SIZE]
    if len(chunks) != SELECTION_SIZE:
        raise ValueError(f"Expected {SELECTION_SIZE} valid papers; found {len(chunks)}")
    manifest = subset_manifest(chunks)
    rejection_reasons = Counter(
        "duplicate PMID" if reason.startswith("duplicate PMID") else reason
        for item in rejections
        for reason in (item["reason"],)
    )
    selection = {
        **manifest,
        "rejected_count": len(rejections),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "parent_corpus": str(corpus_path),
        "selection_policy": "first 200 valid unique records in source order",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    return chunks, selection


def bm25_search(
    chunks: list[dict[str, Any]], query: str, limit: int
) -> list[RetrievalResult]:
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    tokenized: list[list[str]] = [tokenize(c["chunk_text"]) for c in chunks]
    lengths = [len(tokens) for tokens in tokenized]
    average_length = sum(lengths) / len(lengths) if lengths else 0.0
    doc_freq: Counter[str] = Counter()
    frequencies: list[Counter[str]] = [Counter(tokens) for tokens in tokenized]
    for counts in frequencies:
        doc_freq.update(counts.keys())
    count = len(chunks)
    k1, b = 1.5, 0.75
    query_tokens = tokenize(query)
    scores: list[float] = []
    for length, counts in zip(lengths, frequencies, strict=True):
        score = 0.0
        for token in query_tokens:
            frequency = counts[token]
            if not frequency:
                continue
            inverse_frequency = math.log1p(
                (count - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5)
            )
            normalization = (
                k1 * (1 - b + b * length / average_length) if average_length else k1
            )
            score += (
                inverse_frequency * frequency * (k1 + 1) / (frequency + normalization)
            )
        scores.append(score)
    ranked = sorted(
        range(len(chunks)), key=lambda i: (-float(scores[i]), chunks[i]["id"])
    )
    return [
        {
            "pmid": chunks[i]["id"],
            "title": chunks[i]["title"],
            "source_text": chunks[i]["chunk_text"],
            "corpus_hash": chunks[i]["meta"]["hash"],
            "model_id": None,
            "method": "bm25",
            "score": float(scores[i]),
            "rank": rank,
        }
        for rank, i in enumerate(ranked[:limit], start=1)
    ]


def validate_cached_vectors(
    chunks: list[dict[str, Any]], archive: Path
) -> tuple[dict[str, tuple[dict[str, Any], Any]], str]:
    """Validate the complete archive and return selected vectors plus its parent hash."""
    import numpy as np

    if not archive.exists():
        raise FileNotFoundError(f"Embedding archive not found: {archive}")
    expected = {chunk["id"]: chunk for chunk in chunks}
    archive_manifest_path = archive.with_suffix(".manifest.json")
    if not archive_manifest_path.exists():
        raise ValueError(f"Embedding manifest not found: {archive_manifest_path}")
    archive_manifest: dict[str, Any] = json.loads(
        archive_manifest_path.read_text(encoding="utf-8")
    )
    parent_hash = archive_manifest.get("hash")
    parent_pmids = _unique_strings(archive_manifest.get("pmids"))
    if (
        not isinstance(parent_hash, str)
        or not parent_hash
        or parent_pmids is None
        or len(parent_pmids) != archive_manifest.get("count")
    ):
        raise ValueError("Embedding manifest has no corpus identity")
    with np.load(archive, allow_pickle=True) as saved:
        if "d" not in saved:
            raise ValueError("Embedding archive does not contain records")
        records: list[dict[str, Any]] = saved["d"].tolist()

    vectors: dict[str, tuple[dict[str, Any], Any]] = {}
    archive_pmids: list[str] = []
    for record in records:
        pmid = record.get("id")
        if not isinstance(pmid, str):
            raise TypeError("Embedding archive contains a record without a PMID")
        archive_pmids.append(pmid)
        vector = np.asarray(record.get("embedding"), dtype=np.float32)
        if (
            pmid in archive_pmids[:-1]
            or record.get("meta", {}).get("hash") != parent_hash
            or record.get("meta", {}).get("embedding_model_id") != MODEL_ID
            or vector.shape != (1536,)
            or not np.isfinite(vector).all()
            or not np.isclose(np.linalg.norm(vector), 1.0)
        ):
            raise ValueError(
                f"Embedding cache mismatch or invalid vector for PMID {pmid}"
            )
        if pmid in expected:
            chunk = expected[pmid]
            if record.get("chunk_text") != chunk["chunk_text"]:
                raise ValueError(f"Embedding text mismatch for selected PMID {pmid}")
            vectors[pmid] = (record, vector)
    if archive_pmids != parent_pmids:
        raise ValueError(
            "Embedding records do not match manifest PMID membership and order"
        )
    missing = set(expected) - set(vectors)
    if missing:
        raise ValueError(
            f"Cached embeddings missing selected PMIDs: {sorted(missing)[:10]}"
        )

    return vectors, parent_hash


def index_vectors(
    chunks: list[dict[str, Any]],
    selection: dict[str, Any],
    archive: Path,
    database_url: str,
) -> int:
    """Validate cached archive identity, then idempotently upsert selected vectors."""
    import psycopg  # pyright: ignore[reportMissingImports]
    from pgvector.psycopg import register_vector  # pyright: ignore[reportUnknownVariableType]

    psycopg_any = cast(Any, psycopg)
    register_vector = cast(Any, register_vector)
    expected = {chunk["id"]: chunk for chunk in chunks}
    vectors, parent_hash = validate_cached_vectors(chunks, archive)
    with psycopg_any.connect(database_url) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        conn.execute("""CREATE TABLE IF NOT EXISTS paper_chunks (
            chunk_id TEXT PRIMARY KEY, pmid TEXT NOT NULL UNIQUE, title TEXT NOT NULL,
            source_text TEXT NOT NULL, embedding VECTOR(1536) NOT NULL,
            corpus_hash TEXT NOT NULL, model_id TEXT NOT NULL,
            source_corpus_hash TEXT NOT NULL
        )""")
        with conn.cursor() as cursor:
            cursor.executemany(
                """INSERT INTO paper_chunks
                (chunk_id, pmid, title, source_text, embedding, corpus_hash, model_id, source_corpus_hash)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (chunk_id) DO UPDATE SET title=excluded.title,
                source_text=excluded.source_text, embedding=excluded.embedding,
                corpus_hash=excluded.corpus_hash, model_id=excluded.model_id,
                source_corpus_hash=excluded.source_corpus_hash""",
                [
                    (
                        pmid,
                        pmid,
                        expected[pmid]["title"],
                        expected[pmid]["chunk_text"],
                        vector,
                        selection["hash"],
                        MODEL_ID,
                        parent_hash,
                    )
                    for pmid, (_, vector) in vectors.items()
                ],
            )
    return len(vectors)


def dense_search(
    query: str,
    limit: int,
    database_url: str,
    corpus_hash: str,
    allow_paid_query: bool,
) -> list[RetrievalResult]:
    """Embed one query and ask Postgres for exact cosine nearest neighbors."""
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    if not allow_paid_query:
        raise ValueError(
            "Dense search makes one paid question embedding request; "
            "pass --allow-paid-query to run this single-request trial"
        )
    import psycopg  # pyright: ignore[reportMissingImports]
    from openai import OpenAI
    from pgvector.psycopg import register_vector  # pyright: ignore[reportUnknownVariableType]

    psycopg_any = cast(Any, psycopg)
    register_vector = cast(Any, register_vector)
    response = OpenAI().embeddings.create(
        model="text-embedding-3-small", input=query, dimensions=1536
    )
    if response.model != "text-embedding-3-small" or len(response.data) != 1:
        raise ValueError("Embedding response model or item count did not match request")
    import numpy as np

    vector = np.asarray(response.data[0].embedding, dtype=np.float32)
    if vector.shape != (1536,) or not np.isfinite(vector).all():
        raise ValueError("Question embedding has invalid dimensions or values")
    with psycopg_any.connect(database_url) as conn:
        register_vector(conn)
        rows: list[tuple[Any, ...]] = conn.execute(
            """SELECT pmid, title, source_text, corpus_hash, model_id,
            1 - (embedding <=> %s) AS score FROM paper_chunks
            WHERE corpus_hash = %s AND model_id = %s
            ORDER BY embedding <=> %s, pmid LIMIT %s""",
            (vector, corpus_hash, MODEL_ID, vector, limit),
        ).fetchall()
    return [
        {
            "pmid": row[0],
            "title": row[1],
            "source_text": row[2],
            "corpus_hash": row[3],
            "model_id": row[4],
            "method": "dense",
            "score": float(row[5]),
            "rank": i,
        }
        for i, row in enumerate(rows, start=1)
    ]


def provenance(result: RetrievalResult) -> str:
    return (
        f"corpus={result['corpus_hash']} method={result['method']} "
        f"model={result['model_id'] or 'none'}"
    )


def main(args: Any) -> None:
    chunks, selection = load_selection(Path(args.selection), Path(args.corpus))
    if args.command == "index":
        count = index_vectors(chunks, selection, Path(args.archive), args.database_url)
        print(f"Indexed {count} cached vectors for corpus {selection['hash']}")
        return
    results = (
        bm25_search(chunks, args.query, args.limit)
        if args.method == "bm25"
        else dense_search(
            args.query,
            args.limit,
            args.database_url,
            selection["hash"],
            args.allow_paid_query,
        )
    )
    for result in results:
        print(f"{result['rank']}. {result['title']} (PMID {result['pmid']})")
        print(f"   score={result['score']:.6f} {provenance(result)}")
        print(f"   {result['source_text'].replace(chr(10), ' ')[:240]}")
