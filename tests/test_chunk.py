import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from pubmed_rag import chunk
from pubmed_rag.chunk import atomic_save, load_chunks, save_embeddings, subset_manifest


def write_corpus(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def paper(pmid, abstract="An abstract"):
    return {"pmid": pmid, "title": "A title", "abstract": abstract}


@pytest.fixture(autouse=True)
def offline_tokenizer(monkeypatch):
    monkeypatch.setattr(
        "pubmed_rag.chunk.tiktoken.encoding_for_model",
        lambda model: SimpleNamespace(encode=lambda text, **kwargs: list(text)),
    )


def test_selection_skips_invalid_and_duplicate_records(tmp_path, caplog):
    corpus = tmp_path / "corpus.jsonl"
    write_corpus(
        corpus,
        [paper(None), paper("1", " "), paper("2"), paper("2"), paper("3"), paper("4")],
    )
    chunks = list(load_chunks(corpus, 2))
    assert [chunk["id"] for chunk in chunks] == ["2", "3"]
    assert chunks[0]["chunk_text"] == "A title\n\nAn abstract"
    assert "duplicate PMID" in caplog.text
    assert "blank abstract" in caplog.text


def test_unlimited_selection(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    write_corpus(corpus, [paper("1"), paper("2", " "), paper("1"), paper("3")])
    assert [record["id"] for record in load_chunks(corpus, -1)] == ["1", "3"]


@pytest.mark.parametrize("record", [None, [], "invalid", 42])
def test_non_object_records_are_rejected(tmp_path, caplog, record):
    corpus = tmp_path / "corpus.jsonl"
    write_corpus(corpus, [record, paper("1")])
    assert [item["id"] for item in load_chunks(corpus, -1)] == ["1"]
    assert "record must be a JSON object" in caplog.text


@pytest.mark.parametrize("limit", [0, -2])
def test_invalid_selection_limit(tmp_path, limit):
    with pytest.raises(ValueError, match="must be -1 or greater than zero"):
        list(load_chunks(tmp_path / "unused.jsonl", limit))


def test_unlimited_cli_and_manifest_resume(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.jsonl"
    cache = tmp_path / "vectors.npz"
    write_corpus(corpus, [paper("1"), paper("2")])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rag-chunk",
            "--corpus-path",
            str(corpus),
            "--embedding-path",
            str(cache),
            "--limit-abstracts",
            "-1",
        ],
    )
    monkeypatch.setattr(chunk, "OpenAI", Mock())
    save = Mock()
    monkeypatch.setattr(chunk, "save_embeddings", save)
    chunk.main()
    manifest = json.loads(cache.with_suffix(".manifest.json").read_text())
    assert manifest["count"] == 2
    assert manifest["pmids"] == ["1", "2"]
    write_corpus(corpus, [paper("2"), paper("1")])
    chunk.main()
    assert [record["id"] for record in save.call_args.args[0]] == ["1", "2"]
    save.reset_mock()
    write_corpus(corpus, [paper("1"), paper("2"), paper("3")])
    chunk.main()
    assert [record["id"] for record in save.call_args.args[0]] == ["1", "2", "3"]
    refreshed = json.loads(cache.with_suffix(".manifest.json").read_text())
    assert refreshed["count"] == 3
    assert refreshed["hash"] != manifest["hash"]


def test_subset_hash_and_frozen_membership(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    write_corpus(corpus, [paper("1"), paper("2"), paper("3")])
    original = subset_manifest(list(load_chunks(corpus, 2)))
    write_corpus(corpus, [paper("new"), paper("2"), paper("1"), paper("3", "Changed")])
    selected = list(load_chunks(corpus, 2, original["pmids"]))
    selected.sort(key=lambda chunk: original["pmids"].index(chunk["id"]))
    assert subset_manifest(selected) == original
    selected[0]["chunk_text"] += " changed"
    assert subset_manifest(selected)["hash"] != original["hash"]
    write_corpus(corpus, [paper("1")])
    with pytest.raises(ValueError, match="missing or invalid PMIDs"):
        list(load_chunks(corpus, 2, original["pmids"]))


def prepared_chunks(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    write_corpus(corpus, [paper("1"), paper("2")])
    chunks = list(load_chunks(corpus, 2))
    subset_manifest(chunks)
    return chunks


def response(*indices):
    return SimpleNamespace(
        data=[SimpleNamespace(index=index, embedding=[3.0, 4.0]) for index in indices]
    )


def test_resume_after_failed_batch_without_repeating_saved_requests(tmp_path):
    chunks = prepared_chunks(tmp_path)
    cache = tmp_path / "vectors.npz"
    client = Mock()
    client.embeddings.create.side_effect = [response(0), RuntimeError("offline")]
    with pytest.raises(RuntimeError, match="offline"):
        save_embeddings(chunks, client, cache, batch_size=1, embed_dim=2)
    with np.load(cache, allow_pickle=True) as saved:
        assert [record["id"] for record in saved["d"]] == ["1"]
    client.embeddings.create.side_effect = None
    client.embeddings.create.return_value = response(0)
    client.embeddings.create.reset_mock()
    save_embeddings(chunks, client, cache, batch_size=1, embed_dim=2)
    assert client.embeddings.create.call_args.kwargs["input"] == [
        chunks[1]["chunk_text"]
    ]
    assert client.embeddings.create.call_args.kwargs["dimensions"] == 2
    client.embeddings.create.reset_mock()
    save_embeddings(chunks, client, cache, embed_dim=2)
    client.embeddings.create.assert_not_called()
    save_embeddings(
        chunks,
        client,
        cache,
        batch_size=1,
        embed_dim=2,
        embed_model="text-embedding-3-large",
    )
    assert client.embeddings.create.call_count == 2


def test_changed_corpus_rebuilds_cache_and_preserves_it_on_failure(tmp_path):
    chunks = prepared_chunks(tmp_path)
    cache = tmp_path / "vectors.npz"
    client = Mock()
    client.embeddings.create.return_value = response(0, 1)
    save_embeddings(chunks, client, cache, embed_dim=2)
    original_cache = cache.read_bytes()

    chunks[1]["chunk_text"] += " changed"
    manifest = subset_manifest(chunks)
    client.embeddings.create.side_effect = RuntimeError("offline")
    with pytest.raises(RuntimeError, match="offline"):
        save_embeddings(chunks, client, cache, embed_dim=2)
    assert cache.read_bytes() == original_cache

    client.embeddings.create.side_effect = None
    client.embeddings.create.reset_mock()
    save_embeddings(chunks, client, cache, embed_dim=2)
    assert client.embeddings.create.call_args.kwargs["input"] == [
        c["chunk_text"] for c in chunks
    ]
    with np.load(cache, allow_pickle=True) as saved:
        assert len(saved["d"]) == 2
        assert all(record["meta"]["hash"] == manifest["hash"] for record in saved["d"])


def test_response_indices_and_incomplete_response(tmp_path):
    chunks = prepared_chunks(tmp_path)
    cache = tmp_path / "vectors.npz"
    client = Mock()
    client.embeddings.create.return_value = response(0)
    with pytest.raises(ValueError, match="does not match"):
        save_embeddings(chunks, client, cache, embed_dim=2)
    assert not cache.exists()
    client.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(index=1, embedding=[0.0, 2.0]),
            SimpleNamespace(index=0, embedding=[3.0, 0.0]),
        ]
    )
    save_embeddings(chunks, client, cache, embed_dim=2)
    with np.load(cache, allow_pickle=True) as saved:
        assert saved["d"][0]["embedding"].tolist() == [1.0, 0.0]
        assert saved["d"][1]["embedding"].tolist() == [0.0, 1.0]


def test_failed_atomic_write_preserves_checkpoint(tmp_path):
    cache = tmp_path / "vectors.npz"
    cache.write_bytes(b"previous checkpoint")

    def fail(fp):
        fp.write(b"incomplete replacement")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        atomic_save(cache, fail)
    assert cache.read_bytes() == b"previous checkpoint"
    assert list(tmp_path.iterdir()) == [cache]
