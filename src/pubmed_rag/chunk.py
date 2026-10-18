from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from openai import OpenAI as OpenAIType


class _TiktokenProxy:
    @staticmethod
    def encoding_for_model(model: str) -> Any:
        import tiktoken

        return tiktoken.encoding_for_model(model)


tiktoken = _TiktokenProxy()


class OpenAI:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from openai import OpenAI as OpenAIClient

        return OpenAIClient(*args, **kwargs)


logger = logging.getLogger(__name__)

MAX_TOKEN_LENGTH = 8191
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
BATCH_SIZE = 128
LIMIT_ABSTRACTS = 500


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero")
    return number


def abstract_limit(value: str) -> int:
    number = int(value)
    if number != -1 and number <= 0:
        raise argparse.ArgumentTypeError("Value must be -1 or greater than zero")
    return number


def parsed_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rag-chunk", description="Select PubMed chunks and cache their embeddings."
    )
    parser.add_argument(
        "--corpus-path", type=Path, required=True, help="Corpus JSONL file"
    )
    parser.add_argument(
        "--embedding-path", type=Path, required=True, help="Output .npz file"
    )
    parser.add_argument("--batch-size", type=positive_int, default=BATCH_SIZE)
    parser.add_argument(
        "--limit-abstracts",
        type=abstract_limit,
        default=LIMIT_ABSTRACTS,
        help="Number of usable abstracts to select; -1 selects the entire corpus",
    )
    return parser.parse_args()


def load_chunks(
    corpus_path: Path,
    limit_abstracts: int = LIMIT_ABSTRACTS,
    selected_pmids: list[str] | None = None,
    rejections: list[dict[str, str]] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield valid, unique records; a limit of -1 reads the entire corpus."""
    if limit_abstracts != -1 and limit_abstracts <= 0:
        raise ValueError("limit_abstracts must be -1 or greater than zero")
    wanted = set(selected_pmids) if selected_pmids is not None else None
    seen: set[str] = set()
    with corpus_path.open(encoding="utf-8") as corpus:
        for line_number, line in enumerate(corpus, start=1):
            try:
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    raise TypeError("record must be a JSON object")
                record = cast(dict[str, object], parsed)
                pmid = record.get("pmid")
                if not isinstance(pmid, str) or not pmid.strip():
                    raise ValueError("missing or blank PMID")
                pmid = pmid.strip()
                if wanted is not None and pmid not in wanted:
                    continue
                if pmid in seen:
                    raise ValueError(f"duplicate PMID {pmid}")
                abstract = record.get("abstract")
                if not isinstance(abstract, str) or not abstract.strip():
                    raise ValueError(f"missing or blank abstract for PMID {pmid}")
                title = record.get("title")
                if not isinstance(title, str) or not title.strip():
                    raise ValueError(f"missing or blank title for PMID {pmid}")
            except (TypeError, ValueError) as exc:
                logger.warning("Rejected line %s: %s", line_number, exc)
                if rejections is not None:
                    rejections.append({"line": str(line_number), "reason": str(exc)})
                continue
            seen.add(pmid)
            yield {
                "id": pmid,
                "title": title,
                "abstract": abstract,
                "chunk_text": f"{title}\n\n{abstract}",
                "meta": {
                    "pmid": pmid,
                    **{
                        key: record.get(key)
                        for key in (
                            "journal",
                            "pub_year",
                            "pub_types",
                            "mesh_terms",
                            "doi",
                        )
                    },
                },
            }
            if seen == wanted or (wanted is None and len(seen) == limit_abstracts):
                return
    if wanted is not None and seen != wanted:
        raise ValueError(
            f"Frozen subset has missing or invalid PMIDs: {sorted(wanted - seen)}"
        )


def subset_manifest(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Hash the ordered PMIDs and exact text sent for embedding."""
    digest = hashlib.sha256()
    for chunk in chunks:
        payload = json.dumps(
            {"pmid": chunk["id"], "chunk_text": chunk["chunk_text"]},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update((payload + "\n").encode("utf-8"))
    corpus_hash = digest.hexdigest()
    for chunk in chunks:
        chunk["meta"]["hash"] = corpus_hash
    return {
        "pmids": [c["id"] for c in chunks],
        "count": len(chunks),
        "hash": corpus_hash,
    }


def atomic_save(path: Path, write: Callable[[IO[bytes]], object]) -> None:
    """Replace a local checkpoint only after its complete contents reach disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, mode="wb", delete=False
        ) as fp:
            temporary = Path(fp.name)
            write(fp)
            fp.flush()
            os.fsync(fp.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_embeddings(
    chunks: list[dict[str, Any]],
    oai_client: OpenAIType,
    save_embed: Path,
    batch_size: int = BATCH_SIZE,
    embed_model: str = EMBED_MODEL,
    embed_dim: int = EMBED_DIM,
) -> None:
    import numpy as np
    from tqdm import tqdm

    ## Check input validation
    if batch_size <= 0 or embed_dim <= 0:
        raise ValueError("batch_size and embed_dim must be greater than zero")
    if save_embed.suffix != ".npz":
        raise ValueError("Embedding output must have a .npz extension")
    if not chunks:
        raise ValueError("No usable chunks to embed")

    expected = {c["id"]: c for c in chunks}
    if len(expected) != len(chunks):
        raise ValueError("Duplicate chunk IDs")
    model_id = f"{embed_model}_{embed_dim}"
    existing: list[dict[str, Any]] = []
    done: set[str] = set()

    ##
    if save_embed.exists():
        # Only load trusted local archives: object arrays use pickle.
        with np.load(save_embed, allow_pickle=True) as saved:
            cached = saved["d"].tolist()

        for record in cached:
            source = expected.get(record["id"])
            vector = np.asarray(record["embedding"])

            ## Check the checkpoint contradicts the new corpus:
            if (
                source is None
                or record["id"] in done
                or record["meta"].get("hash") != source["meta"]["hash"]
                or record["meta"].get("embedding_model_id") != model_id
                or record["chunk_text"] != source["chunk_text"]
                or vector.shape != (embed_dim,)
                or not np.isfinite(vector).all()
                or not np.isclose(np.linalg.norm(vector), 1.0)
            ):
                logger.info(
                    "Corpus or embedding cache changed; regenerating all selected embeddings"
                )
                existing.clear()
                done.clear()
                break
            existing.append(record)
            done.add(record["id"])
    pending = [c for c in chunks if c["id"] not in done]

    if not pending:
        return

    encoding = tiktoken.encoding_for_model(embed_model)
    for chunk in pending:
        ## Safety check:
        if (
            len(encoding.encode(chunk["chunk_text"], disallowed_special=()))
            > MAX_TOKEN_LENGTH
        ):
            raise ValueError(f"PMID {chunk['id']} exceeds the token limit")

    for idx in tqdm(range(0, len(pending), batch_size)):
        batch = pending[idx : idx + batch_size]

        ## Add re-try and wait-retry logic:
        response = oai_client.embeddings.create(
            model=embed_model,
            input=[c["chunk_text"] for c in batch],
            dimensions=embed_dim,
        )
        by_index = {item.index: item for item in response.data}

        if len(response.data) != len(batch) or set(by_index) != set(range(len(batch))):
            raise ValueError("Embedding response does not match the requested batch")
        completed: list[dict[str, Any]] = []

        for index, chunk in enumerate(batch):
            vector = np.asarray(by_index[index].embedding, dtype=np.float32)
            norm = np.linalg.norm(vector)
            if vector.shape != (embed_dim,) or not np.isfinite(norm) or norm == 0:
                raise ValueError(f"Invalid embedding for PMID {chunk['id']}")

            completed.append(
                {
                    **chunk,
                    "embedding": vector / norm,
                    "meta": {**chunk["meta"], "embedding_model_id": model_id},
                }
            )
        checkpoint = existing + completed
        atomic_save(
            save_embed,
            lambda fp, checkpoint=checkpoint: np.savez(
                fp, d=np.array(checkpoint, dtype=object)
            ),
        )
        existing = checkpoint


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    args = parsed_args()
    manifest_path = args.embedding_path.with_suffix(".manifest.json")
    frozen = None

    if manifest_path.exists():
        frozen = json.loads(manifest_path.read_text(encoding="utf-8"))

    selected = frozen["pmids"] if frozen is not None else None
    if (
        frozen is not None
        and args.limit_abstracts != -1
        and frozen["count"] != args.limit_abstracts
    ):
        selected = None

    try:
        chunks = list(
            load_chunks(
                args.corpus_path,
                args.limit_abstracts,
                selected if args.limit_abstracts != -1 else None,
            )
        )
    except ValueError:
        if selected is None or args.limit_abstracts == -1:
            raise
        logger.info("Frozen selection is no longer available; selecting records again")
        selected = None
        chunks = list(load_chunks(args.corpus_path, args.limit_abstracts))

    if not chunks:
        raise ValueError("No usable chunks to embed")
    if args.limit_abstracts != -1 and len(chunks) != args.limit_abstracts:
        raise ValueError(
            f"Requested {args.limit_abstracts} usable records; found {len(chunks)}"
        )

    if selected is not None:
        positions = {pmid: index for index, pmid in enumerate(selected)}
        chunks.sort(key=lambda chunk: positions.get(chunk["id"], len(positions)))

    manifest = subset_manifest(chunks)
    if frozen is not None and manifest != frozen:
        logger.info("Frozen subset contents changed; refreshing the corpus manifest")

    if manifest != frozen:
        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        atomic_save(manifest_path, lambda fp: fp.write(manifest_bytes))
    logger.info("Selected %s chunks; corpus hash %s", len(chunks), manifest["hash"])

    save_embeddings(chunks, OpenAI(), args.embedding_path, batch_size=args.batch_size)
    logger.info("Completed saving embeddings")


if __name__ == "__main__":
    main()
