"""Reproducible ranking transformations over frozen-corpus retrieval results."""

import math

from pubmed_rag.search import RetrievalResult

CANDIDATE_LIMIT = 50
RRF_CONSTANT = 60


def fuse_rankings(
    dense: list[RetrievalResult], lexical: list[RetrievalResult]
) -> list[RetrievalResult]:
    """Fuse the source top-50 lists by PMID and retain the shared C/D pool."""
    by_pmid: dict[str, RetrievalResult] = {}
    corpus_hashes = {row["corpus_hash"] for row in dense + lexical}
    if len(corpus_hashes) > 1:
        raise ValueError("Retrieval corpus identity differs between candidates")
    for method, rows in (("dense", dense), ("bm25", lexical)):
        seen: set[str] = set()
        for rank, row in enumerate(
            sorted(rows, key=lambda row: row["rank"])[:CANDIDATE_LIMIT], 1
        ):
            pmid = row["pmid"]
            if pmid in seen or row["rank"] != rank or row["method"] != method:
                raise ValueError(
                    "Expected unique PMIDs and consecutive one-based source ranks"
                )
            if not math.isfinite(row["score"]):
                raise ValueError("Retrieval scores must be finite")
            seen.add(pmid)
            previous = by_pmid.get(pmid)
            if previous is not None and any(
                previous[key] != row[key]
                for key in ("title", "source_text", "corpus_hash")
            ):
                raise ValueError(f"Retrieval paper identity differs for PMID {pmid}")
            if pmid not in by_pmid:
                by_pmid[pmid] = {
                    **row,
                    "method": "hybrid",
                    "score": 0.0,
                    "sources": {},
                }
            result = by_pmid[pmid]
            result.setdefault("sources", {})[method] = {
                "rank": row["rank"],
                "score": row["score"],
                "model_id": row["model_id"],
            }
            result["score"] += 1 / (RRF_CONSTANT + row["rank"])
    ordered = sorted(by_pmid.values(), key=lambda row: (-row["score"], row["pmid"]))
    return [
        {**row, "rank": rank, "fused_rank": rank, "fused_score": row["score"]}
        for rank, row in enumerate(ordered[:CANDIDATE_LIMIT], 1)
    ]
