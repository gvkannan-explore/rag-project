import json
import sys
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest
import yaml

from pubmed_rag import ingest

QUERY_CONFIG = Path(__file__).resolve().parents[1] / "configs/test_query.yaml"


@pytest.mark.parametrize("email", [None, "", "   "])
def test_ingest_requires_email(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, email: str | None
) -> None:
    if email is None:
        monkeypatch.delenv("NCBI_EMAIL", raising=False)
    else:
        monkeypatch.setenv("NCBI_EMAIL", email)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rag-ingest",
            "--save_dir",
            str(tmp_path),
            "--query_config",
            str(QUERY_CONFIG),
            "--timeout",
            "5",
        ],
    )
    with pytest.raises(ValueError, match="Set NCBI_EMAIL"):
        ingest.main()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("record_count", "rejected_pmid"),
    [(2, None), (2, "2"), (201, None)],
    ids=["two-papers", "missing-abstract", "multiple-batches"],
)
def test_ingest_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_count: int,
    rejected_pmid: str | None,
) -> None:
    with QUERY_CONFIG.open(encoding="utf-8") as source:
        query = yaml.safe_load(source)

    # Use the real test configuration, increasing only the multi-batch limit.
    config_path = QUERY_CONFIG
    if record_count != query["retmax"]:
        query["retmax"] = record_count
        config_path = tmp_path / "batch_query.yaml"
        config_path.write_text(yaml.safe_dump(query), encoding="utf-8")

    requested_pmids = [str(index) for index in range(1, record_count + 1)]
    fetched_batches: list[list[str]] = []
    search_count = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal search_count
        assert request.url.host == "eutils.ncbi.nlm.nih.gov"
        if request.url.path.endswith("/esearch.fcgi"):
            assert request.method == "GET"
            assert dict(request.url.params) == {
                **{key: str(value) for key, value in query.items()},
                "email": "learner@example.test",
                "api_key": "",
            }
            search_count += 1
            return httpx.Response(
                200,
                json={"esearchresult": {"idlist": requested_pmids[::-1]}},
            )

        assert request.url.path.endswith("/efetch.fcgi")
        assert request.method == "POST"
        form = parse_qs(request.content.decode(), keep_blank_values=True)
        assert form["db"] == ["pubmed"]
        assert form["retmode"] == ["xml"]
        assert form["email"] == ["learner@example.test"]
        assert form["api_key"] == [""]
        batch = form["id"][0].split(",")
        fetched_batches.append(batch)
        articles = []
        for pmid in batch:
            abstract = (
                ""
                if pmid == rejected_pmid
                else f'<Abstract><AbstractText Label="BACKGROUND">'
                f"Evidence for paper {pmid}.</AbstractText></Abstract>"
            )
            articles.append(
                f"<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID>"
                f"<Article><ArticleTitle>Paper <i>{pmid}</i></ArticleTitle>"
                f"{abstract}<Journal><Title>Test Journal</Title><JournalIssue>"
                "<PubDate><Year>2026</Year></PubDate></JournalIssue></Journal>"
                "<PublicationTypeList><PublicationType>Journal Article</PublicationType>"
                "</PublicationTypeList></Article><MeshHeadingList><MeshHeading>"
                "<DescriptorName>Test Topic</DescriptorName></MeshHeading>"
                "</MeshHeadingList></MedlineCitation><PubmedData><ArticleIdList>"
                f'<ArticleId IdType="doi">10.0000/test.{pmid}</ArticleId>'
                "</ArticleIdList></PubmedData></PubmedArticle>"
            )
        return httpx.Response(
            200, text=f"<PubmedArticleSet>{''.join(articles)}</PubmedArticleSet>"
        )

    original_client = httpx.Client

    def offline_client(*, timeout: float) -> httpx.Client:
        return original_client(timeout=timeout, transport=httpx.MockTransport(respond))

    monkeypatch.setattr(ingest.httpx, "Client", offline_client)
    monkeypatch.setenv("NCBI_EMAIL", "learner@example.test")
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rag-ingest",
            "--save_dir",
            str(tmp_path),
            "--query_config",
            str(config_path),
            "--timeout",
            "5",
        ],
    )

    ingest.main()

    assert search_count == 1
    assert fetched_batches == [
        requested_pmids[start : start + 200] for start in range(0, record_count, 200)
    ]
    corpus_files = list(tmp_path.glob("corpus_*.jsonl"))
    assert len(corpus_files) == 1
    records = [json.loads(line) for line in corpus_files[0].read_text().splitlines()]
    expected_pmids = [pmid for pmid in requested_pmids if pmid != rejected_pmid]
    assert [record["pmid"] for record in records] == expected_pmids
    for record in records:
        pmid = record["pmid"]
        assert record == {
            "pmid": pmid,
            "title": f"Paper {pmid}",
            "abstract": f"BACKGROUND: Evidence for paper {pmid}.",
            "journal": "Test Journal",
            "pub_year": "2026",
            "pub_types": ["Journal Article"],
            "mesh_terms": ["Test Topic"],
            "doi": f"10.0000/test.{pmid}",
        }

    rejection_files = list(tmp_path.glob("rejectedAbstract_*.json"))
    assert len(rejection_files) == 1
    rejected = json.loads(rejection_files[0].read_text())
    assert rejected == (
        []
        if rejected_pmid is None
        else [{"pmid": rejected_pmid, "reason": "Invalid abstract"}]
    )
