import json
import subprocess
import sys
from pathlib import Path


def test_cli_help() -> None:
    command = Path(sys.executable).with_name("rag")

    result = subprocess.run(
        [str(command), "--help"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "PubMed RAG command-line tool." in result.stdout
    assert "--help" in result.stdout


def test_search_cli_exposes_hybrid_modes_and_deep_json_rankings(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "".join(
            json.dumps(
                {"pmid": str(i), "title": f"Paper {i}", "abstract": "Topic study"}
            )
            + "\n"
            for i in range(200)
        )
    )
    command = Path(sys.executable).with_name("rag")
    args = [
        str(command),
        "search",
        "topic",
        "--corpus",
        str(corpus),
        "--selection",
        str(tmp_path / "selection.json"),
    ]
    default = subprocess.run(
        args + ["--method", "B"], capture_output=True, text=True, check=False, timeout=5
    )
    assert default.returncode == 0, default.stderr
    assert default.stdout.count("(PMID ") == 5
    deep = subprocess.run(
        args + ["--method", "bm25", "--limit", "20", "--json"],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert deep.returncode == 0, deep.stderr
    payload = json.loads(deep.stdout)
    assert payload["configuration"] == "B"
    assert len(payload["results"]) == 20
    assert len(payload["context_pmids"]) == 5
    blocked = subprocess.run(
        args + ["--method", "C"], capture_output=True, text=True, check=False, timeout=5
    )
    assert blocked.returncode == 2
    assert "paid question embedding" in blocked.stderr
