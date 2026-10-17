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
