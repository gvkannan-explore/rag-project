import os
import select
import subprocess
import time
from pathlib import Path

import pytest


@pytest.fixture
def chosen_environment(tmp_path: Path) -> Path:
    environment = tmp_path / "chosen environment"
    (environment / "bin").mkdir(parents=True)
    for name in ("rag", "python"):
        command = environment / "bin" / name
        command.write_text(
            "#!/usr/bin/env bash\nprintf 'fixture failure\\n' >&2\nexit 7\n"
        )
        command.chmod(0o755)
    return environment


@pytest.mark.parametrize(
    "mode,operation",
    [
        ("bm25", "BM25 retrieval"),
        ("index", "Cached-vector indexing"),
        ("dense", "Cached-vector indexing"),
    ],
)
def test_search_runfile_uses_chosen_environment_and_preserves_failure(
    chosen_environment: Path,
    mode: str,
    operation: str,
) -> None:
    result = subprocess.run(
        ["bash", "runs/run_search.sh", mode, "question"],
        env={
            **os.environ,
            "RAG_ENV_DIR": str(chosen_environment),
            "RAG_NONINTERACTIVE": "1",
            "ALLOW_PAID_QUERY": "1",
            "QUERY_BUDGET_USD": "0.001",
            "OPENAI_API_KEY": "test",
        },
        capture_output=True,
        check=False,
        text=True,
        timeout=5,
    )
    assert result.returncode == 7
    assert "fixture failure" in result.stderr
    assert operation in result.stderr
    assert "7" in result.stderr


@pytest.mark.skipif(os.name != "posix", reason="Uses a POSIX terminal")
def test_interactive_runfile_waits_before_returning_failure(
    chosen_environment: Path,
) -> None:
    master, slave = os.openpty()
    process = subprocess.Popen(
        ["bash", "runs/run_search.sh", "bm25", "question"],
        env={
            **os.environ,
            "RAG_ENV_DIR": str(chosen_environment),
            "RAG_NONINTERACTIVE": "0",
        },
        stdin=slave,
        stdout=slave,
        stderr=slave,
    )
    os.close(slave)
    output = b""
    try:
        deadline = time.monotonic() + 5
        while b"Press Enter" not in output and time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                try:
                    output += os.read(master, 4096)
                except OSError:
                    break
        assert b"fixture failure" in output
        assert b"Press Enter" in output
        assert process.poll() is None
        os.write(master, b"\n")
        assert process.wait(timeout=5) == 7
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)
