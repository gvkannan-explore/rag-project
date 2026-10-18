# Retrieval setup and checks

Use an existing Python environment. The runfiles default to `.data/`; set
`RAG_ENV_DIR` to another existing environment when needed. No runfile creates
an environment or installs packages.

The maintained code lives in `src/pubmed_rag/`. The notebooks contain older
experiments and are outside this change.

## Dependencies and offline checks

After choosing `.data/`, install the declared dependencies into it:

```sh
UV_PROJECT_ENVIRONMENT="$PWD/.data" uv sync --inexact --group dev --group vector-store
bash runs/perform_ci.sh "$PWD/.data"
```

The check runner scans for secrets, tests the Git hooks, checks runfile rules
and Bash syntax, runs Ruff and Pyright, and runs the offline tests. It reports
each failure and returns a nonzero status if any check fails. Pre-push runs
this suite once and scans each outgoing ref. Integration checks run separately.

Runfiles print the failed operation and exit status. When attached to an
interactive terminal, they wait for Enter at completion. Set
`RAG_NONINTERACTIVE=1` to suppress that pause.

## Frozen data and indexing

Supply the existing corpus, embedding archive, and archive manifest under the
ignored `data/` directory. `config/mvp_corpus.json` records the ordered 200-paper
selection without abstract text. Keep that file fixed across comparisons.

```sh
bash runs/run_search.sh bm25 "HER2 treatment"
docker compose up -d db
bash runs/run_search.sh index
```

Indexing checks selected text, model identity, archive membership, dimensions,
and vector validity before writing to Postgres. It reuses the cached document
embeddings and makes no paid calls.

## Database integration checks

Run the known-vector checks against the local Postgres 16 service:

```sh
RAG_TEST_DATABASE_URL=postgresql://rag:rag_project@localhost:5432/pubmed_rag \
  .data/bin/pytest tests/integration/test_vectors.py -q
```

Each test creates and removes its own schema. It indexes synthetic document
vectors twice and returns results through dense retrieval with a fixed question
embedding. It checks cosine scores, stable ties, corpus/model filtering, and
retrieval depth without paid calls.

To also restart the Compose database and verify persistence, add
`RAG_TEST_RESTART=1` to the command's environment. That check briefly interrupts
other clients of the same database. It retains the existing database volume.

## Optional paid query

Agree on a spending ceiling before a live trial. Set `OPENAI_API_KEY` through
your local credential setup, then pass both `--allow-paid-query` and
`--query-budget-usd` to `rag search --method dense`. The runfile equivalents
are `ALLOW_PAID_QUERY=1` and `QUERY_BUDGET_USD`.

The command requires a finite positive ceiling, counts question tokens, and
rejects an estimate above that ceiling. It checks that the selected corpus and
model are indexed before creating the API client. It sends one request with
SDK retries disabled and prints the token count and estimated cost to stderr.
The ceiling applies to one command invocation. Repeating a command can incur
another charge. Ticket 07 covers budgets across evaluation runs.

The estimate uses the standard input price of $0.02 per million tokens for
`text-embedding-3-small`, checked on 2026-10-18 against the
[official model page](https://developers.openai.com/api/docs/models/text-embedding-3-small).
Refresh the price before future paid trials. The command submits the counted
token IDs so the estimate and request refer to the same input.

No paid model call was made while implementing these checks. This project is
a research demonstration and has no clinical use.
