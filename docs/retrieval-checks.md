# Retrieval setup and checks

Use an existing Python environment. The runfiles default to `.data/`; set
`RAG_ENV_DIR` to another existing environment when needed. No runfile creates
an environment or installs packages.

The maintained code lives in `src/pubmed_rag/`. The notebooks contain older
experiments and are outside this change.

## Dependencies and offline checks

After choosing `.data/`, install the declared dependencies into it:

```sh
UV_PROJECT_ENVIRONMENT="$PWD/.data" uv sync --inexact --group dev --group vector-store --group reranking
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

## Hybrid retrieval and reranking

`rag search --method` accepts `A` or `dense`, `B` or `bm25`, `C` or `hybrid`,
and `D` or `reranked`. A, C, and D require the existing paid-query opt-in and
per-command budget. B makes no model calls. For example:

```sh
.data/bin/rag search "HER2 treatment" --method B --limit 20 --json
.data/bin/rag search "HER2 treatment" --method C --limit 20 --json \
  --allow-paid-query --query-budget-usd 0.001
.data/bin/rag search "HER2 treatment" --method D --limit 20 --json \
  --allow-paid-query --query-budget-usd 0.001
```

Use the paid examples only after agreeing on the spending ceiling. Separate
C and D commands each make their own question-embedding request. The examples
do not authorize paid calls. The future evaluation runner must account for
all requests and reuse only correctly identified cached inputs.

C and D retrieve 50 papers per source, fuse their union by PMID with
`1 / (60 + source_rank)`, then retain the best 50 fused candidates. Source
ranks start at one. A paper contributes once per source where it appears.
Equal final scores sort by PMID as a string. Both configurations use the
same candidate construction, regardless of the display limit.

The Python `retrieve` entry point returns deep rankings in `run.results` and
five papers in `run.context`. The CLI displays five by default. Use
`--limit 20` for ticket 08 labeling pools or `--limit 50` for the entire C/D
pool. C/D reject limits above 50. A/B allow deeper limits. JSON output
contains the requested display depth and the ordered generation-context PMIDs.
It includes source text, so save it only in ignored local storage.

Each fused result preserves the source ranks, scores, and model IDs in
`sources`. `fused_score` and `fused_rank` preserve the C ranking. D keeps
`score` equal to the fused retrieval score, adds `reranker_score`, and assigns
its final `rank` by descending reranker score. Failed model scoring or
non-finite scores cause an error. There is no automatic fallback to C.

D uses `cross-encoder/ms-marco-MiniLM-L6-v2`, pinned to revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`, through Sentence Transformers.
The [official model list](https://sbert.net/docs/cross_encoder/pretrained_models.html)
and [CrossEncoder reference](https://sbert.net/docs/package_reference/cross_encoder/model.html)
confirm the model ID and question-document scoring interface. The model runs
on CPU with batches of 16 and raw logit scores. Each pair contains the question
and the full title-plus-abstract chunk. The tokenizer limits the pair to 512
tokens, including special tokens, with longest-first truncation on the right.
Later abstract text can therefore fall outside the model input. Source text
in the returned result remains unchanged.

Model files stay under ignored `data/models/`. The first D invocation may
download model weights. Loading includes imports, download/cache lookup,
and model initialization. `rerank_seconds` covers pair construction, model
prediction, score validation, and sorting. It excludes candidate retrieval
and model loading. Metadata records these boundaries separately, together
with the model revision, CPU device, batch size, and truncation settings.
Weights load into memory with `disable_mmap=True`. The CPU trial found that
mapped weights had unaligned storage and produced non-finite scores on this
machine. In-memory loading preserved the values and returned finite scores.

Run the CPU trial with synthetic text and no paid requests:

```sh
.data/bin/python runs/trial_reranker.py
```

It scores 50 candidates, checks candidate membership and a long input's
512-token truncation, and saves metadata in ignored `data/reranker_trial.json`.
This checks the model runtime, not retrieval quality or a paid end-to-end D run.

On 2026-10-18, a cache-only CPU trial on macOS 26.3 arm64 with Python 3.14.3,
Sentence Transformers 5.7.0, Transformers 5.20.1, and PyTorch 2.10.0 loaded the
model in 4.67 seconds and reranked 50 synthetic candidates in 0.29 seconds.
One 3,012-token pair truncated to 512 tokens. All candidate identities survived,
all scores were finite, and the explicit Mars passage ranked first. These are
single-trial observations, not latency percentiles or biomedical quality results.

## Verification record

On 2026-10-18, the implementation at `8037079` plus its working review fixes
passed the following checks. The tree was dirty during verification.

- `RAG_NONINTERACTIVE=1 bash runs/perform_ci.sh "$PWD/.data"` passed 44 offline
  tests, Ruff lint and formatting, Pyright, shell syntax and runfile rules,
  secret scans, and hook regression checks.
- The database integration command above with `RAG_TEST_RESTART=1` passed
  both tests. The restart check verified a changed Postgres startup timestamp
  before comparing retrieval results.
- `.data/bin/rag index` validated and indexed 200 cached vectors. After the
  restart, SQL confirmed 200 rows with distinct chunk IDs and PMIDs for the
  selected corpus and model.
- Both review axes accepted the final changes. The spec review's restart
  verification finding was fixed and rechecked. ShellCheck was unavailable.

These results do not establish live question-embedding behavior. A paid trial
and its user-agreed spending ceiling remain pending.
