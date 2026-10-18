# PubMed RAG — Learning MVP

Created: 2026-10-17.

## Purpose

Build a small PubMed RAG CLI by hand. Understand the flow from abstracts to
retrieval, generated answers, and citation checks. Produce a short demo and
an honest comparison of retrieval methods for a LinkedIn post.

This document defines the reduced MVP scope. It takes priority over the
milestone scope and acceptance gates in [SPEC.md](SPEC.md) for this build.
The original spec remains the roadmap for later work. Completing this MVP
does not mean that its original M0–M7 gates have passed.

The MVP includes a vector database, reranking, and a full evaluation workflow.
The earlier 4–6 hour estimate no longer applies. Plan short learning sessions;
do not promise completion under ten hours for handwritten implementation,
labeling, judge calibration, and debugging. Keep the corpus and model choices
small to control effort without removing these learning goals.

## Learning agreement

- The learner writes the code.
- The assistant does not write or edit code, supply implementation snippets,
  or provide executable commands unless explicitly asked.
- Use `$learn` for focused questions, explanations, progressive hints, and
  feedback on the learner's work.
- Diagnose understanding from explanations, predictions, and attempts.
- Ask exactly three in-depth questions after a major code change or completed
  code review. Cover mechanism, design judgment, and failure with verification.
- Use `$adhd` when an open design choice warrants broader exploration.
- Work through one small task at a time. Do not add infrastructure solely to
  satisfy the original scaffold checklist.

## MVP workflow

Existing local abstracts → one chunk per paper → Postgres vector storage and
BM25 → retrieval → optional fusion and reranking → top five papers → generated
claims with quotes → validation → evaluation report.

Support four configurations: A dense, B BM25, C hybrid, and D hybrid with a
reranker. Use the same corpus, questions, and final result count for all four.
The primary experiment measures whether reranking improves hybrid retrieval
and answer support, and what latency and cost it adds.

## Scope

### Corpus

- First inspect and validate the existing local `data/corpus.jsonl`.
- Freeze a subset of 200–500 usable abstracts. Do not build a new ingestion
  client for this MVP.
- Retain each paper's PMID, title, and abstract.
- Require unique PMIDs and nonempty abstracts. Report rejected records.
- Keep one chunk per paper: title, a blank line, then abstract.
- Record the subset's PMIDs, record count, and corpus hash. Reuse that exact
  subset for the comparison.
- Keep abstract text, embeddings, local environments, and secrets out of Git.
  Untrack an existing corpus file while preserving its local copy.

### Database and retrieval

- Use the existing Postgres 16 Compose service with pgvector. Pinecone is an
  alternative for a later deployment need; do not implement two stores.
- Verify database readiness and enable the vector extension in the database.
- Persist chunk IDs, PMIDs, text, vectors, and corpus/model identifiers.
- Upsert by stable chunk ID. Indexing twice must not create duplicate rows.
- Query Postgres for dense retrieval. Start with exact cosine search. Defer
  HNSW tuning while the corpus is small.

- Use `rank_bm25` for keyword retrieval, as in the original spec.
- Lowercase and split tokens on non-alphanumeric characters. Preserve tokens
  such as `her2` and `brca1`.
- Use one embedding model for documents and questions. Record its exact ID.
- Cache document embeddings locally. Associate vectors with their PMIDs and
  record the model ID and corpus hash to detect stale caches.
- Retrieve up to 50 candidates from each method for hybrid retrieval.
- Fuse rankings with Reciprocal Rank Fusion. Use a constant of 60 and ranks
  starting at 1. Add each paper's contribution from each list where it appears.
- Return the top five papers. Preserve PMIDs, scores, and retrieval provenance.
- Resolve ties consistently. Keep ranking order reproducible for fixed inputs.

### Reranking

- Use one MiniLM cross-encoder through Sentence Transformers, following the
  original spec. Verify and pin the exact model ID before implementation.
- For configuration D, rerank the top 50 fused candidates using question–chunk
  pairs. Select the top five for generation and retain at least ten ranked
  results for retrieval evaluation.
- Configuration C uses the same candidates without reranking. Keep all other
  settings fixed so the comparison isolates the reranker's effect.
- Preserve retrieval provenance and store the reranker score separately.
- Record reranking latency. Record input truncation settings and CPU behavior.

### Answers and validation

- Use one generator model. Record its exact ID before the comparison.
- Generate answers using only the five retrieved papers.
- Require structured output with claims, cited PMIDs, quotes with source PMIDs,
  and an abstention status. Retain the retrieved context PMIDs.
- Instruct the generator to abstain when the context cannot answer the question.
- Do not allow outside knowledge to fill gaps, even when the model is confident.
- Treat the question and source text as data rather than instructions.
- Validate claims without another model call. Drop a claim if it has no
  citations, cites a PMID outside the retrieved context, has no quotes, names
  a quote PMID outside its citations, or includes a quote absent from the
  corresponding cited abstract.
- Match quotes after collapsing whitespace. Keep matching case-sensitive.
- Record each dropped claim and its reason. Abstain if no valid claims remain.
- Keep generator abstentions explicit when the context is insufficient.

Citation and quote checks establish source membership and quote existence.
They do not establish that a quote supports a claim or that an answer is
medically correct. State this limit in the demo documentation.

### CLI and development checks

- Extend the existing CLI. Keep its help output and offline smoke test.
- Let the user select a retrieval mode and inspect retrieved papers without
  generation. Also support a question that produces a validated answer.
- Provide separate indexing, evaluation, and report operations. Keep the
  evaluation runner separate from interactive CLI presentation.
- Keep Ruff lint, Ruff formatting, Pyright, and pytest in one local check script.
- Make a failed check cause the script to fail.
- Verify model interfaces against official documentation when implementing
  their calls. Do not select model IDs by guessing.
- Choose a spending limit before paid calls. Run a small trial before embedding
  the full subset or generating all demonstration answers.

## Build sessions

### 1. Load papers and implement BM25 — about 1 hour

Validate and freeze the subset. Load papers into a simple representation.
Retrieve the top five papers for a question with BM25.

Completion check: Results show titles, PMIDs, and scores. Every returned PMID
belongs to the frozen subset. A fixture test works without network access.

### 2. Store vectors and add dense retrieval

Verify Postgres and pgvector. Embed the subset once. Persist chunks and vectors,
then query the database with exact cosine search.

Completion check: Database contents survive a restart. Reindexing creates no
duplicates. A repeat run reuses cached document embeddings. Dense retrieval
returns the expected ranking on a small fixture with known vectors.

### 3. Fuse and rerank

Add Reciprocal Rank Fusion, then the cross-encoder stage.

Completion check: All four configurations return ranked papers. Hand-made
rankings verify fusion, including a paper present in only one list. Tests
verify that reranking preserves candidate identities and selects by reranker
score. A live trial confirms model loading and records latency.

### 4. Generate and validate answers

Generate claims with source quotes. Apply deterministic validation and show
abstentions and dropped-claim reasons.

Completion check: Offline tests cover valid claims, missing citations or quotes,
PMIDs outside context, mismatched quote PMIDs, fabricated quotes, and abstention
when every claim is dropped. Also exercise a context-insufficient question.

### 5. Build and run the evaluation harness

Build a reusable dataset loader, runner, evaluators, and report generator.
First exercise the entire harness on three questions. Then run the frozen
evaluation datasets. A full harness means the workflow below is implemented;
it does not require every domain model or hosted integration in `SPEC.md`.

#### Datasets and labels

- Store versioned datasets in `evals/datasets/` with stable item IDs.
- Start with ten manually labeled retrieval questions. Expand to at least 30
  before presenting the final comparison. Each has a question, relevant PMIDs,
  and labeling notes. Require at least one relevant paper per retrieval item.
- Build a labeling pool from the union of the top 20 results from all four
  configurations. The learner supplies relevance labels. Freeze the labels
  before measuring the final comparison and disclose incomplete pool coverage.
- The QA set contains the answerable questions, five unanswerable questions,
  and three adversarial questions. Record expected abstention and case type.
- Keep a separate set of 20 claim–abstract pairs with human support labels to
  calibrate the judge. Models must not supply gold labels.
- Synthetic questions may help draft candidates. The learner must review them,
  verify their answerability against the frozen corpus, and supply the final
  labels. Keep the evaluation set fixed across configurations.

#### Evaluators

- Evaluate retrieval independently from end-to-end answer behavior. A relevant
  paper at a high rank does not by itself establish a supported answer.
- Score Recall@5, Recall@10, and MRR@10 for configurations A–D. Retain ten
  results even though generation receives only five.
- Recall@k is the share of labeled relevant PMIDs in the first k results.
  MRR@10 is the reciprocal rank of the first relevant result, or zero if absent.
- Run generation evaluation for C and D to isolate reranking's effect and
  control cost. Leave generation metrics for A and B explicitly not applicable.
- Check citations against the frozen corpus. Score context membership and
  verbatim quotes both before and after validation.
- Score expected abstention. Judge claim support and resistance to instructions
  planted in adversarial questions. Report these separately from quote validity.
- Pin the judge model and prompt. Record the prompt hash and run calibration.
  Report percent agreement and Cohen's kappa against the 20 human labels.
  Flag agreement below 80 percent. Report undefined kappa as such.
- Return not applicable for checks without applicable claims or quotes. Do not
  count empty abstentions as successful citation or quote checks. Score their
  abstention behavior separately.
- Exclude errored and not-applicable scores from metric denominators. Report
  their counts and item IDs so failures cannot improve a hidden denominator.

#### Runner and reporting

- Use bounded concurrency, bounded retries, and a configured spending ceiling.
  Track all paid attempts, including retries and judge calls. Record estimated
  usage and stop scheduling new paid work when the budget is exhausted.
- Give each run its own result and intermediate-output directory. Reuse only
  caches whose corpus, model, and input identities match. Validate citations
  against the frozen corpus rather than prior run artifacts.
- Record config and corpus hashes, model IDs, dataset version, judge prompt
  hash, concurrency, and the source revision. Flag a dirty working tree.
- Save per-item retrieval ranks, raw and validated answers, evaluator scores,
  errors, stage latency, token usage, and estimated cost locally. Result files
  containing abstract quotes stay out of Git. Commit sanitized reports and
  metadata without source text.
- Retain the exact ordered context PMIDs supplied to generation. Compare C and
  D on each question, including ranking changes, top-five context changes,
  answer support, validation drops, and abstention. Include representative
  improvements and regressions in the report alongside aggregate metrics.
- Sort outputs by item ID. Verify stable ordering and repeatable deterministic
  scoring using saved responses; do not claim live model outputs are identical.
- Produce `results/ablation.md` from saved results without rerunning models.
  Include four configuration rows, retrieval and answer metrics, sample counts,
  errors, p50 latency, cost per question, and judge calibration results.
- Document metric applicability, latency boundaries, concurrency, small-sample
  uncertainty, and incomplete relevance labels. Do not tune labels to favor a
  method. Improvement is an experimental result, not a completion requirement.
- Test the harness offline with stubbed model calls. Cover metric arithmetic,
  not-applicable scores, failures, retry limits, budget exhaustion, ordering,
  and aggregation. Keep database integration checks separate.

Completion check: One evaluation operation produces saved results for A–D.
One report operation produces the comparison from those results. Errors and
budget-stopped items remain visible. Retrieval and answer results are reported
separately. Per-question comparisons explain representative failures. Offline
harness tests pass.

#### Failure analysis

When retrieval metrics improve without an improvement in answer support:

1. Inspect whether the top five papers supplied to generation changed. MRR@10
   can improve while the context membership stays the same.
2. Inspect whether those abstracts contain enough evidence to answer the whole
   question. Relevance labels alone do not establish sufficient evidence.
3. Compare generated claims with that evidence. Separate failure to use available
   evidence from failure to retrieve it.
4. Check citation membership, verbatim quotes, dropped claims, and abstention.
   Valid citations and copied quotes do not by themselves establish support.
5. Inspect judge explanations and human calibration before attributing a change
   to the pipeline. Report uncertainty rather than assuming one metric proves
   overall answer quality.

### 6. Demonstrate the result

Record a short terminal demo showing retrieved evidence, a validated answer,
an abstention, and the generated comparison report. Explain where reranking
helped or failed and what it cost. State the limits of the evidence.

## Deliverables and completion gate

- [ ] A fixed subset of 200–500 validated abstracts with recorded PMIDs and hash.
- [ ] Persistent Postgres/pgvector storage with verified readiness and extension.
- [ ] Repeat indexing produces no duplicates and survives database restart.
- [ ] Dense, BM25, hybrid, and hybrid-with-reranking configurations.
- [ ] A CLI that displays retrieved evidence and produces a validated answer.
- [ ] Explicit abstention and recorded reasons for dropped claims.
- [ ] Passing local lint, formatting, type, and offline behavior checks.
- [ ] At least 30 manually labeled retrieval questions with frozen labels.
- [ ] QA cases include five unanswerable and three adversarial questions.
- [ ] Twenty human-labeled judge calibration pairs and reported agreement.
- [ ] A reusable evaluation runner with metrics, errors, budgets, and saved runs.
- [ ] A four-configuration report with retrieval, answer, latency, and cost results.
- [ ] Per-question comparisons with representative improvements and regressions.
- [ ] A short README with setup, workflow, observed results, and limitations.
- [ ] A short terminal recording showing evidence, an answer, and abstention.
- [ ] No committed secrets, abstract text, or embedding caches.

Present the work as a small learning experiment. Report the actual results,
including cases where hybrid retrieval does not outperform a single method.
State that the tool is for research demonstration and has no clinical use.

## Deferred work

Defer summary indexing, new PubMed ingestion, the full upstream client port,
HNSW tuning, MedCPT,
Pinecone or a second database, LangSmith synchronization, hosted CI, Makefile
wrappers, additional CLI frameworks, a public library API, and deployment
extensions. The evaluation harness and automated judging are in scope; a
hosted evaluation platform is not required.

Add these only after the MVP works and a specific learning or product need
justifies them. They remain part of the larger roadmap in `SPEC.md`.

## First learning task

Before implementation, explain what information a loaded paper must retain
so that a generated quote can be checked against the correct source. Then
inspect the existing corpus and propose the smallest suitable representation.

## Implementation references

- [Claude Cookbook RAG guide](https://platform.claude.com/cookbook/capabilities-retrieval-augmented-generation-guide):
  use its separation of retrieval and end-to-end evaluation, incremental
  comparisons, and detailed result inspection. Adapt its LLM reranking example
  to the cross-encoder selected here. Keep Postgres rather than its in-memory
  store. Keep human-reviewed labels and strict grounding rather than adopting
  its synthetic dataset and outside-knowledge allowance directly. Summary
  indexing remains a later experiment because this corpus uses abstracts.
- [pgvector documentation](https://github.com/pgvector/pgvector): persistent
  vector storage and exact cosine search in Postgres.
- [Sentence Transformers retrieve and rerank guide](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html):
  cross-encoder reranking after candidate retrieval.
- [Original evaluation requirements](SPEC.md): Section 7 supplies the metric,
  judge calibration, runner, and reporting conventions used here.
