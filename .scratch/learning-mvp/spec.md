# Learning MVP ticket context

Source of truth: [MVP_SPEC.md](../../MVP_SPEC.md). This ticket set implements its reduced scope rather than the full build spec.

The user approved the twelve-ticket breakdown and supplied existing corpus and embedding artifacts. Reuse them; do not repeat ingestion or paid document embedding merely to start the MVP.

## Existing artifacts

- Frozen corpus: `data/corpus_Oct2026.jsonl` (exists).
- Embeddings: `data/vectors/vector_embeddings.npz` (exists). The user referred to `vector_embedding.npz`; this is the archive found locally.
- Manifest: `data/vectors/vector_embeddings.manifest.json` (exists), reporting 5,000 PMIDs and corpus hash `5e39d0131a0b0891ae2be0080a44b3c4d8b91cd5dfab9039143cdaed3c0dc9ec`.
- Ticket 02 records archive verification and database integration evidence. Consult its latest dated comment for current verification status.

The manifest count exceeds the source spec's 200–500-paper limit. Resolve this explicitly in ticket 01: either select and record an MVP-sized view of the frozen corpus and reuse matching cached vectors, or obtain an explicit scope amendment for all 5,000 papers. Preserve the original artifacts. No scope amendment is assumed.

## Working agreement

- The learner writes implementation code. Tickets use `ready-for-human` to preserve that agreement; the assistant tutors and reviews unless explicitly asked for implementation.
- Before running Python or environment-specific tools, confirm which existing environment to use, as required by the root agent instructions.
- Use local Markdown tickets only. No GitHub Issues.
- Paid calls require an agreed spending limit and a small trial. Verify model interfaces and exact model IDs against official documentation during implementation.
- Keep the same frozen comparison corpus and datasets across configurations A–D. Generation evaluation applies only to C and D.
- Retain the existing CLI help, offline smoke test, and local lint, formatting, type, and test checks throughout. Keep database integration checks separate.
- Keep source abstracts, quotes, embeddings, secrets, and local environments out of Git. Commit only sanitized reports and permitted metadata.
- Each ticket's acceptance checks are completion requirements, not claims that verification has already run.

## Dependencies

01 → 02 → 03 → 04 → 06

01 → 05; 05 + 06 → 07; 06 → 08; 07 + 08 → 09 → 10 → 11 → 12.

Work any ticket whose blockers are complete. Add `## Comments` to individual tickets for progress and decisions.
