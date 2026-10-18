# Implementation and review workflow

## Start a task

Read the referenced ticket and its source spec. Record the baseline commit,
included files, and exclusions before changing code or reviewing it. Use the
maintained implementation in `src/pubmed_rag/`; notebooks are historical
experiments and are outside ticket 01–02 follow-up reviews.

Confirm an existing Python environment once per session, as required by
`AGENTS.md`. Reuse that confirmation for checks and tests. Runfiles accept
`RAG_ENV_DIR` as the chosen existing environment; resolve it to an absolute
path before using its tools.

## Record completion evidence

For every acceptance check marked complete, append evidence to the ticket:

- Check: exact command or manual procedure used.
- Result: observed outcome, including failures and skipped checks.
- Scope: offline behavior, live database integration, or paid model trial.
- Revision: tested source revision and whether the tree was dirty.
- Remaining: anything the evidence does not establish.

An implementation or a fake database response is not evidence that live
database ordering or restart persistence works. Keep those checks separate.
Paid trials require an agreed spending ceiling; offline tests use fixed
responses and make no paid calls.

## Deliver changes

Prefer one implementation commit per ticket and separate commits for review
fixes. Stage named files so unrelated work and notebook outputs stay outside
the change. Review against the recorded baseline, and resolve findings before
delivery. The pre-push hook scans all outgoing refs and runs the existing
local check suite once for a push containing commits.

## Runfile behavior

Runfiles use `set -u -o pipefail` with explicit failure handling. Report the
failed operation and exit status. When stdin and stdout are terminals, pause
at completion so the output can be read; noninteractive runs return the
operation's failure status without waiting. `RAG_NONINTERACTIVE=1` suppresses
the pause for automation. Keep these behaviors in the shared runfile helper.

## Explore efficiently

Find filenames before reading content. Bound search output and line lengths
when inspecting notebooks, corpus files, generated artifacts, or logs. Use
targeted ranges rather than dumping whole artifacts into agent context.
