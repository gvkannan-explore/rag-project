# Issue tracker: Local Markdown

Issues and specs for this repo live as Markdown files in `.scratch/`.
Do not use GitHub Issues for tracking work in this repo.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`.
- The spec is `.scratch/<feature-slug>/spec.md`.
- Implementation issues are one file per ticket at `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`. Never combine tickets into one file.
- Triage state is a `Status:` line near the top of each issue file. Use the role strings in `docs/agents/triage-labels.md`.
- Append comments and conversation history under a `## Comments` heading.

## Publishing and fetching

When a skill says "publish to the issue tracker", create the appropriate file under `.scratch/<feature-slug>/`, creating directories as needed.

When a skill says "fetch the relevant ticket", read the referenced local file. Resolve issue numbers within the referenced feature directory; ask for the feature or path when ambiguous.

## Wayfinding operations

Used by the `wayfinder` skill. The map is a file with one child file per ticket.

- Map: `.scratch/<effort>/map.md`, containing Notes, Decisions-so-far, and Fog.
- Child ticket: `.scratch/<effort>/issues/NN-<slug>.md`, numbered from `01`, with the question in the body.
- Type: a `Type:` line records `research`, `prototype`, `grilling`, or `task`.
- Status: wayfinding lifecycle statuses are `open`, `claimed`, and `resolved`, separate from the triage roles.
- Blocking: a `Blocked by: NN, NN` line near the top. A ticket is unblocked when every listed ticket is `resolved`.
- Frontier: scan the effort's issues for open, unblocked, unclaimed tickets; choose the lowest number first.
- Claim: set `Status: claimed` and save before starting work.
- Resolve: append the answer under `## Answer`, set `Status: resolved`, then append a summary and link to Decisions-so-far in the map.
