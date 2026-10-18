# Domain docs

This repo uses a single-context layout: root `GLOSSARY.md` and `docs/adr/`.

## Before exploring

- Read root `GLOSSARY.md` if it exists.
- Read ADRs in `docs/adr/` that relate to the area being explored.

If these files or directories do not exist, proceed silently. Do not flag their absence or suggest creating them upfront. The `domain-modeling` skill, reached through `grill-with-docs` or `improve-codebase-architecture`, creates them when terms or decisions are resolved.

## Use the glossary's vocabulary

Use terms defined in `GLOSSARY.md` for domain concepts in issues, refactor proposals, hypotheses, tests, and other output. Avoid synonyms the glossary explicitly rejects.

If a needed concept is missing, reconsider whether it belongs to the project's language. Note real gaps for `domain-modeling`.

## Flag ADR conflicts

Explicitly identify any proposal that contradicts an existing ADR, name the ADR, and explain why the decision should be reconsidered. Do not silently override it.
