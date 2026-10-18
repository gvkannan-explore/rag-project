# Python environment

- Ask the user which existing Python environment to use before running Python
  or environment-specific tools. Suggest `.data/` in the project root, but wait
  for confirmation. Reuse that confirmation during the session unless the user
  changes it or the environment becomes unavailable.
- Use its Python interpreter and installed tools for execution and tests.
- When using `uv`, set `UV_PROJECT_ENVIRONMENT` to the absolute path of
  the confirmed environment. Do not let `uv` create a default `.venv`.
- Do not create a separate virtual environment, including a temporary one.
- If the confirmed environment is missing or unusable, report the issue and ask
  which existing environment to use before running Python.
  Do not create it or switch environments automatically.

## Agent skills

### Learning and review workflow

- Before implementation, tutoring, or review, read the working agreement in
  `.scratch/learning-mvp/spec.md` and the workflow in `docs/agents/workflow.md`.

### Shell runfiles

- Keep interactive terminals open after runfiles complete. Use `set -u -o pipefail` with explicit failure handling; never use `set -e` in a runfile.

### Issue tracker

Track issues and specs as local Markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the five default triage roles as local issue statuses. See `docs/agents/triage-labels.md`.

### Domain docs

Use a single-context layout: root `GLOSSARY.md` and `docs/adr/`. See `docs/agents/domain.md`.
