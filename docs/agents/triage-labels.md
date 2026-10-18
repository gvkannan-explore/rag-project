# Triage labels

Use the following default role strings in each local issue's `Status:` line:

- `needs-triage`: maintainer needs to evaluate the issue.
- `needs-info`: waiting for more information.
- `ready-for-agent`: fully specified and ready for an agent to implement.
- `ready-for-human`: requires human implementation.
- `wontfix`: will not be actioned.

When a skill says to apply a triage label, update the local issue's `Status:` line to the corresponding role. Do not create remote labels or GitHub Issues.

Wayfinding tickets use their own lifecycle statuses as described in `issue-tracker.md`.
