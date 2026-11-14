# Code review: combined notes (J, A, D)

**Branch:** `review/code-audit` (one branch for everyone's review notes and fixes)
**Base:** `main` @ 742ed89
**Started:** 2026-11-14 by A

How to use this file:
- One row per finding. Use the next free ID (`CR-xx`) and put your agent letter in "By".
- Severity: **bug** (wrong result or crash), **inconsistency** (code/docs/contracts disagree),
  **simplification** (dead or duplicate code).
- Fix it in the same branch when you can, then put the commit hash in "Fix". If you can't fix it,
  write `open` and say who should.
- Pull before you edit (`git pull --rebase origin review/code-audit`) and keep commits small.

## Findings

| ID | By | Severity | File | Finding | Fix |
|---|---|---|---|---|---|

## Verification log

| When | By | Command | Result |
|---|---|---|---|
