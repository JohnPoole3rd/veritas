# Per-Agent Runbooks

Read your own file first. It is more specific than the shared plan.

| Agent | Runbook | Owns |
|---|---|---|
| J | [`J.md`](J.md) | config, data, encoders, alignment, retrieval, integration |
| A | [`A.md`](A.md) | ResNet-18, majority, Upper Arlington tiles, local evaluation |
| D | [`D.md`](D.md) | Gradio demo, write-up, video |

## Common setup

Run from the repository root.

```sh
git fetch --all
git switch main
git pull --rebase origin main
```

Then read:

1. `plan/DECISIONS.md`
2. `plan/INTERFACES.md`
3. `plan/TASKS.md`
4. Your runbook

## Rules

- One owner per module. Do not edit another agent's files.
- If you need a function in another owner's module, add a `TODO(owner)` stub or write a learning file. Do not implement it yourself.
- Normal task branches only: `j/...`, `a/...`, `d/...`.
- No message board.
- No direct push to `main` except J.
- Every task writes `results/Txx.md`.
- Important findings go to `plan/learnings/Txx-topic.md`.
- If a command fails, copy the exact error into the task report or learning file.
- If you need a change in another agent's module, write a learning file and ask J.
