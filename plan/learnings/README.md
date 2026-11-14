# Learnings

This folder is the durable knowledge base for Platonic Earth. It is **not** a message board.

Use it when a finding is important enough that future agents should know it, but it does not fit cleanly into an existing plan section.

## When to create a learning file

- A task reveals a constraint that changes later work.
- A dataset or model behaves differently than expected.
- A metric changes the interpretation of the project.
- A failure mode becomes important.
- A new artifact or script becomes part of the workflow.
- A plan assumption turns out to be wrong.

## Naming

```text
Txx-topic.md
general-topic.md
```

Examples:

```text
T04-dinov2-resolution-limit.md
T10-upper-arlington-cloud-cover.md
general-procrustes-centering-bug.md
```

## Rules

- Any agent may create a learning file.
- Do not edit another agent's learning file.
- Keep it short: what happened, evidence, impact, next action.
- If it changes a locked decision, write the learning and ask J/human.
- J reviews and merges learnings with the task branch.
- Update the relevant plan file if the change is clear.

## Template

Copy `_template.md`.
