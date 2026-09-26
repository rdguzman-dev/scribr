# Agent Instructions

## Project

Scribr is a CSC211H5 course project exploring different approaches
(algorithmic, supervised deep learning, and LLM-based) to automated music
transcription for quantized monophonic audio. Prioritize clarity, correctness,
and understandability over unnecessary abstraction or cleverness.

## Code Style

- Follow existing code conventions before introducing new patterns.
- Prefer simple, explicit implementations over abstractions without a
  demonstrated need.
- Comments should explain why, not restate what the code does.
- Use Google-style docstrings.
- Use concise docstrings and do not add documentation mechanically.
- Private helpers generally do not need docstrings unless their behavior
  is non-obvious or there is an important implementation detail to explain.
- Use backticks for Python identifiers in prose and docstrings.
- Do not add `Args:`, `Returns:`, or `Raises:` sections when they merely
  repeat information already apparent from the signature.
- For anything not listed above, default to the PEP 8 guidelines.

## Architecture

- Keep the canonical symbolic representation independent of dataset,
  synthesis, transcription, and model-specific implementation details.
- Prefer existing domain types over introducing duplicate representations.
- Do not introduce abstractions unless they solve a concrete problem.

## Working Style

- Inspect the existing implementation before making changes.
- Make the smallest change that satisfies the task.
- Preserve existing behavior unless the task explicitly requires changing it.
- Do not modify unrelated code merely to satisfy personal style preferences.
- Before considering a task complete, run the relevant tests and checks.

## Git conventions

This repository follows [Conventional Branch](https://conventionalbranch.org/)
for branch names and [Conventional Commits](https://www.conventionalcommits.org/)
for commits.

Preferred branch format:

```text
<type>/<verb>-<feature>
```

When proposing Git metadata, follow these conventions for branch names,
commit groups, commit types/messages, PR titles, and merge commit
titles/descriptions. Group changes into logically independent commits.

The agent may use read-only Git commands such as `git status`, but must not
create branches, stage files, commit, push, merge, rebase, or otherwise
modify Git state.

PR descriptions and merge commit descriptions must use the same
description and follow this structure:

```text
Add ... layer, including:

- ...
```

Regular commits should use only a concise one-line message and do not need
an extended description.

## Dependencies

- Do not add a dependency without first explaining why it is necessary.
- Prefer the project's existing dependencies and standard library where
  practical, but do not reinvent the wheel when a new external dependency
  would cleanly solve the given problem.
