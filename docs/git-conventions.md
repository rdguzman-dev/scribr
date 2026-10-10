# Git conventions

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

PR descriptions and merge commit descriptions must use the same
description and follow this structure:

```text
<Verb> ..., including:

- ...
```

Each bullet point in the merge commit description should begin with a capital
letter (unless it is escaped in backticks like `src/...`) and end with a
period.

Merge commit titles should start with `[merge]` and end with the PR number. For
example:

```text
[merge] refactor(experiments): remove the dataset candidate prefix #20
```

Regular commits should not use an extended description.
