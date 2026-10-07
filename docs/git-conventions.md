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

The agent may use read-only Git commands such as `git status`, but must not
create branches, stage files, commit, push, merge, rebase, or otherwise
modify Git state.

PR descriptions and merge commit descriptions must use the same
description and follow this structure:

```text
Add ..., including:

- ...
```

Regular commits should use only a concise one-line message and do not need
an extended description.
