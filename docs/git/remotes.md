# Remotes

- Inspect the configured remotes and branch tracking before pushing.
- Preserve existing remotes unless changing them is explicitly requested.
- If the configured SSH remote is inaccessible, use an existing HTTPS remote
  or an explicit HTTPS URL.
- Do not change the existing `origin` URL merely to work around an
  environment-specific authentication issue.
