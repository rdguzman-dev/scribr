# Issues

Create GitHub Issues with `gh issue create`. Every issue carries one type
label, one priority label, and `👀 Needs Review` unless the maintainer has
already triaged it.

## Type

Apply exactly one:

- `🐛 bug`: existing behavior is wrong.
- `📘 docs`: documentation only.
- `♻️ refactor`: internal change that keeps behavior the same.
- `✨ feature`: a new capability.
- `🔬 experiment`: something to try, with an uncertain outcome.

## Priority

Apply exactly one:

- `🔴 High Priority`
- `🟡 Mid Priority`
- `🟢 Low Priority`

## Titles

A title describes the change and has no type prefix. The type label covers the
type, so `Add MIDI export` is right and `feat: add MIDI export` is wrong.
Write the title in the imperative mood, same as a commit subject.

## Descriptions

Keep the description brief, but include what a maintainer or their agent needs
to act on it: the problem or goal, the code it applies to, and any known
constraints. Leave out background that the referenced code or docs already
provide.

```bash
gh issue create \
  --title "Raise a clear error for a missing split" \
  --label "🐛 bug" \
  --label "🟡 Mid Priority" \
  --label "👀 Needs Review" \
  --body "load_dataset raises a bare AssertionError when a split is missing, which hides the split name. Raise ValueError with the split in the message so callers can tell which input was wrong."
```
