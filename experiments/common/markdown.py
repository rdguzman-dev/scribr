"""Render plain values as Markdown."""

from collections.abc import Iterable, Sequence


def markdown_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
) -> str:
    """Render a pipe table with `headers` and `rows`."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def format_score(value: float | None) -> str:
    """Format a score to four decimals, or empty when missing."""
    return "" if value is None else f"{value:.4f}"


def display(value: object | None) -> str:
    """Format a value for display, or `n/a` when missing."""
    return "n/a" if value is None else str(value)
