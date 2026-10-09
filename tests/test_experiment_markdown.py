"""Tests for the shared Markdown rendering helpers."""

from experiments.common.markdown import (
    display,
    format_score,
    markdown_table,
)


def test_markdown_table_renders_headers_and_rows() -> None:
    rendered = markdown_table(
        ("Metric", "Score"),
        (("F-measure", "1.0000"), ("Recall", "")),
    )

    assert rendered == (
        "| Metric | Score |\n"
        "| --- | --- |\n"
        "| F-measure | 1.0000 |\n"
        "| Recall |  |"
    )


def test_format_score_rounds_to_four_decimals() -> None:
    assert format_score(0.5) == "0.5000"
    assert format_score(None) == ""


def test_display_falls_back_to_na() -> None:
    assert display(120.0) == "120.0"
    assert display(None) == "n/a"
