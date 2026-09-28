"""Render an experiment evaluation as JSON and Markdown reports."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from scribr.synthesis import SOUNDFONT_ENV_VAR

from .evaluation import ExperimentEvaluation
from .materialize import Manifest
from .predictions import TextPredictions

# Metric display order; anything else follows in the order it appears.
_METRIC_ORDER = (
    "F-measure_no_offset",
    "Precision_no_offset",
    "Recall_no_offset",
    "Average_Overlap_Ratio_no_offset",
    "Onset_F-measure",
    "Onset_Precision",
    "Onset_Recall",
    "F-measure",
    "Precision",
    "Recall",
    "Average_Overlap_Ratio",
    "Offset_F-measure",
    "Offset_Precision",
    "Offset_Recall",
)

_RELEASE_TAIL_NOTE = (
    "Offset-aware metrics are noisier because synthesized instrument "
    "release tails should not be transcribed as notes."
)


def write_report(
    manifest: Manifest,
    evaluation: ExperimentEvaluation,
    predictions: TextPredictions,
    artifacts_dir: str | Path,
    *,
    soundfont_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write `report.json` and `report.md` into `artifacts_dir`.

    Returns:
        The JSON and Markdown report paths.
    """
    artifacts = Path(artifacts_dir)
    data = _report_dict(manifest, evaluation, predictions, soundfont_path)

    json_path = artifacts / "report.json"
    json_path.write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )

    markdown_path = artifacts / "report.md"
    markdown_path.write_text(_markdown(data), encoding="utf-8")

    return json_path, markdown_path


def _report_dict(
    manifest: Manifest,
    evaluation: ExperimentEvaluation,
    predictions: TextPredictions,
    soundfont_path: str | Path | None,
) -> dict:
    group_counts = _group_counts(evaluation)
    package_versions = {
        name: _package_version(name) for name in ("scribr", "mir_eval")
    }

    return {
        "experiment": manifest.experiment,
        "soundfont_sha256": manifest.soundfont_sha256,
        "metadata": {
            "git_commit": _git_commit(),
            "python_version": platform.python_version(),
            "package_versions": package_versions,
            "soundfont_path": _resolved_soundfont(soundfont_path),
            "tempo": evaluation.tempo,
            "metric_options": dict(evaluation.metric_options),
            "num_examples": len(evaluation.per_example),
        },
        "overall": dict(evaluation.metrics),
        "per_instrument": {
            group: {
                "num_examples": group_counts[group],
                "metrics": dict(metrics),
            }
            for group, metrics in evaluation.grouped_metrics.items()
        },
        "examples": [
            {
                "id": example.example_id,
                "instrument": example.group,
                "off_grid_onsets": example.off_grid_onsets,
                "off_grid_offsets": example.off_grid_offsets,
                "metrics": dict(example.metrics),
            }
            for example in evaluation.per_example
        ],
        "missing": list(predictions.missing),
        "parse_errors": [
            {"id": example_id, "error": message}
            for example_id, message in predictions.errors.items()
        ],
    }


def _markdown(data: Mapping) -> str:
    metadata = data["metadata"]
    tempo = metadata["tempo"]
    options = metadata["metric_options"]

    lines = [
        f"# {_report_title(data['experiment'])} report",
        "",
        "## Metadata",
        "",
        _markdown_table(
            ("Field", "Value"),
            (
                ("Git commit", _display(metadata["git_commit"])),
                ("Python", metadata["python_version"]),
                ("scribr", _display(metadata["package_versions"]["scribr"])),
                (
                    "mir_eval",
                    _display(metadata["package_versions"]["mir_eval"]),
                ),
                ("Tempo (BPM)", str(tempo)),
                (
                    "Onset tolerance (s)",
                    str(options["onset_tolerance"]),
                ),
                (
                    "Pitch tolerance (cents)",
                    str(options["pitch_tolerance"]),
                ),
                ("Offset ratio", _display(options["offset_ratio"])),
                (
                    "Offset min tolerance (s)",
                    str(options["offset_min_tolerance"]),
                ),
                ("SoundFont SHA-256", _display(data["soundfont_sha256"])),
                ("SoundFont path", _display(metadata["soundfont_path"])),
                ("Examples evaluated", str(metadata["num_examples"])),
            ),
        ),
        "",
        "## Overall metrics",
        "",
        _metric_table(data["overall"]),
        "",
        _RELEASE_TAIL_NOTE,
        "",
        "## Per-instrument metrics",
        "",
        _group_count_line(data["per_instrument"]),
        "",
    ]

    per_instrument_rows = []
    for instrument, entry in data["per_instrument"].items():
        metrics = entry["metrics"]
        per_instrument_rows.append(
            (instrument, str(entry["num_examples"]))
            + tuple(
                _format_score(metrics.get(name))
                for name in _ordered_metrics(metrics)
            )
        )

    lines.extend(
        [
            _markdown_table(
                (
                    ("Instrument", "N")
                    + _metric_headers(data["per_instrument"].values())
                ),
                per_instrument_rows,
            ),
            "",
            "## Examples",
            "",
            _markdown_table(
                (
                    ("Example", "Instrument")
                    + _metric_headers(data["examples"])
                    + ("Off-grid onsets", "Off-grid offsets")
                ),
                tuple(
                    (
                        example["id"],
                        example["instrument"],
                    )
                    + tuple(
                        _format_score(example["metrics"].get(name))
                        for name in _ordered_metrics(example["metrics"])
                    )
                    + (
                        str(example["off_grid_onsets"]),
                        str(example["off_grid_offsets"]),
                    )
                    for example in data["examples"]
                ),
            ),
            "",
        ]
    )

    if data["missing"]:
        lines.extend(
            [
                "## Missing transcriptions",
                "",
                *(f"- `{example_id}`" for example_id in data["missing"]),
                "",
            ]
        )

    if data["parse_errors"]:
        lines.extend(
            [
                "## Parse errors",
                "",
                *(
                    f"- `{error['id']}`: {error['error']}"
                    for error in data["parse_errors"]
                ),
                "",
            ]
        )

    return "\n".join(lines)


def _metric_headers(entries: Iterable[Mapping]) -> tuple[str, ...]:
    metrics: dict[str, float] = {}

    for entry in entries:
        metrics.update(entry["metrics"])

    return tuple(_ordered_metrics(metrics))


def _metric_table(metrics: Mapping[str, float]) -> str:
    if not metrics:
        return "No examples were evaluated."

    return _markdown_table(
        ("Metric", "Score"),
        tuple(
            (name, _format_score(metrics[name]))
            for name in _ordered_metrics(metrics)
        ),
    )


def _group_count_line(per_instrument: Mapping) -> str:
    counts = [entry["num_examples"] for entry in per_instrument.values()]

    if not counts:
        return "No examples were evaluated."

    unique = set(counts)

    if len(unique) == 1:
        count = unique.pop()
        noun = "example" if count == 1 else "examples"

        return (
            "Per-instrument means are based on "
            f"{count} {noun} each."
        )

    return "Per-instrument means are based on varying example counts."


def _ordered_metrics(metrics: Mapping[str, float]) -> list[str]:
    ordered = [name for name in _METRIC_ORDER if name in metrics]
    ordered.extend(name for name in metrics if name not in _METRIC_ORDER)

    return ordered


def _markdown_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _format_score(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _report_title(experiment: str) -> str:
    return experiment.replace("_", " ").strip().capitalize()


def _display(value: object | None) -> str:
    return "n/a" if value is None else str(value)


def _group_counts(evaluation: ExperimentEvaluation) -> dict[str, int]:
    counts: dict[str, int] = {}

    for example in evaluation.per_example:
        counts[example.group] = counts.get(example.group, 0) + 1

    return counts


def _git_commit() -> str | None:
    repository = Path(__file__).resolve().parents[2]

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=repository,
        )

    except OSError:
        return None

    if result.returncode != 0:
        return None

    return result.stdout.strip() or None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)

    except importlib.metadata.PackageNotFoundError:
        return None


def _resolved_soundfont(
    soundfont_path: str | Path | None,
) -> str | None:
    configured = (
        str(soundfont_path)
        if soundfont_path is not None
        else os.environ.get(SOUNDFONT_ENV_VAR)
    )

    if not configured:
        return None

    return str(Path(configured).expanduser())
