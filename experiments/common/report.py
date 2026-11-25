"""Render an experiment evaluation as JSON and Markdown reports."""

from __future__ import annotations

import importlib.metadata
import json
import platform
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .markdown import display, format_score, markdown_table
from .materialize import Manifest
from .predictions import TextPredictions
from .scoring import ExperimentEvaluation, group_counts
from .synthesis import resolve_soundfont_path
from .tracking import git_commit

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

# Top-level report keys that callers cannot override through `extra`.
_REPORT_KEYS = frozenset(
    {
        "experiment",
        "soundfont_sha256",
        "metadata",
        "overall",
        "per_instrument",
        "examples",
        "missing",
        "parse_errors",
    }
)


@dataclass(frozen=True, slots=True)
class ExampleReport:
    """One per-example row in a report.

    Attributes:
        id: Example ID.
        instrument: Instrument or other group the example belongs to.
        off_grid_onsets: Estimate onsets not on the dataset grid.
        off_grid_offsets: Estimate offsets not on the dataset grid.
        metrics: `mir_eval` metrics for the example.
    """

    id: str
    instrument: str
    off_grid_onsets: int
    off_grid_offsets: int
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", dict(self.metrics))

    def to_dict(self) -> dict[str, Any]:
        """Return the example as a JSON-compatible mapping."""
        return {
            "id": self.id,
            "instrument": self.instrument,
            "off_grid_onsets": self.off_grid_onsets,
            "off_grid_offsets": self.off_grid_offsets,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class InstrumentReport:
    """Per-instrument means and the number of examples they cover.

    Attributes:
        num_examples: Number of evaluated examples for the instrument.
        metrics: Mean of every metric across the instrument's examples.
    """

    num_examples: int
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", dict(self.metrics))

    def to_dict(self) -> dict[str, Any]:
        """Return the instrument record as a JSON-compatible mapping."""
        return {
            "num_examples": self.num_examples,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class ReportData:
    """Sections shared by experiment reports.

    Attributes:
        title: Markdown title.
        experiment: Experiment name written to JSON.
        metadata: Typed metadata written to JSON.
        metadata_rows: Metadata table rows in display order.
        overall: Overall metric means.
        per_instrument: Per-instrument records, in display order.
        examples: Per-example records, in evaluation order.
        missing: Example IDs without a transcription, or `None` to omit
            the section from the report.
        parse_errors: Parse error per example ID, or `None` to omit the
            section from the report.
        notes: Notes rendered after the overall metric table.
        show_group_counts: Render the group-count sentence before the
            per-instrument table.
        show_examples: Render the per-example table.
    """

    title: str
    experiment: str
    metadata: Mapping[str, Any]
    metadata_rows: tuple[tuple[str, str], ...]
    overall: Mapping[str, float]
    per_instrument: Mapping[str, InstrumentReport]
    examples: tuple[ExampleReport, ...]
    missing: tuple[str, ...] | None = None
    parse_errors: Mapping[str, str] | None = None
    notes: tuple[str, ...] = ()
    show_group_counts: bool = True
    show_examples: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "metadata_rows", tuple(self.metadata_rows))
        object.__setattr__(self, "overall", dict(self.overall))
        object.__setattr__(self, "per_instrument", dict(self.per_instrument))
        object.__setattr__(self, "examples", tuple(self.examples))

        if self.missing is not None:
            object.__setattr__(self, "missing", tuple(self.missing))

        if self.parse_errors is not None:
            object.__setattr__(self, "parse_errors", dict(self.parse_errors))

        object.__setattr__(self, "notes", tuple(self.notes))

    def to_dict(
        self, *, provenance: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Return the report as a JSON-compatible mapping.

        Args:
            provenance: Top-level fields written after `experiment`,
                such as the evaluation split or checkpoint.
        """
        data: dict[str, Any] = {"experiment": self.experiment}

        if provenance:
            data.update(provenance)

        data.update(
            {
                "metadata": dict(self.metadata),
                "overall": dict(self.overall),
                "per_instrument": {
                    instrument: entry.to_dict()
                    for instrument, entry in self.per_instrument.items()
                },
                "examples": [
                    example.to_dict() for example in self.examples
                ],
            }
        )

        if self.missing is not None:
            data["missing"] = list(self.missing)

        if self.parse_errors is not None:
            data["parse_errors"] = [
                {"id": example_id, "error": message}
                for example_id, message in self.parse_errors.items()
            ]

        return data


def write_report(
    manifest: Manifest,
    evaluation: ExperimentEvaluation,
    predictions: TextPredictions,
    artifacts_dir: str | Path,
    *,
    soundfont_path: str | Path | None = None,
    extra: Mapping[str, Any] | None = None,
    package_versions: Mapping[str, str] | None = None,
) -> tuple[Path, Path]:
    """Write `report.json` and `report.md` into `artifacts_dir`.

    Returns:
        The JSON and Markdown report paths.
    """
    report = _human_baseline_report(
        manifest,
        evaluation,
        predictions,
        soundfont_path,
        package_versions=package_versions,
    )

    return write_report_data(
        report,
        artifacts_dir,
        provenance={"soundfont_sha256": manifest.soundfont_sha256},
        extra=extra,
    )


def write_report_data(
    report: ReportData,
    artifacts_dir: str | Path,
    *,
    provenance: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write `report.json` and `report.md` into `artifacts_dir`.

    Args:
        report: Shared report sections to render.
        artifacts_dir: Directory for the reports.
        provenance: Top-level fields written after `experiment`.
        extra: Additional top-level fields written after the shared
            report sections.

    Returns:
        The JSON and Markdown report paths.
    """
    if extra:
        reserved = _REPORT_KEYS.union(provenance or {})
        conflicts = sorted(reserved.intersection(extra))

        if conflicts:
            raise ValueError(
                "extra keys collide with report keys: "
                + ", ".join(conflicts)
            )

    data = report.to_dict(provenance=provenance)

    if extra:
        data.update(extra)

    artifacts = Path(artifacts_dir)

    json_path = artifacts / "report.json"
    json_path.write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )

    markdown_path = artifacts / "report.md"
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    return json_path, markdown_path


def render_markdown(report: ReportData) -> str:
    """Render a report as Markdown."""
    lines = [
        f"# {report.title}",
        "",
        "## Metadata",
        "",
        markdown_table(("Field", "Value"), report.metadata_rows),
        "",
        "## Overall metrics",
        "",
        metric_table(report.overall),
        "",
    ]

    if report.notes:
        lines.extend([*report.notes, ""])

    lines.extend(["## Per-instrument metrics", ""])

    if report.show_group_counts:
        lines.extend([_group_count_line(report.per_instrument), ""])

    lines.extend(
        [
            _per_instrument_table(
                report.per_instrument,
                table_when_empty=report.show_group_counts,
            ),
            "",
        ]
    )

    if report.show_examples:
        lines.extend(
            [
                "## Examples",
                "",
                _examples_table(report.examples),
                "",
            ]
        )

    if report.missing:
        lines.extend(
            [
                "## Missing transcriptions",
                "",
                *(f"- `{example_id}`" for example_id in report.missing),
                "",
            ]
        )

    if report.parse_errors:
        lines.extend(
            [
                "## Parse errors",
                "",
                *(
                    f"- `{example_id}`: {message}"
                    for example_id, message in report.parse_errors.items()
                ),
                "",
            ]
        )

    return "\n".join(lines)


def metric_headers(
    metrics: Iterable[Mapping[str, float]],
) -> tuple[str, ...]:
    """Return the metric names across `metrics`, in display order."""
    names: dict[str, float] = {}

    for entry in metrics:
        names.update(entry)

    return tuple(ordered_metrics(names))


def metric_table(metrics: Mapping[str, float]) -> str:
    """Render overall metrics as a two-column table."""
    if not metrics:
        return "No examples were evaluated."

    return markdown_table(
        ("Metric", "Score"),
        tuple(
            (name, format_score(metrics[name]))
            for name in ordered_metrics(metrics)
        ),
    )


def ordered_metrics(metrics: Mapping[str, float]) -> list[str]:
    """Order metric names for display, with unknown names last."""
    ordered = [name for name in _METRIC_ORDER if name in metrics]
    ordered.extend(name for name in metrics if name not in _METRIC_ORDER)

    return ordered


def _human_baseline_report(
    manifest: Manifest,
    evaluation: ExperimentEvaluation,
    predictions: TextPredictions,
    soundfont_path: str | Path | None,
    *,
    package_versions: Mapping[str, str] | None = None,
) -> ReportData:
    counts = group_counts(evaluation)
    resolved_soundfont = resolve_soundfont_path(
        soundfont_path, required=False
    )
    versions = {
        name: _package_version(name) for name in ("scribr", "mir_eval")
    }

    if package_versions:
        versions.update(package_versions)

    metadata = {
        "git_commit": git_commit(),
        "python_version": platform.python_version(),
        "package_versions": versions,
        "soundfont_path": (
            None if resolved_soundfont is None else str(resolved_soundfont)
        ),
        "tempo": evaluation.tempo,
        "metric_options": dict(evaluation.metric_options),
        "num_examples": len(evaluation.per_example),
    }
    options = metadata["metric_options"]

    return ReportData(
        title=f"{_report_title(manifest.experiment)} report",
        experiment=manifest.experiment,
        metadata=metadata,
        metadata_rows=(
            ("Git commit", display(metadata["git_commit"])),
            ("Python", metadata["python_version"]),
            ("scribr", display(metadata["package_versions"]["scribr"])),
            (
                "mir_eval",
                display(metadata["package_versions"]["mir_eval"]),
            ),
            ("Tempo (BPM)", str(metadata["tempo"])),
            (
                "Onset tolerance (s)",
                str(options["onset_tolerance"]),
            ),
            (
                "Pitch tolerance (cents)",
                str(options["pitch_tolerance"]),
            ),
            ("Offset ratio", display(options["offset_ratio"])),
            (
                "Offset min tolerance (s)",
                str(options["offset_min_tolerance"]),
            ),
            ("SoundFont SHA-256", display(manifest.soundfont_sha256)),
            ("SoundFont path", display(metadata["soundfont_path"])),
            ("Examples evaluated", str(metadata["num_examples"])),
        ),
        overall=evaluation.metrics,
        per_instrument={
            group: InstrumentReport(
                num_examples=counts[group],
                metrics=metrics,
            )
            for group, metrics in evaluation.grouped_metrics.items()
        },
        examples=tuple(
            ExampleReport(
                id=example.example_id,
                instrument=example.group,
                off_grid_onsets=example.off_grid_onsets,
                off_grid_offsets=example.off_grid_offsets,
                metrics=example.metrics,
            )
            for example in evaluation.per_example
        ),
        missing=predictions.missing,
        parse_errors=predictions.errors,
        notes=(_RELEASE_TAIL_NOTE,),
    )


def _per_instrument_table(
    per_instrument: Mapping[str, InstrumentReport],
    *,
    table_when_empty: bool,
) -> str:
    if not per_instrument and not table_when_empty:
        return "No examples were evaluated."

    headers = metric_headers(
        entry.metrics for entry in per_instrument.values()
    )

    return markdown_table(
        ("Instrument", "N") + headers,
        tuple(
            (instrument, str(entry.num_examples))
            + tuple(
                format_score(entry.metrics.get(name))
                for name in headers
            )
            for instrument, entry in per_instrument.items()
        ),
    )


def _examples_table(examples: Iterable[ExampleReport]) -> str:
    examples = tuple(examples)
    headers = metric_headers(example.metrics for example in examples)

    return markdown_table(
        (
            ("Example", "Instrument")
            + headers
            + ("Off-grid onsets", "Off-grid offsets")
        ),
        tuple(
            (example.id, example.instrument)
            + tuple(
                format_score(example.metrics.get(name))
                for name in headers
            )
            + (
                str(example.off_grid_onsets),
                str(example.off_grid_offsets),
            )
            for example in examples
        ),
    )


def _group_count_line(
    per_instrument: Mapping[str, InstrumentReport],
) -> str:
    counts = [entry.num_examples for entry in per_instrument.values()]

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


def _report_title(experiment: str) -> str:
    return experiment.replace("_", " ").strip().capitalize()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)

    except importlib.metadata.PackageNotFoundError:
        return None
