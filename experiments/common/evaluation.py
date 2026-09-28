"""Batch scoring of transcriptions against reference melodies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from scribr.evaluation import (
    DEFAULT_OFFSET_MIN_TOLERANCE_SECONDS,
    DEFAULT_OFFSET_RATIO,
    DEFAULT_ONSET_TOLERANCE_SECONDS,
    DEFAULT_PITCH_TOLERANCE_CENTS,
    evaluate,
)
from scribr.representation import STEPS_PER_QUARTER_NOTE, Melody

# A time is on the dataset grid when it lands on a step boundary.
_GRID_EPSILON_STEPS = 1e-9


@dataclass(frozen=True, slots=True)
class ExampleEvaluation:
    """Scores and off-grid counts for one example.

    Attributes:
        example_id: Example ID from the manifest.
        group: Group the example belongs to, such as its instrument.
        metrics: `mir_eval` metrics for the example.
        off_grid_onsets: Estimate onsets not on the dataset grid.
        off_grid_offsets: Estimate offsets not on the dataset grid.
    """

    example_id: str
    group: str
    metrics: Mapping[str, float]
    off_grid_onsets: int
    off_grid_offsets: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", dict(self.metrics))


@dataclass(frozen=True, slots=True)
class ExperimentEvaluation:
    """Per-example scores with overall and grouped means.

    Attributes:
        per_example: One record per evaluated example, in input order.
        metrics: Mean of every metric across all examples.
        grouped_metrics: Mean of every metric per group.
        tempo: Tempo in BPM used to convert beats to seconds.
        metric_options: `mir_eval` options used for the run.
    """

    per_example: tuple[ExampleEvaluation, ...]
    metrics: Mapping[str, float]
    grouped_metrics: Mapping[str, Mapping[str, float]]
    tempo: float
    metric_options: Mapping[str, float | None]

    def __post_init__(self) -> None:
        object.__setattr__(self, "per_example", tuple(self.per_example))
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(
            self,
            "grouped_metrics",
            {
                group: dict(metrics)
                for group, metrics in self.grouped_metrics.items()
            },
        )
        object.__setattr__(self, "metric_options", dict(self.metric_options))


def evaluate_predictions(
    items: Iterable[tuple[str, Melody, Melody, str]],
    *,
    tempo: float,
    onset_tolerance: float = DEFAULT_ONSET_TOLERANCE_SECONDS,
    pitch_tolerance: float = DEFAULT_PITCH_TOLERANCE_CENTS,
    offset_ratio: float | None = DEFAULT_OFFSET_RATIO,
    offset_min_tolerance: float = DEFAULT_OFFSET_MIN_TOLERANCE_SECONDS,
) -> ExperimentEvaluation:
    """Score `(example_id, reference, estimate, group)` items.

    Estimation times are scored exactly as written: nothing is snapped
    or quantized. Off-grid onset and offset counts are collected as
    informational data alongside the metrics.
    """
    per_example: list[ExampleEvaluation] = []
    totals: dict[str, float] = {}
    grouped_totals: dict[str, dict[str, float]] = {}
    grouped_counts: dict[str, int] = {}

    for example_id, reference, estimate, group in items:
        scores = evaluate(
            reference,
            estimate,
            tempo=tempo,
            onset_tolerance=onset_tolerance,
            pitch_tolerance=pitch_tolerance,
            offset_ratio=offset_ratio,
            offset_min_tolerance=offset_min_tolerance,
        )

        per_example.append(
            ExampleEvaluation(
                example_id=example_id,
                group=group,
                metrics=scores,
                off_grid_onsets=sum(
                    not _on_grid(note.onset) for note in estimate.notes
                ),
                off_grid_offsets=sum(
                    not _on_grid(note.offset) for note in estimate.notes
                ),
            )
        )

        group_totals = grouped_totals.setdefault(group, {})

        for name, score in scores.items():
            totals[name] = totals.get(name, 0.0) + score
            group_totals[name] = group_totals.get(name, 0.0) + score

        grouped_counts[group] = grouped_counts.get(group, 0) + 1

    count = len(per_example)

    overall = (
        {name: total / count for name, total in totals.items()}
        if count
        else {}
    )
    grouped = {
        group: {
            name: total / grouped_counts[group]
            for name, total in group_metrics.items()
        }
        for group, group_metrics in grouped_totals.items()
    }

    return ExperimentEvaluation(
        per_example=tuple(per_example),
        metrics=overall,
        grouped_metrics=grouped,
        tempo=tempo,
        metric_options={
            "onset_tolerance": onset_tolerance,
            "pitch_tolerance": pitch_tolerance,
            "offset_ratio": offset_ratio,
            "offset_min_tolerance": offset_min_tolerance,
        },
    )


def _on_grid(time: float) -> bool:
    steps = time * STEPS_PER_QUARTER_NOTE

    return abs(steps - round(steps)) < _GRID_EPSILON_STEPS
