"""Evaluate a `Transcriber` over a set of reference melodies."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ..representation import MelodyExample
from ..synthesis import Synthesizer
from ..transcription import Transcriber
from .metrics import (
    DEFAULT_OFFSET_MIN_TOLERANCE_SECONDS,
    DEFAULT_OFFSET_RATIO,
    DEFAULT_ONSET_TOLERANCE_SECONDS,
    DEFAULT_PITCH_TOLERANCE_CENTS,
    evaluate,
)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Mean metrics across the evaluated examples.

    Attributes:
        num_examples: Number of examples evaluated.
        metrics: Mean of each `mir_eval` metric across the examples.
    """

    num_examples: int
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", dict(self.metrics))


def evaluate_transcriber(
    transcriber: Transcriber,
    examples: Iterable[MelodyExample],
    synthesizer: Synthesizer,
    *,
    onset_tolerance: float = DEFAULT_ONSET_TOLERANCE_SECONDS,
    pitch_tolerance: float = DEFAULT_PITCH_TOLERANCE_CENTS,
    offset_ratio: float | None = DEFAULT_OFFSET_RATIO,
    offset_min_tolerance: float = DEFAULT_OFFSET_MIN_TOLERANCE_SECONDS,
) -> EvaluationReport:
    """Run a transcriber over examples and average the metrics.

    Each reference melody is rendered to audio with `synthesizer`, and
    the sample rate and tempo used for the rendering are passed to
    `Transcriber.transcribe`, so estimates come back in the same units as
    the reference.

    Every metric returned by `evaluate` is averaged across the examples.
    `mir_eval` scores empty references and estimates as 0, so silent
    examples lower the mean; filter them out beforehand if that is not
    wanted.

    Args:
        transcriber: Approach to evaluate.
        examples: Reference examples to transcribe.
        synthesizer: Synthesizer used to render each reference melody.
        onset_tolerance: Allowed onset deviation in seconds.
        pitch_tolerance: Allowed pitch deviation in cents.
        offset_ratio: Allowed offset deviation as a fraction of the
            reference note duration, or `None` to ignore offsets.
        offset_min_tolerance: Lower bound for the offset tolerance in
            seconds.

    Returns:
        Mean metrics across the examples.

    Raises:
        ValueError: If `examples` is empty.
    """
    totals: dict[str, float] = {}
    count = 0

    for example in examples:
        audio = synthesizer.synthesize(example.melody)
        estimate = transcriber.transcribe(
            audio,
            synthesizer.sample_rate,
            synthesizer.tempo,
        )

        scores = evaluate(
            example.melody,
            estimate,
            tempo=synthesizer.tempo,
            onset_tolerance=onset_tolerance,
            pitch_tolerance=pitch_tolerance,
            offset_ratio=offset_ratio,
            offset_min_tolerance=offset_min_tolerance,
        )

        for name, score in scores.items():
            totals[name] = totals.get(name, 0.0) + score

        count += 1

    if count == 0:
        raise ValueError("examples must contain at least one melody")

    return EvaluationReport(
        num_examples=count,
        metrics={name: total / count for name, total in totals.items()},
    )
