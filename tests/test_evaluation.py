"""Tests for mir_eval-based transcription evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from scribr.evaluation import (
    evaluate,
    evaluate_transcriber,
    melody_to_mir_eval,
)
from scribr.representation import Melody, MelodyExample, Note


def hz(pitch: int) -> float:
    return 440.0 * 2.0 ** ((pitch - 69) / 12.0)


def test_melody_to_mir_eval_converts_units() -> None:
    melody = Melody((Note(69, 0.5, 1.0),))

    intervals, pitches = melody_to_mir_eval(melody, tempo=60.0)

    np.testing.assert_allclose(intervals, [[0.5, 1.0]])
    np.testing.assert_allclose(pitches, [440.0])


def test_melody_to_mir_eval_sorts_notes_by_onset() -> None:
    melody = Melody((Note(62, 1.0, 1.25), Note(60, 0.0, 0.25)))

    intervals, pitches = melody_to_mir_eval(melody, tempo=120.0)

    np.testing.assert_allclose(intervals, [[0.0, 0.125], [0.5, 0.625]])
    np.testing.assert_allclose(pitches, [hz(60), hz(62)])


def test_melody_to_mir_eval_handles_empty_melody() -> None:
    intervals, pitches = melody_to_mir_eval(Melody(), tempo=120.0)

    assert intervals.shape == (0, 2)
    assert pitches.shape == (0,)


def test_melody_to_mir_eval_rejects_non_positive_tempo() -> None:
    with pytest.raises(ValueError, match="tempo"):
        melody_to_mir_eval(Melody(), tempo=0.0)


def test_perfect_transcription_scores_one() -> None:
    melody = Melody((Note(60, 0.0, 0.5), Note(62, 1.0, 1.5)))

    scores = evaluate(melody, melody, tempo=120.0)

    assert scores["F-measure"] == 1.0
    assert scores["F-measure_no_offset"] == 1.0
    assert scores["Onset_F-measure"] == 1.0


def test_semitone_error_is_not_a_pitch_match() -> None:
    reference = Melody((Note(60, 0.0, 0.5),))
    estimate = Melody((Note(61, 0.0, 0.5),))

    scores = evaluate(reference, estimate, tempo=120.0)

    assert scores["F-measure"] == 0.0
    assert scores["F-measure_no_offset"] == 0.0
    # Onset-only metrics ignore pitch entirely.
    assert scores["Onset_F-measure"] == 1.0


def test_onset_tolerance_is_configurable() -> None:
    reference = Melody((Note(60, 0.0, 1.0),))
    estimate = Melody((Note(60, 0.12, 1.12),))  # 60 ms late at 120 BPM

    late = evaluate(reference, estimate, tempo=120.0)
    tolerated = evaluate(reference, estimate, tempo=120.0, onset_tolerance=0.1)

    assert late["F-measure_no_offset"] == 0.0
    assert tolerated["F-measure_no_offset"] == 1.0


def test_note_metrics_require_matching_offsets() -> None:
    reference = Melody((Note(60, 0.0, 0.5),))
    estimate = Melody((Note(60, 0.0, 1.0),))

    scores = evaluate(reference, estimate, tempo=120.0)

    assert scores["F-measure_no_offset"] == 1.0
    assert scores["Onset_F-measure"] == 1.0
    assert scores["F-measure"] == 0.0
    assert scores["Offset_F-measure"] == 0.0


def test_offset_ratio_none_skips_offset_metrics() -> None:
    reference = Melody((Note(60, 0.0, 0.5),))
    estimate = Melody((Note(60, 0.0, 1.0),))

    scores = evaluate(reference, estimate, tempo=120.0, offset_ratio=None)

    assert "Precision" not in scores
    assert "Offset_F-measure" not in scores
    assert scores["F-measure_no_offset"] == 1.0


def test_empty_estimate_scores_zero() -> None:
    reference = Melody((Note(60, 0.0, 0.5),))

    with pytest.warns(UserWarning, match="Estimated notes are empty"):
        scores = evaluate(reference, Melody(), tempo=120.0)

    assert scores["F-measure"] == 0.0
    assert scores["Onset_F-measure"] == 0.0


class FakeSynthesizer:
    """Minimal `Synthesizer` stand-in that avoids loading FluidSynth."""

    def __init__(
        self, tempo: float = 120.0, sample_rate: int = 22_050
    ) -> None:
        self.tempo = tempo
        self.sample_rate = sample_rate
        self.synthesized: list[Melody] = []

    def synthesize(self, melody: Melody) -> np.ndarray:
        self.synthesized.append(melody)
        return np.zeros(8, dtype=np.float32)


class FakeTranscriber:
    def __init__(self, estimates: list[Melody]) -> None:
        self.estimates = list(estimates)
        self.calls: list[tuple[int, float]] = []

    def transcribe(
        self, audio: np.ndarray, sample_rate: int, tempo: float
    ) -> Melody:
        self.calls.append((sample_rate, tempo))
        return self.estimates.pop(0)


def test_evaluate_transcriber_averages_metrics_over_examples() -> None:
    first = Melody((Note(60, 0.0, 0.5),))
    second = Melody((Note(62, 0.25, 0.75),))
    examples = [MelodyExample(first), MelodyExample(second)]
    synthesizer = FakeSynthesizer(tempo=90.0, sample_rate=16_000)
    transcriber = FakeTranscriber(
        [first, Melody((Note(65, 0.0, 0.5),))]
    )

    report = evaluate_transcriber(transcriber, examples, synthesizer)

    assert report.num_examples == 2
    assert report.metrics["F-measure"] == pytest.approx(0.5)
    assert report.metrics["F-measure_no_offset"] == pytest.approx(0.5)
    assert report.metrics["Onset_F-measure"] == pytest.approx(0.5)
    assert synthesizer.synthesized == [first, second]
    assert transcriber.calls == [(16_000, 90.0), (16_000, 90.0)]


def test_evaluate_transcriber_forwards_metric_options() -> None:
    melody = Melody((Note(60, 0.0, 0.5),))
    examples = [MelodyExample(melody)]
    transcriber = FakeTranscriber([melody])

    report = evaluate_transcriber(
        transcriber, examples, FakeSynthesizer(), offset_ratio=None
    )

    assert "F-measure" not in report.metrics
    assert report.metrics["F-measure_no_offset"] == 1.0


def test_evaluate_transcriber_rejects_empty_examples() -> None:
    with pytest.raises(ValueError, match="at least one"):
        evaluate_transcriber(FakeTranscriber([]), [], FakeSynthesizer())
