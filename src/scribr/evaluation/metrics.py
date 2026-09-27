"""Note-level transcription metrics computed with `mir_eval`.

`mir_eval` measures time in seconds and pitch in Hertz, while a `Melody`
measures time in quarter-note beats and pitch in MIDI note numbers. This
module converts both before scoring. Passing MIDI numbers through as-is
would understate pitch error: `mir_eval` compares pitches in cents, and
a semitone computed from MIDI numbers looks like 28.6 cents instead of
100.
"""

import mir_eval
import numpy as np

from ..representation import Melody

# Defaults mirrored from `mir_eval.transcription.evaluate`.
DEFAULT_ONSET_TOLERANCE_SECONDS = 0.05
DEFAULT_PITCH_TOLERANCE_CENTS = 50.0
DEFAULT_OFFSET_RATIO = 0.2
DEFAULT_OFFSET_MIN_TOLERANCE_SECONDS = 0.05

_A4_PITCH = 69
_A4_HZ = 440.0


def melody_to_mir_eval(
    melody: Melody,
    tempo: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a `Melody` to `mir_eval` interval and pitch arrays.

    Args:
        melody: Melody to convert.
        tempo: Tempo in BPM used to convert quarter-note beats to
            seconds.

    Returns:
        A `(len(melody), 2)` float array of onset/offset times in
        seconds and a `(len(melody),)` float array of pitches in Hertz.
        Notes are sorted by onset.

    Raises:
        ValueError: If `tempo` is not positive.
    """
    if tempo <= 0:
        raise ValueError("tempo must be positive")

    notes = sorted(melody.notes, key=lambda note: (note.onset, note.pitch))
    seconds_per_beat = 60.0 / tempo

    intervals = np.zeros((len(notes), 2), dtype=float)
    pitches = np.zeros(len(notes), dtype=float)

    for index, note in enumerate(notes):
        intervals[index] = (
            note.onset * seconds_per_beat,
            note.offset * seconds_per_beat,
        )
        pitches[index] = _A4_HZ * 2.0 ** ((note.pitch - _A4_PITCH) / 12.0)

    return intervals, pitches


def evaluate(
    reference: Melody,
    estimate: Melody,
    *,
    tempo: float,
    onset_tolerance: float = DEFAULT_ONSET_TOLERANCE_SECONDS,
    pitch_tolerance: float = DEFAULT_PITCH_TOLERANCE_CENTS,
    offset_ratio: float | None = DEFAULT_OFFSET_RATIO,
    offset_min_tolerance: float = DEFAULT_OFFSET_MIN_TOLERANCE_SECONDS,
) -> dict[str, float]:
    """Score an estimated melody against a reference melody.

    Both melodies are converted to `mir_eval` units (seconds and Hertz)
    with `melody_to_mir_eval` before scoring. The result contains
    offset-aware note metrics (`Precision`, `Recall`, `F-measure`),
    onset-and-pitch metrics that ignore offsets (`*_no_offset`),
    onset-only metrics (`Onset_*`), and offset-only metrics (`Offset_*`).
    Set `offset_ratio=None` to drop the offset-aware and offset-only
    metrics.

    Following `mir_eval`, an empty reference or estimate scores 0 on
    every metric.

    Args:
        reference: Ground-truth melody.
        estimate: Transcribed melody.
        tempo: Tempo in BPM shared by both melodies.
        onset_tolerance: Allowed onset deviation in seconds.
        pitch_tolerance: Allowed pitch deviation in cents.
        offset_ratio: Allowed offset deviation as a fraction of the
            reference note duration, or `None` to ignore offsets.
        offset_min_tolerance: Lower bound for the offset tolerance in
            seconds.

    Returns:
        A mapping from metric name to score.
    """
    ref_intervals, ref_pitches = melody_to_mir_eval(reference, tempo)
    est_intervals, est_pitches = melody_to_mir_eval(estimate, tempo)

    scores = mir_eval.transcription.evaluate(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=pitch_tolerance,
        offset_ratio=offset_ratio,
        offset_min_tolerance=offset_min_tolerance,
    )

    return {name: float(score) for name, score in scores.items()}
