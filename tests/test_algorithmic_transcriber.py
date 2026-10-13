"""Audio-to-note behavior and segmentation edge cases."""

import numpy as np
import pytest
from scribr.evaluation import evaluate
from scribr.representation import Melody, Note

from scribr.algorithmic import AlgorithmicConfig, AlgorithmicTranscriber
from scribr.algorithmic.transcriber import _segment_notes


def _tone(pitch, duration=0.5, sample_rate=22050):
    time = np.arange(round(duration * sample_rate)) / sample_rate
    frequency = 440 * 2 ** ((pitch - 69) / 12)
    return (0.5 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


@pytest.mark.parametrize(
    "audio", [np.array([], dtype=np.float32), np.zeros(4096)]
)
def test_silence(audio):
    assert AlgorithmicTranscriber().transcribe(audio, 22050, 120) == Melody()


@pytest.mark.parametrize(
    "audio,sample_rate,tempo",
    [
        (np.zeros((2, 10)), 22050, 120),
        (np.array([np.nan]), 22050, 120),
        (np.array([np.inf]), 22050, 120),
        (np.array([1.1]), 22050, 120),
        (np.zeros(10), 0, 120),
        (np.zeros(10), 22050, 0),
        (np.zeros(10), 22050, float("nan")),
    ],
)
def test_invalid_inputs(audio, sample_rate, tempo):
    with pytest.raises(ValueError):
        AlgorithmicTranscriber().transcribe(audio, sample_rate, tempo)


@pytest.mark.parametrize(
    "options",
    [
        {"hop_length": 0},
        {"frame_length": 10},
        {"max_pitch": 128},
        {"min_pitch": 70, "max_pitch": 69},
        {"silence_db": 0},
        {"min_note_seconds": 0},
        {"max_gap_seconds": float("nan")},
        {"sample_rate": 8000},
    ],
)
def test_invalid_config(options):
    with pytest.raises(ValueError):
        AlgorithmicConfig(**options)


def test_repeated_pitch_attacks_stay_separate():
    estimate = _segment_notes(
        np.full(100, 69), np.array([0, 50]), 0.01, 1, 120, AlgorithmicConfig()
    )
    assert estimate == Melody((Note(69, 0, 1), Note(69, 1, 2)))


def test_rests_and_short_voicing_dropout():
    pitches = np.r_[
        np.full(20, 60),
        [-1, -1],
        np.full(20, 60),
        np.full(15, -1),
        np.full(20, 62),
    ]
    estimate = _segment_notes(
        pitches, np.array([]), 0.01, 0.77, 60, AlgorithmicConfig()
    )
    assert [note.pitch for note in estimate] == [60, 62]
    np.testing.assert_allclose(
        [(note.onset, note.offset) for note in estimate],
        [(0, 0.42), (0.57, 0.77)],
    )


def test_pitch_change_without_attack_and_end_clipping():
    estimate = _segment_notes(
        np.r_[np.full(30, 60), np.full(30, 64)],
        np.array([]),
        0.01,
        0.595,
        120,
        AlgorithmicConfig(),
    )
    assert estimate == Melody((Note(60, 0, 0.6), Note(64, 0.6, 1.19)))


def test_brief_pitch_excursion_is_not_a_note():
    pitches = np.r_[np.full(20, 60), [72], np.full(20, 60)]
    estimate = _segment_notes(
        pitches, np.array([]), 0.01, 0.41, 60, AlgorithmicConfig()
    )
    assert estimate == Melody((Note(60, 0, 0.41),))


@pytest.mark.parametrize("sample_rate,tempo", [(22050, 120), (44100, 60)])
def test_real_pyin_tracks_a_tone_after_leading_silence(sample_rate, tempo):
    audio = np.r_[
        np.zeros(round(0.2 * sample_rate)),
        _tone(69, sample_rate=sample_rate),
        np.zeros(round(0.2 * sample_rate)),
    ].astype(np.float32)
    estimate = AlgorithmicTranscriber().transcribe(audio, sample_rate, tempo)
    reference = Melody((Note(69, 0.2 * tempo / 60, 0.7 * tempo / 60),))
    assert len(estimate) == 1
    assert estimate.notes[0].offset <= len(audio) / sample_rate * tempo / 60
    assert (
        evaluate(reference, estimate, tempo=tempo)["F-measure_no_offset"] == 1
    )


def test_real_pyin_distinguishes_notes_and_rest():
    audio = np.r_[_tone(60), np.zeros(4410), _tone(64), np.zeros(4410)].astype(
        np.float32
    )
    estimate = AlgorithmicTranscriber().transcribe(audio, 22050, 120)
    reference = Melody((Note(60, 0, 1), Note(64, 1.4, 2.4)))
    assert [note.pitch for note in estimate] == [60, 64]
    assert estimate.notes[0].offset < estimate.notes[1].onset
    assert evaluate(reference, estimate, tempo=120)["F-measure_no_offset"] == 1


def test_real_repeated_pitch_with_new_attack():
    tone = _tone(69)
    envelope = np.exp(-np.arange(len(tone)) / 5000)
    note = (tone * envelope).astype(np.float32)
    audio = np.r_[note, note, np.zeros(4410)].astype(np.float32)
    estimate = AlgorithmicTranscriber().transcribe(audio, 22050, 120)
    reference = Melody((Note(69, 0, 1), Note(69, 1, 2)))
    assert [note.pitch for note in estimate] == [69, 69]
    assert evaluate(reference, estimate, tempo=120)["F-measure_no_offset"] == 1
