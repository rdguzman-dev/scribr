"""Tests for the pitch-sequence class mapping."""

from __future__ import annotations

import pytest

from scribr.deep_learning import (
    HOLD_CLASS,
    NOTE_OFF_CLASS,
    NUM_CLASSES,
    NUM_PITCH_CLASSES,
    class_to_token,
    classes_to_pitch_sequence,
    melody_from_classes,
    pitch_sequence_to_classes,
    token_to_class,
)
from scribr.representation import (
    DATASET_MAX_PITCH,
    DATASET_MIN_PITCH,
    HOLD_TOKEN,
    NOTE_OFF_TOKEN,
    Melody,
    decode_pitch_sequence,
)


def test_every_dataset_token_round_trips() -> None:
    tokens = [
        *range(DATASET_MIN_PITCH, DATASET_MAX_PITCH + 1),
        HOLD_TOKEN,
        NOTE_OFF_TOKEN,
    ]

    for token in tokens:
        assert class_to_token(token_to_class(token)) == token


def test_vocabulary_boundaries() -> None:
    assert NUM_PITCH_CLASSES == 88
    assert NUM_CLASSES == 90
    assert token_to_class(DATASET_MIN_PITCH) == 0
    assert token_to_class(DATASET_MAX_PITCH) == 87
    assert token_to_class(HOLD_TOKEN) == HOLD_CLASS == 88
    assert token_to_class(NOTE_OFF_TOKEN) == NOTE_OFF_CLASS == 89


@pytest.mark.parametrize("token", [0, 20, 109, 127, 130])
def test_invalid_token_raises(token: int) -> None:
    with pytest.raises(ValueError):
        token_to_class(token)


@pytest.mark.parametrize("index", [-1, NUM_CLASSES])
def test_invalid_class_raises(index: int) -> None:
    with pytest.raises(ValueError):
        class_to_token(index)


def test_sequence_conversion_round_trips() -> None:
    sequence = (60, HOLD_TOKEN, 62, NOTE_OFF_TOKEN, 129)

    assert classes_to_pitch_sequence(
        pitch_sequence_to_classes(sequence)
    ) == sequence


def test_melody_from_classes_matches_the_codec() -> None:
    sequence = (60, 128, 62, 129)
    classes = pitch_sequence_to_classes(sequence)

    assert melody_from_classes(classes) == Melody(
        decode_pitch_sequence(sequence)
    )
