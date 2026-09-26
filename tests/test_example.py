"""Tests for turning decoded TFRecord records into MelodyExample objects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tfrecord.torch import TFRecordDataset

from scribr.data.example import (
    ATTRIBUTE_NAMES,
    CONTEXT_DESCRIPTION,
    PITCH_SEQUENCE_KEY,
    SEQUENCE_DESCRIPTION,
    melody_example_from_features,
)
from scribr.representation.melody import Melody, Note

EXPECTED_SEQUENCE = (
    63,
    128,
    128,
    64,
    128,
    128,
    61,
    128,
    128,
    128,
    128,
    128,
    128,
    128,
    128,
    129,
    129,
    129,
    57,
    128,
    128,
    61,
    128,
    128,
    64,
    128,
    128,
    129,
    66,
    128,
    128,
    128,
    68,
    128,
    128,
    128,
    69,
    128,
    128,
    128,
    128,
    128,
    68,
    128,
    128,
    66,
    128,
    128,
    64,
    128,
    128,
    128,
    128,
    128,
    128,
    128,
    128,
    128,
    63,
    128,
    128,
    61,
    128,
    128,
)

EXPECTED_NOTES = (
    Note(63, 0.0, 0.75),
    Note(64, 0.75, 1.5),
    Note(61, 1.5, 3.75),
    Note(57, 4.5, 5.25),
    Note(61, 5.25, 6.0),
    Note(64, 6.0, 6.75),
    Note(66, 7.0, 8.0),
    Note(68, 8.0, 9.0),
    Note(69, 9.0, 10.5),
    Note(68, 10.5, 11.25),
    Note(66, 11.25, 12.0),
    Note(64, 12.0, 14.5),
    Note(63, 14.5, 15.25),
    Note(61, 15.25, 16.0),
)


def _first_record(path: Path):
    dataset = TFRecordDataset(
        str(path),
        None,
        description=CONTEXT_DESCRIPTION,
        sequence_description=SEQUENCE_DESCRIPTION,
    )
    return next(iter(dataset))


def test_parses_real_fixture_record(fixture_head: Path) -> None:
    context, features = _first_record(fixture_head)
    example = melody_example_from_features(context, features)

    assert example.melody.notes == EXPECTED_NOTES
    assert example.pitch_sequence == EXPECTED_SEQUENCE
    assert example.melody.duration == 16.0


def test_attributes_are_preserved(fixture_head: Path) -> None:
    context, features = _first_record(fixture_head)
    example = melody_example_from_features(context, features)

    assert set(ATTRIBUTE_NAMES) <= set(example.attributes)
    assert example.attributes["note_density"] == pytest.approx(0.21875, abs=1e-6)
    assert example.attributes["ratio_hold_note_steps"] == pytest.approx(
        0.71875, abs=1e-6
    )
    assert all(isinstance(value, float) for value in example.attributes.values())


def test_empty_context_value_becomes_nan() -> None:
    context = {name: np.zeros(1, dtype=np.float32) for name in ATTRIBUTE_NAMES}
    context["toussaint"] = np.zeros(0, dtype=np.float32)
    features = {PITCH_SEQUENCE_KEY: [np.array([60, 128], dtype=np.int64)]}

    example = melody_example_from_features(context, features)

    assert np.isnan(example.attributes["toussaint"])
    assert example.melody == Melody((Note(60, 0.0, 0.5),))


def test_missing_pitch_sequence_raises() -> None:
    context = {name: np.zeros(1, dtype=np.float32) for name in ATTRIBUTE_NAMES}

    with pytest.raises(KeyError, match=PITCH_SEQUENCE_KEY):
        melody_example_from_features(context, {})
