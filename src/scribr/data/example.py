"""Conversion from decoded TFRecord records to `MelodyExample`.

The `tfrecord[torch]` reader returns each `SequenceExample` record as a
`(context, features)` pair of numpy arrays. The context holds the 13
attributes computed by the dataset authors; the feature list holds the
64-step pitch sequence. This module maps that layout onto the canonical
`MelodyExample`, so the rest of Scribr never sees the library's output
shape.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from ..representation.encoding import decode_pitch_sequence
from ..representation.melody import Melody, MelodyExample

# Feature-list key holding the 64-step pitch sequence.
PITCH_SEQUENCE_KEY = "pitch_seq"

# The 13 context attributes documented by the dataset authors.
ATTRIBUTE_NAMES: tuple[str, ...] = (
    "toussaint",
    "note_density",
    "pitch_range",
    "contour",
    "note_change_ratio",
    "dynamic_range",
    "len_longest_rep_section",
    "repetitive_section_ratio",
    "ratio_hold_note_steps",
    "ratio_note_off_steps",
    "unique_notes_ratio",
    "unique_bigrams_ratio",
    "unique_trigrams_ratio",
)

# Descriptions for `tfrecord.torch.TRRecordDataset` decoding.
CONTEXT_DESCRIPTION: dict[str, str] = dict.fromkeys(ATTRIBUTE_NAMES, "float")
SEQUENCE_DESCRIPTION: dict[str, str] = {PITCH_SEQUENCE_KEY: "int"}


def melody_example_from_features(
    context: Mapping[str, np.ndarray],
    features: Mapping[str, Sequence[np.ndarray]],
) -> MelodyExample:
    """Convert one decoded dataset record into a `MelodyExample`."""
    sequence = _pitch_sequence(features)

    return MelodyExample(
        melody=Melody(decode_pitch_sequence(sequence)),
        pitch_sequence=sequence,
        attributes=_attributes(context),
    )


def _pitch_sequence(
    features: Mapping[str, Sequence[np.ndarray]],
) -> tuple[int, ...]:
    return tuple(
        int(token)
        for feature in features[PITCH_SEQUENCE_KEY]
        for token in feature
    )


def _attributes(context: Mapping[str, np.ndarray]) -> dict[str, float]:
    return {
        key: float(values[0]) if values.size else float("nan")
        for key, values in context.items()
    }
