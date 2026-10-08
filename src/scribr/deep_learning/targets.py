"""Class mapping between dataset pitch-sequence tokens and model targets.

The model predicts one class per dataset step. The 88 playable pitches
map to classes `0-87`, `HOLD` to `88`, and `NOTE_OFF` to `89`. Softmax
over these 90 classes encodes the fact that at most one note sounds at a
time. Decoding a class sequence with `melody_from_classes` goes through
the dataset codec, so training targets and decoded melodies cannot drift
apart.
"""

from collections.abc import Sequence

from ..representation import (
    DATASET_MAX_PITCH,
    DATASET_MIN_PITCH,
    HOLD_TOKEN,
    NOTE_OFF_TOKEN,
    Melody,
    decode_pitch_sequence,
)

NUM_PITCH_CLASSES = DATASET_MAX_PITCH - DATASET_MIN_PITCH + 1
HOLD_CLASS = NUM_PITCH_CLASSES
NOTE_OFF_CLASS = NUM_PITCH_CLASSES + 1
NUM_CLASSES = NUM_PITCH_CLASSES + 2


def token_to_class(token: int) -> int:
    """Convert a dataset pitch-sequence token to a class index.

    Args:
        token: Pitch (`21-108`), `HOLD_TOKEN`, or `NOTE_OFF_TOKEN`.

    Raises:
        ValueError: If the token is not a valid dataset token.
    """
    if token == HOLD_TOKEN:
        return HOLD_CLASS

    if token == NOTE_OFF_TOKEN:
        return NOTE_OFF_CLASS

    if DATASET_MIN_PITCH <= token <= DATASET_MAX_PITCH:
        return token - DATASET_MIN_PITCH

    raise ValueError(
        f"token {token} is not a dataset pitch-sequence token; expected "
        f"a pitch in [{DATASET_MIN_PITCH}, {DATASET_MAX_PITCH}], "
        f"{HOLD_TOKEN}, or {NOTE_OFF_TOKEN}"
    )


def class_to_token(index: int) -> int:
    """Convert a class index back to a dataset token.

    Args:
        index: Class index in `[0, NUM_CLASSES)`.

    Raises:
        ValueError: If the index is out of range.
    """
    if not 0 <= index < NUM_CLASSES:
        raise ValueError(
            f"class {index} is outside [0, {NUM_CLASSES})"
        )

    if index == HOLD_CLASS:
        return HOLD_TOKEN

    if index == NOTE_OFF_CLASS:
        return NOTE_OFF_TOKEN

    return DATASET_MIN_PITCH + index


def pitch_sequence_to_classes(sequence: Sequence[int]) -> tuple[int, ...]:
    """Convert a dataset pitch sequence to class indices."""
    return tuple(token_to_class(token) for token in sequence)


def classes_to_pitch_sequence(classes: Sequence[int]) -> tuple[int, ...]:
    """Convert class indices back to a dataset pitch sequence."""
    return tuple(class_to_token(index) for index in classes)


def melody_from_classes(classes: Sequence[int]) -> Melody:
    """Decode per-step class indices into a `Melody`."""
    return Melody(decode_pitch_sequence(classes_to_pitch_sequence(classes)))
