"""TFRecord dataset access for Scribr."""

from .dataset import (
    DATA_ROOT_ENV_VAR,
    SPLITS,
    MelodyDataset,
    default_data_root,
)
from .example import (
    ATTRIBUTE_NAMES,
    CONTEXT_DESCRIPTION,
    PITCH_SEQUENCE_KEY,
    SEQUENCE_DESCRIPTION,
    melody_example_from_features,
)

__all__ = [
    "ATTRIBUTE_NAMES",
    "CONTEXT_DESCRIPTION",
    "DATA_ROOT_ENV_VAR",
    "PITCH_SEQUENCE_KEY",
    "SEQUENCE_DESCRIPTION",
    "SPLITS",
    "MelodyDataset",
    "default_data_root",
    "melody_example_from_features",
]
