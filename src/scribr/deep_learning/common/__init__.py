"""Shared per-step pitch-sequence machinery for deep learning architectures."""

from .dataset import SynthesizedMelodyDataset
from .features import FeatureExtractor, fix_length
from .targets import (
    HOLD_CLASS,
    NOTE_OFF_CLASS,
    NUM_CLASSES,
    NUM_PITCH_CLASSES,
    class_to_token,
    class_weights,
    classes_to_pitch_sequence,
    melody_from_classes,
    pitch_sequence_to_classes,
    token_group_masks,
    token_to_class,
)
from .transcriber import DeepLearningTranscriber

__all__ = [
    "DeepLearningTranscriber",
    "FeatureExtractor",
    "HOLD_CLASS",
    "NOTE_OFF_CLASS",
    "NUM_CLASSES",
    "NUM_PITCH_CLASSES",
    "SynthesizedMelodyDataset",
    "class_to_token",
    "class_weights",
    "classes_to_pitch_sequence",
    "fix_length",
    "melody_from_classes",
    "pitch_sequence_to_classes",
    "token_group_masks",
    "token_to_class",
]
