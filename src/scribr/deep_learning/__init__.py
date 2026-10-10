"""Supervised deep learning transcription."""

from .dataset import SynthesizedMelodyDataset
from .features import (
    DEFAULT_F_MAX,
    DEFAULT_F_MIN,
    DEFAULT_HOP_LENGTH,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_NUM_FRAMES,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_TOP_DB,
    FeatureExtractor,
    LogMelSpectrogram,
    fix_length,
)
from .model import (
    DEFAULT_CHANNELS,
    DEFAULT_DROPOUT,
    PitchSequenceCNN,
)
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
    "DEFAULT_CHANNELS",
    "DEFAULT_DROPOUT",
    "DEFAULT_F_MAX",
    "DEFAULT_F_MIN",
    "DEFAULT_HOP_LENGTH",
    "DEFAULT_N_FFT",
    "DEFAULT_N_MELS",
    "DEFAULT_NUM_FRAMES",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_TOP_DB",
    "DeepLearningTranscriber",
    "FeatureExtractor",
    "HOLD_CLASS",
    "LogMelSpectrogram",
    "NOTE_OFF_CLASS",
    "NUM_CLASSES",
    "NUM_PITCH_CLASSES",
    "PitchSequenceCNN",
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
