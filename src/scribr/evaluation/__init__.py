"""Transcription evaluation against reference melodies with `mir_eval`."""

from .metrics import (
    DEFAULT_OFFSET_MIN_TOLERANCE_SECONDS,
    DEFAULT_OFFSET_RATIO,
    DEFAULT_ONSET_TOLERANCE_SECONDS,
    DEFAULT_PITCH_TOLERANCE_CENTS,
    evaluate,
    melody_to_mir_eval,
)
from .runner import EvaluationReport, evaluate_transcriber

__all__ = [
    "DEFAULT_OFFSET_MIN_TOLERANCE_SECONDS",
    "DEFAULT_OFFSET_RATIO",
    "DEFAULT_ONSET_TOLERANCE_SECONDS",
    "DEFAULT_PITCH_TOLERANCE_CENTS",
    "EvaluationReport",
    "evaluate",
    "evaluate_transcriber",
    "melody_to_mir_eval",
]
