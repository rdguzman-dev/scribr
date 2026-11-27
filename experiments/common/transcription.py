"""Load verified manifest recordings and score any `Transcriber`."""

import hashlib
import io
import wave
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scribr.representation import Melody, melody_from_text
from scribr.transcription import Transcriber

from .materialize import Manifest
from .scoring import ExperimentEvaluation, evaluate_predictions


@dataclass(frozen=True, slots=True, eq=False)
class Recording:
    """One materialized example, decoded and ready to transcribe.

    Attributes:
        example_id: Stable manifest ID, `<sample index>-<instrument>`.
        reference: Reference melody read from the note table.
        audio: Mono `float32` samples in `[-1, 1]`.
        sample_rate: Analysis sample rate from the WAV header.
        group: Group used for grouped metrics, currently the instrument.
    """

    example_id: str
    reference: Melody
    audio: np.ndarray
    sample_rate: int
    group: str


def _read_verified(path: Path, expected_hash: str) -> bytes:
    data = path.read_bytes()

    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"artifact hash mismatch: {path}")

    return data


def load_recordings(
    manifest: Manifest, artifacts_dir: Path
) -> Iterator[Recording]:
    """Yield one decoded `Recording` per manifest example.

    WAV and reference files are verified against the manifest hashes
    before being decoded.
    """
    if not manifest.examples:
        raise ValueError("manifest must contain at least one example")

    for example in manifest.examples:
        wav_path = artifacts_dir / example.wav
        wav_bytes = _read_verified(wav_path, example.wav_sha256)

        with wave.open(io.BytesIO(wav_bytes), "rb") as source:
            if (
                source.getnchannels() != 1
                or source.getsampwidth() != 2
                or source.getcomptype() != "NONE"
            ):
                raise ValueError("manifest WAV files must be mono 16-bit PCM")

            sample_rate = source.getframerate()
            if sample_rate != manifest.config.synthesis.sample_rate:
                raise ValueError("WAV sample rate does not match manifest")

            audio = (
                np.frombuffer(
                    source.readframes(source.getnframes()), dtype="<i2"
                ).astype(np.float32)
                / 32768
            )

        reference = melody_from_text(
            _read_verified(
                artifacts_dir / example.reference,
                example.reference_sha256,
            ).decode("utf-8")
        )

        yield Recording(
            example_id=example.id,
            reference=reference,
            audio=audio,
            sample_rate=sample_rate,
            group=example.instrument.value,
        )


def score_recordings(
    recordings: Iterable[Recording],
    transcriber: Transcriber,
    tempo: float,
    *,
    verbose: bool = False,
) -> tuple[ExperimentEvaluation, dict[str, Melody]]:
    """Transcribe and score recordings, returning scores and estimates.

    Args:
        recordings: Decoded recordings to score.
        transcriber: Approach to run on each recording.
        tempo: Tempo in BPM used to convert beats to seconds.
        verbose: Print one line per transcribed example.

    Returns:
        The batch evaluation and the estimate per example ID.
    """
    estimates: dict[str, Melody] = {}

    def items() -> Iterator[tuple[str, Melody, Melody, str]]:
        for recording in recordings:
            estimate = transcriber.transcribe(
                recording.audio, recording.sample_rate, tempo
            )
            estimates[recording.example_id] = estimate

            if verbose:
                print(
                    f"Transcribed {recording.example_id}: "
                    f"{len(estimate)} note(s)",
                    flush=True,
                )

            yield (
                recording.example_id,
                recording.reference,
                estimate,
                recording.group,
            )

    return evaluate_predictions(items(), tempo=tempo), estimates
