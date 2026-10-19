"""Evaluate the algorithmic transcriber on a split or a baseline manifest."""

import argparse
import hashlib
import json
import wave
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np

from experiments.common.materialize import Manifest, materialize
from experiments.common.predictions import TextPredictions
from experiments.common.report import write_report
from experiments.common.scoring import (
    ExperimentEvaluation,
    evaluate_predictions,
)
from experiments.common.spec import ExperimentSpec
from scribr.algorithmic import AlgorithmicConfig, AlgorithmicTranscriber
from scribr.representation import Melody, melody_from_text
from scribr.transcription import Transcriber

_DIR = Path(__file__).resolve().parent
_ARTIFACTS_DIR = _DIR / "artifacts"


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
        _read_verified(wav_path, example.wav_sha256)

        with wave.open(str(wav_path), "rb") as source:
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

    def items():
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path)
    source.add_argument(
        "--manifest",
        type=Path,
        help="score the exact materialized human-baseline audio",
    )
    parser.add_argument("--soundfont", type=Path)
    parser.add_argument(
        "--parameters",
        type=Path,
        help="JSON mapping of AlgorithmicConfig parameters",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=_ARTIFACTS_DIR,
        help="directory for materialized artifacts and reports",
    )
    args = parser.parse_args()

    config = AlgorithmicConfig(
        **json.loads(args.parameters.read_text()) if args.parameters else {}
    )
    transcriber = AlgorithmicTranscriber(config)
    artifacts_dir = args.artifacts_dir

    if args.manifest:
        manifest = Manifest.load(args.manifest)
        spec = manifest.config
        source_dir = args.manifest.parent

    else:
        spec = ExperimentSpec.load(args.config or _DIR / "config.json")
        manifest = materialize(
            spec, artifacts_dir, soundfont_path=args.soundfont
        )
        source_dir = artifacts_dir

    recordings = load_recordings(manifest, source_dir)
    evaluation, estimates = score_recordings(
        recordings, transcriber, spec.synthesis.tempo, verbose=True
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    provenance = {"manifest": manifest.to_dict()}

    json_path, markdown_path = write_report(
        manifest,
        evaluation,
        TextPredictions(melodies=estimates, missing=(), errors={}),
        artifacts_dir,
        soundfont_path=args.soundfont,
        extra={
            "parameters": asdict(config),
            "source": provenance,
        },
        package_versions={
            "librosa": librosa.__version__,
            "numpy": np.__version__,
        },
    )

    print(
        f"F-measure_no_offset: {evaluation.metrics['F-measure_no_offset']:.4f}"
    )
    print(f"F-measure: {evaluation.metrics['F-measure']:.4f}")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
