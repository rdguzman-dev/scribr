"""Materialize a reproducible set of audio and reference artifacts.

Materialization renders one WAV file and one reference note table per
sampled dataset example and records both in a deterministic manifest.
The manifest is the entry point for later scoring: it maps stable
example IDs to files and hashes without embedding absolute paths or
environment-specific metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

from scribr.data import MelodyDataset
from scribr.representation import MelodyExample, melody_to_text
from scribr.synthesis import (
    SOUNDFONT_ENV_VAR,
    Instrument,
    Synthesizer,
)
from scribr.wav import write_wav

from .spec import ExperimentSpec

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ManifestExample:
    """One materialized example.

    Attributes:
        id: Stable example ID, `<sample index>-<instrument>`.
        index: Position in the sampled order.
        candidate_index: Position in the candidate prefix.
        instrument: Instrument the example was rendered with.
        program: General MIDI program number of the instrument.
        wav: WAV path relative to the artifacts directory.
        reference: Reference note-table path relative to the artifacts
            directory.
        wav_sha256: SHA-256 hex digest of the WAV file.
        reference_sha256: SHA-256 hex digest of the reference file.
        num_notes: Number of notes in the reference melody.
    """

    id: str
    index: int
    candidate_index: int
    instrument: Instrument
    program: int
    wav: str
    reference: str
    wav_sha256: str
    reference_sha256: str
    num_notes: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ManifestExample:
        """Build an example record from its JSON-compatible mapping."""
        return cls(
            id=data["id"],
            index=data["index"],
            candidate_index=data["candidate_index"],
            instrument=Instrument(data["instrument"]),
            program=data["program"],
            wav=data["wav"],
            reference=data["reference"],
            wav_sha256=data["wav_sha256"],
            reference_sha256=data["reference_sha256"],
            num_notes=data["num_notes"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the example record as a JSON-compatible mapping."""
        return {
            "id": self.id,
            "index": self.index,
            "candidate_index": self.candidate_index,
            "instrument": self.instrument.value,
            "program": self.program,
            "wav": self.wav,
            "reference": self.reference,
            "wav_sha256": self.wav_sha256,
            "reference_sha256": self.reference_sha256,
            "num_notes": self.num_notes,
        }


@dataclass(frozen=True, slots=True)
class Manifest:
    """Deterministic record of a materialized artifact set."""

    experiment: str
    soundfont_sha256: str | None
    config: ExperimentSpec
    examples: tuple[ManifestExample, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "examples", tuple(self.examples))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Manifest:
        """Build a manifest from its JSON-compatible mapping."""
        if data["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported manifest schema_version "
                f"{data['schema_version']!r}; expected {SCHEMA_VERSION}"
            )

        return cls(
            experiment=data["experiment"],
            soundfont_sha256=data["soundfont_sha256"],
            config=ExperimentSpec.from_dict(data["config"]),
            examples=tuple(
                ManifestExample.from_dict(example)
                for example in data["examples"]
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest as a JSON-compatible mapping."""
        return {
            "schema_version": self.schema_version,
            "experiment": self.experiment,
            "soundfont_sha256": self.soundfont_sha256,
            "config": self.config.to_dict(),
            "examples": [example.to_dict() for example in self.examples],
        }

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        """Load a manifest from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        """Write the manifest to a JSON file."""
        text = json.dumps(self.to_dict(), indent=2) + "\n"
        Path(path).write_text(text, encoding="utf-8")


def materialize(
    spec: ExperimentSpec,
    artifacts_dir: str | Path,
    *,
    dataset: Iterable[MelodyExample] | None = None,
    synthesizers: Mapping[Instrument, Synthesizer] | None = None,
    soundfont_path: str | Path | None = None,
    force: bool = False,
) -> Manifest:
    """Render each sampled example to a WAV and a reference note table.

    Experiments should use `spec` for configuration. The keyword-only
    parameters are intended for testing, dependecy injection, and other
    advanced use cases.

    An existing manifest is reused when it matches `spec`; pass
    `force=True` to rebuild the artifacts. Sampling uses
    `random.Random(spec.dataset.seed).sample` over the candidate prefix,
    so identical inputs always produce identical artifacts.

    Args:
        spec: Experiment configuration.
        artifacts_dir: Directory holding `manifest.json`, `wav/`, and
            `reference/`.
        dataset: Candidate source, defaulting to
            `MelodyDataset(split=spec.dataset.split)`.
        synthesizers: Per-instrument synthesizers, defaulting to
            FluidSynth synthesizers built from the configured SoundFont.
            If provided, no SoundFont hash will be saved to the
            manifest.
        soundfont_path: SoundFont override; falls back to
            `SCRIBR_SOUNDFONT` when synthesizers are not injected.
        force: Rebuild even when a matching manifest already exists.

    Returns:
        The manifest describing the materialized artifacts.

    Raises:
        ValueError: If the existing manifest was built from a different
            config, or fewer than `sample_size` candidates exist.
    """
    artifacts = Path(artifacts_dir)
    manifest_path = artifacts / "manifest.json"

    if manifest_path.is_file() and not force:
        manifest = Manifest.load(manifest_path)

        if manifest.config != spec:
            raise ValueError(
                f"{manifest_path} was materialized from a different config; "
                "pass force=True to rebuild"
            )

        print(
            f"Using existing manifest at {manifest_path} "
            "(pass force=True to rebuild)."
        )

        return manifest

    if dataset is None:
        dataset = MelodyDataset(split=spec.dataset.split)

    candidates = list(islice(dataset, spec.dataset.prefix_size))

    if len(candidates) < spec.dataset.sample_size:
        raise ValueError(
            f"only {len(candidates)} candidate(s) in the first "
            f"{spec.dataset.prefix_size} record(s); "
            f"sample_size is {spec.dataset.sample_size}"
        )

    # Sample positions so the manifest can record where each example
    # came from in the candidate prefix.
    chosen_indices = random.Random(spec.dataset.seed).sample(
        range(len(candidates)), spec.dataset.sample_size
    )

    if synthesizers is None:
        synthesizers, resolved_soundfont = _build_synthesizers(
            spec, soundfont_path
        )
        soundfont_sha256 = _sha256_file(resolved_soundfont)

    else:
        soundfont_sha256 = None

    wav_dir = artifacts / "wav"
    reference_dir = artifacts / "reference"
    wav_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    manifest_examples: list[ManifestExample] = []

    for index, candidate_index in enumerate(chosen_indices):
        example = candidates[candidate_index]
        instrument = spec.instruments[index % len(spec.instruments)]
        example_id = f"{index:03d}-{instrument.value}"

        wav_path = wav_dir / f"{example_id}.wav"
        reference_path = reference_dir / f"{example_id}.txt"

        samples = synthesizers[instrument].synthesize(example.melody)
        write_wav(samples, spec.synthesis.sample_rate, wav_path)
        reference_path.write_text(
            melody_to_text(example.melody), encoding="utf-8"
        )

        manifest_examples.append(
            ManifestExample(
                id=example_id,
                index=index,
                candidate_index=candidate_index,
                instrument=instrument,
                program=instrument.program,
                wav=f"wav/{example_id}.wav",
                reference=f"reference/{example_id}.txt",
                wav_sha256=_sha256_file(wav_path),
                reference_sha256=_sha256_file(reference_path),
                num_notes=len(example.melody),
            )
        )

    manifest = Manifest(
        experiment=spec.experiment,
        soundfont_sha256=soundfont_sha256,
        config=spec,
        examples=tuple(manifest_examples),
    )
    manifest.save(manifest_path)

    return manifest


def _build_synthesizers(
    spec: ExperimentSpec,
    soundfont_path: str | Path | None,
) -> tuple[dict[Instrument, Synthesizer], Path]:
    configured = (
        soundfont_path
        if soundfont_path is not None
        else os.environ.get(SOUNDFONT_ENV_VAR)
    )

    if configured is None:
        raise ValueError(
            "No SoundFont configured. Pass soundfont_path=... or set the "
            f"{SOUNDFONT_ENV_VAR} environment variable."
        )

    resolved = Path(configured).expanduser()
    synthesizers = {
        instrument: Synthesizer(
            resolved,
            sample_rate=spec.synthesis.sample_rate,
            tempo=spec.synthesis.tempo,
            program=instrument.program,
            velocity=spec.synthesis.velocity,
            gain=spec.synthesis.gain,
            release_tail_seconds=spec.synthesis.release_tail_seconds,
        )
        for instrument in spec.instruments
    }

    return synthesizers, resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()
