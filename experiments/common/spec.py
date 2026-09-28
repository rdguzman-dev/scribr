"""Frozen experiment configuration with JSON persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from scribr.data import SPLITS
from scribr.synthesis import Instrument
from scribr.synthesis.synthesizer import (
    DEFAULT_GAIN,
    DEFAULT_RELEASE_TAIL_SECONDS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_TEMPO_BPM,
    DEFAULT_VELOCITY,
)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Dataset slice used by an experiment.

    Attributes:
        split: Dataset split to draw from.
        prefix_size: Number of leading records treated as candidates.
        sample_size: Number of examples to select from the candidates.
        seed: Seed for the deterministic candidate draw.
    """

    split: str = "test"
    prefix_size: int = 1000
    sample_size: int = 20
    seed: int = 42

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(
                f"unknown split {self.split!r}; expected one of "
                f"{', '.join(SPLITS)}"
            )

        if self.prefix_size < 1:
            raise ValueError("prefix_size must be positive")

        if self.sample_size < 1:
            raise ValueError("sample_size must be positive")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetSpec:
        """Build a spec from its JSON-compatible mapping."""
        return cls(
            split=data["split"],
            prefix_size=data["prefix_size"],
            sample_size=data["sample_size"],
            seed=data["seed"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the spec as a JSON-compatible mapping."""
        return {
            "split": self.split,
            "prefix_size": self.prefix_size,
            "sample_size": self.sample_size,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class SynthesisSpec:
    """Synthesis parameters shared by every example in an experiment.

    The values default to the corresponding `Synthesizer` defaults.
    """

    sample_rate: int = DEFAULT_SAMPLE_RATE
    tempo: float = DEFAULT_TEMPO_BPM
    velocity: int = DEFAULT_VELOCITY
    gain: float = DEFAULT_GAIN
    release_tail_seconds: float = DEFAULT_RELEASE_TAIL_SECONDS

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        if self.tempo <= 0:
            raise ValueError("tempo must be positive")

        if not 1 <= self.velocity <= 127:
            raise ValueError("velocity must be in [1, 127]")

        if self.gain <= 0:
            raise ValueError("gain must be positive")

        if self.release_tail_seconds < 0:
            raise ValueError("release_tail_seconds must be non-negative")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SynthesisSpec:
        """Build a spec from its JSON-compatible mapping."""
        return cls(
            sample_rate=data["sample_rate"],
            tempo=data["tempo"],
            velocity=data["velocity"],
            gain=data["gain"],
            release_tail_seconds=data["release_tail_seconds"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the spec as a JSON-compatible mapping."""
        return {
            "sample_rate": self.sample_rate,
            "tempo": self.tempo,
            "velocity": self.velocity,
            "gain": self.gain,
            "release_tail_seconds": self.release_tail_seconds,
        }


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """Full configuration for one experiment run."""

    experiment: str
    dataset: DatasetSpec
    instruments: tuple[Instrument, ...]
    synthesis: SynthesisSpec

    def __post_init__(self) -> None:
        if not self.experiment:
            raise ValueError("experiment name must be non-empty")

        try:
            instruments = tuple(Instrument(name) for name in self.instruments)

        except ValueError as error:
            raise ValueError(f"invalid instrument: {error}") from error

        object.__setattr__(self, "instruments", instruments)

        if not instruments:
            raise ValueError("at least one instrument is required")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentSpec:
        """Build a spec from its JSON-compatible mapping."""
        return cls(
            experiment=data["experiment"],
            dataset=DatasetSpec.from_dict(data["dataset"]),
            instruments=tuple(data["instruments"]),
            synthesis=SynthesisSpec.from_dict(data["synthesis"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the spec as a JSON-compatible mapping."""
        return {
            "experiment": self.experiment,
            "dataset": self.dataset.to_dict(),
            "instruments": [
                instrument.value for instrument in self.instruments
            ],
            "synthesis": self.synthesis.to_dict(),
        }

    @classmethod
    def load(cls, path: str | Path) -> ExperimentSpec:
        """Load a spec from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        """Write the spec to a JSON file."""
        text = json.dumps(self.to_dict(), indent=2) + "\n"
        Path(path).write_text(text, encoding="utf-8")
