"""Shared training configuration with JSON persistence.

The schema mirrors `experiments.common.spec`: plain dataclasses with a
`from_dict`/`to_dict` pair. `TrainingConfig` is architecture-agnostic;
the `features` and `model` sections are plain mappings that each
experiment's builders turn into a feature extractor and an `nn.Module`.
The checkpoint embeds the whole configuration, so an evaluation can
rebuild the exact model without a separate config file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from experiments.common.spec import SynthesisSpec
from scribr.data import SPLITS
from scribr.synthesis.instruments import Instrument


@dataclass(frozen=True, slots=True)
class DataSpec:
    """One split slice of the dataset.

    Attributes:
        split: Dataset split to stream.
        max_examples: Number of leading records to keep. The subset is a
            prefix of the stored shard order, so it does not depend on
            worker count.
        num_workers: `DataLoader` workers used for this split.
    """

    split: str = "train"
    max_examples: int = 10_000
    num_workers: int = 4

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(
                f"unknown split {self.split!r}; expected one of "
                f"{', '.join(SPLITS)}"
            )

        if self.max_examples < 1:
            raise ValueError("max_examples must be positive")

        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DataSpec:
        """Build a spec from its JSON-compatible mapping."""
        return cls(
            split=data["split"],
            max_examples=data["max_examples"],
            num_workers=data["num_workers"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the spec as a JSON-compatible mapping."""
        return {
            "split": self.split,
            "max_examples": self.max_examples,
            "num_workers": self.num_workers,
        }


@dataclass(frozen=True, slots=True)
class OptimizationSpec:
    """Training loop and optimizer settings.

    Attributes:
        class_weights: Whether to weight the loss by token frequency.
        class_weight_power: Exponent applied to inverse token
            frequency. `1.0` is full inverse frequency, `0.0` is
            uniform, and `0.5` tempers the correction for rare pitch
            tokens.
        primary_instrument: Instrument whose `F-measure_no_offset`
            selects the best checkpoint, or `None` for the mean over
            all evaluation instruments.
    """

    seed: int = 42
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    min_learning_rate: float = 1e-4
    gradient_clip: float = 5.0
    log_every_steps: int = 50
    metric_examples: int = 128
    metric_interval: int = 1
    class_weights: bool = True
    class_weight_power: float = 0.5
    primary_instrument: str | None = None
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

        if self.epochs < 1:
            raise ValueError("epochs must be positive")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

        if self.min_learning_rate < 0:
            raise ValueError("min_learning_rate must be non-negative")

        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")

        if self.log_every_steps < 1:
            raise ValueError("log_every_steps must be positive")

        if self.metric_examples < 1:
            raise ValueError("metric_examples must be positive")

        if self.metric_interval < 1:
            raise ValueError("metric_interval must be positive")

        if not 0.0 <= self.class_weight_power <= 1.0:
            raise ValueError("class_weight_power must be in [0, 1]")

        if self.primary_instrument is not None and not self.primary_instrument:
            raise ValueError("primary_instrument must be non-empty when set")

        if not self.device:
            raise ValueError("device must be non-empty")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OptimizationSpec:
        """Build a spec from its JSON-compatible mapping."""
        return cls(
            seed=data["seed"],
            batch_size=data["batch_size"],
            epochs=data["epochs"],
            learning_rate=data["learning_rate"],
            weight_decay=data["weight_decay"],
            min_learning_rate=data["min_learning_rate"],
            gradient_clip=data["gradient_clip"],
            log_every_steps=data["log_every_steps"],
            metric_examples=data["metric_examples"],
            metric_interval=data["metric_interval"],
            class_weights=data["class_weights"],
            class_weight_power=data.get("class_weight_power", 0.5),
            primary_instrument=data.get("primary_instrument"),
            device=data["device"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the spec as a JSON-compatible mapping."""
        return {
            "seed": self.seed,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "min_learning_rate": self.min_learning_rate,
            "gradient_clip": self.gradient_clip,
            "log_every_steps": self.log_every_steps,
            "metric_examples": self.metric_examples,
            "metric_interval": self.metric_interval,
            "class_weights": self.class_weights,
            "class_weight_power": self.class_weight_power,
            "primary_instrument": self.primary_instrument,
            "device": self.device,
        }


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Full configuration for one training run.

    `features` and `model` are validated by the experiment's builders and
    by the feature extractor and model constructors they call.

    Attributes:
        name: Experiment name, used for the run and report titles.
        instrument: Instrument used to synthesize training audio.
        evaluation_instruments: Instruments scored by the metric pass.
            Training on one instrument and evaluating on several turns
            the evaluation into a timbre-transfer check.
        features: Feature extractor parameters.
        model: Model architecture parameters.
    """

    name: str
    instrument: Instrument
    evaluation_instruments: tuple[Instrument, ...]
    dataset: DataSpec
    validation: DataSpec
    synthesis: SynthesisSpec
    features: Mapping[str, Any]
    model: Mapping[str, Any]
    optimization: OptimizationSpec

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")

        try:
            instrument = Instrument(self.instrument)
            evaluation_instruments = tuple(
                Instrument(value) for value in self.evaluation_instruments
            )

        except ValueError as error:
            raise ValueError(f"invalid instrument: {error}") from error

        object.__setattr__(self, "instrument", instrument)
        object.__setattr__(
            self, "evaluation_instruments", evaluation_instruments
        )
        object.__setattr__(self, "features", dict(self.features))
        object.__setattr__(self, "model", dict(self.model))

        if not evaluation_instruments:
            raise ValueError("at least one evaluation instrument is required")

        primary_instrument = self.optimization.primary_instrument

        if (
            primary_instrument is not None
            and primary_instrument not in evaluation_instruments
        ):
            raise ValueError(
                "primary_instrument must be one of evaluation_instruments"
            )

        feature_sample_rate = self.features.get("sample_rate")

        if feature_sample_rate is None:
            raise ValueError("features.sample_rate is required")

        if self.synthesis.sample_rate != feature_sample_rate:
            raise ValueError(
                "synthesis.sample_rate and features.sample_rate must match; "
                f"got {self.synthesis.sample_rate} and "
                f"{feature_sample_rate}"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainingConfig:
        """Build a config from its JSON-compatible mapping."""
        return cls(
            name=data["name"],
            instrument=data["instrument"],
            evaluation_instruments=tuple(data["evaluation_instruments"]),
            dataset=DataSpec.from_dict(data["dataset"]),
            validation=DataSpec.from_dict(data["validation"]),
            synthesis=SynthesisSpec.from_dict(data["synthesis"]),
            features=data["features"],
            model=data["model"],
            optimization=OptimizationSpec.from_dict(data["optimization"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a JSON-compatible mapping."""
        return {
            "name": self.name,
            "instrument": self.instrument.value,
            "evaluation_instruments": [
                instrument.value for instrument in self.evaluation_instruments
            ],
            "dataset": self.dataset.to_dict(),
            "validation": self.validation.to_dict(),
            "synthesis": self.synthesis.to_dict(),
            "features": dict(self.features),
            "model": dict(self.model),
            "optimization": self.optimization.to_dict(),
        }

    @classmethod
    def load(cls, path: str | Path) -> TrainingConfig:
        """Load a config from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        """Write the config to a JSON file."""
        text = json.dumps(self.to_dict(), indent=2) + "\n"
        Path(path).write_text(text, encoding="utf-8")
