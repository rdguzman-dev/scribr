"""Tests for the log-mel CNN training experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from mlflow.tracking import MlflowClient

from experiments.common.spec import SynthesisSpec
from experiments.deep_learning.logmel_cnn.data import (
    build_features,
    build_model,
    load_examples,
    subset_fingerprint,
)
from experiments.deep_learning.logmel_cnn.evaluation import (
    run_evaluation,
    write_report,
)
from experiments.deep_learning.logmel_cnn.spec import (
    DataSpec,
    FeaturesSpec,
    ModelSpec,
    OptimizationSpec,
    TrainingConfig,
)
from experiments.deep_learning.logmel_cnn.train import (
    EXPERIMENT,
    flatten_config,
    resolve_device,
    train,
)
from scribr.representation import Melody
from scribr.synthesis import Instrument


class FakeSynthesizer:
    """Deterministic tone renderer that needs no SoundFont."""

    def __init__(
        self,
        sample_rate: int = 22_050,
        tempo: float = 120.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.tempo = tempo

    def synthesize(self, melody: Melody) -> np.ndarray:
        if melody.notes:
            seconds = melody.duration * 60.0 / self.tempo
            pitch = melody.notes[0].pitch
        else:
            seconds = 0.25
            pitch = 60

        frames = max(1, round(seconds * self.sample_rate))
        time = np.arange(frames, dtype=np.float32) / self.sample_rate
        frequency = 440.0 * 2.0 ** ((pitch - 69) / 12.0)

        return (0.1 * np.sin(2 * np.pi * frequency * time)).astype(
            np.float32
        )


def fake_factory(instrument: Instrument) -> FakeSynthesizer:
    return FakeSynthesizer()


def tiny_config() -> TrainingConfig:
    return TrainingConfig(
        name="test_run",
        instrument=Instrument.PIANO,
        evaluation_instruments=(Instrument.PIANO, Instrument.VIOLIN),
        dataset=DataSpec(split="train", max_examples=8, num_workers=0),
        validation=DataSpec(
            split="validation", max_examples=4, num_workers=0
        ),
        synthesis=SynthesisSpec(
            sample_rate=22_050,
            tempo=120.0,
            velocity=80,
            gain=1.0,
            release_tail_seconds=0.5,
        ),
        features=FeaturesSpec(
            sample_rate=22_050,
            n_fft=256,
            hop_length=64,
            n_mels=16,
            f_min=27.5,
            f_max=8_000.0,
            top_db=80.0,
            num_frames=32,
        ),
        model=ModelSpec(channels=(4, 8), dropout=0.0),
        optimization=OptimizationSpec(
            seed=7,
            batch_size=2,
            epochs=1,
            learning_rate=0.001,
            weight_decay=0.0,
            min_learning_rate=0.0001,
            gradient_clip=1.0,
            log_every_steps=1,
            metric_examples=2,
            metric_interval=1,
            device="cpu",
        ),
    )


def test_config_round_trips_through_json(tmp_path: Path) -> None:
    config = tiny_config()
    path = tmp_path / "config.json"

    config.save(path)

    assert TrainingConfig.load(path) == config


@pytest.mark.parametrize(
    "build",
    [
        lambda: DataSpec(split="bogus"),
        lambda: DataSpec(max_examples=0),
        lambda: DataSpec(num_workers=-1),
        lambda: ModelSpec(channels=()),
        lambda: ModelSpec(dropout=1.0),
        lambda: OptimizationSpec(epochs=0),
        lambda: OptimizationSpec(metric_examples=0),
        lambda: OptimizationSpec(device=""),
    ],
)
def test_invalid_config_values_raise(build) -> None:
    with pytest.raises(ValueError):
        build()


def test_flatten_config_uses_dot_separated_names() -> None:
    flat = flatten_config(
        {"name": "run", "model": {"dropout": 0.1, "head": {"size": 2}}}
    )

    assert flat == {
        "name": "run",
        "model.dropout": 0.1,
        "model.head.size": 2,
    }


def test_subset_fingerprint_is_stable_and_sensitive(
    data_root: Path,
) -> None:
    examples = load_examples(
        DataSpec(split="test", max_examples=4, num_workers=0),
        root=data_root,
    )

    assert subset_fingerprint(examples) == subset_fingerprint(examples)
    assert subset_fingerprint(examples[:-1]) != subset_fingerprint(examples)


def test_resolve_device_accepts_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_train_logs_run_and_writes_artifacts(
    tmp_path: Path,
    data_root: Path,
) -> None:
    config = tiny_config()
    tracking_uri = tmp_path / "mlruns"
    artifacts_dir = tmp_path / "artifacts"

    result = train(
        config,
        artifacts_dir=artifacts_dir,
        run_name="unit",
        tracking_uri=tracking_uri,
        data_root=data_root,
        device="cpu",
        synthesizer_factory=fake_factory,
    )

    assert (artifacts_dir / "best.pt").is_file()
    assert (artifacts_dir / "last.pt").is_file()
    assert (artifacts_dir / "config.json").is_file()
    assert (artifacts_dir / "history.json").is_file()
    assert len(result.history) == 1
    assert "val/loss" in result.history[0]

    payload = torch.load(
        result.best_checkpoint, map_location="cpu", weights_only=True
    )

    assert payload["config"]["name"] == "test_run"
    assert payload["epoch"] == 1
    assert payload["metric"] is not None

    client = MlflowClient(tracking_uri=tracking_uri.as_uri())
    experiment = client.get_experiment_by_name(EXPERIMENT)

    assert experiment is not None

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    assert len(runs) == 1
    run = runs[0]

    assert run.data.tags["instrument"] == "piano"
    assert run.data.tags["device"] == "cpu"
    assert run.data.params["model.channels"] == "[4, 8]"
    assert run.data.params["optimization.learning_rate"] == "0.001"
    assert "val/loss" in run.data.metrics
    assert "val/F-measure_no_offset" in run.data.metrics
    assert "val/piano/F-measure_no_offset" in run.data.metrics
    assert len(client.list_artifacts(run.info.run_id)) == 3


def test_run_evaluation_writes_reports(
    tmp_path: Path,
    data_root: Path,
) -> None:
    config = tiny_config()
    features = build_features(config.features)
    model = build_model(config.model, n_mels=features.n_mels)
    examples = load_examples(
        DataSpec(split="test", max_examples=2, num_workers=0),
        root=data_root,
    )

    evaluation = run_evaluation(
        model,
        features,
        torch.device("cpu"),
        examples,
        config,
        fake_factory,
        batch_size=2,
        num_workers=0,
    )

    json_path, markdown_path = write_report(
        evaluation,
        config,
        split="test",
        checkpoint="best.pt",
        device="cpu",
        artifacts_dir=tmp_path / "report",
    )

    data = json.loads(json_path.read_text(encoding="utf-8"))

    assert data["split"] == "test"
    assert data["metadata"]["num_examples"] == 4
    assert set(data["per_instrument"]) == {"piano", "violin"}
    assert "F-measure_no_offset" in data["overall"]
    assert markdown_path.read_text(encoding="utf-8").startswith(
        "# test_run report (test)"
    )
