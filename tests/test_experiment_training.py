"""Tests for the log-mel CNN training experiment."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from mlflow.tracking import MlflowClient
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.common.spec import SynthesisSpec
from experiments.deep_learning.logmel_cnn.ablation import (
    AblationArm,
    ArmResult,
    _winner,
    run_ablation,
    write_ablation_report,
)
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
    _class_weights,
    _train_epoch,
    flatten_config,
    resolve_device,
    train,
)
from scribr.deep_learning import HOLD_CLASS, token_to_class
from scribr.representation import HOLD_TOKEN, Melody, MelodyExample
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
        lambda: OptimizationSpec(class_weight_power=-0.1),
        lambda: OptimizationSpec(class_weight_power=1.1),
        lambda: OptimizationSpec(primary_instrument=""),
        lambda: OptimizationSpec(device=""),
    ],
)
def test_invalid_config_values_raise(build) -> None:
    with pytest.raises(ValueError):
        build()


def test_primary_instrument_must_be_evaluated() -> None:
    config = tiny_config()

    with pytest.raises(ValueError):
        replace(
            config,
            optimization=replace(
                config.optimization,
                primary_instrument="acoustic_guitar",
            ),
        )


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


def test_class_weights_temper_the_inverse_frequency_correction() -> None:
    examples = [
        MelodyExample(
            Melody(),
            pitch_sequence=(60, HOLD_TOKEN, HOLD_TOKEN, HOLD_TOKEN),
        )
    ]
    inverse = _class_weights(examples, power=1.0)
    tempered = _class_weights(examples, power=0.5)
    uniform = _class_weights(examples, power=0.0)

    pitch = token_to_class(60)
    hold = HOLD_CLASS

    assert float(uniform.mean()) == pytest.approx(1.0)
    assert float(tempered.mean()) == pytest.approx(1.0)
    assert uniform[pitch] == pytest.approx(1.0)
    assert uniform[hold] == pytest.approx(1.0)
    assert inverse[hold] < tempered[hold] < 1.0
    assert 1.0 < tempered[pitch] < inverse[pitch]


def test_resolve_device_accepts_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


class FixedLogits(nn.Module):
    """Select a fixed logit row using the index in the first feature."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = nn.Parameter(logits)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.logits[features[:, 0].long()]


@pytest.mark.parametrize("use_class_weights", [False, True])
def test_train_epoch_aggregates_losses_over_tokens(
    use_class_weights: bool,
) -> None:
    torch.manual_seed(0)
    steps, classes = 4, 5
    logits = torch.randn(5, steps, classes)
    targets = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [0, 2, 4, 1],
            [3, 0, 1, 4],
            [2, 3, 0, 1],
        ]
    )
    loss_weight = (
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        if use_class_weights
        else None
    )
    model = FixedLogits(logits)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    dataset = TensorDataset(
        torch.arange(5, dtype=torch.float32).unsqueeze(1), targets
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    metrics, step = _train_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        OptimizationSpec(epochs=1, log_every_steps=100),
        start_step=0,
        loss_weight=loss_weight,
    )

    token_losses = F.cross_entropy(
        logits.transpose(1, 2), targets, reduction="none"
    )

    if loss_weight is None:
        expected_loss = float(token_losses.mean())
    else:
        train_weights = loss_weight[targets]
        expected_loss = float(
            (token_losses * train_weights).sum() / train_weights.sum()
        )

    expected_unweighted = float(token_losses.mean())
    expected_accuracy = float((logits.argmax(-1) == targets).float().mean())

    assert step == 3
    assert metrics["train/loss"] == pytest.approx(expected_loss, abs=1e-6)
    assert metrics["train/loss_unweighted"] == pytest.approx(
        expected_unweighted, abs=1e-6
    )
    assert metrics["train/token_accuracy"] == pytest.approx(
        expected_accuracy, abs=1e-6
    )


def test_train_logs_run_and_writes_artifacts(
    tmp_path: Path,
    data_root: Path,
) -> None:
    config = tiny_config()
    tracking_uri = tmp_path / "mlflow.db"
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
    assert "train/loss_unweighted" in result.history[0]

    payload = torch.load(
        result.best_checkpoint, map_location="cpu", weights_only=True
    )

    assert payload["config"]["name"] == "test_run"
    assert payload["epoch"] == 1
    assert payload["metric"] is not None

    client = MlflowClient(tracking_uri=f"sqlite:///{tracking_uri}")
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
    assert "train/loss" in run.data.metrics
    assert "train/loss_unweighted" in run.data.metrics
    assert "val/F-measure_no_offset" in run.data.metrics
    assert "val/piano/F-measure_no_offset" in run.data.metrics
    assert len(client.list_artifacts(run.info.run_id)) == 3


def test_train_selects_checkpoint_on_primary_instrument(
    tmp_path: Path,
    data_root: Path,
) -> None:
    config = tiny_config()
    config = replace(
        config,
        optimization=replace(
            config.optimization,
            primary_instrument="piano",
        ),
    )
    tracking_uri = tmp_path / "mlflow.db"

    result = train(
        config,
        artifacts_dir=tmp_path / "artifacts",
        run_name="primary",
        tracking_uri=tracking_uri,
        data_root=data_root,
        device="cpu",
        synthesizer_factory=fake_factory,
    )
    payload = torch.load(
        result.best_checkpoint, map_location="cpu", weights_only=True
    )
    piano_scores = [
        record["per_instrument"]["piano"]["F-measure_no_offset"]
        for record in result.history
        if "per_instrument" in record
    ]

    assert payload["metric"] == pytest.approx(max(piano_scores))

    client = MlflowClient(tracking_uri=f"sqlite:///{tracking_uri}")
    experiment = client.get_experiment_by_name(EXPERIMENT)

    assert experiment is not None

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    assert runs[0].data.tags["metric"] == "piano/F-measure_no_offset"


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


def _arm_result(name: str, score: float) -> ArmResult:
    return ArmResult(
        arm=AblationArm(name=name, power=0.5),
        run_name=name,
        artifacts_dir=name,
        best_epoch=1,
        best_score=score,
        epoch_scores=((1, score),),
    )


def test_ablation_winner_reports_higher_score_and_ties() -> None:
    inverse = _arm_result("inverse", 0.2)
    tempered = _arm_result("tempered", 0.3)

    assert _winner((inverse, tempered)) == "tempered"
    assert _winner((inverse, _arm_result("tempered", 0.2))) == "tie"


def test_run_ablation_trains_both_arms_and_writes_report(
    tmp_path: Path,
    data_root: Path,
) -> None:
    config = tiny_config()
    ablation_dir = tmp_path / "ablation"

    result = run_ablation(
        config,
        artifacts_dir=ablation_dir,
        run_name="ab",
        tracking_uri=tmp_path / "mlflow.db",
        data_root=data_root,
        device="cpu",
        synthesizer_factory=fake_factory,
    )

    assert [arm.arm.name for arm in result.arms] == ["inverse", "tempered"]
    assert [arm.arm.power for arm in result.arms] == [1.0, 0.5]
    assert result.metric == "piano/F-measure_no_offset"
    assert result.winner in {"inverse", "tempered", "tie"}
    assert result.scores == {
        arm.arm.name: arm.best_score for arm in result.arms
    }

    for arm_result in result.arms:
        payload = torch.load(
            Path(arm_result.artifacts_dir) / "best.pt",
            map_location="cpu",
            weights_only=True,
        )
        arm_config = payload["config"]

        assert arm_result.epoch_scores
        assert (
            arm_config["optimization"]["class_weight_power"]
            == arm_result.arm.power
        )
        assert arm_config["optimization"]["primary_instrument"] == "piano"

    json_path, markdown_path = write_ablation_report(
        result,
        artifacts_dir=ablation_dir,
        config=config,
    )
    data = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert data["primary_metric"] == "piano/F-measure_no_offset"
    assert [arm["name"] for arm in data["arms"]] == ["inverse", "tempered"]
    assert data["winner"] == result.winner
    assert "Winner:" in markdown
    assert "| Epoch | inverse | tempered |" in markdown
