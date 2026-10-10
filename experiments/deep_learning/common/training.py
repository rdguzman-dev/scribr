"""Training loop shared by supervised deep learning experiments.

The loop is architecture-agnostic: the experiment injects its feature
extractor and model builders, and the loop owns the optimizer schedule,
loss aggregation, validation, checkpointing, and MLflow logging.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from experiments.common.tracking import (
    DEFAULT_TRACKING_URI,
    git_metadata,
    log_artifact,
    log_metrics,
    log_params,
    start_run,
)
from scribr.deep_learning import (
    FeatureExtractor,
    SynthesizedMelodyDataset,
    class_weights,
    token_group_masks,
)
from scribr.representation import MelodyExample
from scribr.synthesis import Instrument, Synthesizer

from .data import (
    default_synthesizer_factory,
    load_examples,
    subset_fingerprint,
)
from .inference import run_evaluation
from .spec import OptimizationSpec, TrainingConfig

EXPERIMENT = "scribr-deep-learning"


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Paths and scores from one training run.

    Attributes:
        run_name: MLflow run name.
        artifacts_dir: Directory holding the checkpoints and history.
        best_metric: Best primary-metric score seen during training.
        best_checkpoint: Checkpoint with the best metric.
        history: One record per epoch with losses and metric scores.
    """

    run_name: str
    artifacts_dir: Path
    best_metric: float
    best_checkpoint: Path
    history: tuple[dict[str, Any], ...]


def train(
    config: TrainingConfig,
    *,
    build_features: Callable[[Mapping[str, Any]], FeatureExtractor],
    build_model: Callable[
        [Mapping[str, Any], Mapping[str, Any]], nn.Module
    ],
    artifacts_dir: str | Path,
    run_name: str,
    tracking_uri: str | Path = DEFAULT_TRACKING_URI,
    soundfont_path: str | Path | None = None,
    data_root: str | Path | None = None,
    device: str | None = None,
    synthesizer_factory: Callable[[Instrument], Synthesizer] | None = None,
) -> TrainingResult:
    """Train one model and log the run to MLflow.

    Args:
        config: Full training configuration.
        build_features: Builds the feature extractor from
            `config.features`.
        build_model: Builds the model from `config.model` and
            `config.features`.
        artifacts_dir: Directory for checkpoints, config, and history.
        run_name: MLflow run name.
        tracking_uri: MLflow tracking URI, or a SQLite database path.
        soundfont_path: SoundFont override; defaults to
            `SCRIBR_SOUNDFONT`. Ignored when `synthesizer_factory` is
            given.
        data_root: Dataset root override, for tests and non-default
            checkouts.
        device: Device override; defaults to `config.optimization.device`.
        synthesizer_factory: Per-instrument synthesizer factory, for
            tests and other dependency injection. The training
            instrument is used for train and validation audio.

    Returns:
        The run name, artifact paths, best metric, and epoch history.

    Raises:
        ValueError: If no SoundFont is configured and no factory is
            injected.
    """
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_device(device or config.optimization.device)
    torch.manual_seed(config.optimization.seed)

    train_examples = load_examples(config.dataset, root=data_root)
    validation_examples = load_examples(config.validation, root=data_root)
    features = build_features(config.features)
    model = build_model(config.model, config.features)
    model.to(resolved_device)

    factory = (
        synthesizer_factory
        if synthesizer_factory is not None
        else default_synthesizer_factory(config, soundfont_path)
    )

    train_loader = _make_loader(
        train_examples,
        factory,
        features,
        config.instrument,
        config,
        shuffle=True,
        num_workers=config.dataset.num_workers,
    )
    validation_loader = _make_loader(
        validation_examples,
        factory,
        features,
        config.instrument,
        config,
        shuffle=False,
        num_workers=config.validation.num_workers,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.optimization.epochs,
        eta_min=config.optimization.min_learning_rate,
    )

    history: list[dict[str, Any]] = []
    best_metric = float("-inf")
    best_checkpoint = artifacts / "best.pt"
    last_checkpoint = artifacts / "last.pt"
    global_step = 0

    loss_weight = None
    primary_instrument = config.optimization.primary_instrument

    if config.optimization.class_weights:
        loss_weight = class_weights(
            train_examples,
            power=config.optimization.class_weight_power,
        ).to(resolved_device)

    metric_label = (
        f"{primary_instrument}/F-measure_no_offset"
        if primary_instrument is not None
        else "F-measure_no_offset"
    )
    tags = {
        "device": str(resolved_device),
        "instrument": config.instrument.value,
        "metric": metric_label,
        "dataset.subset_sha256": subset_fingerprint(train_examples),
        **git_metadata(),
    }

    with start_run(
        EXPERIMENT,
        run_name=run_name,
        tracking_uri=tracking_uri,
        tags=tags,
    ):
        log_params(flatten_config(config.to_dict()))

        for epoch in range(1, config.optimization.epochs + 1):
            started = time.perf_counter()
            model.train()
            train_metrics, global_step = _train_epoch(
                model,
                train_loader,
                optimizer,
                resolved_device,
                config.optimization,
                start_step=global_step,
                loss_weight=loss_weight,
            )
            scheduler.step()

            val_metrics = _validation_metrics(
                model, validation_loader, resolved_device
            )
            val_loss = val_metrics["val/loss"]
            epoch_seconds = time.perf_counter() - started

            record = {
                "epoch": epoch,
                "step": global_step,
                **train_metrics,
                "epoch_seconds": epoch_seconds,
                **val_metrics,
            }
            metrics = {
                **train_metrics,
                "epoch_seconds": epoch_seconds,
                **val_metrics,
            }

            if _should_evaluate_metrics(epoch, config.optimization):
                evaluation = run_evaluation(
                    model,
                    features,
                    resolved_device,
                    validation_examples[: config.optimization.metric_examples],
                    config,
                    factory,
                    batch_size=config.optimization.batch_size,
                    num_workers=config.validation.num_workers,
                )
                metrics.update(
                    {
                        f"val/{name}": score
                        for name, score in evaluation.metrics.items()
                    }
                )
                metrics.update(
                    {
                        f"val/{instrument}/F-measure_no_offset": grouped[
                            "F-measure_no_offset"
                        ]
                        for instrument, grouped in (
                            evaluation.grouped_metrics.items()
                        )
                    }
                )
                record["metrics"] = dict(evaluation.metrics)
                record["per_instrument"] = {
                    instrument: dict(grouped)
                    for instrument, grouped in (
                        evaluation.grouped_metrics.items()
                    )
                }

                if primary_instrument is None:
                    score = _best_metric_score(evaluation.metrics)
                else:
                    score = _best_metric_score(
                        evaluation.grouped_metrics[primary_instrument]
                    )

                if score > best_metric:
                    best_metric = score
                    _save_checkpoint(
                        best_checkpoint,
                        config,
                        model,
                        epoch=epoch,
                        val_loss=val_loss,
                        metric=score,
                    )

                model.train()

            log_metrics(metrics, step=global_step)
            history.append(record)
            _save_checkpoint(
                last_checkpoint,
                config,
                model,
                epoch=epoch,
                val_loss=val_loss,
                metric=None,
            )

        config_path = artifacts / "config.json"
        config.save(config_path)
        history_path = artifacts / "history.json"
        history_path.write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )

        log_artifact(config_path)
        log_artifact(history_path)
        log_artifact(best_checkpoint)

    return TrainingResult(
        run_name=run_name,
        artifacts_dir=artifacts,
        best_metric=best_metric,
        best_checkpoint=best_checkpoint,
        history=tuple(history),
    )


def flatten_config(
    data: Mapping[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten nested configuration into dot-separated parameter
    names."""
    flat: dict[str, Any] = {}

    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key

        if isinstance(value, Mapping):
            flat.update(flatten_config(value, name))
        else:
            flat[name] = value

    return flat


def resolve_device(requested: str) -> torch.device:
    """Resolve `"auto"` to MPS when available, otherwise CPU.

    Explicit `"mps"` and `"cuda"` requests fail when the backend is not
    available, instead of falling back silently to CPU.
    """
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available on this machine")

    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this machine")

    return torch.device(requested)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    optimization: OptimizationSpec,
    *,
    start_step: int,
    loss_weight: torch.Tensor | None,
) -> tuple[dict[str, float], int]:
    """Train for one epoch and return its metrics and final step.

    The weighted loss divides by the sum of the sample weights seen in
    the epoch, not by the token count, so the epoch mean stays exact
    when the loader shuffles examples. `train/loss_unweighted` drops
    the weights and is the counterpart of the unweighted `val/loss`.
    """
    weighted_total = 0.0
    weight_total = 0.0
    unweighted_total = 0.0
    correct = 0
    positions = 0
    step = start_step

    for batch_features, targets in loader:
        batch_features = batch_features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_features)
        token_losses = F.cross_entropy(
            logits.transpose(1, 2), targets, reduction="none"
        )

        if loss_weight is None:
            loss = token_losses.mean()
            batch_weight = float(targets.numel())
        else:
            batch_weights = loss_weight[targets]
            loss = (token_losses * batch_weights).sum() / batch_weights.sum()
            batch_weight = float(batch_weights.detach().sum())

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), optimization.gradient_clip
        )
        optimizer.step()

        step += 1
        weighted_total += float(loss.detach()) * batch_weight
        weight_total += batch_weight
        unweighted_total += float(token_losses.detach().sum())
        correct += int((logits.detach().argmax(-1) == targets).sum())
        positions += targets.numel()

        if step % optimization.log_every_steps == 0:
            log_metrics(
                {
                    "train/loss": float(loss.detach()),
                    "train/loss_unweighted": float(
                        token_losses.detach().mean()
                    ),
                    "train/grad_norm": float(grad_norm),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                },
                step=step,
            )

    if positions == 0:
        raise ValueError("training loader produced no examples")

    return (
        {
            "train/loss": weighted_total / weight_total,
            "train/loss_unweighted": unweighted_total / positions,
            "train/token_accuracy": correct / positions,
        },
        step,
    )


def _validation_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Return validation metrics, including per-group token accuracy.

    The loss is unweighted even when training uses class weights, so
    `val/loss` means the same thing across weighted and unweighted runs.
    `train/loss_unweighted` follows the same convention.
    Pitch, hold, and note-off accuracy are logged separately because
    overall token accuracy hides a collapse to one class.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    positions = 0
    group_correct: dict[str, int] = {}
    group_total: dict[str, int] = {}

    with torch.no_grad():
        for batch_features, targets in loader:
            batch_features = batch_features.to(device)
            targets = targets.to(device)

            logits = model(batch_features)
            loss = F.cross_entropy(
                logits.transpose(1, 2), targets, reduction="sum"
            )
            predictions = logits.argmax(-1)

            total_loss += float(loss)
            correct += int((predictions == targets).sum())
            positions += targets.numel()

            for name, mask in token_group_masks(targets):
                group_total[name] = group_total.get(name, 0) + int(
                    mask.sum()
                )
                group_correct[name] = group_correct.get(name, 0) + int(
                    (predictions[mask] == targets[mask]).sum()
                )

    if positions == 0:
        raise ValueError("validation loader produced no examples")

    metrics = {
        "val/loss": total_loss / positions,
        "val/token_accuracy": correct / positions,
    }

    for name, total in group_total.items():
        if total:
            metrics[f"val/{name}_token_accuracy"] = group_correct[name] / total

    return metrics


def _make_loader(
    examples: Sequence[MelodyExample],
    factory: Callable[[Instrument], Synthesizer],
    features: FeatureExtractor,
    instrument: Instrument,
    config: TrainingConfig,
    *,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = SynthesizedMelodyDataset(
        examples,
        partial(factory, instrument),
        features,
    )
    generator = torch.Generator()
    generator.manual_seed(config.optimization.seed)

    return DataLoader(
        dataset,
        batch_size=config.optimization.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator if shuffle else None,
        persistent_workers=num_workers > 0,
    )


def _save_checkpoint(
    path: Path,
    config: TrainingConfig,
    model: nn.Module,
    *,
    epoch: int,
    val_loss: float,
    metric: float | None,
) -> None:
    torch.save(
        {
            "config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "metric": metric,
        },
        path,
    )


def _should_evaluate_metrics(
    epoch: int,
    optimization: OptimizationSpec,
) -> bool:
    return (
        epoch % optimization.metric_interval == 0
        or epoch == optimization.epochs
    )


def _best_metric_score(metrics: Mapping[str, float]) -> float:
    score = float(metrics.get("F-measure_no_offset", 0.0))

    return score if math.isfinite(score) else 0.0
