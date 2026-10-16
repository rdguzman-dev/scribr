"""Tests for the Optuna algorithmic parameter search."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import optuna
import pytest
from mlflow.tracking import MlflowClient
from optuna.samplers import RandomSampler

from experiments.algorithmic import tune
from experiments.common.scoring import ExperimentEvaluation
from scribr.algorithmic import AlgorithmicConfig, AlgorithmicTranscriber
from scribr.representation import DATASET_MAX_PITCH, DATASET_MIN_PITCH, Melody


def _recordings(count: int):
    return [
        (
            f"{index:03d}-piano",
            Melody(),
            np.zeros(8, dtype=np.float32),
            22050,
            "piano",
        )
        for index in range(count)
    ]


def _evaluation(metrics: dict[str, float], tempo: float) -> ExperimentEvaluation:
    return ExperimentEvaluation(
        per_example=(),
        metrics=metrics,
        grouped_metrics={},
        tempo=tempo,
        metric_options={},
    )


def _fake_score(recordings, transcriber, tempo, *, verbose=False):
    metrics = {"F-measure_no_offset": 0.75, "F-measure": 0.5}
    return _evaluation(metrics, tempo), {}


def _stage_aware_score(recordings, transcriber, tempo, *, verbose=False):
    config = transcriber.config

    if len(recordings) == 2:
        score = 0.5 + config.silence_db / 100.0
    elif len(recordings) == 3:
        score = 0.5 - config.silence_db / 100.0
    else:
        score = 0.42

    metrics = {"F-measure_no_offset": score, "F-measure": score / 2.0}

    return _evaluation(metrics, tempo), {}


def _tuned(number: int, score: float) -> tune.TunedTrial:
    return tune.TunedTrial(
        number=number, config=AlgorithmicConfig(), score=score
    )


def test_suggest_config_respects_the_fixed_fields() -> None:
    study = optuna.create_study(
        direction="maximize", sampler=RandomSampler(seed=1)
    )
    trial = study.ask()
    config = tune._suggest_config(trial)

    assert config.sample_rate == tune.ANALYSIS_SAMPLE_RATE
    assert config.min_pitch == DATASET_MIN_PITCH
    assert config.max_pitch == DATASET_MAX_PITCH
    assert config.quantize is True
    assert config.frame_length in tune.FRAME_LENGTH_CHOICES
    assert config.hop_length in tune.HOP_LENGTH_CHOICES
    assert config.rms_window_hops in tune.RMS_WINDOW_HOP_CHOICES


def test_default_trial_params_cover_the_search_space() -> None:
    params = tune._default_trial_params()
    assert set(params) == {
        "frame_length",
        "hop_length",
        "silence_db",
        "min_note_seconds",
        "max_gap_seconds",
        "onset_merge_seconds",
        "attack_reject_ratio",
        "attack_backtrack_seconds",
        "rms_window_hops",
    }


def test_build_transcriber_uses_the_track_cache() -> None:
    config = AlgorithmicConfig()
    cache: dict = {}

    cached = tune._build_transcriber(config, cache)
    plain = tune._build_transcriber(config, None)

    assert isinstance(cached, tune._CachedAlgorithmicTranscriber)
    assert type(plain) is AlgorithmicTranscriber


def test_cached_track_reuses_pyin_across_segmentation_configs(
    monkeypatch,
) -> None:
    calls = []

    def fake_track(self, waveform):
        calls.append((self.config.frame_length, self.config.hop_length))
        return object()

    monkeypatch.setattr(
        tune.AlgorithmicTranscriber, "_track_pitch", fake_track
    )
    audio = np.zeros(8, dtype=np.float32)
    cache: dict = {}

    first = tune._CachedAlgorithmicTranscriber(
        AlgorithmicConfig(frame_length=2048, hop_length=256), cache
    )
    track = first._track_pitch(audio)

    assert calls == [(2048, 256)]

    # Only segmentation parameters differ, so the pYIN track is reused.
    second = tune._CachedAlgorithmicTranscriber(
        AlgorithmicConfig(
            frame_length=2048, hop_length=256, silence_db=-30.0
        ),
        cache,
    )

    assert second._track_pitch(audio) is track
    assert calls == [(2048, 256)]

    # A different hop changes the pYIN input, so it recomputes.
    third = tune._CachedAlgorithmicTranscriber(
        AlgorithmicConfig(frame_length=2048, hop_length=512), cache
    )

    assert third._track_pitch(audio) is not track
    assert calls == [(2048, 256), (2048, 512)]

    # Different audio under the same config is a separate entry.
    assert (
        first._track_pitch(np.ones(8, dtype=np.float32)) is not track
    )
    assert calls == [(2048, 256), (2048, 512), (2048, 256)]


def test_objective_records_stage_one_metrics(monkeypatch) -> None:
    monkeypatch.setattr(tune, "score_recordings", _fake_score)
    study = optuna.create_study(
        direction="maximize", sampler=RandomSampler(seed=0)
    )
    objective = tune._make_objective(_recordings(1), 120.0)
    trial = study.ask()

    value = objective(trial)

    assert value == 0.75
    assert trial.user_attrs[tune.PRIMARY_METRIC] == 0.75
    assert trial.user_attrs[tune.SECONDARY_METRIC] == 0.5

    restored = AlgorithmicConfig(**trial.user_attrs["config"])

    assert asdict(restored) == trial.user_attrs["config"]
    study.tell(trial, value)


def test_promote_finalists_keeps_the_top_k_by_stage_one() -> None:
    trials = [
        _tuned(index, score)
        for index, score in enumerate([0.1, 0.9, 0.5, 0.7])
    ]

    promoted = tune.promote_finalists(trials, top_k=2)

    assert [item.number for item in promoted] == [1, 3]


def test_select_champion_uses_stage_two_scores() -> None:
    trials = [_tuned(0, 0.9), _tuned(1, 0.8)]
    finalists = [
        tune.Finalist(trial=trials[0], score_no_offset=0.3, score=0.2),
        tune.Finalist(trial=trials[1], score_no_offset=0.6, score=0.5),
    ]

    assert tune.select_champion(finalists).trial.number == 1


@pytest.mark.parametrize(
    "screen,final,top_k,total",
    [
        (0, 2, 1, 2),
        (3, 2, 1, 3),
        (1, 4, 1, 3),
        (1, 2, 0, 3),
    ],
)
def test_validate_limits_rejects_inconsistent_sizes(
    screen: int, final: int, top_k: int, total: int
) -> None:
    with pytest.raises(ValueError):
        tune.validate_limits(screen, final, top_k, total)


def test_validate_limits_accepts_a_nested_prefix() -> None:
    tune.validate_limits(2, 3, 1, 3)


def test_best_parameters_json_round_trips(tmp_path: Path) -> None:
    config = AlgorithmicConfig(silence_db=-33.0, rms_window_hops=4)
    path = tmp_path / "best_parameters.json"
    path.write_text(json.dumps(asdict(config)), encoding="utf-8")

    restored = AlgorithmicConfig(**json.loads(path.read_text(encoding="utf-8")))

    assert restored == config


def test_run_tuning_writes_outputs_and_logs_both_stages(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(tune, "score_recordings", _stage_aware_score)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    result = tune.run_tuning(
        _recordings(3),
        120.0,
        trials=3,
        timeout=None,
        seed=0,
        n_jobs=1,
        screen_examples=2,
        final_examples=3,
        top_k=2,
        artifacts_dir=tmp_path,
        tracking_uri=tracking_uri,
        run_name="test",
        validation_recordings=_recordings(4),
        sampler=RandomSampler(seed=0),
    )

    assert result.champion == result.finalists[0].trial.config
    assert result.champion_no_offset == result.finalists[0].score_no_offset
    assert result.champion_score == result.finalists[0].score
    assert result.validation_metrics["F-measure_no_offset"] == 0.42

    restored = AlgorithmicConfig(
        **json.loads(result.best_parameters_path.read_text(encoding="utf-8"))
    )
    assert restored == result.champion
    assert result.trials_path.is_file()
    assert result.finalists_path.is_file()
    assert result.validation_path is not None
    assert result.validation_path.is_file()

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(tune.EXPERIMENT)
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    stages = {run.data.tags.get("stage") for run in runs}

    assert "screen" in stages
    assert "final" in stages
    assert "summary" in stages

    parents = {
        run.data.tags.get("mlflow.parentRunId")
        for run in runs
        if run.data.tags.get("stage") is not None
    }

    assert len(parents) == 1
    assert None not in parents
