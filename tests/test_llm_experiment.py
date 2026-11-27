"""Offline manifest-to-report tests for the bounded LLM experiment."""

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from google.genai import types

from experiments.common.materialize import Manifest, ManifestExample
from experiments.common.spec import DatasetSpec, ExperimentSpec, SynthesisSpec
from experiments.llm import evaluate as experiment
from scribr.llm import transcriber as implementation
from scribr.representation import Melody, Note, melody_to_text
from scribr.synthesis import Instrument
from scribr.wav import write_wav


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    client = MagicMock()
    client.models.generate_content.return_value = (
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    finish_reason=types.FinishReason.STOP,
                    content=types.Content(parts=[types.Part(text="A4, 0, 1")]),
                )
            ],
        )
    )
    monkeypatch.setattr(
        implementation.genai, "Client", MagicMock(return_value=client)
    )
    monkeypatch.setattr(implementation.time, "sleep", MagicMock())
    monkeypatch.setenv("GEMINI_API_KEY", "offline-key")
    monkeypatch.setenv("SCRIBR_LLM_MODEL", "offline-model")
    return client


def _manifest(directory: Path, count: int = 5) -> Manifest:
    directory.mkdir(parents=True, exist_ok=True)
    reference = Melody((Note(69, 0, 1),))
    examples = []
    for index in range(count):
        wav = write_wav(
            np.full(200, 0.25, np.float32), 100, directory / f"{index}.wav"
        )
        table = directory / f"{index}.txt"
        table.write_text(melody_to_text(reference))
        examples.append(
            ManifestExample(
                f"{index:03d}-piano",
                index,
                Instrument.PIANO,
                0,
                wav.name,
                table.name,
                hashlib.sha256(wav.read_bytes()).hexdigest(),
                hashlib.sha256(table.read_bytes()).hexdigest(),
                1,
            )
        )
    config = ExperimentSpec(
        "human_baseline",
        DatasetSpec("test", count),
        (Instrument.PIANO,),
        SynthesisSpec(sample_rate=100),
    )
    manifest = Manifest("human_baseline", None, config, tuple(examples))
    manifest.save(directory / "manifest.json")
    return manifest


def _run(
    monkeypatch: pytest.MonkeyPatch, source: Path, output: Path, *extra: str
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--manifest",
            str(source / "manifest.json"),
            "--artifacts-dir",
            str(output),
            *extra,
        ],
    )
    experiment.main()


def test_manifest_default_cap_shared_scores_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: MagicMock
) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    manifest = _manifest(source)
    # The unselected example must not be opened or sent to Gemini.
    (source / "4.wav").unlink()
    _run(monkeypatch, source, output)
    report = json.loads((output / "report.json").read_text())
    assert report["experiment"] == "llm_gemini"
    assert report["overall"]["F-measure_no_offset"] == 1
    assert report["overall"]["F-measure"] == 1
    assert report["metadata"]["num_examples"] == 4
    assert report["source"]["manifest"] == manifest.to_dict()
    assert report["llm"]["selected_ids"] == [
        example.id for example in manifest.examples[:4]
    ]
    assert report["parameters"]["model"] == "offline-model"
    assert "google-genai" in report["metadata"]["package_versions"]
    assert len(list((output / "predictions").glob("*.txt"))) == 4
    assert (output / "report.md").is_file()
    assert "offline-key" not in (output / "report.json").read_text()
    # Identical audio is deliberately shared by these fixtures.
    assert (report["llm"]["api_calls"], report["llm"]["cache_hits"]) == (1, 3)
    provider.close.assert_called_once()
    monkeypatch.delenv("GEMINI_API_KEY")
    _run(monkeypatch, source, output)
    rerun = json.loads((output / "report.json").read_text())
    assert (rerun["llm"]["api_calls"], rerun["llm"]["cache_hits"]) == (0, 4)
    assert provider.models.generate_content.call_count == 1


def test_custom_cap_and_disabled_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: MagicMock
) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    _manifest(source)
    _run(
        monkeypatch,
        source,
        output,
        "--max-examples",
        "2",
        "--no-cache",
        "--unquantized",
    )
    report = json.loads((output / "report.json").read_text())
    assert report["metadata"]["num_examples"] == 2
    assert report["parameters"]["quantize"] is False
    assert report["llm"]["cache_enabled"] is False
    assert provider.models.generate_content.call_count == 2
    assert not (output / "responses").exists()


def test_config_is_capped_before_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: MagicMock
) -> None:
    manifest = _manifest(tmp_path / "output", 2)
    full_spec = replace(manifest.config, dataset=DatasetSpec("test", None))
    config_path = tmp_path / "config.json"
    full_spec.save(config_path)
    materialize = MagicMock(return_value=manifest)
    monkeypatch.setattr(experiment, "materialize", materialize)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--config",
            str(config_path),
            "--max-examples",
            "2",
            "--artifacts-dir",
            str(tmp_path / "output"),
        ],
    )
    experiment.main()
    assert materialize.call_args.args[0].dataset.sample_size == 2
    assert materialize.call_args.args[0].dataset.split == "test"


def test_bad_manifest_hash_fails_before_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: MagicMock
) -> None:
    source = tmp_path / "source"
    _manifest(source)
    (source / "0.wav").write_bytes(b"edited")
    with pytest.raises(ValueError, match="hash mismatch"):
        _run(monkeypatch, source, tmp_path / "output")
    provider.models.generate_content.assert_not_called()
    assert not (tmp_path / "output" / "report.json").exists()


def test_api_failure_does_not_write_partial_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: MagicMock
) -> None:
    from scribr.llm import LLMTranscriptionError

    source, output = tmp_path / "source", tmp_path / "output"
    _manifest(source)
    provider.models.generate_content.return_value = (
        types.GenerateContentResponse()
    )
    with pytest.raises(LLMTranscriptionError):
        _run(monkeypatch, source, output, "--max-attempts", "1")
    assert not (output / "report.json").exists()
    provider.close.assert_called_once()


def test_default_config_matches_baseline_synthesis() -> None:
    config = ExperimentSpec.load(
        Path(experiment.__file__).parent / "config.json"
    )
    algorithmic = ExperimentSpec.load("experiments/algorithmic/config.json")
    assert config.dataset == DatasetSpec("test", 4)
    assert config.instruments == algorithmic.instruments
    assert config.synthesis == algorithmic.synthesis


def test_qwen_cli_records_provider_and_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    _manifest(source, 1)
    received = []

    class OfflineQwen:
        api_calls = 0
        cache_hits = 0

        def __init__(self, config: object, *, cache_dir: Path) -> None:
            received.append(config)

        def transcribe(
            self, audio: np.ndarray, sample_rate: int, tempo: float
        ) -> Melody:
            self.api_calls += 1
            return Melody((Note(69, 0, 1),))

        def close(self) -> None:
            pass

    monkeypatch.setattr(experiment, "LLMTranscriber", OfflineQwen)
    _run(
        monkeypatch,
        source,
        output,
        "--provider",
        "qwen",
        "--model",
        "qwen3.8-omni-flash",
        "--base-url",
        "https://ws-test.ap-southeast-1.maas.aliyuncs.com/api/v1",
    )
    report = json.loads((output / "report.json").read_text())
    assert report["experiment"] == "llm_qwen"
    assert report["llm"]["provider"] == "qwen"
    assert report["overall"]["F-measure_no_offset"] == 1
    assert report["parameters"]["qwen_base_url"].endswith(
        "/compatible-mode/v1/"
    )
    assert received[0].provider == "qwen"


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_nonpositive_limit_is_rejected(
    limit: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluate", "--max-examples", limit])
    with pytest.raises(SystemExit):
        experiment.main()
