"""Golden checks for the Basic Pitch port against the ONNX export.

The tests are skipped unless `SCRIBR_BASIC_PITCH_ONNX` points at
`nmp.onnx` from `basic-pitch==0.4.0`. The artifact is not committed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn.functional as F
from onnx import TensorProto, helper, numpy_helper

from scripts.convert_basic_pitch import find_graph_tensors, load_graph
from scribr.deep_learning import BasicPitchFrontend, BasicPitchTrunk

pytestmark = pytest.mark.basic_pitch

SAMPLE_RATE = 22_050
SAMPLES = 43_844


@pytest.fixture(scope="module")
def artifact() -> Path:
    configured = os.environ.get("SCRIBR_BASIC_PITCH_ONNX")

    if not configured:
        pytest.skip("set SCRIBR_BASIC_PITCH_ONNX to run the ONNX checks")

    path = Path(configured).expanduser()

    if not path.is_file():
        pytest.skip(f"SCRIBR_BASIC_PITCH_ONNX does not exist: {path}")

    return path


def _probe_audio() -> np.ndarray:
    samples = np.arange(SAMPLES)
    seconds = samples / SAMPLE_RATE
    rng = np.random.default_rng(0)

    return (
        0.3 * np.sin(2 * np.pi * 440 * seconds)
        + 0.1 * np.sin(2 * np.pi * 220 * seconds)
        + 0.02 * rng.standard_normal(SAMPLES)
    ).astype(np.float32)


@pytest.fixture(scope="module")
def probe(
    artifact: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Any, dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    graph = load_graph(artifact)
    tensors = find_graph_tensors(graph)

    for name in (
        tensors.pre_stack,
        tensors.harmonic_stack,
        tensors.trunk_output,
        tensors.note_output,
        tensors.onset_output,
    ):
        graph.graph.output.append(
            helper.make_tensor_value_info(
                name,
                TensorProto.FLOAT,
                None,
            )
        )

    path = tmp_path_factory.mktemp("basic-pitch") / "probe.onnx"
    onnx.save(graph, path)
    session = ort.InferenceSession(str(path))
    audio = _probe_audio()
    outputs = dict(
        zip(
            (output.name for output in session.get_outputs()),
            session.run(
                None,
                {session.get_inputs()[0].name: audio[None, :, None]},
            ),
        )
    )
    initializers = {
        tensor.name: numpy_helper.to_array(tensor)
        for tensor in graph.graph.initializer
    }

    return tensors, outputs, audio, initializers


def _conv(
    initializers: dict[str, np.ndarray],
    tensors: Any,
    name: str,
    inputs: torch.Tensor,
    padding: tuple[int, int],
    stride: tuple[int, int] = (1, 1),
) -> torch.Tensor:
    node = tensors.layers[name]

    return F.conv2d(
        inputs,
        torch.tensor(initializers[node.input[1]]),
        torch.tensor(initializers[node.input[2]]),
        padding=padding,
        stride=stride,
    )


def test_frontend_matches_onnx_pre_stack(probe: Any) -> None:
    """Gate 1: measured max difference 1.1e-4, tolerance 1e-3."""
    tensors, outputs, audio, _ = probe
    frontend = BasicPitchFrontend()
    waveform = torch.from_numpy(audio).unsqueeze(0)
    normalized = frontend._normalized_log(
        frontend._cqt_magnitude(waveform)
    )
    mine = normalized.squeeze(0).transpose(0, 1).numpy()
    reference = np.squeeze(outputs[tensors.pre_stack])
    difference = np.abs(mine - reference)

    assert mine.shape == reference.shape
    assert difference.max() < 1e-3
    assert difference.mean() < 1e-5


def test_harmonic_stack_matches_onnx(probe: Any) -> None:
    """Measured max difference 2.7e-4, tolerance 1e-3."""
    tensors, outputs, audio, _ = probe
    frontend = BasicPitchFrontend()
    stack = frontend(audio).unsqueeze(0)
    reference = torch.from_numpy(
        np.squeeze(outputs[tensors.harmonic_stack])
    ).permute(2, 1, 0)
    difference = (stack.squeeze(0) - reference).abs()

    assert difference.max() < 1e-3
    assert difference.mean() < 1e-5


def test_trunk_matches_onnx(probe: Any) -> None:
    """Gate 2: measured max difference 4.2e-6, tolerance 1e-4."""
    tensors, outputs, audio, _ = probe
    frontend = BasicPitchFrontend()
    trunk = BasicPitchTrunk(pretrained=True)
    stack = frontend(audio).unsqueeze(0)
    mine = trunk(stack.transpose(2, 3)).detach().numpy()
    reference = outputs[tensors.trunk_output]
    difference = np.abs(mine - reference)

    assert mine.shape == reference.shape
    assert difference.max() < 1e-4


def test_heads_match_onnx(probe: Any) -> None:
    """Gate 3: measured note 2.5e-6 and onset 7.0e-6, tolerance 1e-4."""
    tensors, outputs, audio, initializers = probe
    frontend = BasicPitchFrontend()
    trunk = BasicPitchTrunk(pretrained=True)
    stack = frontend(audio).unsqueeze(0).transpose(2, 3)
    trunk_output = trunk(stack)

    note = torch.sigmoid(
        _conv(initializers, tensors, "note", trunk_output, (3, 1))
    )
    onset = torch.relu(
        _conv(
            initializers,
            tensors,
            "onset",
            stack,
            (2, 1),
            (1, 3),
        )
    )
    onset = torch.sigmoid(
        _conv(
            initializers,
            tensors,
            "onset_final",
            torch.cat([note, onset], dim=1),
            (1, 1),
        )
    )

    note_reference = outputs[tensors.note_output]
    onset_reference = outputs[tensors.onset_output]

    assert note.shape == note_reference.shape
    assert onset.shape == onset_reference.shape
    assert np.abs(note.detach().numpy() - note_reference).max() < 1e-4
    assert np.abs(onset.detach().numpy() - onset_reference).max() < 1e-4
