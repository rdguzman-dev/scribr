"""Tests for the Basic Pitch trunk and pitch-sequence model."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scribr.deep_learning import (
    NUM_CLASSES,
    BasicPitchPitchSequenceCNN,
    BasicPitchTrunk,
)
from scribr.deep_learning.basic_pitch import (
    N_HARMONICS,
    N_STACK_BINS,
    _load_weights,
)


def tiny_model(**overrides) -> BasicPitchPitchSequenceCNN:
    defaults = {
        "steps": 8,
        "dropout": 0.0,
        "pretrained": True,
    }
    defaults.update(overrides)

    return BasicPitchPitchSequenceCNN(**defaults)


def features(batch: int = 2, frames: int = 40) -> torch.Tensor:
    return torch.randn(batch, N_HARMONICS, N_STACK_BINS, frames)


def test_forward_shape_from_harmonic_features() -> None:
    model = tiny_model()
    logits = model(features())

    assert logits.shape == (2, 8, NUM_CLASSES)


def test_gradients_reach_every_parameter() -> None:
    model = tiny_model()
    logits = model(features())

    logits.square().mean().backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name


def test_pretrained_weights_are_loaded() -> None:
    trunk = BasicPitchTrunk(pretrained=True)
    weights = _load_weights()

    assert np.allclose(
        trunk.conv2.weight.detach().numpy(),
        weights["conv2.weight"],
    )
    assert np.allclose(
        trunk.conv4.bias.detach().numpy(),
        weights["conv4.bias"],
    )


def test_fresh_trunk_differs_from_pretrained() -> None:
    torch.manual_seed(0)
    pretrained = BasicPitchTrunk(pretrained=True)
    torch.manual_seed(0)
    fresh = BasicPitchTrunk(pretrained=False)

    assert not torch.equal(
        pretrained.conv2.weight,
        fresh.conv2.weight,
    )


def test_steps_can_differ_from_the_input_length() -> None:
    model = tiny_model(steps=64)
    logits = model(features(batch=1, frames=733))

    assert logits.shape == (1, 64, NUM_CLASSES)


@pytest.mark.parametrize(
    ("overrides", "shape"),
    [
        ({"steps": 0}, (1, N_HARMONICS, N_STACK_BINS, 40)),
        ({"dropout": 1.0}, (1, N_HARMONICS, N_STACK_BINS, 40)),
        ({"dropout": -0.1}, (1, N_HARMONICS, N_STACK_BINS, 40)),
        ({}, (1, N_HARMONICS, 100, 40)),
        ({}, (1, N_HARMONICS, N_STACK_BINS)),
    ],
)
def test_invalid_parameters_or_features_raise(
    overrides: dict,
    shape: tuple[int, ...],
) -> None:
    if overrides:
        with pytest.raises(ValueError):
            tiny_model(**overrides)
        return

    model = tiny_model()

    with pytest.raises(ValueError):
        model(torch.randn(*shape))
