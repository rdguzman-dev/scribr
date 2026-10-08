"""Tests for the pitch-sequence CNN."""

from __future__ import annotations

import pytest
import torch

from scribr.deep_learning import NUM_CLASSES, PitchSequenceCNN


def tiny_model(**overrides) -> PitchSequenceCNN:
    defaults = {
        "n_mels": 16,
        "channels": (4, 8),
        "steps": 8,
        "dropout": 0.0,
    }
    defaults.update(overrides)

    return PitchSequenceCNN(**defaults)


def test_forward_shape_from_three_dimensional_input() -> None:
    model = tiny_model()
    logits = model(torch.randn(2, 16, 40))

    assert logits.shape == (2, 8, NUM_CLASSES)


def test_forward_shape_from_four_dimensional_input() -> None:
    model = tiny_model()
    logits = model(torch.randn(2, 1, 16, 40))

    assert logits.shape == (2, 8, NUM_CLASSES)


def test_gradients_reach_every_parameter() -> None:
    model = tiny_model()
    logits = model(torch.randn(2, 16, 40))

    logits.square().mean().backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name


def test_steps_can_differ_from_the_input_length() -> None:
    model = tiny_model(steps=64)
    logits = model(torch.randn(1, 16, 733))

    assert logits.shape == (1, 64, NUM_CLASSES)


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_mels": 0},
        {"channels": ()},
        {"channels": (4, 0)},
        {"steps": 0},
        {"dropout": 1.0},
        {"dropout": -0.1},
    ],
)
def test_invalid_model_parameters_raise(overrides: dict) -> None:
    with pytest.raises(ValueError):
        tiny_model(**overrides)
