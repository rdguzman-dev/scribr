# Basic Pitch transfer learning

Basic Pitch's pretrained CQT frontend and contour trunk, with its
polyphonic heads replaced by Scribr's per-step 90-class classifier. The
model maps `(batch, 8, 264, frames)` harmonic CQT features to
`(batch, 64, 90)` logits, so `melody_from_classes`, the shared training
loop, and the `mir_eval` evaluator work unchanged.

This is approach 4 in the notes on forcing monophonic output: keep the
pretrained representation, make the target monophonic by construction.

No full run has been recorded yet. The smoke run and the transfer
ablation commands below are the next steps.

## Setup

Synthesis needs a SoundFont, same as the rest of Scribr:

```bash
export SCRIBR_SOUNDFONT=/path/to/GeneralUser-GS.sf2
```

The runtime path needs neither `basic-pitch` nor TensorFlow. The
converted 36 KB weight archive is committed at
`src/scribr/deep_learning/basic_pitch/weights/basic_pitch_trunk.npz`,
and the frontend kernels are rebuilt from NumPy and SciPy constants at
import time.

## Smoke run

```bash
uv run python -m experiments.deep_learning.basic_pitch.train \
  --run-name basic-pitch-smoke \
  --epochs 2 \
  --train-examples 1000 \
  --validation-examples 256
```

Artifacts land in `artifacts/<run name>/`: `best.pt`, `last.pt`,
`config.json`, and `history.json`. Watch the first validation decode:
it should produce an actual melody, not only `HOLD` and `NOTE_OFF`. If
every decode is empty after the smoke run, check that class weights are
on and compare the first validation loss with the log-mel run. Two
epochs is a short run for a randomly initialized head. If the smoke
still stays all-`HOLD`, overfit 32 examples before starting the full
run. That separates a slow start from a wiring problem.

## Full run

The committed `config.json` uses 10,000 training melodies and 1,000
validation melodies for 30 epochs. The pretrained trunk uses `1e-4`,
while the new `temporal` and `classifier` layers use
`head_learning_rate = 1e-3`. The head starts from random weights, so it
needs to move faster than the trunk. Do not freeze the trunk: it is
small and its weights need to move toward FluidSynth timbres.

```bash
uv run python -m experiments.deep_learning.basic_pitch.train
```

## Transfer ablation

Train the same config a second time with `model.pretrained` set to
`false`, keeping the seed and subset sizes identical:

```bash
cp experiments/deep_learning/basic_pitch/config.json /tmp/basic-pitch-control.json
# Set "pretrained": false and "name": "basic_pitch_control" in the copy.
uv run python -m experiments.deep_learning.basic_pitch.train \
  --config /tmp/basic-pitch-control.json \
  --run-name basic-pitch-control
```

Compare best validation `F-measure_no_offset` on the same primary
instrument and check that the `dataset.subset_sha256` tags match. The
CQT frontend is a strong pitch estimator by itself. Without the control
arm, a gain could come from the frontend rather than the pretrained
weights.

## Reference checks

The ONNX export is a development artifact. It is not needed at runtime
and is not committed. Fetch it from the wheel:

```bash
uv venv /tmp/bp-venv
VIRTUAL_ENV=/tmp/bp-venv uv pip install --no-deps \
  --target /tmp/bp-artifact basic-pitch==0.4.0
```

Convert the weights again, or just inspect the manifest:

```bash
uv run python -m scripts.convert_basic_pitch \
  --onnx /tmp/bp-artifact/basic_pitch/saved_models/icassp_2022/nmp.onnx
```

Run the golden tests with the artifact path:

```bash
SCRIBR_BASIC_PITCH_ONNX=/tmp/bp-artifact/basic_pitch/saved_models/icassp_2022/nmp.onnx \
  uv run pytest tests/test_basic_pitch_reference.py -q
```

The tests check three cut points:

| Gate | Comparison | Tolerance | Measured |
| --- | --- | --- | --- |
| 1 | Normalized log CQT before the frontend affine | `1e-3` | `1.1e-4` |
| 2 | Trunk output against `Relu__42` | `1e-4` | `4.2e-6` |
| 3 | Note and onset heads against the ONNX outputs | `1e-4` | `2.5e-6`, `7.0e-6` |

The tests skip when `SCRIBR_BASIC_PITCH_ONNX` is unset.

## What the export actually contains

The export is not the layer list in the paper or in a quick reading of
`models.py`. Two details matter for anyone changing this code:

- The first `Conv2D(32, (5, 5))` in `models.py` writes to
  `x_contours`, then the next convolution reads `x` instead. That
  variable is dead, so the exported contour trunk starts at the 8-filter
  `(3, 39)` convolution. The 32-filter `(5, 5)` tensor in the export
  belongs to the onset branch, not to the trunk.
- Every BatchNorm is folded into the preceding convolution. The trunk is
  `conv2 -> ReLU -> conv3 -> sigmoid -> conv4 -> ReLU` with no separate
  running statistics. The frontend's scalar BatchNorm survives as the
  `scale` and `shift` buffers applied between `NormalizedLog` and the
  harmonic stack.

`scripts/convert_basic_pitch.py` resolves the weight tables by graph
traversal, checks each kernel shape and stride against a `MANIFEST`, and
records the output tensor names. Do not resolve them by node-name
substring: the names are mangled and several layers share prefixes.

## Feature and model boundary

`BasicPitchFrontend` is the experiment's `FeatureExtractor`, so it runs
inside `SynthesizedMelodyDataset` where the log-mel extractor runs for
`logmel_cnn`. It returns `(8, 264, frames)` for one example, with time
on the last axis. The model receives that tensor and transposes the
frequency and time axes once before the trunk. The tempo path in
`DeepLearningTranscriber` therefore resamples the CQT time axis, not raw
audio.

The frontend has no trainable parameters. Its buffers, including the
fixed `scale` and `shift`, stay frozen.

## Layout

```text
experiments/deep_learning/basic_pitch/
├── SPEC.md      # original handoff spec; fold into this README after a run
├── config.json  # full training configuration
├── pipeline.py  # feature and model builders
├── train.py     # training entry point
├── evaluate.py  # checkpoint scoring entry point
└── artifacts/   # gitignored checkpoints and reports
```

The shared training loop, inference, configuration, and CLI live in
[`experiments/deep_learning/common`](../common/).
