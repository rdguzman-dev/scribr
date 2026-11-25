# Basic Pitch transfer learning: implementation spec

Status: implemented; full training runs pending. This file is the
handoff spec for the experiment. Once the experiment runs, fold this
content into `README.md` and drop the spec.

## Goal

Fine-tune Spotify's Basic Pitch as a monophonic transcriber for Scribr by
keeping its pretrained CQT frontend and convolutional trunk, and replacing
its polyphonic output heads with the project's per-step 90-class classifier.
Call this approach 4 in the notes on forcing monophonic output.

The result must be a drop-in for the existing pipeline: `(batch, features,
time)` in, `(batch, STEPS_PER_MELODY, NUM_CLASSES)` logits out, decoded by
`melody_from_classes`, trained by `experiments/deep_learning/common/training.py`,
scored by `mir_eval` through `common/inference.py`.

## Non-goals

- Polyphonic transcription, pitch bends, note contours, onset detection as
  a first-class output.
- Real recorded audio. The dataset is FluidSynth-rendered.
- Depending on `basic-pitch` or TensorFlow at runtime. The port is pure
  torch and numpy, and `onnx` is a development dependency only.
- Frame-level heads. If a later experiment wants onsets, the pretrained
  onset branch is still available from the same cut point.

## Why the per-step head

`PitchSequenceCNN` already frames transcription as 90-way softmax per
dataset step, which makes monophony structural: at most one pitch class can
be active per step, and `HOLD`/`NOTE_OFF` are explicit classes. Basic
Pitch's own heads are per-bin sigmoids trained with per-pixel BCE, so they
are polyphonic by parametrization, and no amount of post-processing turns
that into a guarantee.

Keeping the project's target encoding means `class_weights`,
`token_group_masks`, `melody_from_classes`, the checkpoint format, the
report writer, and the MLflow contract all work unchanged. The experiment
is then comparable to `logmel_cnn` with one variable changed: pretrained
trunk versus random init.

## Upstream facts

All measurements below come from `basic_pitch-0.4.0`'s `nmp.onnx`
(SHA-256 of the wheel: `738adb50...425db0`, ONNX file size 230,444 bytes),
inspected with `onnx` and `onnxruntime` 1.32.

| Property | Value |
| --- | --- |
| Input | `(batch, 43844, 1)` float audio at 22050 Hz, fixed length |
| CQT | fmin 27.5 Hz, 309 bins, 36 bins/octave, hop 256, computed inside the graph |
| Frame rate | 22050 / 256 = 86.13 fps |
| Head outputs | `note` (172, 88), `onset` (172, 88), `contour` (172, 264), all sigmoid |
| Conv parameters | 16,625 total: 15,656 in the trunk path used here |
| Training data | mixtures of Slakh, MAESTRO, GuitarSet, MedleyDB-Pitch, iKala |
| Licence | Apache 2.0 (repository `LICENSE`) |

The pretrained model accepts exactly 43,844 samples per call. Feeding
44,100 or 187,425 samples raises `INVALID_ARGUMENT` from onnxruntime,
because the export baked the length into the graph. A torch port has no
such restriction, which is one reason to port rather than wrap.

The trunk cut point is the tensor that feeds the note head (`Relu__42` in
the export), shape `(N, 32, 172, 88)`: 32 channels, 172 frames, and an
88-bin semitone axis that already matches `DATASET_MIN_PITCH` (21) through
`DATASET_MAX_PITCH` (108). No pitch remapping is needed anywhere.

Trunk weight tensors present in the export, keyed by their `models.py` layers
(shapes as they appear in `nmp.onnx`):

| layer | kernel | stride | weight shape |
| --- | --- | --- | --- |
| conv1 | (5, 5) | 1 | (32, 8, 5, 5) |
| bn1 | | | 32 channels |
| conv2 | (3, 39) | 1 | (8, 8, 3, 39) |
| bn2 | | | 8 channels |
| conv3 + sigmoid | (5, 5) | 1 | (1, 8, 5, 5) |
| conv4 | (7, 7) | (1, 3) | (32, 1, 7, 7) |

Every conv keeps the time axis at 172 frames and only conv4 changes the
frequency axis, 264 to 88.

### Numerics that took measuring

CQT parameters for this configuration:

- `Q = 1 / (2 ** (1/36) - 1) = 51.4386`
- `n_octaves = ceil(309 / 36) = 9`, and the top octave is truncated to 21
  bins, giving 309 in total
- 36 complex kernels of `fftLen = 256` taps per octave, windowed with a
  periodic Hann and normalized to unit L1 norm, scaled by `1/l` and
  modulated by the complex exponential at the bin frequency
- hop 256, halved at every octave; the signal is lowpassed with the same
  256-tap `firwin2` filter and decimated by 2 before each octave
- no early downsampling: `early_downsample_count` evaluates to 0 for these
  parameters, so that branch does not need porting
- reflect padding of `fftLen // 2` on both sides before each octave's
  correlation
- output is scaled by `sqrt(ceil(Q * fs / f))` per bin, which is librosa's
  normalization

Frame counts from the verified prototype: 43,844 samples give 172 frames,
8.0 s gives 690, 8.5 s gives 733.

Two conventions are easy to get backwards and cost a debugging session if
they are wrong:

1. `tf.nn.conv1d` and `torch.nn.functional.conv1d` compute cross-correlation.
   `scipy.signal.convolve` and `fftconvolve` flip the kernel. With the
   asymmetric complex CQT kernels, mixing them up produces a wrong but
   plausible CQT.
2. Harmonic stacking shifts content down in bin index by
   `s = round(36 * log2(h))`. Shifts are `[-36, 0, 36, 57, 72, 84, 93, 101]`
   for `h in [0.5, 1, 2, 3, 4, 5, 6, 7]`. Note that `h=3` gives 57, not 68.
   Positive `s` pads zeros at the high-frequency end, negative `s` pads at
   the low-frequency end, and the result is truncated to 264 bins.

Verified against the ONNX graph: the CQT plus `NormalizedLog` matches to
max 3.1e-4 and mean 9.8e-5, and harmonic-stack channels 0, 1 and 2
correlate 1.0000 with the predicted shifts.

Do not substitute `librosa.cqt`. It correlates only 0.94 with the TF
frontend and deviates by up to 0.46 on a [0, 1] scale, because librosa
resamples with soxr for its early-downsampling path while Basic Pitch uses
a FIR lowpass. Loading pretrained weights onto a mismatched frontend
changes the input distribution under the first conv, which is exactly the
thing transfer learning is supposed to preserve.

`torchaudio.transforms.Cqt` was removed before torchaudio 2.11, so there is
no library shortcut anyway.

## Dependency decision

No `basic-pitch` dependency, no TensorFlow, no fork.

Resolving `basic-pitch` for Python 3.13 fails outright: the package requires
`tensorflow>=2.4.1,<2.15.1`, and TensorFlow only
started publishing cp313 wheels at 2.20.0. On macOS it also pulls
`coremltools` unconditionally and `tensorflow-macos`, which stops at
2.16.2. Independently of Python, TensorFlow 2.15 requires numpy < 2 while
this project requires `numpy>=2.5.3`. Even with the pin relaxed, the shipped
`nmp` SavedModel is Keras 2 format and TensorFlow 2.16+ ships Keras 3,
which cannot load it without the `tf-keras` backport.

Add to `[dependency-groups].dev`:

```toml
onnx = ">=1.16"           # one-off weight extraction and golden tests
onnxruntime = ">=1.20"    # reference outputs for the golden tests
```

Both install cleanly on Python 3.13 with numpy 2.5. Neither becomes a
runtime dependency. `onnx` is only used by `scripts/convert_basic_pitch.py`;
`onnxruntime` is only used by the golden tests, which skip when the artifact
is absent.

Add `scipy` to `[project.dependencies]` as well. It is already installed as
a transitive dependency of librosa, so the lock does not change, but
`BasicPitchFrontend` imports `scipy.signal` for `firwin2` and `get_window`,
and a module should not rely on a dependency it does not declare.

Register a marker rather than reusing `integration`, which already means
"needs FluidSynth and a SoundFont":

```toml
markers = [
    "integration: requires a system FluidSynth install and a SoundFont (set SCRIBR_SOUNDFONT)",
    "basic_pitch: requires the Basic Pitch ONNX artifact for reference checks",
]
```

## Deliverables

```text
scripts/convert_basic_pitch.py                      # ONNX -> .npz, one-off
src/scribr/deep_learning/basic_pitch.py             # frontend, trunk, model
src/scribr/deep_learning/__init__.py                # export the new names
experiments/deep_learning/basic_pitch/pipeline.py   # build_features/build_model
experiments/deep_learning/basic_pitch/config.json   # full run configuration
experiments/deep_learning/basic_pitch/train.py      # shared CLI
experiments/deep_learning/basic_pitch/evaluate.py   # shared CLI
experiments/deep_learning/basic_pitch/README.md     # after the first run
tests/test_basic_pitch_frontend.py                  # kernel/shape unit tests
tests/test_basic_pitch_model.py                     # model contract tests
tests/test_basic_pitch_reference.py                 # golden tests, marked basic_pitch
src/scribr/deep_learning/weights/basic_pitch_trunk.npz  # converted weights, ~70 KB
```

Commit these as separate commits: conversion script and artifact, frontend,
trunk plus model, experiment wiring, tests. Branch name
`feat/port-basic-pitch-trunk`.

## Interface contracts

Boundaries, with the shapes the rest of the pipeline expects:

```text
audio            (batch, samples)      float32, 22050 Hz
CQT              (batch, 309, frames)  magnitude, before NormalizedLog
normalized log   (batch, frames, 309)  min-max over the whole example
harmonic stack   (batch, 8, frames, 264)
trunk            (batch, 32, frames, 88)
step head        (batch, 64, 90)       logits
```

The trunk keeps Basic Pitch's ONNX axis order `(N, C, frames, freq)`, so the
four weight tensors copy across with no spatial permutation. The frontend
returns `(N, C, freq, frames)` to match the rest of the project, so the model
transposes axes 2 and 3 exactly once, in `forward`, before the trunk.

### The `FeatureExtractor` contract

`FeatureExtractor` is documented as producing a 2D `(channels, frames)`
feature map. The HCQT is `(8, 264, frames)`. Two acceptable resolutions:

1. Widen the protocol docstring to "feature map whose last axis is time",
   and have `BasicPitchFrontend` return `(8, 264, frames)`. The transcriber
   only uses `sample_rate`, `num_frames`, `spectrogram`, `_stretch_tempo`
   and `fix_length`, all of which act on the last axis, so nothing else
   changes.
2. Keep the frontend inside the model and give the experiment an identity
   feature extractor that returns the raw waveform.

Prefer (1). It keeps the transcriber, the dataset, and the tempo stretch
shared with `logmel_cnn`, and (2) would silently make `features` meaningless
in the checkpoint config.

`num_frames` should be `None`. The port is fully convolutional in time, and
every synthesized example is 8.5 s, so batches are already rectangular.
`DEFAULT_NUM_FRAMES = 736` belongs to `LogMelSpectrogram` and should not be
reused.

One deviation from the upstream frontend is required for the tempo path:
`_stretch_tempo` resamples the last axis, which for the HCQT is time. That
is correct and needs no change.

## Step 0: fetch the artifact

```bash
uv venv /tmp/bp-venv
VIRTUAL_ENV=/tmp/bp-venv uv pip install --no-deps \
  --target /tmp/bp-artifact basic-pitch==0.4.0
ls /tmp/bp-artifact/basic_pitch/saved_models/icassp_2022/nmp.onnx
```

`--no-deps` is what makes this work; installing the package with its
dependencies fails on Python 3.13, see the dependency section. Unpacking to a
`--target` directory never imports the package, so the missing TensorFlow
does not matter.

The wheel is 758,279 bytes and contains `nmp.onnx`, `nmp.tflite` and
`nmp.mlpackage`. Only the ONNX file matters, and only the ~70 KB derived
`.npz` is committed. If you would rather pin the download, the wheel's
SHA-256 is
`738adb503aae7fdfc7d1e1511aa0ce35052315f260a19531ef4c356708425db0`.

## Step 1: frontend port

Implement `BasicPitchFrontend` in `src/scribr/deep_learning/basic_pitch.py`.
It registers everything as buffers, has no trainable parameters, and
exposes the same `sample_rate`/`num_frames`/`spectrogram`/`__call__`
surface as `LogMelSpectrogram` so it satisfies the widened protocol.

The scipy prototype in [Appendix A](#appendix-a-validated-frontend-prototype)
is the numerical reference. Port it to torch as follows:

- Build the CQT kernels and the two lowpass filters once in `__init__` with
  numpy and `scipy.signal`, then register them as `float32` buffers shaped
  `(out_channels, 1, taps)` for `F.conv1d`.
- Keep each octave's correlation as `F.conv1d(padded, kernel, stride=hop)`.
  That is already cross-correlation, so the kernel flip in the prototype
  disappears.
- Keep the frequency-axis shifts as `F.pad` plus slicing. Do not use
  `torch.roll`, which wraps instead of zero-filling.
- Compute the magnitude with `torch.abs`, not `sqrt(re**2 + im**2)` on
  split real/imaginary tensors.
- Keep `NormalizedLog` exactly as written: `10 * log10(power + 1e-10)`,
  subtract the per-example minimum over both axes, divide by the maximum,
  and guard a zero maximum.

The frontend runs in `DataLoader` workers inside
`SynthesizedMelodyDataset`, which is where the log-mel extractor already
runs. Cost is negligible: the whole 9-octave frontend is one strided
correlation plus eight decimate-and-correlate passes, roughly 50 M MACs for
an 8.5 s example.

`SynthesizedMelodyDataset` checks that the synthesizer sample rate matches
`features.sample_rate`. The default is 22050 on both sides, so no change.

**Gate 1.** Add the pre-stack tensor as a graph output, run both
implementations on the same 43,844-sample probe, and require
`max |diff| < 1e-3` on the normalized log CQT. The measured value for the
prototype is 3.1e-4.

## Step 2: trunk and weight conversion

`BasicPitchTrunk` is `conv1 -> bn1 -> relu -> conv2 -> bn2 -> relu ->
conv3 -> sigmoid -> conv4 -> relu`, keeping the axis order
`(N, C, frames, freq)`. Note that the sigmoid on conv3's output is part of
the trunk: the upstream graph feeds the sigmoid contour map into conv4.

`scripts/convert_basic_pitch.py` writes a `.npz` with float32 arrays. Rules
that matter:

- ONNX Conv weights are `(out, in, kH, kW)` and map straight into
  `nn.Conv2d`. No permutation. Biases come from the Conv node's third
  input.
- Resolve each weight table by graph traversal, not by name. The exported
  node names are mangled, several distinct nodes share a prefix, and
  substring matching silently returns the wrong tensor. Build a
  producer map from output tensor name to node and walk from the graph
  inputs. I hit this while probing and it produced a plausible but wrong
  layer order.
- BatchNorm: TF exports `FusedBatchNormV3` as `(x - mean) * gamma /
  sqrt(var + eps) + beta` with mean, variance, gamma and beta as
  initializers. Map to `nn.BatchNorm2d(weight=gamma, bias=beta,
  running_mean=mean, running_var=var)` and set `eps=1e-3`.
  TensorFlow's BatchNorm default is 1e-3 and torch's is 1e-5. Getting this
  wrong shifts every pretrained activation slightly, trains fine, and hides
  the bug.
- The frontend's own BatchNorm (from `get_cqt(..., use_batchnorm=True)`) is
  a scalar affine on a single-channel map. Fold it into two float buffers,
  `scale` and `shift`, applied after `NormalizedLog` and before harmonic
  stacking. Keep it frozen; there is nothing to learn at that scale.
- Record the ONNX tensor and node names used for each table in a
  `MANIFEST` dict inside the script, and assert the expected shapes. Names
  are stable for the 0.4.0 artifact and give the next person an audit
  trail.

Take the architecture from `basic_pitch/models.py` in the wheel, not from the
shape of the exported graph. While probing I found a `(8, 8, 3, 39)` conv
whose input appears to be the harmonic stack rather than the 32-channel
output of the first conv, which `models.py` does not describe. That is most
likely the exporter's name mangling, but it is exactly why the layer order
has to come from `models.py` and be confirmed numerically instead of
inferred from node names or shapes.

Settle it first, before writing the torch module, because the wiring of the
first two convs decides how the weights are loaded. Run each candidate chain
on the verified frontend output and compare against the exported `Relu__35`
and `Relu__38` tensors: `conv1(32 filters) -> BN -> ReLU -> conv2(8 filters)`
from `models.py` should reproduce them if the export is faithful, and
`conv2` applied directly to the harmonic stack should reproduce them if the
shape inference is right. Whichever chain passes at 1e-6 under a numpy
convolution is the one to implement. Record the answer in the script's
`MANIFEST`.

**Gate 2.** Feed identical audio to the torch trunk and to the ONNX graph
with `Relu__42` exposed as an output. Require `max |diff| < 1e-4`. This
gate covers the stacking direction, the folded BatchNorm, the eps, the
layer order and the axis convention at once. If it fails, diagnose with
per-channel shifts before rewriting anything: correlate each of the 8
harmonic-stack channels against a shifted copy of the reference, and guard
slice bounds, because a negative start index in numpy wraps silently and
produces meaningless correlations.

## Step 3: reassemble the original heads

Before building the new model, reproduce the full upstream network in torch:
add the conv3 sigmoid contour head, the note head `conv(1, (7, 3))`
sigmoid, the onset branch `conv(32, (5, 5), stride (1, 3))` plus its BN and
the `concat` plus `conv(1, (3, 3))` sigmoid, and apply `FlattenFreqCh`.

**Gate 3.** Require `max |diff| < 1e-4` on both the note and onset heads
against onnxruntime. This is the acceptance test for the weight mapping.
Without it, a 0.02 mismatch in the trunk looks like "fine-tuning noise" and
the experiment silently loses its point.

Keep this reassembled reference model in `tests/test_basic_pitch_reference.py`
and skip it when the ONNX artifact is missing. Mark it `basic_pitch`, a new
marker registered alongside `integration`.

Fallback if Gate 3 cannot be made to pass in one session: cut the ONNX graph
at `Relu__42` with `onnx.utils.extract_model` and use onnxruntime as a
frozen feature extractor to unblock training, while the torch trunk stays
under test. This gives up unfreezing the trunk but not the schedule.

### Exposing intermediate tensors

The recipe for every gate:

```python
graph = onnx.load(MODEL)
graph.graph.output.append(
    helper.make_tensor_value_info(TENSOR, TensorProto.FLOAT, None)
)
onnx.save(graph, tmp)  # onnxruntime now returns that tensor too
```

Pick probe audio that separates the failure modes: a 440 Hz sine puts its
CQT peak at bin 144 (`36 * log2(440 / 27.5)`), which makes axis and shift
errors obvious, and a FluidSynth render of a real dataset example catches
edge behavior in the release tail.

## Step 4: the new model

```python
class BasicPitchPitchSequenceCNN(nn.Module):
    """Pretrained Basic Pitch trunk with a per-step pitch-sequence head.

    Args:
        steps: Number of dataset steps to predict.
        num_classes: Size of the per-step class vocabulary.
        dropout: Dropout probability before the classifier.
        pretrained: Load the converted trunk weights. `False` keeps the
            same architecture with fresh initialization, which is the
            control arm for the transfer ablation.
    """

    def __init__(
        self,
        *,
        steps: int = STEPS_PER_MELODY,
        num_classes: int = NUM_CLASSES,
        dropout: float = DEFAULT_DROPOUT,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.frontend = BasicPitchFrontend()
        self.trunk = BasicPitchTrunk(pretrained=pretrained)
        self.temporal = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, features: Tensor) -> Tensor:
        stack = self.frontend(features)                # (N, 8, 264, frames)
        hidden = self.trunk(stack.transpose(2, 3))     # (N, 32, frames, 88)
        hidden = hidden.mean(dim=-1)                   # pool pitch bins
        hidden = self.temporal(hidden)
        hidden = F.interpolate(
            hidden, size=steps, mode="linear", align_corners=False
        )
        return self.classifier(hidden.transpose(1, 2))
```

Pooling the 88-bin pitch axis before classification mirrors
`PitchSequenceCNN`, which also means over its frequency axis. The first
variant should keep that for comparability.

Variants to try only after the baseline runs end to end:

- Keep the pretrained note head as an 88-channel feature instead of
  averaging: interpolate `(N, 88, frames)` onto 64 steps and concatenate it
  with the pooled trunk features before the classifier.
- Add a second head predicting note onsets per step, using the pretrained
  onset branch as initialization.

Unit tests, mirroring `tests/test_deep_learning_model.py`:

- forward shape from a 3D features tensor
- gradients reach every trainable parameter, including the trunk when it is
  unfrozen
- `pretrained=False` produces different trunk weights than `pretrained=True`
  for the same seed
- the frontend has no parameters and stays `requires_grad=False`
- the frontend accepts 43,844 samples, 8.0 s and 8.5 s inputs and returns
  172, 690 and 733 frames respectively

## Step 5: experiment wiring

Mirror `experiments/deep_learning/logmel_cnn/` exactly. `pipeline.py`
exports the two builders:

```python
def build_features(spec: Mapping[str, Any]) -> BasicPitchFrontend:
    return BasicPitchFrontend(
        sample_rate=spec["sample_rate"],
        num_frames=spec["num_frames"],
    )


def build_model(
    spec: Mapping[str, Any], features: Mapping[str, Any]
) -> BasicPitchPitchSequenceCNN:
    return BasicPitchPitchSequenceCNN(
        steps=STEPS_PER_MELODY,
        dropout=spec["dropout"],
        pretrained=spec["pretrained"],
    )
```

`config.json` starts from `logmel_cnn/config.json` and changes three things:

- `features`: `{"sample_rate": 22050, "num_frames": null}`. The frontend has
  no tunable fft or mel parameters; everything is fixed by the pretrained
  weights and must not be exposed as config knobs.
- `model`: `{"dropout": 0.1, "pretrained": true}`.
- `optimization.learning_rate`: `1e-4`, not `1e-3`, for the pretrained
  trunk. The trunk is only 15,656 parameters. Set
  `optimization.head_learning_rate` to `1e-3` so the randomly initialized
  `temporal` and `classifier` layers move faster without retuning the
  trunk.

Keep `class_weight_power = 0.5`, batch size 64, and the cosine schedule from
the shared loop. Do not freeze the trunk: with 15,656 parameters freezing
buys nothing, and the BatchNorm running statistics need to move toward
FluidSynth timbres anyway.

Smoke run before anything else:

```bash
export SCRIBR_SOUNDFONT=/path/to/GeneralUser-GS.sf2
uv run python -m experiments.deep_learning.basic_pitch.train \
  --run-name basic-pitch-smoke --epochs 2 \
  --train-examples 1000 --validation-examples 256
```

Two cheap signals that the port is wired correctly, before any real
training: an overfit test (train on 32 examples until the loss is near zero;
if it plateaus, the head cannot represent the grammar and something upstream
is wrong), and the first validation decode of a handful of examples, which
should produce non-empty melodies within the first epoch. The `logmel_cnn`
notes describe the failure mode to expect otherwise: unweighted
cross-entropy collapses onto `HOLD` and decodes to empty melodies.

### Release tail

8.5 s at 120 BPM is 8 s of notes plus a 0.5 s release, and the head
interpolates all 733 frames onto 64 steps, so the tail gets stretched across
the grid. `logmel_cnn` has the same behavior with
`DEFAULT_NUM_FRAMES = 736`. If the last step or two of decoded melodies are
consistently wrong, crop the audio to 8 s before the CQT rather than
changing the interpolation.

## Step 6: transfer ablation

Run the same config twice, identical seed and subset, with
`model.pretrained` true and false. Compare best validation
`F-measure_no_offset` on the same primary instrument, and record the subset
fingerprint from the MLflow tags to prove the data matched.

This is not optional. The trunk is 15,656 parameters, the CQT frontend is a
strong pitch estimator whether or not it was pretrained, and the training
set has 10 million examples. "Transfer learning helped" is a claim that
needs the control arm.

If the control arm matches the pretrained arm, the honest conclusion is that
the gain came from the CQT frontend, not the weights, and the report should
say so.

## Acceptance gates

| Gate | Compares | Tolerance | Covers |
| --- | --- | --- | --- |
| 1 | torch frontend vs ONNX pre-stack tensor | 1e-3 | CQT kernels, normalization, padding, frame alignment |
| 2 | torch trunk vs ONNX `Relu__42` | 1e-4 | stacking direction, folded BN, eps, layer order, axes |
| 3 | torch note/onset heads vs ONNX | 1e-4 | weight mapping end to end |
| 4 | 32-example overfit | loss near zero | head can represent the target grammar |
| 5 | 2-epoch smoke run | non-empty decodes | full pipeline integration |

Gates 1 to 3 are the ones that cannot be skipped or approximated. Gates 4
and 5 are cheap and catch wiring errors that tolerances do not.

## Risks and open questions

- The first two convs' wiring is unresolved in the export, see Step 2. It is
  a 20-minute check against the exported ReLU tensors and it blocks the
  weight mapping, so do it before writing `BasicPitchTrunk`.
- Weight redistribution. The repository is Apache 2.0 and the wheel ships no
  separate model card. Keeping the converted `.npz` in the repo is
  consistent with that licence, but confirm before publishing anything that
  redistributes the weights outside the repository.
- The 3D `FeatureExtractor` contract is a real (if small) API change. If it
  turns out to leak into more than the docstring, stop and reconsider
  approach 2 (frontend inside the model) rather than growing the protocol.
- Basic Pitch was trained on recordings, Scribr's audio is a GM soundfont.
  Expect fine-tuning to converge fast. If validation stalls at a score far
  below `logmel_cnn`, check Gate 2 before touching hyperparameters.
- The port fixes the frontend, so the only hyperparameters left are
  `dropout`, the head width, the temporal kernel, and the optimizer. Resist
  adding CQT knobs; they are a property of the pretrained weights, and
  retuning them invalidates the transfer.

## Appendix A: validated frontend prototype

This is the numpy/scipy version whose numerics were checked against the ONNX
graph (max 3.1e-4 on the normalized log CQT). Port it to torch as described
in Step 1. It is not intended to ship as-is: it loops over bins with
`fftconvolve`, which is fine for a 36-bin octave and unnecessary in the final
implementation.

```python
import numpy as np
import scipy.signal

SR, HOP, FMIN, NBINS, BPO = 22050, 256, 27.5, 309, 36


def create_lowpass_filter(band_center=0.5, kernel_length=256,
                          transition_bandwidth=0.03):
    p = band_center / (1 + transition_bandwidth)
    s = band_center * (1 + transition_bandwidth)
    return scipy.signal.firwin2(kernel_length, [0.0, p, s, 1.0],
                                [1.0, 1.0, 0.0, 0.0])


def create_cqt_kernels(Q, fs, fmin, n_bins, bpo, norm=1, window="hann"):
    fft_len = 2 ** int(np.ceil(np.log2(np.ceil(Q * fs / fmin))))
    freqs = fmin * 2.0 ** (np.arange(n_bins) / float(bpo))
    kernels = np.zeros((int(n_bins), int(fft_len)), dtype=np.complex64)
    for k in range(int(n_bins)):
        freq = freqs[k]
        length = np.ceil(Q * fs / freq)
        start = int(np.ceil(fft_len / 2.0 - length / 2.0)) - int(length % 2)
        sig = (
            scipy.signal.get_window(window, int(length), fftbins=True)
            * np.exp(np.r_[-length // 2:length // 2] * 1j * 2 * np.pi * freq / fs)
            / length
        )
        kernels[k, start:start + int(length)] = sig / np.linalg.norm(sig, norm)
    return kernels, fft_len


def downsample(x, kernel, n):  # FIR lowpass + decimation, TF-style padding
    pad = (kernel.shape[-1] - 1) // 2
    padded = np.pad(x[0, 0], (pad, pad))
    return scipy.signal.fftconvolve(padded, kernel, mode="valid")[::n][None, None, :]


def cqt_complex(x, real, imag, hop, pad):
    padded = np.pad(x[0, 0], (pad, pad), mode="reflect")
    # [::-1] turns convolution into the cross-correlation that TF and torch use
    re = np.stack([scipy.signal.fftconvolve(padded, k[::-1], mode="valid")[::hop]
                   for k in real])
    im = -np.stack([scipy.signal.fftconvolve(padded, k[::-1], mode="valid")[::hop]
                    for k in imag])
    return (re + 1j * im)[None, :, :]


def cqt2010(y):
    Q = 1.0 / (2 ** (1 / BPO) - 1)
    lowpass = create_lowpass_filter(0.5, 256, 0.001)
    n_octaves = int(np.ceil(NBINS / BPO))
    fmin_t = FMIN * 2 ** (n_octaves - 1)
    remainder = NBINS % BPO
    fmax_t = (fmin_t * 2 ** ((remainder - 1) / BPO) if remainder
              else fmin_t * 2 ** ((BPO - 1) / BPO))
    fmin_t = fmax_t / 2 ** (1 - 1 / BPO)

    # early downsampling: evaluates to a factor of 1 for these parameters
    cutoff = fmax_t * (1 + 0.5 * 1.5 / Q)
    count1 = max(0, int(np.ceil(np.log2(0.85 * (SR // 2) / cutoff)) - 1) - 1)
    count2 = max(0, int(np.ceil(np.log2(HOP))) - n_octaves + 1)
    ds_factor = 2 ** min(count1, count2)

    sr, hop = SR / ds_factor, HOP // ds_factor
    basis, n_fft = create_cqt_kernels(Q, sr, fmin_t, min(BPO, NBINS), BPO)
    x = y[None, None, :].astype(np.float32)
    if ds_factor != 1:
        x = downsample(x, create_lowpass_filter(1 / ds_factor, 256, 0.03),
                       ds_factor)

    cqt = cqt_complex(x, basis.real, basis.imag, hop, n_fft // 2)
    x_down = x
    for _ in range(n_octaves - 1):
        hop = hop // 2
        x_down = downsample(x_down, lowpass, 2)
        cqt = np.concatenate(
            [cqt_complex(x_down, basis.real, basis.imag, hop, n_fft // 2), cqt],
            axis=1,
        )

    cqt = cqt[:, -NBINS:, :] * ds_factor
    lengths = np.ceil(Q * sr / (FMIN * 2.0 ** (np.arange(NBINS) / BPO)))
    cqt = cqt * np.sqrt(lengths).reshape(1, -1, 1)
    return np.transpose(np.abs(cqt), (0, 2, 1))  # (batch, frames, bins)


def normalized_log(magnitude):
    db = 10 * np.log10(magnitude ** 2 + 1e-10)
    offset = db - db.min()
    return offset / np.where(offset.max() == 0, 1, offset.max())


def shift_bins(x, k):
    """Return x shifted so that result[i] = x[i - k], zero filled."""
    if k == 0:
        return x
    if k > 0:
        return np.concatenate(
            [np.zeros_like(x[:, :k]), x[:, :x.shape[1] - k]], axis=1)
    return np.concatenate([x[:, -k:], np.zeros_like(x[:, :-k])], axis=1)


def harmonic_stack(normalized, n_out=264):
    shifts = [int(np.round(36 * np.log2(h))) for h in
              [0.5, 1, 2, 3, 4, 5, 6, 7]]
    return np.stack([shift_bins(normalized, -s) for s in shifts],
                    axis=-1)[:, :n_out, :]
```

## Checklist

- [ ] `onnx` and `onnxruntime` in the dev dependency group, `scipy` declared
      in `[project.dependencies]`
- [ ] the first-two-convs wiring settled against the exported ReLU tensors
      and recorded in the conversion script
- [ ] `scripts/convert_basic_pitch.py` writes
      `src/scribr/deep_learning/weights/basic_pitch_trunk.npz` with shape
      assertions and a recorded name manifest
- [ ] confirm `uv build` ships the `.npz` inside the wheel
- [ ] `BasicPitchFrontend` with no parameters, buffer-only, sample-rate
      contract matching the synthesizer
- [ ] `BasicPitchTrunk` with `eps=1e-3` BatchNorm and the folded frontend
      affine
- [ ] `BasicPitchPitchSequenceCNN` with `pretrained` as a constructor flag
- [ ] Gate 1, Gate 2, Gate 3 passing, tolerances recorded in the test
      docstrings
- [ ] Unit tests for shapes, gradients, and frame counts
- [ ] `pipeline.py`, `config.json`, `train.py`, `evaluate.py`
- [ ] 32-example overfit test
- [ ] 2-epoch smoke run logged to MLflow
- [ ] Full run, then the `pretrained=False` control arm
- [ ] `README.md` written from this spec once the runs exist
