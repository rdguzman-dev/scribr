# Log-mel CNN

The first deep learning experiment. A small CNN reads log-mel
spectrograms of synthesized monophonic audio and predicts one class per
dataset step: a MIDI pitch, `HOLD`, or `NOTE_OFF`. Predictions decode
with `decode_pitch_sequence` and score with the shared `mir_eval` batch
scorer, so results are comparable to the human baseline.

Training is piano-only. Evaluation renders the same references with all
four instruments, so the per-instrument split measures timbre transfer:
the model never sees guitar or violin audio during training.

Runs are tracked in a local MLflow SQLite store. Nothing leaves the
machine.

## Setup

Synthesis needs a SoundFont, same as the rest of Scribr:

```bash
export SCRIBR_SOUNDFONT=/path/to/GeneralUser-GS.sf2
```

## Smoke run

Before the full training run, check the pipeline with a couple of
minutes of work:

```bash
uv run python -m experiments.deep_learning.logmel_cnn.train \
  --run-name logmel-cnn-smoke \
  --epochs 2 \
  --train-examples 1000 \
  --validation-examples 256
```

Artifacts land in `artifacts/<run name>/`: `best.pt`, `last.pt`,
`config.json`, and `history.json`. `best.pt` is the checkpoint with the
best validation `F-measure_no_offset`.

## Training run

The committed `config.json` is the first real run: 10,000 training
melodies and 1,000 validation melodies, 30 epochs, AdamW with a cosine
schedule.

```bash
uv run python -m experiments.deep_learning.logmel_cnn.train
```

Scale the training subset only if validation loss is still falling at
the end. Double `dataset.max_examples` and rerun; keep the validation
subset fixed so runs stay comparable.

Training weighs the cross-entropy loss by tempered inverse token
frequency. `HOLD` and `NOTE_OFF` together are the majority of the 90
classes, and without weights a short run settles on predicting a single
majority class, which decodes to an empty melody. Full inverse
frequency over-corrects, though: rare pitch tokens get weights large
enough to push the model toward isolated notes. The weight for a class
is `count ** -class_weight_power`, normalized to mean one, so `1.0` is
the original full inverse frequency, `0.0` is unweighted, and the
`0.5` default tempers the correction. The weights are computed from
the training subset itself and stored in the run parameters through
`class_weight_power`. Pass `--no-class-weights` for diagnostic runs
and `--class-weight-power` to override the exponent. Validation loss
stays unweighted so `val/loss` is comparable across weighted and
unweighted runs. `train/loss_unweighted` uses the same unweighted
convention, which makes it the training loss to compare against
`val/loss`; `train/loss` reports the weighted objective.

An 8-epoch diagnostic on 1,024 melodies scored 0.117
`F-measure_no_offset` with full inverse-frequency weights and 0.011
without. The unweighted run stayed on the all-`HOLD` policy until
epoch 6.

## Class-weight ablation

The ablation trains the same configuration twice, once with
`class_weight_power=1.0` (the old inverse-frequency scheme) and once
with the tempered default, and compares the best piano-only
`F-measure_no_offset` on the validation subset. Both arms share the
seed, data, and schedule.

```bash
uv run python -m experiments.deep_learning.logmel_cnn.ablation
```

The defaults are the 8-epoch diagnostic scale: 1,024 training melodies
and 128 validation melodies. Pass `--epochs 30 --train-examples 10000
--validation-examples 1000` to match the full run. Each arm writes its
usual artifacts under `artifacts/<run name>/<arm>/`, and the
comparison is written as `ablation.json` and `ablation.md` in
`artifacts/<run name>/`. A summary run tagged `ablation=class-weights`
collects both scores in MLflow.

Arm checkpoints are selected on piano-only `F-measure_no_offset`.
Set `optimization.primary_instrument` in a config to apply that rule
to a regular run; `null` keeps the overall mean.

## Tracking

Every run logs to the `scribr-deep-learning` MLflow experiment:

- params: the flattened training config
- metrics: `train/loss`, `train/loss_unweighted`, `train/grad_norm`,
  `train/learning_rate`, `val/loss`, overall and per-group (`pitch`,
  `hold`, `note_off`) token accuracy, the full `mir_eval` suite as
  `val/<metric>`, and `val/<instrument>/F-measure_no_offset`
- artifacts: `best.pt`, `config.json`, `history.json`
- tags: git commit and dirty state, device, training instrument, the
  primary metric label, and a SHA-256 fingerprint of the training
  subset

View runs locally:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Run metadata lives in `mlflow.db` at the repository root. Artifacts stay
in `mlruns/` beside the database; the migrated file store paths in the
database point there.

Tags and params are the reproducibility contract. The subset fingerprint
covers the training pitch sequences, so two runs with the same seed,
subset size, and fingerprint saw the same data even if the shard layout
changed underneath.

## Evaluation

Score a checkpoint on a held-out split:

```bash
uv run python -m experiments.deep_learning.logmel_cnn.evaluate \
  --checkpoint artifacts/logmel-cnn-smoke/best.pt \
  --split test \
  --max-examples 512
```

The config is loaded from the checkpoint, so the model and features are
rebuilt exactly as trained. The command writes `report.json` and
`report.md` next to the checkpoint and logs a separate MLflow run tagged
with the split. `F-measure_no_offset` is the headline metric; offsets
from synthesized audio are noisy because instrument release tails should
not be transcribed as notes. Only evaluate the test split with a frozen
config.

To compare against the human baseline directly, score the same 20 test
melodies: split `test`, 20 examples (the baseline uses the first 20
test records).

## Layout

```text
experiments/deep_learning/logmel_cnn/
├── config.json  # the full training run's configuration
├── pipeline.py  # log-mel feature and CNN builders
├── train.py     # training entry point
├── ablation.py  # old vs tempered class weights
├── evaluate.py  # checkpoint scoring entry point
└── artifacts/   # gitignored checkpoints and reports
```

The shared training loop, inference, configuration, and CLI live in
[`experiments/deep_learning/common`](../common/). See the
[deep learning README](../README.md) for how to add an architecture.
The model itself lives in `src/scribr/deep_learning/`, next to the
`DeepLearningTranscriber` that implements the `Transcriber` protocol.
