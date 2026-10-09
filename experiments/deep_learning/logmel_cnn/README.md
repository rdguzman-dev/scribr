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

Training weighs the cross-entropy loss by inverse token frequency.
`HOLD` and `NOTE_OFF` together are the majority of the 90 classes, and
without weights a short run settles on predicting a single majority
class, which decodes to an empty melody. The weights are computed from
the training subset itself and stored in the run parameters through the
`class_weights` flag. Pass `--no-class-weights` for diagnostic runs.
Validation loss stays unweighted so `val/loss` is comparable across
weighted and unweighted runs. `train/loss_unweighted` uses the same
unweighted convention, which makes it the training loss to compare
against `val/loss`; `train/loss` reports the weighted objective.

An 8-epoch diagnostic on 1,024 melodies scored 0.117
`F-measure_no_offset` with weights and 0.011 without. The unweighted
run stayed on the all-`HOLD` policy until epoch 6.

## Tracking

Every run logs to the `scribr-deep-learning` MLflow experiment:

- params: the flattened training config
- metrics: `train/loss`, `train/loss_unweighted`, `train/grad_norm`,
  `train/learning_rate`, `val/loss`, overall and per-group (`pitch`,
  `hold`, `note_off`) token accuracy, the full `mir_eval` suite as
  `val/<metric>`, and `val/<instrument>/F-measure_no_offset`
- artifacts: `best.pt`, `config.json`, `history.json`
- tags: git commit and dirty state, device, training instrument, and a
  SHA-256 fingerprint of the training subset

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
melodies: split `test`, 20 examples (the baseline sampled the first
1,000 records, seed 42).

## Layout

```text
experiments/deep_learning/logmel_cnn/
├── config.json  # the first run's configuration
├── spec.py      # typed config with JSON persistence
├── data.py      # subset loading, fingerprints, model and feature builders
├── train.py     # training loop and MLflow logging
├── evaluate.py  # checkpoint scoring and reports
└── artifacts/   # gitignored checkpoints and reports
```

The model itself lives in `src/scribr/deep_learning/`, next to the
`DeepLearningTranscriber` that implements the `Transcriber` protocol.
