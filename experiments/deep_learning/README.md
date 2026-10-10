# Deep learning experiments

Each subdirectory is one model architecture built on the shared
infrastructure in `common/`:

- `common/spec.py` defines `TrainingConfig` with the training and
  validation `DataSpec`s. The `features` and `model` sections are plain
  mappings that the architecture's builders turn into objects.
- `common/data.py` loads dataset subsets, fingerprints them, and builds
  the per-instrument synthesizer factory.
- `common/training.py` owns the training loop, checkpointing, and MLflow
  logging.
- `common/evaluation.py` scores checkpoints with `mir_eval` and writes
  `report.json` and `report.md`.
- `common/cli.py` provides the shared `train` and `evaluate` entry
  points.

## Adding an architecture

1. Add the model class in `src/scribr/deep_learning/`, next to
   `model.py`, and export it from `src/scribr/deep_learning/__init__.py`.
   The model must map `(batch, n_features, time)` features to
   `(batch, steps, num_classes)` logits.
2. Create `experiments/deep_learning/<architecture>/` with a `data.py`
   exposing two builders:

   ```python
   def build_features(spec: Mapping[str, Any]) -> FeatureExtractor:
       ...

   def build_model(
       spec: Mapping[str, Any], features: Mapping[str, Any]
   ) -> nn.Module:
       ...
   ```

3. Add a `config.json` with `features`, `model`, `dataset`,
   `validation`, `synthesis`, and `optimization` sections. The
   `logmel_cnn/config.json` is a complete example.
4. Add thin `train.py` and `evaluate.py` modules that pass the builders
   to `common.cli.train_main` and `common.cli.evaluate_main`, mirroring
   `logmel_cnn/`.
5. Add a `README.md` describing setup, the smoke run, and the full run.

The training loop, evaluation, reports, and MLflow logging come from
`common/`, so a new architecture only supplies its builders and config.

The shared loop assumes the per-step pitch-sequence target: cross
entropy over the 90-class vocabulary, pitch/hold/note-off accuracy
groups, and `melody_from_classes` decoding. A different target encoding
would need its own hooks before it can reuse `common/training.py` and
`common/evaluation.py`.
