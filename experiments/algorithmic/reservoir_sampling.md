# Reservoir sampling

Reservoir sampling draws a uniform sample from the whole training split in
one streaming pass, keeping only `sample_size` items in memory. The
algorithmic tuning experiment uses this approach so that the 2,000-example
sample is not limited to the first records of the 10.1 million-record train
split. The held-out validation split still uses prefix selection; this
document describes the tuning sample.

`tuning_config.json` sets `sample_size` to 2,000, `seed` to 42, and
`sampling` to `"reservoir"`. The seed makes the sample reproducible. The
selected set is returned in uniformly random order, so the stage-1 screen
(a prefix of the materialized sample) is also a uniform sample rather than
an early slice of the dataset.

Reservoir sampling is implemented by
`scribr.data.sampling.reservoir_sample` using Algorithm R.

## Resulting sample

With seed 42, the 2,000 selected dataset indices range from 8,239 to
10,120,518. The ten smallest selected indices are:

```text
8239, 12205, 19220, 27577, 28587, 29494, 29518, 29672, 37040, 38627
```

The largest gap between consecutive selected indices is 39,486; the median
gap is 3,535. The sample is composed as follows:

| Property | Value |
| --- | --- |
| Examples | 2,000 |
| Dataset indices | 8,239-10,120,518 |
| Piano | 500 |
| Acoustic guitar | 500 |
| Electric guitar | 500 |
| Violin | 500 |

The instrument counts stay exactly balanced because materialization assigns
instruments round-robin over the selected order.

The selected indices are spread across the full split:

| Index range | Selected |
| --- | --- |
| 0-2,025,334 | 400 |
| 2,025,335-4,050,669 | 396 |
| 4,050,670-6,076,005 | 443 |
| 6,076,006-8,101,340 | 394 |
| 8,101,341-10,126,675 | 367 |

Attribute means, from the dataset attributes and decoded note counts:

| Attribute | Reservoir sample | Full train split | Difference |
| --- | --- | --- | --- |
| Note count | 24.9535 | 24.8460 | +0.43% |
| Note density | 0.3899 | 0.3882 | +0.43% |
| Pitch range | 0.1619 | 0.1630 | -0.70% |
| Contour | 1.1137 | 1.1302 | -1.45% |
| Note change ratio | 0.2669 | 0.2702 | -1.22% |
| Dynamic range | 0.1742 | 0.1742 | +0.02% |

Compared with the prefix, the reservoir sample is closer to the full train
split on all six attributes. Its largest difference is 1.45%, compared with
up to 7.60% for the prefix.

## Reproduce

The committed `tuning_config.json` already uses this sample. To print the
selected indices without materializing audio:

```bash
uv run python - <<'PY'
from scribr.data import MelodyDataset
from scribr.data.sampling import reservoir_sample

dataset = MelodyDataset(split="train")
selected = reservoir_sample(enumerate(dataset), 2000, seed=42)
print([index for index, _ in selected])
PY
```

The full split is scanned once, but only the 2,000 selected examples are
kept in memory.
