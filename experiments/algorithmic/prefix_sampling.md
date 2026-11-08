# Prefix sampling

Prefix sampling takes the first `sample_size` records of the training
split. The algorithmic tuning experiment used this approach before adding
reservoir sampling: with `sample_size: 2000`, it selected dataset indices
0-1999 and never looked at later records. The held-out validation config
still uses prefix selection; this document describes the tuning sample.

Prefix selection is deterministic without a seed, and it keeps nested
tuning and validation prefixes reproducible. Its weakness is coverage:
the sample is entirely drawn from the beginning of the split.

## Resulting sample

The 2,000-example tuning prefix is composed as follows:

| Property | Value |
| --- | --- |
| Examples | 2,000 |
| Dataset indices | 0-1999 |
| Piano | 500 |
| Acoustic guitar | 500 |
| Electric guitar | 500 |
| Violin | 500 |
| Index quartile 0-2,531,668 | 2,000 |
| Index quartile 2,531,669-5,063,337 | 0 |
| Index quartile 5,063,338-7,595,006 | 0 |
| Index quartile 7,595,007-10,126,675 | 0 |

The round-robin instrument assignment makes the sample exactly balanced.
It covers only the first 0.02% of the 10,126,676 training records.

Attribute means, from the dataset attributes and decoded note counts:

| Attribute | Prefix sample | Full train split | Difference |
| --- | --- | --- | --- |
| Note count | 25.1615 | 24.8460 | +1.27% |
| Note density | 0.3931 | 0.3882 | +1.27% |
| Pitch range | 0.1591 | 0.1630 | -2.42% |
| Contour | 1.0443 | 1.1302 | -7.60% |
| Note change ratio | 0.2745 | 0.2702 | +1.59% |
| Dynamic range | 0.1624 | 0.1742 | -6.77% |

The prefix is close to the full split in note count, density, and note
change ratio. It has a smaller contour and dynamic range, and a slightly
smaller pitch range.

## Reproduce

The committed `tuning_config.json` now uses reservoir sampling. To select
the prefix instead, use this dataset block:

```json
{
  "split": "train",
  "sample_size": 2000,
  "sampling": "prefix"
}
```

With `--config PATH` pointing at that spec and a fresh `--artifacts-dir`,
run the normal tuning command. The manifest records dataset indices 0-1999,
and the stage-1 screen is the first 500 of those records.
