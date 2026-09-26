# Test fixtures

These fixtures are **byte-identical prefixes of real shards** from the
published *4 Bars Monophonic Melodies Dataset (Pitch Sequence)*
(<https://zenodo.org/records/13369389>, CC-BY-4.0), not synthetic data. They
are committed so the parser and dataset tests run without downloading
713 MB of data.

| File | Origin | Contents |
| --- | --- | --- |
| `melodies_head.tfrecord` | `test_pitchseq-00000-of-00008.tfrecord`, first 4358 bytes | 8 complete records |
| `data/train/train_pitchseq-00000-of-00002.tfrecord` | copy of `melodies_head.tfrecord` | 8 records |
| `data/train/train_pitchseq-00001-of-00002.tfrecord` | `test_pitchseq-00001-of-00008.tfrecord`, first 2148 bytes | 4 records |
| `data/validation/validation_pitchseq-00000-of-00001.tfrecord` | copy of `melodies_head.tfrecord` | 8 records |
| `data/test/test_pitchseq-00000-of-00001.tfrecord` | `test_pitchseq-00001-of-00008.tfrecord`, first 2148 bytes | 4 records |

Record boundaries are 16 bytes of TFRecord framing plus the payload, so the
prefix length is `sum(payload lengths) + 16 * n_records`. Because the files
are byte prefixes, the CRC32C checksums and payloads are exactly as
published. The `data/` directory mimics the on-disk layout expected by
`scribr.data.MelodyDataset`; the shard filenames are renamed only so the
miniature "splits" are self-describing.

Expected decoded values used by the tests were produced independently from
the same published bytes and are hard-coded in `tests/test_example.py` and
`tests/test_dataset.py`.
