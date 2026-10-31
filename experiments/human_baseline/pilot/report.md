# Human baseline report

## Metadata

| Field | Value |
| --- | --- |
| Git commit | 6185489c7f4d3516e3a33ebcb8b41e8733e95b80 |
| Python | 3.13.11 |
| scribr | 0.1.0 |
| mir_eval | 0.8.2 |
| Tempo (BPM) | 120.0 |
| Onset tolerance (s) | 0.05 |
| Pitch tolerance (cents) | 50.0 |
| Offset ratio | 0.2 |
| Offset min tolerance (s) | 0.05 |
| SoundFont SHA-256 | 9575028c7a1f589f5770fccc8cff2734566af40cd26ed836944e9a5152688cfe |
| SoundFont path | n/a |
| Examples evaluated | 20 |

## Overall metrics

| Metric | Score |
| --- | --- |
| F-measure_no_offset | 0.9478 |
| Precision_no_offset | 0.9492 |
| Recall_no_offset | 0.9465 |
| Average_Overlap_Ratio_no_offset | 0.9828 |
| Onset_F-measure | 0.9875 |
| Onset_Precision | 0.9889 |
| Onset_Recall | 0.9862 |
| F-measure | 0.9152 |
| Precision | 0.9166 |
| Recall | 0.9139 |
| Average_Overlap_Ratio | 0.9987 |
| Offset_F-measure | 0.9603 |
| Offset_Precision | 0.9619 |
| Offset_Recall | 0.9588 |

Offset-aware metrics are noisier because synthesized instrument release tails should not be transcribed as notes.

## Per-instrument metrics

Per-instrument means are based on 5 examples each.

| Instrument | N | F-measure_no_offset | Precision_no_offset | Recall_no_offset | Average_Overlap_Ratio_no_offset | Onset_F-measure | Onset_Precision | Onset_Recall | F-measure | Precision | Recall | Average_Overlap_Ratio | Offset_F-measure | Offset_Precision | Offset_Recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| piano | 5 | 0.9852 | 0.9852 | 0.9852 | 0.9986 | 1.0000 | 1.0000 | 1.0000 | 0.9852 | 0.9852 | 0.9852 | 0.9986 | 1.0000 | 1.0000 | 1.0000 |
| acoustic_guitar | 5 | 0.9724 | 0.9724 | 0.9724 | 0.9327 | 1.0000 | 1.0000 | 1.0000 | 0.8422 | 0.8422 | 0.8422 | 0.9960 | 0.8698 | 0.8698 | 0.8698 |
| electric_guitar | 5 | 0.9359 | 0.9359 | 0.9359 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9359 | 0.9359 | 0.9359 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| violin | 5 | 0.8977 | 0.9033 | 0.8926 | 1.0000 | 0.9500 | 0.9556 | 0.9448 | 0.8977 | 0.9033 | 0.8926 | 1.0000 | 0.9714 | 0.9778 | 0.9655 |

## Examples

| Example | Instrument | F-measure_no_offset | Precision_no_offset | Recall_no_offset | Average_Overlap_Ratio_no_offset | Onset_F-measure | Onset_Precision | Onset_Recall | F-measure | Precision | Recall | Average_Overlap_Ratio | Offset_F-measure | Offset_Precision | Offset_Recall | Off-grid onsets | Off-grid offsets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 000-piano | piano | 1.0000 | 1.0000 | 1.0000 | 0.9929 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9929 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| 001-acoustic_guitar | acoustic_guitar | 1.0000 | 1.0000 | 1.0000 | 0.7381 | 1.0000 | 1.0000 | 1.0000 | 0.4762 | 0.4762 | 0.4762 | 1.0000 | 0.4762 | 0.4762 | 0.4762 | 0 | 0 |
| 002-electric_guitar | electric_guitar | 0.8889 | 0.8889 | 0.8889 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8889 | 0.8889 | 0.8889 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| 003-violin | violin | 0.9844 | 0.9844 | 0.9844 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9844 | 0.9844 | 0.9844 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| 004-piano | piano | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| 005-acoustic_guitar | acoustic_guitar | 0.9333 | 0.9333 | 0.9333 | 0.9524 | 1.0000 | 1.0000 | 1.0000 | 0.8667 | 0.8667 | 0.8667 | 0.9872 | 0.9333 | 0.9333 | 0.9333 | 0 | 0 |
| 006-electric_guitar | electric_guitar | 0.9286 | 0.9286 | 0.9286 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9286 | 0.9286 | 0.9286 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| 007-violin | violin | 0.8636 | 0.8636 | 0.8636 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8636 | 0.8636 | 0.8636 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| 008-piano | piano | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| 009-acoustic_guitar | acoustic_guitar | 1.0000 | 1.0000 | 1.0000 | 0.9930 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9930 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| 010-electric_guitar | electric_guitar | 0.9286 | 0.9286 | 0.9286 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9286 | 0.9286 | 0.9286 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| 011-violin | violin | 0.7500 | 0.7778 | 0.7241 | 1.0000 | 0.7500 | 0.7778 | 0.7241 | 0.7500 | 0.7778 | 0.7241 | 1.0000 | 0.8571 | 0.8889 | 0.8276 | 0 | 0 |
| 012-piano | piano | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| 013-acoustic_guitar | acoustic_guitar | 0.9286 | 0.9286 | 0.9286 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9286 | 0.9286 | 0.9286 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| 014-electric_guitar | electric_guitar | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| 015-violin | violin | 0.8906 | 0.8906 | 0.8906 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8906 | 0.8906 | 0.8906 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| 016-piano | piano | 0.9259 | 0.9259 | 0.9259 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9259 | 0.9259 | 0.9259 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| 017-acoustic_guitar | acoustic_guitar | 1.0000 | 1.0000 | 1.0000 | 0.9798 | 1.0000 | 1.0000 | 1.0000 | 0.9394 | 0.9394 | 0.9394 | 1.0000 | 0.9394 | 0.9394 | 0.9394 | 0 | 0 |
| 018-electric_guitar | electric_guitar | 0.9333 | 0.9333 | 0.9333 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9333 | 0.9333 | 0.9333 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| 019-violin | violin | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
