"""`Transcriber` backed by a trained `PitchSequenceCNN`.

Inference is three steps: extract natural-length log-mel features from
the audio, stretch them onto the tempo grid the model was trained on,
and decode the per-step class predictions with the dataset codec. The
model only ever sees the training feature layout.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor, nn

from ..representation import Melody
from ..synthesis.synthesizer import DEFAULT_TEMPO_BPM
from .features import LogMelSpectrogram, fix_length
from .targets import melody_from_classes


class DeepLearningTranscriber:
    """Transcribe mono audio into a `Melody` with a pitch-sequence model.

    Args:
        model: Trained `PitchSequenceCNN`.
        features: Feature extractor the model was trained with.
        training_tempo: Tempo in BPM used during training. Audio
            rendered at another tempo is resampled onto that grid.
        device: Torch device to run inference on.
    """

    def __init__(
        self,
        model: nn.Module,
        features: LogMelSpectrogram,
        *,
        training_tempo: float = DEFAULT_TEMPO_BPM,
        device: str | torch.device = "cpu",
    ) -> None:
        if training_tempo <= 0:
            raise ValueError("training_tempo must be positive")

        self.features = features
        self.training_tempo = float(training_tempo)
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        tempo: float,
    ) -> Melody:
        """Transcribe one melody from its synthesized audio.

        Args:
            audio: Mono `float32` waveform in `[-1, 1]`.
            sample_rate: Sample rate of `audio` in Hz.
            tempo: Tempo the audio was rendered at, in BPM.

        Returns:
            Estimated melody on the dataset's step grid.
        """
        if tempo <= 0:
            raise ValueError("tempo must be positive")

        waveform = torch.as_tensor(audio, dtype=torch.float32)

        if waveform.numel() == 0:
            return Melody(())

        if sample_rate != self.features.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                self.features.sample_rate,
            )

        features = self.features.spectrogram(waveform)

        if tempo != self.training_tempo:
            features = _stretch_tempo(features, tempo / self.training_tempo)

        if self.features.num_frames is not None:
            features = fix_length(features, self.features.num_frames)

        with torch.no_grad():
            logits = self.model(features.unsqueeze(0).to(self.device))

        return melody_from_classes(logits.argmax(dim=-1)[0].tolist())


def _stretch_tempo(features: Tensor, factor: float) -> Tensor:
    """Resample the time axis so `tempo` maps onto the training grid."""
    length = max(1, round(features.shape[-1] * factor))

    return F.interpolate(
        features.unsqueeze(0),
        size=length,
        mode="linear",
        align_corners=False,
    ).squeeze(0)
