"""The inference contract shared by all transcription approaches.

`Transcriber` is the seam between an approach and everything that
consumes it (evaluation, demos, future tooling). The contract fixes the
audio format and the output type so that the algorithmic, deep learning,
and LLM approaches stay interchangeable.
"""

from typing import Protocol

import numpy as np

from .representation import Melody


class Transcriber(Protocol):
    """Audio in, canonical `Melody` out.

    An approach can do anything internally (spectral analysis, a neural
    network, an LLM prompt) as long as `transcribe` accepts raw synthesis
    output and returns notes in the canonical units.
    """

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        tempo: float,
    ) -> Melody:
        """Transcribe one monophonic melody from audio.

        Args:
            audio: Mono `float32` waveform in `[-1, 1]`, exactly as
                produced by `Synthesizer.synthesize`.
            sample_rate: Sample rate of `audio` in Hz.
            tempo: Tempo the audio was rendered at, in BPM. `Melody`
                times are quarter-note beats, so an approach needs the
                tempo to convert its own time estimates to beats.

        Returns:
            Estimated melody. Times are quarter-note beats measured from
            the first sample of `audio`, matching the reference melodies
            produced by `scribr.data`. An empty `Melody` is the valid
            estimate for silence.
        """
        ...
