"""Optional WAV export for synthesized audio.

WAV is an *export* format only. Dataset loading, training, evaluation,
and synthesis never require it. This module exists so the output of
`Synthesizer.synthesize` can be saved and opened in an audio editor or
shared with other tools.

Files are mono, 16-bit PCM. Floating-point samples outside `[-1, 1]` are
clipped before conversion.
"""

import wave
from pathlib import Path

import numpy as np

# Scale factor for converting normalized float audio to signed 16-bit PCM.
_FLOAT_TO_INT16 = 32767.0


def write_wav(
    samples: np.ndarray,
    sample_rate: int,
    path: str | Path,
) -> Path:
    """Write a mono normalized waveform to `path` as a 16-bit PCM WAV.

    Args:
        samples: 1-D floating-point array of normalized audio samples,
            such as the output of `Synthesizer.synthesize`.
        sample_rate: Sample rate in Hz.
        path: Destination path for the WAV file.

    Returns:
        The path to the written WAV file.
    """
    waveform = np.asarray(samples)

    if waveform.ndim != 1:
        raise ValueError("samples must be a 1-D mono waveform")

    if not np.issubdtype(waveform.dtype, np.floating):
        raise ValueError("samples must be floating-point values in [-1, 1]")

    if not np.all(np.isfinite(waveform)):
        raise ValueError("samples must contain only finite values")

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    output = Path(path)

    pcm = np.clip(waveform, -1.0, 1.0)
    pcm = np.round(pcm * _FLOAT_TO_INT16).astype(np.dtype("<i2"))

    with wave.open(str(output), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())

    return output
