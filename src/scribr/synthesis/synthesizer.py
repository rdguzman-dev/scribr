"""Synthesis of Scribr melodies to audio with FluidSynth.

This module converts `Melody` objects directly to mono `float32`
waveforms. No MIDI or WAV files are written to disk.

Requirements:

- FluidSynth shared library
- `pyfluidsynth` Python package
- SoundFont (`.sf2` or `.sf3`)

The SoundFont is never committed to the repository. Provide one
explicitly with `soundfont_path` or set the `SCRIBR_SOUNDFONT`
environment variable.
"""

import os
import platform
from pathlib import Path
from typing import Any

import numpy as np

from ..representation.melody import Melody

DEFAULT_SAMPLE_RATE = 22_050
DEFAULT_TEMPO_BPM = 120.0
DEFAULT_PROGRAM = 0
DEFAULT_VELOCITY = 80
DEFAULT_GAIN = 1.0
DEFAULT_RELEASE_TAIL_SECONDS = 0.5
DEFAULT_MIDI_CHANNEL = 0

# Environment variable used to locate the FluidSynth SoundFont.
SOUNDFONT_ENV_VAR = "SCRIBR_SOUNDFONT"

# Scale factor for converting signed 16-bit PCM to normalized audio.
_INT16_SCALE = 32768.0


class Synthesizer:
    """Synthesizer for rendering `Melody` objects to mono `float32`
    waveforms.

    Args:
        soundfont_path: Path to a General MIDI SoundFont. If `None`,
            falls back to `SCRIBR_SOUNDFONT`.
        sample_rate: Output sample rate in Hz.
        tempo: Tempo in BPM used to convert beat-based note times to
            seconds.
        program: Zero-based General MIDI program number (`0` = Acoustic
            Grand Piano).
        velocity: Note-on velocity.
        gain: FluidSynth synthesizer gain.
        release_tail_seconds: Audio rendered after the final note-off so the
            instrument release is not cut off.
        midi_channel: MIDI channel used for rendering.
    """

    def __init__(
        self,
        soundfont_path: str | Path | None = None,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        tempo: float = DEFAULT_TEMPO_BPM,
        program: int = DEFAULT_PROGRAM,
        velocity: int = DEFAULT_VELOCITY,
        gain: float = DEFAULT_GAIN,
        release_tail_seconds: float = DEFAULT_RELEASE_TAIL_SECONDS,
        midi_channel: int = DEFAULT_MIDI_CHANNEL,
    ) -> None:
        if soundfont_path is None:
            soundfont_path = os.environ.get(SOUNDFONT_ENV_VAR)

        if soundfont_path is None:
            raise ValueError(
                "No SoundFont configured. Pass soundfont_path=... or set the "
                f"{SOUNDFONT_ENV_VAR} environment variable to a .sf2/.sf3 file."
            )

        path = Path(soundfont_path).expanduser()

        if not path.is_file():
            raise FileNotFoundError(f"SoundFont not found: {path}")

        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        if tempo <= 0:
            raise ValueError("tempo must be positive")

        if not 0 <= program <= 127:
            raise ValueError("program must be in [0, 127]")

        if not 1 <= velocity <= 127:
            raise ValueError("velocity must be in [1, 127]")

        if gain <= 0:
            raise ValueError("gain must be positive")

        if release_tail_seconds < 0:
            raise ValueError("release_tail_seconds must be non-negative")

        if not 0 <= midi_channel <= 15:
            raise ValueError("midi_channel must be in [0, 15]")

        self.soundfont_path = path
        self.sample_rate = sample_rate
        self.tempo = tempo
        self.program = program
        self.velocity = velocity
        self.gain = gain
        self.release_tail_seconds = release_tail_seconds
        self.midi_channel = midi_channel

        self._fluidsynth = _load_pyfluidsynth()

    def synthesize(self, melody: Melody) -> np.ndarray:
        """Render a melody to a mono `float32` waveform with normalized
        samples.

        Note onset and offset times are interpreted as quarter-note
        beats and converted to seconds using `self.tempo`.

        A fresh FluidSynth instance is used per call so no state
        (release tails, reverb, controller values) leaks between
        melodies.
        """
        events = _note_events(melody, self.tempo)

        if not events:
            return np.zeros(0, dtype=np.float32)

        synth = self._fluidsynth.Synth(
            gain=float(self.gain),
            samplerate=float(self.sample_rate),
        )

        try:
            soundfont_id = synth.sfload(str(self.soundfont_path))

            if soundfont_id == -1:
                raise RuntimeError(
                    f"FluidSynth failed to load SoundFont {self.soundfont_path}"
                )

            synth.program_select(
                chan=self.midi_channel,
                sfid=soundfont_id,
                bank=0,
                preset=self.program,
            )

            chunks: list[np.ndarray] = []
            rendered_frames = 0

            for time_seconds, kind, pitch in events:
                target_frame = int(round(time_seconds * self.sample_rate))
                frames = max(0, target_frame - rendered_frames)

                if frames > 0:
                    chunks.append(
                        np.asarray(synth.get_samples(frames)).reshape(-1)
                    )
                    rendered_frames = target_frame

                if kind == "on":
                    synth.noteon(self.midi_channel, pitch, self.velocity)

                else:
                    synth.noteoff(self.midi_channel, pitch)

            tail_frames = int(
                round(self.release_tail_seconds * self.sample_rate)
            )

            if tail_frames > 0:
                chunks.append(
                    np.asarray(synth.get_samples(tail_frames)).reshape(-1)
                )

        finally:
            synth.delete()

        if not chunks:
            return np.zeros(0, dtype=np.float32)

        # `get_smaples()` returns interleaved stereo signed-16-bit samples.
        samples = np.concatenate(chunks).astype(np.float32) / _INT16_SCALE

        if samples.size % 2 != 0:
            raise RuntimeError(
                "FluidSynth returned an incomplete stereo frame"
            )

        # Average stereo channels to mono.
        samples = samples.reshape(-1, 2).mean(axis=1)

        return np.ascontiguousarray(samples, dtype=np.float32)


def _load_pyfluidsynth() -> Any:
    """Import `pyfluidsynth`, configuring Homebrew library discovery on
    macOS."""
    if (
        platform.system() == "Darwin"
        and "HOMEBREW_PREFIX" not in os.environ
        and Path("/opt/homebrew/lib/libfluidsynth.dylib").exists()
    ):
        os.environ["HOMEBREW_PREFIX"] = "/opt/homebrew"

    try:
        import fluidsynth

    except ImportError as exc:  # pragma: no cover - depends on system
        raise ImportError(
            "FluidSynth is required for audio synthesis but could not be "
            "loaded. Install the system library (e.g. `brew install "
            "fluid-synth` or `apt install libfluidsynth3`) and the Python "
            "binding (`uv sync` installs pyfluidsynth)."
        ) from exc

    return fluidsynth


def _note_events(
    melody: Melody,
    tempo: float,
) -> list[tuple[float, str, int]]:
    """Build a note-on/note-off timeline in seconds.

    Note-off events sort before note-on events at the same instant so
    that repeated/adjacent notes with the same pitch are rendered as
    separate attacks.
    """
    seconds_per_beat = 60.0 / tempo

    # (seconds, sort_priority, kind, pitch)
    events: list[tuple[float, int, str, int]] = []

    for note in melody.notes:
        events.append((note.onset * seconds_per_beat, 1, "on", note.pitch))
        events.append((note.offset * seconds_per_beat, 0, "off", note.pitch))

    events.sort(key=lambda event: (event[0], event[1]))

    return [(seconds, kind, pitch) for seconds, _, kind, pitch in events]
