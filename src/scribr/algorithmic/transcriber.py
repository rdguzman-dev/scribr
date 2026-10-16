"""Monophonic transcription with spectral attacks and pYIN pitch tracking."""

import math
from dataclasses import dataclass
from itertools import pairwise

import librosa
import numpy as np

from ..representation import (
    DATASET_MAX_PITCH,
    DATASET_MIN_PITCH,
    STEPS_PER_QUARTER_NOTE,
    Melody,
    Note,
)


@dataclass(frozen=True, slots=True)
class AlgorithmicConfig:
    """Parameters to tune on training audio before scoring the test split.

    Attributes:
        sample_rate: Analysis sample rate in Hz.
        frame_length: Centered analysis window in samples.
        hop_length: Distance between analysis frames in samples.
        min_pitch: Lowest MIDI pitch searched by pYIN.
        max_pitch: Highest MIDI pitch searched by pYIN.
        silence_db: RMS gate relative to the loudest frame, in dB.
        min_note_seconds: Minimum duration of a retained pitch region.
        max_gap_seconds: Longest missing-pitch gap bridged within a note.
        onset_merge_seconds: Attacks this close to a pitch boundary merge.
        quantize: Snap note times to the dataset grid. Disable only for
            the unquantized ablation.
        attack_reject_ratio: Onset candidate kept when its RMS is at
            least this fraction of the earlier frame's RMS.
        attack_backtrack_seconds: Energy lookback window, in seconds,
            compared against the candidate's RMS.
        rms_window_hops: RMS analysis window, as a multiple of
            `hop_length`.
    """

    sample_rate: int = 22_050
    frame_length: int = 2048
    hop_length: int = 256
    min_pitch: int = DATASET_MIN_PITCH
    max_pitch: int = DATASET_MAX_PITCH
    silence_db: float = -40.0
    min_note_seconds: float = 0.05
    max_gap_seconds: float = 0.03
    onset_merge_seconds: float = 0.04
    quantize: bool = True
    attack_reject_ratio: float = 0.7
    attack_backtrack_seconds: float = 0.05
    rms_window_hops: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.quantize, bool):
            raise ValueError("quantize must be a boolean")

        for name in ("sample_rate", "frame_length", "hop_length"):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")

        if self.frame_length < 2 or self.hop_length > self.frame_length:
            raise ValueError(
                "require frame_length >= 2 and hop_length <= frame_length"
            )

        if not (
            isinstance(self.min_pitch, int)
            and isinstance(self.max_pitch, int)
            and 0 <= self.min_pitch < self.max_pitch <= 127
        ):
            raise ValueError(
                "pitch bounds must be integers in [0, 127], min < max"
            )

        fmin = float(librosa.midi_to_hz(self.min_pitch))
        if librosa.midi_to_hz(self.max_pitch) >= self.sample_rate / 2:
            raise ValueError(
                "max_pitch must be below the analysis Nyquist frequency"
            )

        if self.frame_length <= self.sample_rate / fmin + 1:
            raise ValueError("frame_length is too short for min_pitch")

        if not math.isfinite(self.silence_db) or self.silence_db >= 0:
            raise ValueError("silence_db must be finite and negative")

        if (
            not math.isfinite(self.attack_reject_ratio)
            or not 0.0 <= self.attack_reject_ratio <= 1.0
        ):
            raise ValueError("attack_reject_ratio must be in [0, 1]")

        if (
            not isinstance(self.rms_window_hops, int)
            or isinstance(self.rms_window_hops, bool)
            or self.rms_window_hops < 1
        ):
            raise ValueError("rms_window_hops must be a positive integer")

        for name in (
            "min_note_seconds",
            "max_gap_seconds",
            "onset_merge_seconds",
            "attack_backtrack_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

        if self.min_note_seconds == 0:
            raise ValueError("min_note_seconds must be positive")


@dataclass(frozen=True, slots=True)
class _PitchTrack:
    """Resampled waveform and pYIN output for one recording."""

    waveform: np.ndarray
    f0: np.ndarray
    voiced: np.ndarray
    duration: float


class AlgorithmicTranscriber:
    """Estimate notes without training or access to reference melodies.

    Args:
        config: Analysis and segmentation parameters. Times are snapped
            to the dataset grid unless `quantize` is disabled.
    """

    def __init__(self, config: AlgorithmicConfig | None = None) -> None:
        self.config = config if config is not None else AlgorithmicConfig()

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        tempo: float,
    ) -> Melody:
        """Transcribe mono audio into canonical quarter-note beats.

        Args:
            audio: Finite mono waveform in `[-1, 1]`.
            sample_rate: Input sample rate in Hz.
            tempo: Known synthesis tempo in BPM.

        Returns:
            Non-overlapping notes, or an empty `Melody` for silence.

        Raises:
            ValueError: If audio, sample rate, or tempo is invalid.
        """
        if not math.isfinite(tempo) or tempo <= 0:
            raise ValueError("tempo must be finite and positive")

        track = self._analyze(audio, sample_rate)

        if track is None:
            return Melody()

        return self._decode(track, tempo)

    def _prepare(
        self, audio: np.ndarray, sample_rate: int
    ) -> tuple[np.ndarray, float] | None:
        """Validate the input and return the waveform and duration.

        Returns `None` for silence. The resampled waveform is cheap to
        rebuild, so only the pYIN output is cached by the tuning search.
        """
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")

        audio = np.asarray(audio)
        if audio.ndim != 1 or not np.issubdtype(audio.dtype, np.number):
            raise ValueError("audio must be a numeric mono waveform")

        if np.iscomplexobj(audio) or not np.all(np.isfinite(audio)):
            raise ValueError("audio must contain finite real samples")

        if audio.size and np.max(np.abs(audio)) > 1:
            raise ValueError("audio samples must be in [-1, 1]")

        if not audio.size or not np.any(audio):
            return None

        config = self.config
        duration = len(audio) / sample_rate
        waveform = audio.astype(np.float32)

        if sample_rate != config.sample_rate:
            waveform = librosa.resample(
                waveform, orig_sr=sample_rate, target_sr=config.sample_rate
            )

        return waveform, duration

    def _track_pitch(
        self, waveform: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate f0 and voicing with pYIN.

        Only `frame_length`, `hop_length`, the pitch bounds, and the
        analysis sample rate change this result, so a parameter search
        can cache it.
        """
        config = self.config
        f0, voiced, _ = librosa.pyin(
            waveform,
            sr=config.sample_rate,
            fmin=float(librosa.midi_to_hz(config.min_pitch)),
            fmax=float(librosa.midi_to_hz(config.max_pitch)),
            frame_length=config.frame_length,
            hop_length=config.hop_length,
            center=True,
            fill_na=np.nan,
        )

        return f0, voiced

    def _analyze(
        self, audio: np.ndarray, sample_rate: int
    ) -> _PitchTrack | None:
        """Return the pYIN track, or `None` for silence."""
        prepared = self._prepare(audio, sample_rate)

        if prepared is None:
            return None

        waveform, duration = prepared
        f0, voiced = self._track_pitch(waveform)

        return _PitchTrack(waveform, f0, voiced, duration)

    def _decode(self, track: _PitchTrack, tempo: float) -> Melody:
        config = self.config
        waveform = track.waveform

        rms = librosa.feature.rms(
            y=waveform,
            frame_length=config.rms_window_hops * config.hop_length,
            hop_length=config.hop_length,
            center=True,
        )[0]

        threshold = rms.max() * 10 ** (config.silence_db / 20)
        valid = (
            track.voiced
            & np.isfinite(track.f0)
            & (track.f0 > 0)
            & (rms > threshold)
        )
        pitches = np.full(len(track.f0), -1, dtype=int)
        pitches[valid] = np.rint(librosa.hz_to_midi(track.f0[valid])).astype(
            int
        )
        pitches[
            (pitches < config.min_pitch) | (pitches > config.max_pitch)
        ] = -1

        attacks = librosa.onset.onset_detect(
            y=waveform,
            sr=config.sample_rate,
            hop_length=config.hop_length,
            backtrack=False,
            units="frames",
        )

        # Abrupt note releases also create spectral flux. Reject candidates
        # whose energy has fallen, then backtrack on a short energy window
        # so the longer pitch window does not pull attacks into silence.
        lookback = max(
            1,
            math.ceil(
                config.attack_backtrack_seconds
                * config.sample_rate
                / config.hop_length
            ),
        )
        attacks = attacks[
            (rms[attacks] > threshold)
            & (
                rms[attacks]
                >= config.attack_reject_ratio
                * rms[np.maximum(attacks - lookback, 0)]
            )
        ]

        # `onset_backtrack` raises on an empty list, which happens when a
        # note starts at the first sample and no spectral onset is found.
        if attacks.size:
            attacks = librosa.onset.onset_backtrack(attacks, rms)

        seconds_per_frame = config.hop_length / config.sample_rate
        melody = _segment_notes(
            pitches,
            attacks,
            seconds_per_frame,
            track.duration,
            tempo,
            config,
        )

        if config.quantize:
            return _quantize_notes(melody, track.duration * tempo / 60)

        return melody


def _runs(values: np.ndarray) -> list[tuple[int, int, int]]:
    boundaries = np.r_[0, np.flatnonzero(np.diff(values)) + 1, len(values)]

    return [
        (int(start), int(end), int(values[start]))
        for start, end in pairwise(boundaries)
    ]


def _segment_notes(
    pitches: np.ndarray,
    attacks: np.ndarray,
    seconds_per_frame: float,
    duration: float,
    tempo: float,
    config: AlgorithmicConfig,
) -> Melody:
    pitches = pitches.copy()
    gap_frames = int(config.max_gap_seconds / seconds_per_frame)
    min_frames = max(1, math.ceil(config.min_note_seconds / seconds_per_frame))

    # Only bridge gaps bounded by the same pitch; rests and pitch changes
    # must not become a continuation of the preceding note.
    for start, end, pitch in _runs(pitches):
        if (
            pitch == -1
            and end - start <= gap_frames
            and start > 0
            and end < len(pitches)
            and pitches[start - 1] == pitches[end]
        ):
            pitches[start:end] = pitches[start - 1]

    # A brief tracker excursion inside a stable note is not a new attack.
    for start, end, pitch in _runs(pitches):
        if (
            pitch != -1
            and end - start < min_frames
            and start > 0
            and end < len(pitches)
            and pitches[start - 1] == pitches[end]
        ):
            pitches[start:end] = pitches[start - 1]

    attack_times = np.unique(attacks) * seconds_per_frame
    notes: list[Note] = []

    for start, end, pitch in _runs(pitches):
        if pitch == -1 or end - start < min_frames:
            continue

        onset = start * seconds_per_frame
        offset = min(end * seconds_per_frame, duration)
        nearby = attack_times[
            np.abs(attack_times - onset) <= config.onset_merge_seconds
        ]

        if nearby.size:
            onset = max(0.0, float(nearby[np.argmin(np.abs(nearby - onset))]))

        if notes:
            onset = max(onset, notes[-1].offset * 60 / tempo)

        # Spectral attacks split repeated pitches even when pYIN never
        # becomes unvoiced between them.
        internal = attack_times[
            (attack_times > onset + config.onset_merge_seconds)
            & (attack_times < offset - config.min_note_seconds)
        ]
        boundaries = [onset]
        for attack in internal:
            if attack - boundaries[-1] >= config.min_note_seconds:
                boundaries.append(float(attack))
        boundaries.append(offset)

        for left, right in pairwise(boundaries):
            if right - left >= config.min_note_seconds:
                notes.append(
                    Note(pitch, left * tempo / 60, right * tempo / 60)
                )

    return Melody(tuple(notes))


def _quantize_notes(melody: Melody, duration_beats: float) -> Melody:
    steps = STEPS_PER_QUARTER_NOTE
    last_step = math.floor(duration_beats * steps)
    notes: list[Note] = []

    for note in melody:
        # Use one monotone rounding rule for both boundaries so adjacent
        # notes stay non-overlapping. Half-grid ties round forward.
        onset_step = math.floor(note.onset * steps + 0.5)
        offset_step = min(math.floor(note.offset * steps + 0.5), last_step)

        # A collapsed interval has no grid duration. Drop it rather than
        # inventing a longer note that could overlap a neighboring attack.
        if offset_step > onset_step:
            notes.append(
                Note(note.pitch, onset_step / steps, offset_step / steps)
            )

    return Melody(tuple(notes))
