"""Basic Pitch's CQT frontend and pretrained convolutional trunk.

The port keeps the pretrained CQT frontend and the contour trunk from
Spotify's Basic Pitch, and drops its polyphonic output heads. The
per-step 90-class head from `PitchSequenceCNN` is attached to the
trunk's 32-channel feature instead, so monophony is structural rather
than a post-processing assumption.

Everything here is a faithful reimplementation of the ICASSP 2022
model. The ONNX export folds each BatchNorm into the preceding
convolution, so the trunk is a plain conv/activation chain and its
weights already contain the BatchNorm affine. The frontend's scalar
BatchNorm is kept as two frozen buffers.

The frontend returns Basic Pitch's harmonic CQT as
`(batch, 8, 264, frames)`. The project's feature maps put time on the
last axis, so the model transposes the frequency and time axes once
before the trunk.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..representation import STEPS_PER_MELODY
from .features import fix_length
from .model import DEFAULT_DROPOUT
from .targets import NUM_CLASSES

SAMPLE_RATE = 22_050
HOP_LENGTH = 256
F_MIN = 27.5
N_BINS = 309
BINS_PER_OCTAVE = 36
N_HARMONICS = 8
N_STACK_BINS = 264
LOWPASS_LENGTH = 256

# Harmonic ratios 0.5, 1, 2, ..., 7 map to bin offsets. The offsets are
# positive shifts toward lower bin indices, so a positive shift zero-fills
# the high-frequency end of the stack.
HARMONIC_SHIFTS = tuple(
    int(round(BINS_PER_OCTAVE * np.log2(harmonic)))
    for harmonic in (0.5, 1, 2, 3, 4, 5, 6, 7)
)

_WEIGHTS_RESOURCE = "weights/basic_pitch_trunk.npz"


@lru_cache(maxsize=1)
def _load_weights() -> dict[str, np.ndarray]:
    resource = files("scribr.deep_learning").joinpath(_WEIGHTS_RESOURCE)

    try:
        with resource.open("rb") as handle:
            with np.load(handle) as archive:
                return {
                    name: np.array(archive[name])
                    for name in archive.files
                }

    except FileNotFoundError as error:
        raise FileNotFoundError(
            "Basic Pitch trunk weights are missing; run "
            "`python -m scripts.convert_basic_pitch --onnx <nmp.onnx>`"
        ) from error


def _create_lowpass_filter(
    band_center: float = 0.5,
    kernel_length: int = LOWPASS_LENGTH,
    transition_bandwidth: float = 0.03,
) -> np.ndarray:
    lower = band_center / (1 + transition_bandwidth)
    upper = band_center * (1 + transition_bandwidth)

    return scipy.signal.firwin2(
        kernel_length,
        [0.0, lower, upper, 1.0],
        [1.0, 1.0, 0.0, 0.0],
    )


def _create_cqt_kernels(
    quality: float,
    sample_rate: int,
    f_min: float,
    n_bins: int,
    bins_per_octave: int,
) -> tuple[np.ndarray, int]:
    """Build one octave of complex CQT kernels, as Basic Pitch does."""
    fft_len = 2 ** int(
        np.ceil(np.log2(np.ceil(quality * sample_rate / f_min)))
    )
    freqs = f_min * 2.0 ** (np.arange(n_bins) / float(bins_per_octave))
    kernels = np.zeros(
        (int(n_bins), int(fft_len)),
        dtype=np.complex64,
    )

    for index in range(int(n_bins)):
        freq = freqs[index]
        length = np.ceil(quality * sample_rate / freq)
        start = int(np.ceil(fft_len / 2.0 - length / 2.0)) - int(
            length % 2
        )
        signal = (
            scipy.signal.get_window("hann", int(length), fftbins=True)
            * np.exp(
                np.r_[-length // 2 : length // 2]
                * 1j
                * 2
                * np.pi
                * freq
                / sample_rate
            )
            / length
        )
        kernels[index, start : start + int(length)] = signal / np.linalg.norm(
            signal,
            1,
        )

    return kernels, int(fft_len)


def _cqt_geometry(
    sample_rate: int,
) -> tuple[float, float, int, int]:
    """Return the kernel frequency range and octave/length counts."""
    quality = 1.0 / (2 ** (1 / BINS_PER_OCTAVE) - 1)
    n_octaves = int(np.ceil(N_BINS / BINS_PER_OCTAVE))
    f_min_top = F_MIN * 2 ** (n_octaves - 1)
    remainder = N_BINS % BINS_PER_OCTAVE
    f_max_top = (
        f_min_top * 2 ** ((remainder - 1) / BINS_PER_OCTAVE)
        if remainder
        else f_min_top * 2 ** ((BINS_PER_OCTAVE - 1) / BINS_PER_OCTAVE)
    )
    f_min_top = f_max_top / 2 ** (1 - 1 / BINS_PER_OCTAVE)

    # Basic Pitch's early-downsampling branch evaluates to a factor of
    # one for these parameters, so it is not ported. If the constants ever
    # change, this raises instead of silently producing a wrong CQT.
    cutoff = f_max_top * (1 + 0.5 * 1.5 / quality)
    count1 = max(
        0,
        int(np.ceil(np.log2(0.85 * (sample_rate // 2) / cutoff)) - 1) - 1,
    )
    count2 = max(
        0,
        int(np.ceil(np.log2(HOP_LENGTH))) - n_octaves + 1,
    )
    downsample_factor = 2 ** min(count1, count2)

    if downsample_factor != 1:
        raise RuntimeError(
            "Basic Pitch frontend expected no early downsampling, got "
            f"factor {downsample_factor}"
        )

    return quality, f_min_top, n_octaves, downsample_factor


def _as_waveform_batch(
    audio: np.ndarray | Tensor,
) -> tuple[Tensor, bool]:
    """Return `(batch, samples)` audio and whether the input was one row."""
    waveform = torch.as_tensor(audio, dtype=torch.float32)

    if waveform.ndim == 1:
        return waveform.unsqueeze(0), True

    if waveform.ndim == 2:
        return waveform, False

    raise ValueError(
        "audio must be one- or two-dimensional, got shape "
        f"{tuple(waveform.shape)}"
    )


class BasicPitchFrontend(nn.Module):
    """Basic Pitch's harmonic CQT feature extractor.

    The kernels are fixed by the pretrained model. `sample_rate` must be
    22050 Hz and there are no tunable CQT parameters.

    Args:
        sample_rate: Input sample rate in Hz. Only 22050 is supported.
        num_frames: Fixed length of the time axis, or `None` to keep the
            natural length. Basic Pitch is fully convolutional in time,
            so the experiment uses `None`.
    """

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        num_frames: int | None = None,
    ) -> None:
        super().__init__()

        if int(sample_rate) != SAMPLE_RATE:
            raise ValueError(
                "Basic Pitch frontend only supports "
                f"{SAMPLE_RATE} Hz, got {sample_rate}"
            )

        if num_frames is not None and num_frames < 1:
            raise ValueError("num_frames must be at least 1")

        self.sample_rate = int(sample_rate)
        self.num_frames = None if num_frames is None else int(num_frames)

        quality, f_min_top, n_octaves, _ = _cqt_geometry(self.sample_rate)
        basis, fft_len = _create_cqt_kernels(
            quality,
            self.sample_rate,
            f_min_top,
            min(BINS_PER_OCTAVE, N_BINS),
            BINS_PER_OCTAVE,
        )
        lengths = np.ceil(
            quality
            * self.sample_rate
            / (F_MIN * 2.0 ** (np.arange(N_BINS) / BINS_PER_OCTAVE))
        )

        self.n_octaves = n_octaves
        self.fft_len = fft_len
        self.lowpass_padding = (LOWPASS_LENGTH - 1) // 2
        self.minimum_samples = (fft_len // 2 + 1) * 2 ** (n_octaves - 1)

        weights = _load_weights()

        self.register_buffer(
            "basis_real",
            torch.from_numpy(
                np.ascontiguousarray(basis.real, dtype=np.float32)
            )[:, None, :],
        )
        self.register_buffer(
            "basis_imag",
            torch.from_numpy(
                np.ascontiguousarray(basis.imag, dtype=np.float32)
            )[:, None, :],
        )
        self.register_buffer(
            "lowpass",
            torch.from_numpy(
                _create_lowpass_filter(0.5, LOWPASS_LENGTH, 0.001).astype(
                    np.float32
                )
            )[None, None, :],
        )
        self.register_buffer(
            "sqrt_lengths",
            torch.from_numpy(
                np.sqrt(lengths).astype(np.float32)
            )[None, :, None],
        )
        self.register_buffer(
            "scale",
            torch.tensor(weights["frontend.scale"], dtype=torch.float32),
        )
        self.register_buffer(
            "shift",
            torch.tensor(weights["frontend.shift"], dtype=torch.float32),
        )

    def spectrogram(self, audio: np.ndarray | Tensor) -> Tensor:
        """Extract the harmonic CQT at its natural length.

        Args:
            audio: Mono `float32` waveform, or a `(batch, samples)` batch.

        Returns:
            `(8, 264, frames)` for one waveform, or
            `(batch, 8, 264, frames)` for a batch.
        """
        waveform, single = _as_waveform_batch(audio)

        if waveform.shape[-1] == 0:
            empty = torch.zeros(
                waveform.shape[0],
                N_HARMONICS,
                N_STACK_BINS,
                0,
                dtype=torch.float32,
            )
            return empty.squeeze(0) if single else empty

        if waveform.shape[-1] < self.minimum_samples:
            raise ValueError(
                "audio is too short for the nine-octave CQT; expected at "
                f"least {self.minimum_samples} samples, got "
                f"{waveform.shape[-1]}"
            )

        magnitude = self._cqt_magnitude(waveform)
        normalized = self._normalized_log(magnitude)
        stacked = self._harmonic_stack(
            normalized * self.scale + self.shift
        )

        return stacked.squeeze(0) if single else stacked

    def forward(self, audio: np.ndarray | Tensor) -> Tensor:
        """Extract features cropped or padded to `num_frames`."""
        features = self.spectrogram(audio)

        if self.num_frames is None:
            return features

        return fix_length(features, self.num_frames)

    def _cqt_magnitude(self, waveform: Tensor) -> Tensor:
        """Return the `(batch, 309, frames)` CQT magnitude."""
        signal = waveform.unsqueeze(1)
        padding = self.fft_len // 2

        def correlate(window: Tensor, hop: int) -> Tensor:
            padded = F.pad(window, (padding, padding), mode="reflect")
            real = F.conv1d(padded, self.basis_real, stride=hop)
            imaginary = F.conv1d(padded, self.basis_imag, stride=hop)

            return torch.complex(real, imaginary)

        cqt = correlate(signal, HOP_LENGTH)
        downsampled = signal
        hop = HOP_LENGTH

        for _ in range(self.n_octaves - 1):
            hop //= 2
            # The lowpass filter is symmetric, so convolution and
            # cross-correlation give the same result here.
            downsampled = F.conv1d(
                F.pad(
                    downsampled,
                    (self.lowpass_padding, self.lowpass_padding),
                ),
                self.lowpass,
                stride=2,
            )
            cqt = torch.cat([correlate(downsampled, hop), cqt], dim=1)

        cqt = cqt[:, -N_BINS:, :] * self.sqrt_lengths

        return cqt.abs()

    @staticmethod
    def _normalized_log(magnitude: Tensor) -> Tensor:
        """Apply Basic Pitch's per-example min-max log normalization."""
        decibels = 10.0 * torch.log10(magnitude.square() + 1e-10)
        offset = decibels - decibels.amin(dim=(-2, -1), keepdim=True)
        maximum = offset.amax(dim=(-2, -1), keepdim=True)
        denominator = torch.where(
            maximum == 0,
            torch.ones_like(maximum),
            maximum,
        )

        return offset / denominator

    @staticmethod
    def _shift_frequency(tensor: Tensor, shift: int) -> Tensor:
        """Zero-fill a shift along the frequency axis."""
        if shift == 0:
            return tensor

        if shift > 0:
            return torch.cat(
                [
                    tensor[:, shift:, :],
                    torch.zeros_like(tensor[:, :shift, :]),
                ],
                dim=1,
            )

        offset = -shift

        return torch.cat(
            [
                torch.zeros_like(tensor[:, :offset, :]),
                tensor[:, :-offset, :],
            ],
            dim=1,
        )

    @classmethod
    def _harmonic_stack(cls, normalized: Tensor) -> Tensor:
        """Stack the eight harmonic shifts as `(batch, 8, 264, frames)`."""
        channels = [
            cls._shift_frequency(normalized, shift)
            for shift in HARMONIC_SHIFTS
        ]

        return torch.stack(channels, dim=1)[:, :, :N_STACK_BINS, :]


class BasicPitchTrunk(nn.Module):
    """Basic Pitch's contour trunk, without the polyphonic heads.

    The exported BatchNorms are folded into the convolutions, so the
    chain is `conv2 -> ReLU -> conv3 -> sigmoid -> conv4 -> ReLU`. The
    first `Conv2D(32, (5, 5))` in `basic_pitch/models.py` feeds an
    overwritten variable and is absent from the export.

    Args:
        pretrained: Load the converted Basic Pitch weights. `False`
            keeps the same architecture with fresh initialization, which
            is the control arm for the transfer ablation.
    """

    def __init__(self, *, pretrained: bool = True) -> None:
        super().__init__()

        self.conv2 = nn.Conv2d(
            8,
            8,
            kernel_size=(3, 39),
            stride=(1, 1),
            padding=(1, 19),
        )
        self.conv3 = nn.Conv2d(
            8,
            1,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
        )
        self.conv4 = nn.Conv2d(
            1,
            32,
            kernel_size=(7, 7),
            stride=(1, 3),
            padding=(3, 2),
        )

        if pretrained:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self) -> None:
        weights = _load_weights()

        for name in ("conv2", "conv3", "conv4"):
            module = getattr(self, name)
            weight = torch.from_numpy(weights[f"{name}.weight"])
            bias = torch.from_numpy(weights[f"{name}.bias"])

            with torch.no_grad():
                module.weight.copy_(weight)
                module.bias.copy_(bias)

    def forward(self, stack: Tensor) -> Tensor:
        """Map `(batch, 8, frames, 264)` to `(batch, 32, frames, 88)`."""
        hidden = F.relu(self.conv2(stack))
        hidden = torch.sigmoid(self.conv3(hidden))

        return F.relu(self.conv4(hidden))


class BasicPitchPitchSequenceCNN(nn.Module):
    """Pretrained Basic Pitch trunk with a per-step pitch-sequence head.

    The CQT frontend runs in `SynthesizedMelodyDataset`, like the log-mel
    extractor does for `PitchSequenceCNN`. This model therefore takes the
    harmonic stack `(batch, 8, 264, frames)` rather than raw audio.

    Args:
        steps: Number of dataset steps to predict.
        num_classes: Size of the per-step class vocabulary.
        dropout: Dropout probability before the classifier.
        pretrained: Load the converted trunk weights. `False` keeps the
            same architecture with fresh initialization, which is the
            control arm for the transfer ablation.

    Raises:
        ValueError: If a validation constraint is violated.
    """

    def __init__(
        self,
        *,
        steps: int = STEPS_PER_MELODY,
        num_classes: int = NUM_CLASSES,
        dropout: float = DEFAULT_DROPOUT,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        if steps < 1:
            raise ValueError("steps must be at least 1")

        if num_classes < 1:
            raise ValueError("num_classes must be at least 1")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.steps = int(steps)
        self.trunk = BasicPitchTrunk(pretrained=pretrained)
        self.temporal = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Predict per-step class logits from harmonic CQT features.

        Args:
            features: `(batch, 8, 264, frames)` harmonic CQT.

        Returns:
            A `(batch, steps, num_classes)` float tensor of logits.
        """
        if (
            features.ndim != 4
            or features.shape[1] != N_HARMONICS
            or features.shape[2] != N_STACK_BINS
        ):
            raise ValueError(
                "features must have shape "
                f"(batch, {N_HARMONICS}, {N_STACK_BINS}, frames), got "
                f"{tuple(features.shape)}"
            )

        hidden = self.trunk(features.transpose(2, 3))
        hidden = hidden.mean(dim=-1)
        hidden = self.temporal(hidden)
        hidden = F.interpolate(
            hidden,
            size=self.steps,
            mode="linear",
            align_corners=False,
        )

        return self.classifier(hidden.transpose(1, 2))
