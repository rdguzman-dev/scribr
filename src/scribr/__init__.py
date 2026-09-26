"""Automated music transcription for quantized monophonic audio.

The shared infrastructure lives in four subpackages:

- `scribr.data`: lazy access to the TFRecord dataset.
- `scribr.representation`: the canonical symbolic `Melody` type.
- `scribr.midi`: optional MIDI export.
- `scribr.synthesis`: on-demand FluidSynth audio rendering.

See `README.md` for setup and usage.
"""

__version__ = "0.1.0"


def main() -> None:
    """Entry point for the `scribr` console script."""
    print(
        "Scribr - automated music transcription for quantized monophonic "
        "audio."
    )
    print("See README.md for dataset setup, synthesis, and MIDI export usage.")
