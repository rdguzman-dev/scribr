"""General MIDI instruments used by Scribr experiments."""

from enum import StrEnum


class Instrument(StrEnum):
    """A General MIDI instrument preset.

    The value is the plain instrument name used in experiment configs.
    """

    PIANO = "piano"
    ACOUSTIC_GUITAR = "acoustic_guitar"
    ELECTRIC_GUITAR = "electric_guitar"
    VIOLIN = "violin"

    @property
    def program(self) -> int:
        """Zero-based General MIDI program number."""
        return INSTRUMENT_PROGRAMS[self]


INSTRUMENT_PROGRAMS: dict[Instrument, int] = {
    Instrument.PIANO: 0,
    Instrument.ACOUSTIC_GUITAR: 25,
    Instrument.ELECTRIC_GUITAR: 27,
    Instrument.VIOLIN: 40,
}
