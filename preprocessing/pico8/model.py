from dataclasses import dataclass
from typing import NamedTuple, Optional

NOTES_PER_SFX = 32
N_SLOTS = 64
N_CHANNELS = 4
N_PITCHES = 64
N_INSTRUMENTS = 16
N_VOLUMES = 8
N_EFFECTS = 8
CUSTOM_INSTRUMENT_MIN = 8
TICK_SECONDS = 183.0 / 22050.0

FLAG_LOOP_START = 1
FLAG_LOOP_END = 2
FLAG_STOP = 4
KNOWN_FLAGS = FLAG_LOOP_START | FLAG_LOOP_END | FLAG_STOP

CHANNEL_DISABLED_BIT = 0x40
CHANNEL_SFX_MASK = 0x3F

TERMINATION_STOP = "stop"
TERMINATION_LOOP = "loop"
TERMINATION_LOOP_EXTERNAL = "loop_external"
TERMINATION_END_OF_TABLE = "end_of_table"
TERMINATION_FALLTHROUGH_EMPTY = "fallthrough_empty"

SPEED_NONE = "none"
SPEED_SINGLE = "single"
SPEED_UNIFORM = "uniform"
SPEED_MULTIPLE = "multiple"
SPEED_INCOMPATIBLE = "incompatible"
SPEED_KINDS = (SPEED_NONE, SPEED_SINGLE, SPEED_UNIFORM, SPEED_MULTIPLE, SPEED_INCOMPATIBLE)


class Note(NamedTuple):
    pitch: int
    instrument: int
    volume: int
    effect: int

    @property
    def is_rest(self) -> bool:
        return self.volume == 0

    @property
    def is_custom_instrument(self) -> bool:
        return self.instrument >= CUSTOM_INSTRUMENT_MIN

    @property
    def custom_instrument_sfx_id(self) -> int:
        return self.instrument - CUSTOM_INSTRUMENT_MIN


BLANK_NOTE = Note(0, 0, 0, 0)


@dataclass(frozen=True)
class Sfx:
    speed: int
    loop_start: int
    loop_end: int
    notes: tuple[Note, ...]
    present: bool = True

    @property
    def effective_speed(self) -> int:
        return max(1, self.speed)

    @property
    def loops(self) -> bool:
        return self.loop_end > self.loop_start

    @property
    def len_mode(self) -> bool:
        return self.loop_end == 0 and self.loop_start > 0

    @property
    def n_audible_notes(self) -> int:
        return sum(not note.is_rest for note in self.notes)

    @property
    def custom_instrument_sfx_ids(self) -> set[int]:
        return {n.custom_instrument_sfx_id for n in self.notes
                if n.is_custom_instrument and not n.is_rest}


ABSENT_SFX = Sfx(speed=0, loop_start=0, loop_end=0, notes=(BLANK_NOTE,) * NOTES_PER_SFX, present=False)


def is_channel_disabled(byte: int) -> bool:
    return bool(byte & CHANNEL_DISABLED_BIT)


def channel_sfx_id(byte: int) -> int:
    return byte & CHANNEL_SFX_MASK


@dataclass(frozen=True)
class Pattern:
    id: int
    flags: int
    channels: tuple[int, ...]

    @classmethod
    def empty(cls, pattern_id: int) -> "Pattern":
        return cls.from_sfx_ids(pattern_id, 0, (None,) * N_CHANNELS)

    @classmethod
    def from_sfx_ids(cls, pattern_id: int, flags: int, sfx_ids: tuple[Optional[int], ...]) -> "Pattern":
        channels = tuple(CHANNEL_DISABLED_BIT if sid is None else sid for sid in sfx_ids)
        return cls(pattern_id, flags, channels)

    @property
    def sfx_ids(self) -> tuple[Optional[int], ...]:
        return tuple(None if is_channel_disabled(b) else channel_sfx_id(b) for b in self.channels)

    @property
    def is_empty(self) -> bool:
        return all(is_channel_disabled(b) for b in self.channels)

    @property
    def loop_start(self) -> bool:
        return bool(self.flags & FLAG_LOOP_START)

    @property
    def loop_end(self) -> bool:
        return bool(self.flags & FLAG_LOOP_END)

    @property
    def stop(self) -> bool:
        return bool(self.flags & FLAG_STOP)

    @property
    def has_extra_flags(self) -> bool:
        return bool(self.flags & ~KNOWN_FLAGS)
