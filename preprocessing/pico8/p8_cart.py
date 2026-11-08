import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .model import BLANK_NOTE, N_CHANNELS, N_SLOTS, Note, Pattern, Sfx

SFX_LINE_CHARS = 168
SFX_HEADER_CHARS = 8
NOTE_CHARS = 5
CHANNELS_OFFSET = 3
CHANNEL_CHARS = 2

HEX_LINE = re.compile(r"^[0-9a-fA-F]+$")
MUSIC_LINE = re.compile(r"^[0-9a-fA-F]{2} [0-9a-fA-F]{8}$")


@dataclass(frozen=True)
class P8Sfx:
    editor: int
    sfx: Sfx

    @property
    def is_blank(self) -> bool:
        """All zero apart from the default speed byte."""
        return (self.editor == 0 and self.sfx.loop_start == 0 and self.sfx.loop_end == 0
                and all(note == BLANK_NOTE for note in self.sfx.notes))


@dataclass
class P8Cart:
    version: Optional[str] = None
    has_sfx: bool = False
    has_music: bool = False
    sfx_lines: int = 0
    music_lines: int = 0
    bad_sfx_lines: int = 0
    bad_music_lines: int = 0
    read_error: Optional[str] = None
    sfx: dict[int, P8Sfx] = field(default_factory=dict)
    music: dict[int, Pattern] = field(default_factory=dict)


def _split_sections(lines: list[str], cart: P8Cart) -> tuple[list[str], list[str]]:
    section = None
    sfx_lines, music_lines = [], []
    for line in lines:
        if line.startswith("__") and line.endswith("__"):
            section = line.strip("_")
            cart.has_sfx |= section == "sfx"
            cart.has_music |= section == "music"
        elif section == "sfx":
            sfx_lines.append(line)
        elif section == "music":
            music_lines.append(line)
    return sfx_lines, music_lines


def _drop_trailing_blank(lines: list[str]) -> None:
    while lines and lines[-1] == "":
        lines.pop()


def _parse_sfx_line(line: str) -> P8Sfx:
    notes = []
    for j in range(SFX_HEADER_CHARS, SFX_LINE_CHARS, NOTE_CHARS):
        notes.append(Note(pitch=int(line[j:j + 2], 16), instrument=int(line[j + 2], 16),
                          volume=int(line[j + 3], 16), effect=int(line[j + 4], 16)))
    sfx = Sfx(speed=int(line[2:4], 16), loop_start=int(line[4:6], 16),
              loop_end=int(line[6:8], 16), notes=tuple(notes))
    return P8Sfx(editor=int(line[0:2], 16), sfx=sfx)


def _parse_music_line(index: int, line: str) -> Pattern:
    channels = tuple(int(line[CHANNELS_OFFSET + CHANNEL_CHARS * k:
                              CHANNELS_OFFSET + CHANNEL_CHARS * (k + 1)], 16)
                     for k in range(N_CHANNELS))
    return Pattern(id=index, flags=int(line[0:2], 16), channels=channels)


def parse_p8(path: Path) -> P8Cart:
    cart = P8Cart()
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        cart.read_error = str(e)
        return cart
    lines = raw.decode("latin-1").replace("\r\n", "\n").split("\n")
    if len(lines) > 1 and lines[1].startswith("version"):
        cart.version = lines[1].split()[-1]
    sfx_lines, music_lines = _split_sections(lines, cart)
    _drop_trailing_blank(sfx_lines)
    _drop_trailing_blank(music_lines)
    cart.sfx_lines = len(sfx_lines)
    cart.music_lines = len(music_lines)
    for i, line in enumerate(sfx_lines[:N_SLOTS]):
        if len(line) != SFX_LINE_CHARS or not HEX_LINE.match(line):
            cart.bad_sfx_lines += 1
        else:
            cart.sfx[i] = _parse_sfx_line(line)
    for i, line in enumerate(music_lines[:N_SLOTS]):
        if not MUSIC_LINE.match(line):
            cart.bad_music_lines += 1
        else:
            cart.music[i] = _parse_music_line(i, line)
    return cart


def editor_features(editor: int) -> list[str]:
    """Decode the per-sfx editor byte; the filter bits are mixed radix (zepto8 sfx.cpp)."""
    features = []
    if editor & 1:
        features.append("tracker_mode")
    if editor & 2:
        features.append("filter_noiz")
    if editor & 4:
        features.append("filter_buzz")
    if (editor // 8) % 3:
        features.append("filter_detune")
    if (editor // 24) % 3:
        features.append("filter_reverb")
    if (editor // 72) % 3:
        features.append("filter_dampen")
    return features
