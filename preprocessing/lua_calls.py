import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .pico8.model import N_SLOTS
from .pico8.playback import SongSegment

MUSIC_CALL = re.compile(r"\bmusic\s*\(\s*(-?\d+)?\s*([,)])?")
LITERAL_TERMINATORS = (",", ")")

CALL_STOP = "stop_call"
CALL_OUT_OF_RANGE = "out_of_range"
CALL_SONG_START = "song_start"
CALL_MID_SONG = "mid_song"
CALL_EMPTY_PATTERN = "empty_pattern"


@dataclass(frozen=True)
class LuaMusicCalls:
    literal_targets: list[int]
    n_nonliteral: int


def read_music_calls(lua_root: Path, tid: str) -> Optional[LuaMusicCalls]:
    """None when the cart has no readable Lua file."""
    path = lua_root / tid / f"{tid}.lua"
    if not path.exists():
        return None
    try:
        with open(path, "r", errors="replace") as f:
            source = f.read()
    except OSError:
        return None
    targets, n_nonliteral = set(), 0
    for match in MUSIC_CALL.finditer(source):
        if match.group(1) is not None and match.group(2) in LITERAL_TERMINATORS:
            targets.add(int(match.group(1)))
        else:
            n_nonliteral += 1
    return LuaMusicCalls(sorted(targets), n_nonliteral)


def music_call_rows(tid: str, targets: list[int], segments: list[SongSegment]) -> list[dict]:
    song_at_start = {seg.start: i for i, seg in enumerate(segments)}
    song_at = {p: i for i, seg in enumerate(segments) for p in range(seg.start, seg.end + 1)}
    rows = []
    for target in targets:
        if target < 0:
            kind = CALL_STOP
        elif target >= N_SLOTS:
            kind = CALL_OUT_OF_RANGE
        elif target in song_at_start:
            kind = CALL_SONG_START
        elif target in song_at:
            kind = CALL_MID_SONG
        else:
            kind = CALL_EMPTY_PATTERN
        rows.append({"tid": tid, "pattern": target, "class": kind, "song_idx": song_at.get(target, "")})
    return rows
