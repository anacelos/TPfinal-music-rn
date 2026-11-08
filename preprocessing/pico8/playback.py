from dataclasses import dataclass
from typing import Optional

from .model import (NOTES_PER_SFX, SPEED_INCOMPATIBLE, SPEED_MULTIPLE, SPEED_NONE, SPEED_SINGLE,
                    SPEED_UNIFORM, TERMINATION_END_OF_TABLE, TERMINATION_FALLTHROUGH_EMPTY,
                    TERMINATION_LOOP, TERMINATION_LOOP_EXTERNAL, TERMINATION_STOP, Pattern, Sfx)

NO_SPEED = 0


@dataclass(frozen=True)
class PatternTiming:
    ticks: int
    len_notes: int
    len_mode: bool


@dataclass(frozen=True)
class PatternPlayback:
    sfx_ids: tuple[Optional[int], ...]
    speeds: tuple[int, ...]
    speed_kind: str
    timing: PatternTiming
    absent_sfx_refs: int


@dataclass(frozen=True)
class SongSegment:
    start: int
    end: int
    termination: str
    loop_target: Optional[int]


def pattern_timing(channel_sfx: list[Sfx]) -> PatternTiming:
    looping_ticks = []
    for sfx in channel_sfx:
        if not sfx.loops:
            n = min(NOTES_PER_SFX, sfx.loop_start) if sfx.len_mode else NOTES_PER_SFX
            return PatternTiming(n * sfx.effective_speed, n, sfx.len_mode)
        looping_ticks.append(NOTES_PER_SFX * sfx.effective_speed)
    if looping_ticks:
        return PatternTiming(max(looping_ticks), NOTES_PER_SFX, False)
    return PatternTiming(0, 0, False)


def speed_kind(speeds: tuple[int, ...]) -> str:
    usable = [s for s in speeds if s > NO_SPEED]
    if not usable:
        return SPEED_NONE
    if len(usable) == 1:
        return SPEED_SINGLE
    if len(set(usable)) == 1:
        return SPEED_UNIFORM
    fastest = min(usable)
    return SPEED_MULTIPLE if all(s % fastest == 0 for s in usable) else SPEED_INCOMPATIBLE


def play_pattern(pattern: Pattern, sfx_slots: list[Sfx]) -> PatternPlayback:
    sfx_ids = pattern.sfx_ids
    channel_sfx = [sfx_slots[sid] for sid in sfx_ids if sid is not None]
    speeds = tuple(NO_SPEED if sid is None or not sfx_slots[sid].present
                   else sfx_slots[sid].effective_speed for sid in sfx_ids)
    absent_refs = sum(not sfx.present for sfx in channel_sfx)
    return PatternPlayback(sfx_ids, speeds, speed_kind(speeds), pattern_timing(channel_sfx),
                           absent_refs)


def find_loop_target(patterns: list[Pattern], loop_end: int) -> int:
    target = loop_end
    while target > 0 and not patterns[target].loop_start:
        target -= 1
    return target


def _walk_song(patterns: list[Pattern], start: int) -> SongSegment:
    j = start
    while True:
        pattern = patterns[j]
        if pattern.stop:
            return SongSegment(start, j, TERMINATION_STOP, None)
        if pattern.loop_end:
            target = find_loop_target(patterns, j)
            termination = TERMINATION_LOOP if target >= start else TERMINATION_LOOP_EXTERNAL
            return SongSegment(start, j, termination, target)
        if j + 1 >= len(patterns):
            return SongSegment(start, j, TERMINATION_END_OF_TABLE, None)
        if patterns[j + 1].is_empty:
            return SongSegment(start, j, TERMINATION_FALLTHROUGH_EMPTY, None)
        j += 1


def segment_songs(patterns: list[Pattern]) -> list[SongSegment]:
    segments = []
    i = 0
    while i < len(patterns):
        if patterns[i].is_empty:
            i += 1
            continue
        segment = _walk_song(patterns, i)
        segments.append(segment)
        i = segment.end + 1
    return segments
