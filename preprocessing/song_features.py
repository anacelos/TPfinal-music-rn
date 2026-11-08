import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .pico8.model import (CUSTOM_INSTRUMENT_MIN, N_EFFECTS, N_INSTRUMENTS, N_PITCHES, N_VOLUMES,
                          SPEED_KINDS, TERMINATION_LOOP, TERMINATION_LOOP_EXTERNAL, TICK_SECONDS,
                          Note, Pattern, Sfx)
from .pico8.playback import PatternPlayback, SongSegment, play_pattern

DISABLED_SFX_ID = -1
HASH_HEX_CHARS = 16
PATTERN_SEPARATOR = b"|"
DISABLED_CHANNEL_BYTES = b"\xff"
PITCH, INSTRUMENT, VOLUME, EFFECT = range(len(Note._fields))


@dataclass(frozen=True)
class PrefixStats:
    canonical: bytes
    n_rest: int
    pitch_hist: np.ndarray
    instrument_hist: np.ndarray
    volume_hist: np.ndarray
    effect_hist: np.ndarray
    pitch_sum: int
    pitch_min: Optional[int]
    pitch_max: Optional[int]


def compute_prefix_stats(sfx: Sfx, length: int) -> PrefixStats:
    notes = np.array(sfx.notes[:length], dtype=np.int16).reshape(-1, len(Note._fields))
    rest = notes[:, VOLUME] == 0
    canonical = notes.copy()
    canonical[rest] = 0
    live = notes[~rest]
    return PrefixStats(
        canonical=canonical.astype(np.uint8).tobytes(),
        n_rest=int(rest.sum()),
        pitch_hist=np.bincount(live[:, PITCH], minlength=N_PITCHES)[:N_PITCHES],
        instrument_hist=np.bincount(live[:, INSTRUMENT], minlength=N_INSTRUMENTS)[:N_INSTRUMENTS],
        volume_hist=np.bincount(live[:, VOLUME], minlength=N_VOLUMES)[:N_VOLUMES],
        effect_hist=np.bincount(live[:, EFFECT], minlength=N_EFFECTS)[:N_EFFECTS],
        pitch_sum=int(live[:, PITCH].sum()),
        pitch_min=int(live[:, PITCH].min()) if len(live) else None,
        pitch_max=int(live[:, PITCH].max()) if len(live) else None,
    )


class PrefixStatsCache:
    def __init__(self, sfx_slots: list[Sfx]):
        self._sfx_slots = sfx_slots
        self._cache: dict[tuple[int, int], PrefixStats] = {}

    def get(self, sfx_id: int, length: int) -> PrefixStats:
        key = (sfx_id, length)
        if key not in self._cache:
            self._cache[key] = compute_prefix_stats(self._sfx_slots[sfx_id], length)
        return self._cache[key]


@dataclass
class NoteTotals:
    pitch_hist: np.ndarray
    instrument_hist: np.ndarray
    volume_hist: np.ndarray
    effect_hist: np.ndarray
    n_rest: int = 0
    pitch_sum: int = 0
    pitch_min: Optional[int] = None
    pitch_max: Optional[int] = None

    @property
    def n_live(self) -> int:
        return int(self.pitch_hist.sum())


def _sum_note_totals(usage: Counter, cache: PrefixStatsCache) -> NoteTotals:
    totals = NoteTotals(np.zeros(N_PITCHES, dtype=np.int64), np.zeros(N_INSTRUMENTS, dtype=np.int64),
                        np.zeros(N_VOLUMES, dtype=np.int64), np.zeros(N_EFFECTS, dtype=np.int64))
    for (sfx_id, length), count in usage.items():
        stats = cache.get(sfx_id, length)
        totals.pitch_hist += count * stats.pitch_hist
        totals.instrument_hist += count * stats.instrument_hist
        totals.volume_hist += count * stats.volume_hist
        totals.effect_hist += count * stats.effect_hist
        totals.n_rest += count * stats.n_rest
        totals.pitch_sum += count * stats.pitch_sum
        if stats.pitch_min is not None:
            lows = [v for v in (totals.pitch_min, stats.pitch_min) if v is not None]
            highs = [v for v in (totals.pitch_max, stats.pitch_max) if v is not None]
            totals.pitch_min, totals.pitch_max = min(lows), max(highs)
    return totals


def content_hash(parts: list[bytes]) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(part)
        digest.update(PATTERN_SEPARATOR)
    return digest.hexdigest()[:HASH_HEX_CHARS]


def _canonical_pattern_bytes(playback: PatternPlayback, sfx_slots: list[Sfx],
                             cache: PrefixStatsCache) -> bytes:
    length = playback.timing.len_notes
    part = bytearray([length])
    for sfx_id in playback.sfx_ids:
        if sfx_id is None:
            part += DISABLED_CHANNEL_BYTES
        else:
            speed = sfx_slots[sfx_id].effective_speed
            part += bytes([speed & 0xFF, speed >> 8]) + cache.get(sfx_id, length).canonical
    return bytes(part)


def _pattern_record(pattern: Pattern, playback: PatternPlayback) -> dict:
    return {"id": pattern.id,
            "sfx": [DISABLED_SFX_ID if sid is None else sid for sid in playback.sfx_ids],
            "speed": list(playback.speeds), "kind": playback.speed_kind,
            "ticks": playback.timing.ticks, "len_notes": playback.timing.len_notes,
            "len_mode": playback.timing.len_mode}


def _loop_bounds(segment: SongSegment) -> tuple[Optional[int], Optional[int]]:
    last = segment.end - segment.start
    if segment.termination == TERMINATION_LOOP:
        return segment.loop_target - segment.start, last
    if segment.termination == TERMINATION_LOOP_EXTERNAL:
        return None, last
    return None, None


def _duration_seconds(playbacks: list[PatternPlayback]) -> float:
    return sum(pb.timing.ticks for pb in playbacks) * TICK_SECONDS


def build_song(tid: str, song_idx: int, segment: SongSegment, patterns: list[Pattern],
               sfx_slots: list[Sfx], cache: PrefixStatsCache) -> dict:
    span = patterns[segment.start:segment.end + 1]
    playbacks = [play_pattern(p, sfx_slots) for p in span]
    n_patterns = len(span)
    lo, hi = _loop_bounds(segment)
    loop = slice(lo, hi + 1) if lo is not None else slice(None)

    kinds = Counter(pb.speed_kind for pb in playbacks)
    usage = Counter((sid, pb.timing.len_notes) for pb in playbacks
                    for sid in pb.sfx_ids if sid is not None)
    active = [sum(sid is not None for sid in pb.sfx_ids) for pb in playbacks]
    n_chan_steps = sum(a * pb.timing.len_notes for a, pb in zip(active, playbacks))
    totals = _sum_note_totals(usage, cache)
    n_live = totals.n_live
    canonical_parts = [_canonical_pattern_bytes(pb, sfx_slots, cache) for pb in playbacks]
    all_speeds = [s for pb in playbacks for s in pb.speeds if s > 0]

    loop_target = segment.loop_target if segment.termination == TERMINATION_LOOP_EXTERNAL else None
    return {
        "tid": tid,
        "song_idx": song_idx,
        "pattern_ids": [p.id for p in span],
        "loop_start_idx": lo,
        "loop_end_idx": hi,
        "loop_target_pattern": loop_target,
        "termination": segment.termination,
        "patterns": [_pattern_record(p, pb) for p, pb in zip(span, playbacks)],
        "n_patterns": n_patterns,
        "n_steps_full": sum(pb.timing.len_notes for pb in playbacks),
        "n_steps_loop": sum(pb.timing.len_notes for pb in playbacks[loop]),
        "n_len_mode_patterns": sum(pb.timing.len_mode for pb in playbacks),
        "n_extra_flag_patterns": sum(p.has_extra_flags for p in span),
        "speed_min": min(all_speeds) if all_speeds else None,
        "speed_max": max(all_speeds) if all_speeds else None,
        "n_distinct_speeds": len(set(all_speeds)),
        **{f"n_pat_{kind}": kinds[kind] for kind in SPEED_KINDS},
        "duration_s_full": _duration_seconds(playbacks),
        "duration_s_loop": _duration_seconds(playbacks[loop]),
        "active_ch_mean": sum(active) / n_patterns,
        "active_ch_max": max(active),
        "n_chan_steps": n_chan_steps,
        "n_rest": int(totals.n_rest),
        "n_notes": n_live,
        "rest_ratio": (totals.n_rest / n_chan_steps) if n_chan_steps else None,
        "all_rest": n_live == 0,
        "pitch_min": totals.pitch_min,
        "pitch_max": totals.pitch_max,
        "pitch_mean": (totals.pitch_sum / n_live) if n_live else None,
        "n_custom_instr_notes": int(totals.instrument_hist[CUSTOM_INSTRUMENT_MIN:].sum()),
        "n_blank_sfx_refs": sum(pb.absent_sfx_refs for pb in playbacks),
        "pitch_hist": totals.pitch_hist.tolist(),
        "instr_hist": totals.instrument_hist.tolist(),
        "vol_hist": totals.volume_hist.tolist(),
        "eff_hist": totals.effect_hist.tolist(),
        "content_hash": content_hash(canonical_parts),
        "content_hash_loop": content_hash(canonical_parts[loop]),
    }
