import json
from dataclasses import dataclass
from pathlib import Path

from .model import ABSENT_SFX, BLANK_NOTE, N_SLOTS, NOTES_PER_SFX, Note, Pattern, Sfx

CHANNEL_KEYS = ("ch1", "ch2", "ch3", "ch4")


@dataclass(frozen=True)
class Audio:
    sfx: list[Sfx]
    patterns: list[Pattern]
    n_sfx_entries: int
    stored_patterns: list[Pattern]


def sfx_from_dict(entry: dict) -> Sfx:
    notes = [Note(n["pitch"], n["instrument"], n["volume"], n["effect"])
             for n in entry.get("notes", [])[:NOTES_PER_SFX]]
    notes += [BLANK_NOTE] * (NOTES_PER_SFX - len(notes))
    return Sfx(int(entry["speed"]), int(entry["loop_start"]), int(entry["loop_end"]), tuple(notes))


def pattern_from_dict(entry: dict) -> Pattern:
    return Pattern(entry["id"], entry["flags"], tuple(entry[key] for key in CHANNEL_KEYS))


def sfx_to_dict(sfx_id: int, sfx: Sfx) -> dict:
    return {"id": sfx_id, "speed": sfx.speed, "loop_start": sfx.loop_start,
            "loop_end": sfx.loop_end, "notes": [note._asdict() for note in sfx.notes]}


def pattern_to_dict(pattern: Pattern) -> dict:
    return {"id": pattern.id, "flags": pattern.flags, **dict(zip(CHANNEL_KEYS, pattern.channels))}


def parse_audio(data: dict) -> Audio:
    sfx_entries = data.get("sfx", [])
    sfx = [ABSENT_SFX] * N_SLOTS
    for entry in sfx_entries:
        if 0 <= entry["id"] < N_SLOTS:
            sfx[entry["id"]] = sfx_from_dict(entry)
    stored_patterns = [pattern_from_dict(m) for m in data.get("music", [])]
    by_id = {p.id: p for p in stored_patterns if 0 <= p.id < N_SLOTS}
    patterns = [by_id.get(i, Pattern.empty(i)) for i in range(N_SLOTS)]
    return Audio(sfx, patterns, len(sfx_entries), stored_patterns)


def load_audio(path: Path) -> Audio:
    with open(path) as f:
        return parse_audio(json.load(f))
