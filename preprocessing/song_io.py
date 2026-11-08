"""Read the processed splits and turn a song record back into a PICO-8 audio.json dict."""
import json
from pathlib import Path

from .dataset.records import unpack_sfx
from .paths import PROCESSED
from .pico8.audio_json import pattern_to_dict, sfx_to_dict
from .pico8.model import (FLAG_LOOP_END, FLAG_LOOP_START, FLAG_STOP, TERMINATION_LOOP, Pattern)


def load_split(split: str, path: Path = PROCESSED) -> list[dict]:
    with open(Path(path) / f"{split}.jsonl") as f:
        return [json.loads(line) for line in f]


def _played_range(record: dict, loop_only: bool) -> tuple[int, int]:
    if loop_only and record["loop_start"] is not None:
        return record["loop_start"], record["loop_end"]
    return 0, len(record["patterns"]) - 1


def _pattern_flags(record: dict, position: int, last: int) -> int:
    if record["termination"] == TERMINATION_LOOP and record["loop_start"] is not None:
        return ((FLAG_LOOP_START if position == record["loop_start"] else 0)
                | (FLAG_LOOP_END if position == record["loop_end"] else 0))
    return FLAG_STOP if position == last else 0


def to_audio_json(record: dict, loop_only: bool = False) -> dict:
    """Patterns are renumbered from 0 and sfx keep their original ids, because custom
    instruments address sfx 0-7 by id."""
    first, last = _played_range(record, loop_only)
    music = []
    for new_id, position in enumerate(range(first, last + 1)):
        channels = tuple(record["patterns"][position]["channels"])
        pattern = Pattern.from_sfx_ids(new_id, _pattern_flags(record, position, last), channels)
        music.append(pattern_to_dict(pattern))
    sfx = [sfx_to_dict(int(sid), unpack_sfx(packed))
           for sid, packed in sorted(record["sfx"].items(), key=lambda kv: int(kv[0]))]
    return {"sfx": sfx, "music": music}
