import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

CHANNELS = ("ch1", "ch2", "ch3", "ch4")

Song = dict[str, Any]


def iter_games(root: Path) -> Iterator[tuple[str, list[Song]]]:
    """Yields (tid, songs) in corpus order: games by int(tid), songs by song_id."""
    paths = sorted((root / "audio").glob("*/songs.json"), key=lambda p: int(p.parent.name))
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        yield str(data["tid"]), sorted(data["songs"], key=lambda s: s["song_id"])


def stable_hash(value: Any) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def count_audible(notes: list[dict]) -> int:
    return sum(1 for note in notes if note["volume"] > 0)


def active_channels(pattern: dict) -> list[dict]:
    return [pattern["channels"][ch] for ch in CHANNELS if pattern["channels"][ch]["active"]]


def audible_notes(song: Song) -> int:
    return sum(count_audible(ch["notes_unrolled"]) for p in song["patterns"] for ch in active_channels(p))


def is_incompatible(pattern: dict) -> bool:
    speeds = [ch["speed"] for ch in active_channels(pattern) if ch["speed"] > 0]
    return bool(speeds) and any(s % min(speeds) for s in speeds)


def content_hash(song: Song) -> str:
    """Identifies songs that play the same thing, whatever the game, pattern ids or block ids."""
    patterns = [
        [p["pattern_ticks"],
         [[p["channels"][ch]["speed"], p["channels"][ch]["notes_unrolled"]] if p["channels"][ch]["active"] else None
          for ch in CHANNELS]]
        for p in song["patterns"]
    ]
    return stable_hash([song["loop_start_pattern"] - song["start_pattern"], song["end_reason"], patterns])


def block_keys(song: Song, min_notes: int) -> set[str]:
    """Hashes of the song's sfx blocks with enough audible notes to say two games share material."""
    return {stable_hash(block["notes"]) for block in song["blocks_dictionary"].values()
            if count_audible(block["notes"]) >= min_notes}


def song_features(song: Song, shared_block_min_notes: int) -> dict:
    return {
        "song_id": song["song_id"],
        "audible_notes": audible_notes(song),
        "custom_instruments": bool(song["uses_custom_instruments"]),
        "incompatible_speeds": any(is_incompatible(p) for p in song["patterns"]),
        "content_hash": content_hash(song),
        "block_keys": block_keys(song, shared_block_min_notes),
    }
