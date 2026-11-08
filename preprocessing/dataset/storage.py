import csv
import json
from pathlib import Path


def tid_sort_key(tid: str) -> tuple[int, str]:
    return (int(tid) if tid.isdigit() else 0, tid)


def load_songs(path: Path) -> list[dict]:
    songs = []
    with open(path) as f:
        for line in f:
            song = json.loads(line)
            song["tid"] = str(song["tid"])
            songs.append(song)
    songs.sort(key=lambda s: (*tid_sort_key(s["tid"]), s["song_idx"]))
    return songs


def read_metadata(path: Path) -> dict[str, dict]:
    metadata = {}
    if not path.exists():
        return metadata
    with open(path) as f:
        for row in csv.DictReader(f):
            tags = [t for t in (row.get("tags") or "").split("|") if t]
            stars = row.get("stars")
            metadata[row["tid"]] = {"tags": tags,
                                    "stars": int(float(stars)) if stars not in (None, "") else None}
    return metadata
