import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .dataset.config import Config, load_config
from .dataset.filters import decide_songs, song_key
from .dataset.grouping import group_tids
from .dataset.records import build_record
from .dataset.reports import build_stats, write_filter_report, write_split_tids, write_splits
from .dataset.splits import assign_splits
from .dataset.storage import load_songs, read_metadata, tid_sort_key
from .paths import CONFIG, DATASET, INTERIM, PROCESSED
from .pico8.audio_json import load_audio


def group_by_tid(songs: list[dict]) -> dict[str, list[dict]]:
    by_tid: dict[str, list[dict]] = {}
    for song in songs:
        by_tid.setdefault(song["tid"], []).append(song)
    return by_tid


def build_records(by_tid: dict[str, list[dict]], root: Path, metadata: dict[str, dict]) -> list[dict]:
    records = []
    for tid in sorted(by_tid, key=tid_sort_key):
        audio = load_audio(root / "audio" / tid / "audio.json")
        records += [build_record(song, audio.patterns, audio.sfx, metadata.get(tid, {}))
                    for song in by_tid[tid]]
    return records


def run(config: Config, root: Path, interim: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    songs = load_songs(interim / "songs.jsonl")
    metadata = read_metadata(interim / "tid_metadata.csv")
    reasons = decide_songs(songs, config)
    kept = [s for s in songs if reasons[song_key(s)] is None]
    by_tid = group_by_tid(kept)
    records = build_records(by_tid, root, metadata)

    group_of = group_tids(songs, records, config)
    group_sizes = Counter(group_of[s["tid"]] for s in kept)
    split_of_group = assign_splits(group_sizes, config)
    split_of_tid = {t: split_of_group[g] for t, g in group_of.items() if g in split_of_group}

    counts, steps = write_splits(out, records, split_of_tid, config)
    write_filter_report(out, songs, reasons, split_of_tid)
    write_split_tids(out, split_of_tid, group_of)
    stats = build_stats(config, songs, reasons, kept, list(by_tid), group_sizes, split_of_tid,
                        counts, steps)
    with open(out / "build_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--root", type=Path, default=DATASET)
    parser.add_argument("--interim", type=Path, default=INTERIM)
    parser.add_argument("--out", type=Path, default=PROCESSED)
    args = parser.parse_args()
    stats = run(load_config(args.config), args.root, args.interim, args.out)
    print(json.dumps({k: v for k, v in stats.items() if k != "config"}, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
