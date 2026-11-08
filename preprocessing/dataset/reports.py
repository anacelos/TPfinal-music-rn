import csv
import hashlib
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Optional

from .config import Config
from .filters import SongKey, song_key
from .storage import tid_sort_key


def write_splits(out: Path, records: list[dict], split_of_tid: dict[str, str],
                 config: Config) -> tuple[Counter, dict[str, list[int]]]:
    counts = Counter()
    steps = defaultdict(list)
    with ExitStack() as stack:
        files = {split: stack.enter_context(open(out / f"{split}.jsonl", "w"))
                 for split in config["split_ratios"]}
        for record in records:
            split = split_of_tid[record["tid"]]
            files[split].write(json.dumps({**record, "split": split}, separators=(",", ":")) + "\n")
            counts[split] += 1
            steps[split].append(record["n_steps"])
    return counts, steps


def write_filter_report(out: Path, songs: list[dict], reasons: dict[SongKey, Optional[str]],
                        split_of_tid: dict[str, str]) -> None:
    with open(out / "filter_report.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tid", "song_idx", "content_hash", "decision", "split"])
        for song in songs:
            reason = reasons[song_key(song)]
            writer.writerow([song["tid"], song["song_idx"], song["content_hash"], reason or "keep",
                             split_of_tid.get(song["tid"], "") if reason is None else ""])


def write_split_tids(out: Path, split_of_tid: dict[str, str], group_of: dict[str, str]) -> None:
    with open(out / "split_tids.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tid", "group", "split"])
        for tid in sorted(split_of_tid, key=tid_sort_key):
            writer.writerow([tid, group_of[tid], split_of_tid[tid]])


def median(values: list[int]) -> Optional[int]:
    values = sorted(values)
    return values[len(values) // 2] if values else None


def build_stats(config: Config, songs: list[dict], reasons: dict[SongKey, Optional[str]],
                kept: list[dict], kept_tids: list[str], group_sizes: Counter,
                split_of_tid: dict[str, str], counts: Counter, steps: dict[str, list[int]]) -> dict:
    return {
        "config": config,
        "songs_in": len(songs),
        "drop_reasons": dict(Counter(r for r in reasons.values() if r)),
        "songs_kept": len(kept),
        "tids_kept": len(kept_tids),
        "groups": len(group_sizes),
        "largest_groups": [[g, n] for g, n in group_sizes.most_common(5)],
        "split_songs": dict(counts),
        "split_tids": dict(Counter(split_of_tid[t] for t in kept_tids)),
        "split_median_steps": {k: median(v) for k, v in steps.items()},
        "config_sha": hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10],
    }
