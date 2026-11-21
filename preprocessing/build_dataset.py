import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from .songs import iter_games, song_features
from .splitting import assign_splits, group_games

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "dataset_pico8_v2" / "dataset_pico8_v2"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.json"
OUT_DIR = REPO_ROOT / "data" / "splits"

SPLIT_NAMES = ("train", "val", "test")
DROP_REASONS = ("few_notes", "custom_instruments", "incompatible_speeds", "duplicate")
CONFIG_KEYS = frozenset({"min_notes", "shared_block_min_notes", "split_ratios", "force_train_group_frac", "seed"})

Config = dict[str, Any]


def load_config(path: Path) -> Config:
    with open(path) as f:
        config = json.load(f)
    missing = sorted(CONFIG_KEYS - config.keys())
    unknown = sorted(config.keys() - CONFIG_KEYS)
    if missing or unknown:
        raise ValueError(f"{path}: missing keys {missing}, unknown keys {unknown}")
    ratios = config["split_ratios"]
    if tuple(ratios) != SPLIT_NAMES or abs(sum(ratios.values()) - 1) > 1e-9:
        raise ValueError(f"{path}: split_ratios must be train, val, test in that order and sum to 1")
    return config


def drop_reason(features: dict, config: Config, kept_hashes: set[str]) -> Optional[str]:
    """First filter that rejects the song, in the order of DROP_REASONS."""
    if features["audible_notes"] < config["min_notes"]:
        return "few_notes"
    if features["custom_instruments"]:
        return "custom_instruments"
    if features["incompatible_speeds"]:
        return "incompatible_speeds"
    if features["content_hash"] in kept_hashes:
        return "duplicate"
    return None


def decide(root: Path, config: Config) -> list[dict]:
    rows = []
    kept_hashes: set[str] = set()
    for tid, songs in iter_games(root):
        for song in songs:
            features = song_features(song, config["shared_block_min_notes"])
            reason = drop_reason(features, config, kept_hashes)
            if reason is None:
                kept_hashes.add(features["content_hash"])
            rows.append({**features, "tid": tid, "decision": reason or "keep"})
    return rows


def split_kept(rows: list[dict], config: Config) -> None:
    kept = [row for row in rows if row["decision"] == "keep"]
    groups = group_games([row["tid"] for row in kept], [row["block_keys"] for row in kept])
    sizes = Counter(groups[row["tid"]] for row in kept)
    splits = assign_splits(sizes, config["split_ratios"], config["force_train_group_frac"], config["seed"])
    for row in kept:
        row["group"] = groups[row["tid"]]
        row["split"] = splits[row["group"]]


def write_splits(root: Path, rows: list[dict], out_dir: Path) -> None:
    split_of = {(row["tid"], row["song_id"]): row["split"] for row in rows if row["decision"] == "keep"}
    files = {name: open(out_dir / f"{name}.jsonl", "w") for name in SPLIT_NAMES}
    try:
        for tid, songs in iter_games(root):
            for song in songs:
                name = split_of.get((tid, song["song_id"]))
                if name is not None:
                    files[name].write(json.dumps({**song, "tid": tid}) + "\n")
    finally:
        for f in files.values():
            f.close()


def write_report(rows: list[dict], out_dir: Path) -> None:
    with open(out_dir / "filter_report.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tid", "song_id", "content_hash", "decision", "split", "group"])
        for row in rows:
            writer.writerow([row["tid"], row["song_id"], row["content_hash"], row["decision"],
                             row.get("split", ""), row.get("group", "")])


def build_stats(rows: list[dict], config: Config) -> dict:
    kept = [row for row in rows if row["decision"] == "keep"]
    reasons = Counter(row["decision"] for row in rows)
    group_songs = Counter(row["group"] for row in kept)
    games_of_group: dict[int, set[str]] = {}
    group_split = {}
    for row in kept:
        games_of_group.setdefault(row["group"], set()).add(row["tid"])
        group_split[row["group"]] = row["split"]
    largest = sorted(group_songs, key=lambda g: (-group_songs[g], g))[:5]
    return {
        "songs_in": len(rows),
        "games_in": len({row["tid"] for row in rows}),
        "dropped": {reason: reasons[reason] for reason in DROP_REASONS},
        "kept": len(kept),
        "splits": {
            name: {"songs": sum(1 for row in kept if row["split"] == name),
                   "games": len({row["tid"] for row in kept if row["split"] == name})}
            for name in SPLIT_NAMES
        },
        "groups": len(group_songs),
        "largest_groups": [{"group": g, "songs": group_songs[g], "games": len(games_of_group[g]),
                            "split": group_split[g]} for g in largest],
        "config": config,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/splits/ from the songs.json files")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="folder holding audio/<tid>/songs.json")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="where train/val/test.jsonl and the reports go")
    args = parser.parse_args()
    out_dir = args.out_dir
    try:
        config = load_config(args.config)
    except ValueError as e:
        sys.exit(str(e))

    rows = decide(args.root, config)
    if not rows:
        sys.exit(f"no audio/<tid>/songs.json under {args.root}")
    split_kept(rows, config)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_splits(args.root, rows, out_dir)
    write_report(rows, out_dir)
    stats = build_stats(rows, config)
    with open(out_dir / "build_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")

    print(f"songs in: {stats['songs_in']} ({stats['games_in']} games)")
    for reason, count in stats["dropped"].items():
        print(f"  dropped {reason}: {count}")
    print(f"kept: {stats['kept']} in {stats['groups']} groups")
    for name, split in stats["splits"].items():
        print(f"  {name}: {split['songs']} songs, {split['games']} games")
    print(f"wrote {out_dir}/")


if __name__ == "__main__":
    main()
