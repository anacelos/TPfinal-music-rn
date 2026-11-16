import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "dataset_pico8_v2" / "dataset_pico8_v2"
SPLIT_DIR = REPO_ROOT / "data" / "splits"
SPLITS = ("train", "val", "test")
EXPECTED_SONGS = 24077
EXPECTED_GAMES = 8367
EXPECTED_KEPT = 12384
BALANCE_TOLERANCE = 0.01
# The spec values, kept apart from the pipeline's config so a wrong config cannot vouch for itself.
SPEC_CONFIG = {
    "min_notes": 16,
    "shared_block_min_notes": 4,
    "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
    "force_train_group_frac": 0.01,
    "seed": 831,
}
MODES = ("nlm", "grid_unrolled", "grid_block")
SILENT = None


def sha1_json(value) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def channel_list(pattern: dict) -> list[dict]:
    return [pattern["channels"][f"ch{i}"] for i in range(1, 5)]


def recompute(song: dict, min_notes: int, block_min_notes: int) -> dict:
    audible = 0
    incompatible = False
    for pattern in song["patterns"]:
        active = [ch for ch in channel_list(pattern) if ch["active"]]
        audible += sum(n["volume"] > 0 for ch in active for n in ch["notes_unrolled"])
        speeds = sorted(ch["speed"] for ch in active if ch["speed"] > 0)
        if speeds and not all(s % speeds[0] == 0 for s in speeds):
            incompatible = True
    key = [
        song["loop_start_pattern"] - song["start_pattern"],
        song["end_reason"],
        [[p["pattern_ticks"], [[ch["speed"], ch["notes_unrolled"]] if ch["active"] else None
                               for ch in channel_list(p)]] for p in song["patterns"]],
    ]
    blocks = {sha1_json(b["notes"]) for b in song["blocks_dictionary"].values()
              if sum(n["volume"] > 0 for n in b["notes"]) >= block_min_notes}
    if audible < min_notes:
        reason = "few_notes"
    elif song["uses_custom_instruments"]:
        reason = "custom_instruments"
    elif incompatible:
        reason = "incompatible_speeds"
    else:
        reason = None
    return {"reason": reason, "hash": sha1_json(key), "blocks": blocks}


def corpus(root: Path):
    for path in sorted((root / "audio").glob("*/songs.json"), key=lambda p: int(p.parent.name)):
        with open(path) as f:
            data = json.load(f)
        for song in sorted(data["songs"], key=lambda s: s["song_id"]):
            yield str(data["tid"]), song


class Checks:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def expect(self, check: str, ok: bool, detail: str) -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {check}: {detail}")
        if not ok:
            self.failures.append(check)


def run_checks(root: Path) -> list[str]:
    checks = Checks()
    with open(SPLIT_DIR / "build_stats.json") as f:
        stats = json.load(f)
    config = SPEC_CONFIG
    checks.expect("4 filters", stats["config"] == SPEC_CONFIG,
                  f"build used the spec config: {stats['config'] == SPEC_CONFIG}")
    with open(SPLIT_DIR / "filter_report.csv", newline="") as f:
        report = list(csv.DictReader(f))
    report_by_key = {(r["tid"], int(r["song_id"])): r for r in report}

    checks.expect("2 coverage", len(report) == EXPECTED_SONGS and len(report_by_key) == EXPECTED_SONGS
                  and len({r["tid"] for r in report}) == EXPECTED_GAMES,
                  f"{len(report)} rows, {len(report_by_key)} unique, {len({r['tid'] for r in report})} tids")

    readers = {name: open(SPLIT_DIR / f"{name}.jsonl") for name in SPLITS}
    reasons = Counter()
    kept_hashes: set[str] = set()
    split_hashes = Counter()
    block_splits: dict[str, set[str]] = {}
    decision_mismatch = []
    line_mismatch = []
    format_mismatch = []
    corpus_keys = set()
    try:
        for tid, song in corpus(root):
            key = (tid, song["song_id"])
            corpus_keys.add(key)
            info = recompute(song, config["min_notes"], config["shared_block_min_notes"])
            reason = info["reason"]
            if reason is None and info["hash"] in kept_hashes:
                reason = "duplicate"
            if reason is None:
                kept_hashes.add(info["hash"])
            reasons[reason or "keep"] += 1

            row = report_by_key.get(key)
            if row is None or row["decision"] != (reason or "keep") or row["content_hash"] != info["hash"]:
                decision_mismatch.append(key)
            if row is None or row["decision"] != "keep":
                continue
            split_hashes[info["hash"]] += 1
            line = readers[row["split"]].readline()
            if not line:
                line_mismatch.append(key)
                continue
            out = json.loads(line)
            if (out.get("tid"), out.get("song_id")) != key:
                line_mismatch.append(key)
                continue
            del out["tid"]
            if out != song:
                format_mismatch.append(key)
            for block in info["blocks"]:
                block_splits.setdefault(block, set()).add(row["split"])
        leftover = {name: sum(1 for _ in r) for name, r in readers.items()}
    finally:
        for r in readers.values():
            r.close()

    hash_repeats = sum(n - 1 for n in split_hashes.values())
    checks.expect("2 coverage", corpus_keys == set(report_by_key),
                  f"report keys equal songs.json keys ({len(corpus_keys)})")
    checks.expect("2 coverage", not line_mismatch and not any(leftover.values()),
                  f"jsonl lines match keep rows in their split file: {len(line_mismatch)} mismatches, "
                  f"leftover lines {leftover}")

    leaking_blocks = sum(len(s) > 1 for s in block_splits.values())
    checks.expect("3 leakage", hash_repeats == 0,
                  f"{hash_repeats} repeated content hashes among the {sum(split_hashes.values())} songs in the splits")
    checks.expect("3 leakage", leaking_blocks == 0,
                  f"{leaking_blocks} of {len(block_splits)} qualifying blocks appear in more than one split")
    kept_rows = [r for r in report if r["decision"] == "keep"]
    group_sizes = Counter(r["group"] for r in kept_rows)
    group_split = {r["group"]: r["split"] for r in kept_rows}
    limit = config["force_train_group_frac"] * len(kept_rows)
    big = [g for g, n in group_sizes.items() if n > limit]
    checks.expect("3 leakage", all(group_split[g] == "train" for g in big),
                  f"{len(big)} groups over {limit:.1f} songs, all in train: "
                  f"{all(group_split[g] == 'train' for g in big)}")

    dropped = {reason: reasons[reason] for reason in stats["dropped"]}
    checks.expect("4 filters", dropped == stats["dropped"] and reasons["keep"] == stats["kept"] == EXPECTED_KEPT,
                  f"recomputed {dict(dropped)} kept {reasons['keep']}; build_stats {stats['dropped']} "
                  f"kept {stats['kept']}")
    checks.expect("4 filters", not decision_mismatch,
                  f"{len(decision_mismatch)} songs whose decision or hash differs from filter_report.csv")

    checks.expect("5 format", not format_mismatch,
                  f"{len(format_mismatch)} jsonl lines differ from their songs.json dict")

    total = len(kept_rows)
    shares = {name: sum(r["split"] == name for r in kept_rows) / total for name in SPLITS}
    targets = config["split_ratios"]
    checks.expect("6 balance", all(abs(shares[n] - targets[n]) <= BALANCE_TOLERANCE for n in SPLITS),
                  ", ".join(f"{n} {shares[n]:.2%}" for n in SPLITS))
    return checks.failures


def note_state(note: dict):
    if note["volume"] <= 0:
        return SILENT
    return (note["pitch"], note["instrument"], note["volume"], note["effect"])


def timeline(patterns: list[dict]) -> list[list]:
    """Per channel, run-length encoded (state, ticks) over the whole song: equal runs means equal ticks."""
    runs: list[list] = [[] for _ in range(4)]

    def push(ch: int, state, length: int) -> None:
        if length <= 0:
            return
        if runs[ch] and runs[ch][-1][0] == state:
            runs[ch][-1][1] += length
        else:
            runs[ch].append([state, length])

    for pattern in patterns:
        ticks = pattern["pattern_ticks"]
        for i, ch in enumerate(channel_list(pattern)):
            covered = 0
            if ch["active"]:
                speed = max(1, int(ch["speed"] or 1))
                for k, note in enumerate(ch["notes_unrolled"]):
                    start = k * speed
                    if start >= ticks:
                        break
                    length = min(speed, ticks - start)
                    push(i, note_state(note), length)
                    covered = start + length
            push(i, SILENT, ticks - covered)
    return runs


def expected_loop_start(patterns: list[dict], loop_start_pattern: int):
    """Index of the first sounding pattern at or after the loop start, the pattern the loop jumps to in practice."""
    for i, p in enumerate(patterns):
        if p["pattern_id"] >= loop_start_pattern:
            return i
    return None


_tokenizers = {}


def _init_worker() -> None:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from tokenizer import Pico8Tokenizer
    for mode in MODES:
        _tokenizers[mode] = Pico8Tokenizer(mode=mode, unroll_song_loop=False)


def roundtrip_song(line: str):
    song = json.loads(line)
    sounding = [p for p in song["patterns"] if p["pattern_ticks"] > 0]
    want_ticks = [p["pattern_ticks"] for p in sounding]
    want_runs = timeline(sounding)
    want_loop = song["end_reason"] == "loop_end"
    want_start = expected_loop_start(sounding, song["loop_start_pattern"])
    loop_on_zero_tick = any(p["pattern_id"] == song["loop_start_pattern"] and p["pattern_ticks"] == 0
                            for p in song["patterns"])
    diffs = {}
    for mode, tok in _tokenizers.items():
        decoded = tok.decode_ids(tok.encode_song_to_ids(song))
        got = [p for p in decoded["patterns"] if p["pattern_ticks"] > 0]
        problems = []
        if [p["pattern_ticks"] for p in got] != want_ticks:
            problems.append("pattern_ticks")
        if timeline(got) != want_runs:
            problems.append("notes")
        if (decoded["end_reason"] == "loop_end") != want_loop:
            problems.append("end_reason")
        if decoded["loop_start_pattern"] != want_start:
            problems.append("loop_start")
        if problems:
            diffs[mode] = problems
    return song["tid"], song["song_id"], loop_on_zero_tick, diffs


def run_roundtrip(workers: int) -> list[str]:
    def lines():
        for name in SPLITS:
            with open(SPLIT_DIR / f"{name}.jsonl") as f:
                yield from f

    songs = 0
    known = set()
    failures: dict[str, list] = {mode: [] for mode in MODES}
    with Pool(workers, initializer=_init_worker) as pool:
        for tid, song_id, loop_on_zero_tick, diffs in pool.imap(roundtrip_song, lines(), chunksize=32):
            songs += 1
            if loop_on_zero_tick:
                known.add((tid, song_id))
            for mode, problems in diffs.items():
                failures[mode].append((tid, song_id, problems))

    print(f"round trip over {songs} songs; songs with the loop start on a 0-tick pattern: {sorted(known)}")
    failed = []
    for mode in MODES:
        unexpected = [f for f in failures[mode] if (f[0], f[1]) not in known or f[2] != ["loop_start"]]
        print(f"[{mode}] {len(failures[mode])} mismatches, {len(unexpected)} other than the loop start "
              f"of the 0-tick loop start songs")
        for tid, song_id, problems in failures[mode]:
            print(f"    tid {tid} song {song_id}: {', '.join(problems)}")
        if mode != "grid_unrolled" and unexpected:
            failed.append(f"9 roundtrip {mode}")
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Acceptance checks for data/splits/")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--roundtrip", action="store_true", help="also encode and decode every split song")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    failures = run_checks(args.root)
    if args.roundtrip:
        failures += run_roundtrip(args.workers)
    print("all checks passed" if not failures else f"failed: {sorted(set(failures))}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
