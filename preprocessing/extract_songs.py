import argparse
import json
import os
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from .lua_calls import music_call_rows, read_music_calls
from .paths import DATASET, INTERIM
from .pico8.audio_json import load_audio
from .pico8.playback import segment_songs
from .song_features import PrefixStatsCache, build_song

CALL_COLUMNS = ["tid", "pattern", "class", "song_idx"]
POOL_CHUNK_SIZE = 64


def list_tids(root: Path) -> list[str]:
    tids = [t for t in os.listdir(root / "audio") if (root / "audio" / t / "audio.json").exists()]
    return sorted(tids, key=lambda t: (not t.isdigit(), int(t) if t.isdigit() else 0, t))


def extract_tid(root: Path, tid: str) -> tuple[list[dict], dict, list[dict]]:
    """Songs, one summary row and the literal music() call rows of a single game."""
    try:
        audio = load_audio(root / "audio" / tid / "audio.json")
    except (OSError, ValueError):
        return [], {"tid": tid, "read_error": 1}, []
    segments = segment_songs(audio.patterns)
    cache = PrefixStatsCache(audio.sfx)
    songs = [build_song(tid, i, seg, audio.patterns, audio.sfx, cache) for i, seg in enumerate(segments)]
    lua = read_music_calls(root / "lua", tid)
    targets = lua.literal_targets if lua else []
    summary = {
        "tid": tid,
        "read_error": 0,
        "n_sfx": audio.n_sfx_entries,
        "n_patterns_table": len(audio.stored_patterns),
        "n_empty_patterns": sum(p.is_empty for p in audio.stored_patterns),
        "n_songs": len(songs),
        "has_lua": int(lua is not None),
        "n_lua_literal_targets": sum(t >= 0 for t in targets),
        "n_lua_nonliteral_calls": lua.n_nonliteral if lua else 0,
    }
    return songs, summary, music_call_rows(tid, targets, segments)


def song_summary_row(song: dict) -> dict:
    row = {k: v for k, v in song.items() if not isinstance(v, (list, dict))}
    row["pattern_ids"] = " ".join(map(str, song["pattern_ids"]))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DATASET)
    parser.add_argument("--out", type=Path, default=INTERIM)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tids = list_tids(args.root)
    print(f"{len(tids)} TIDs", file=sys.stderr)

    n_songs = 0
    summary_rows, tid_rows, call_rows = [], [], []
    with Pool(args.workers) as pool, open(args.out / "songs.jsonl", "w") as songs_file:
        for songs, tid_row, calls in pool.imap(partial(extract_tid, args.root), tids,
                                               chunksize=POOL_CHUNK_SIZE):
            for song in songs:
                songs_file.write(json.dumps(song, separators=(",", ":")) + "\n")
                summary_rows.append(song_summary_row(song))
            n_songs += len(songs)
            tid_rows.append(tid_row)
            call_rows.extend(calls)

    pd.DataFrame(summary_rows).to_csv(args.out / "songs_summary.csv", index=False)
    pd.DataFrame(tid_rows).to_csv(args.out / "tids_summary.csv", index=False)
    pd.DataFrame(call_rows, columns=CALL_COLUMNS).to_csv(args.out / "music_calls.csv", index=False)
    print(f"{n_songs} songs written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
