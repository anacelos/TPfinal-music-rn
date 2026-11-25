"""
Render decoded songs (samples.jsonl, or a split .jsonl/.jsonl.gz) to WAV with src/synth.py.

Generated songs can be empty or absurdly long (a few WAIT/REF_SPEED tokens can
mean minutes of audio), so every song is cut at --max-seconds and empty ones are
listed instead of crashing the run. Songs render in parallel (one process per CPU
of the job).

Real songs (those without a "system" key) get their SFX filters (reverb, dampen,
...) from src/sfx_filters.json when it exists, since songs.json does not carry
them; generated songs have none. --sfx-filters none renders real songs without.

Example:
  python3 src/render.py --input runs/nlm/samples-p095/samples.jsonl --out-dir runs/nlm/wav --limit 10
"""

import argparse
import gzip
import json
import os
from multiprocessing import Pool
from pathlib import Path

try:
    from .sfx_filters import DEFAULT_TABLE, attach, load_table
    from .synth import SAMPLE_RATE, SAMPLES_PER_TICK, synthesize_song_to_wav
except ImportError:
    from sfx_filters import DEFAULT_TABLE, attach, load_table
    from synth import SAMPLE_RATE, SAMPLES_PER_TICK, synthesize_song_to_wav


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cut_song(song, max_seconds):
    """Keep whole patterns until the budget is spent; the last one is shortened."""
    budget = int(max_seconds * SAMPLE_RATE / SAMPLES_PER_TICK)
    kept, used = [], 0
    for pat in song.get("patterns", []):
        ticks = int(pat.get("pattern_ticks", 512))
        if used >= budget:
            break
        if used + ticks > budget:
            pat = dict(pat, pattern_ticks=budget - used)
        kept.append(pat)
        used += int(pat["pattern_ticks"])
    return dict(song, patterns=kept), used > 0


def _render_one(job):
    song, path, max_seconds, loops = job
    cut, ok = cut_song(song, max_seconds)
    if not ok:
        return {"song_id": song.get("song_id"), "wav": None, "seconds": 0.0, "empty": True}
    secs = synthesize_song_to_wav(cut, path, loop_extra_repeats=loops)
    return {"song_id": song.get("song_id"), "wav": os.path.basename(path), "seconds": round(secs, 2),
            "empty": False, "cut": secs >= max_seconds - 1e-3}


def render(songs, out_dir, max_seconds=60.0, loops=0, workers=None, sfx_filters=DEFAULT_TABLE):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = load_table(sfx_filters) if sfx_filters else None
    jobs = []
    for i, song in enumerate(songs):
        if table and "system" not in song:
            song = attach(song, table)
        # generated songs carry a unique song_id; real songs are only unique as tid + song_id
        name = str(song["song_id"]) if "system" in song else f"{song.get('tid', 'song')}_{song.get('song_id', i)}"
        jobs.append((song, str(out_dir / f"{name}.wav"), max_seconds, loops))
    workers = workers or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    with Pool(max(1, min(workers, len(jobs)))) as pool:
        rows = pool.map(_render_one, jobs)
    (out_dir / "render_index.json").write_text(json.dumps(rows, indent=2))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description="Render decoded PICO-8 songs to WAV")
    p.add_argument("--input", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-seconds", type=float, default=60.0)
    p.add_argument("--loops", type=int, default=0, help="extra repeats of the loop section")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--sfx-filters", default=str(DEFAULT_TABLE),
                   help="table of SFX filters for real songs (src/sfx_filters.py), or 'none'")
    args = p.parse_args(argv)
    songs = read_jsonl(args.input)[: args.limit]
    filters = None if args.sfx_filters.lower() == "none" else args.sfx_filters
    rows = render(songs, args.out_dir, args.max_seconds, args.loops, args.workers, filters)
    print(json.dumps({"rendered": sum(not r["empty"] for r in rows), "empty": sum(r["empty"] for r in rows)}))


if __name__ == "__main__":
    main()
