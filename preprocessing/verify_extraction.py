"""Check songs.json against the raw carts it was extracted from.

For every kept song (data/splits/split_manifest.csv), parse `__sfx__` and `__music__`
from `p8/<tid>/<tid>.p8` and compare with the song's patterns and blocks:

* channel assignment (which SFX each channel plays, muted channels);
* SFX notes (pitch, instrument, volume, effect), speed and loop points;
* pattern length, against the PICO-8 rule as reconstructed by zepto8: the leftmost
  non-looping channel sets it; if every channel loops, the slowest one does
  (32 notes at its speed);
* SFX filters (noiz, buzz, detune, reverb, dampen), which songs.json does not store.

    python -m preprocessing.verify_extraction --source ../downloads/dataset_pico8_v2.zip
"""

import argparse
import collections
import csv
import json
import zipfile
from pathlib import Path
import sys
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
from sfx_filters import describe  # noqa: E402
MANIFEST = REPO_ROOT / "data" / "splits" / "split_manifest.csv"
CHANNELS = ("ch1", "ch2", "ch3", "ch4")
EMPTY_SFX = {"mode": 0, "speed": 0, "loop_start": 0, "loop_end": 0, "notes": [(0, 0, 0, 0)] * 32}


def _section(text: str, name: str) -> List[str]:
    out, on = [], False
    for line in text.splitlines():
        if line.startswith("__"):
            on = line.strip() == f"__{name}__"
            continue
        if on and line.strip():
            out.append(line.strip())
    return out


def parse_cart(text: str):
    """(sfx, music) of a .p8 cart; carts omit trailing empty SFX, which read as zeros."""
    sfx = {}
    for i, line in enumerate(_section(text, "sfx")[:64]):
        head = bytes.fromhex(line[:8])
        notes = [(int(line[8 + 5 * k:10 + 5 * k], 16), int(line[10 + 5 * k], 16),
                  int(line[11 + 5 * k], 16), int(line[12 + 5 * k], 16)) for k in range(32)]
        sfx[i] = {"mode": head[0], "speed": head[1], "loop_start": head[2], "loop_end": head[3], "notes": notes}
    for i in range(64):
        sfx.setdefault(i, EMPTY_SFX)
    music = {}
    for i, line in enumerate(_section(text, "music")[:64]):
        flags, chans = line.split()
        music[i] = {"flags": int(flags, 16), "ch": [int(chans[2 * k:2 * k + 2], 16) for k in range(4)]}
    return sfx, music


def pattern_ticks(music: dict, sfx: dict, pid: int) -> Optional[int]:
    chans = [c & 0x3F for c in music[pid]["ch"] if not c & 0x40]
    looping = []
    for sid in chans:
        s = sfx[sid]
        speed = max(1, s["speed"])
        if s["loop_end"] > 0 and s["loop_end"] > s["loop_start"]:
            looping.append(32 * speed)
            continue
        length = s["loop_start"] if (s["loop_end"] == 0 and s["loop_start"] > 0) else 32
        return min(32, length) * speed
    return max(looping) if looping else None


def verify(source: Path, manifest: Path = MANIFEST, limit: Optional[int] = None) -> Dict:
    kept = collections.defaultdict(set)
    with open(manifest, newline="") as f:
        for row in csv.DictReader(f):
            kept[row["tid"]].add(int(row["song_id"]))
    tids = sorted(kept, key=int)[:limit]
    bad = collections.Counter()
    filters = collections.Counter()
    examples = collections.defaultdict(list)
    songs_total = songs_filtered = 0
    with zipfile.ZipFile(source) as z:
        names = set(z.namelist())
        prefix = next(n.split("/")[0] for n in names if "/p8/" in n)
        for tid in tids:
            path = f"{prefix}/p8/{tid}/{tid}.p8"
            if path not in names:
                bad["missing_cart"] += 1
                continue
            sfx, music = parse_cart(z.read(path).decode("utf-8", "replace"))
            songs = {s["song_id"]: s for s in json.loads(z.read(f"{prefix}/audio/{tid}/songs.json"))["songs"]}
            for sid in sorted(kept[tid]):
                song, songs_total = songs[sid], songs_total + 1
                ok = dict.fromkeys(("channels", "notes", "speed_loop", "pattern_ticks"), True)
                used = set()
                for pat in song["patterns"]:
                    raw = music.get(pat["pattern_id"])
                    if raw is None:
                        ok["channels"] = False
                        continue
                    for k, name in enumerate(CHANNELS):
                        ch, byte = pat["channels"][name], raw["ch"][k]
                        on = not byte & 0x40
                        if bool(ch["active"]) != on or (on and ch["original_sfx_id"] != byte & 0x3F):
                            ok["channels"] = False
                        if not (on and ch["active"]):
                            continue
                        s = sfx[byte & 0x3F]
                        used.add(byte & 0x3F)
                        block = song["blocks_dictionary"][str(ch["block_id"])]
                        if ch["speed"] != s["speed"] or (block["loop_start"], block["loop_end"]) != (s["loop_start"], s["loop_end"]):
                            ok["speed_loop"] = False
                        if [(n["pitch"], n["instrument"], n["volume"], n["effect"]) for n in block["notes"]] != s["notes"]:
                            ok["notes"] = False
                    want = pattern_ticks(music, sfx, pat["pattern_id"])
                    if want is not None and want != pat["pattern_ticks"]:
                        ok["pattern_ticks"] = False
                for key, good in ok.items():
                    if not good:
                        bad[key] += 1
                        if len(examples[key]) < 5:
                            examples[key].append(f"{tid}/{sid}")
                kinds = {k for u in used for k, v in describe(sfx[u]["mode"]).items() if v}
                if kinds:
                    songs_filtered += 1
                    filters.update(kinds)
    return {"games": len(tids), "songs": songs_total, "mismatch": dict(bad), "examples": dict(examples),
            "songs_with_filters": songs_filtered, "filters": dict(filters)}


def main(argv=None):
    p = argparse.ArgumentParser(description="Compare songs.json with the raw .p8 carts")
    p.add_argument("--source", type=Path, required=True, help="dataset zip (with p8/ and audio/)")
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--limit", type=int, default=None, help="first N games only")
    args = p.parse_args(argv)
    r = verify(args.source, args.manifest, args.limit)
    n = r["songs"]
    print(f"{r['games']} games, {n} kept songs")
    for key in ("channels", "notes", "speed_loop", "pattern_ticks"):
        k = r["mismatch"].get(key, 0)
        print(f"  differs in {key}: {k} songs ({100 * k / n:.1f}%)  e.g. {r['examples'].get(key, [])[:3]}")
    print(f"songs using SFX filters songs.json does not store: {r['songs_with_filters']} "
          f"({100 * r['songs_with_filters'] / n:.1f}%)")
    for k, v in sorted(r["filters"].items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v} ({100 * v / n:.1f}%)")


if __name__ == "__main__":
    main()
