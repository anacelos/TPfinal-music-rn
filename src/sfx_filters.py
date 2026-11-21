"""SFX filters (noiz, buzz, detune, reverb, dampen) that songs.json does not carry.

The extraction kept each SFX's notes, speed and loop points, but not the mode byte
that holds its filters (PICO-8 0.2+). About a third of the kept songs use at least
one filter, so without it the synth plays them drier and brighter than PICO-8.

This module reads the byte from the raw carts (`p8/<tid>/<tid>.p8`, inside the
dataset zip or an extracted dataset folder) into a small table, and attaches it
to the blocks of a song so src/synth.py can play it. The table lives next to this
module (src/sfx_filters.json) so cluster jobs that render audio get it with src/:

    python src/sfx_filters.py --source ../downloads/dataset_pico8_v2.zip

Table format (src/sfx_filters.json): {tid: {sfx_id: mode_byte}}, only for SFX
whose byte has a filter bit set. Byte layout (PICO-8 memory, sfx offset 64):
bit 1 noiz, bit 2 buzz, detune = byte // 8 % 3, reverb = byte // 24 % 3,
dampen = byte // 72 % 3 (bit 0 is the editor mode and has no effect on sound).
"""

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

DEFAULT_TABLE = Path(__file__).resolve().with_name("sfx_filters.json")

Table = Dict[str, Dict[str, int]]


def describe(mode: int) -> Dict[str, int]:
    rest = mode >> 3
    return {"noiz": int(bool(mode & 2)), "buzz": int(bool(mode & 4)),
            "detune": rest % 3, "reverb": (rest // 3) % 3, "dampen": (rest // 9) % 3}


def sfx_modes(p8_text: str) -> Dict[int, int]:
    """Mode byte of every SFX line in a .p8 cart (missing lines are empty SFX, byte 0)."""
    modes, on, i = {}, False, 0
    for line in p8_text.splitlines():
        if line.startswith("__"):
            on = line.strip() == "__sfx__"
            continue
        if on and line.strip():
            if i < 64:
                modes[i] = int(line.strip()[:2], 16)
            i += 1
    return modes


def _carts(source: Path) -> Iterator[Tuple[str, str]]:
    """(tid, cart text) for every cart in the zip or folder."""
    if source.suffix == ".zip":
        with zipfile.ZipFile(source) as z:
            for name in z.namelist():
                parts = name.split("/")
                if len(parts) >= 3 and parts[-3] == "p8" and name.endswith(".p8") and parts[-1] == f"{parts[-2]}.p8":
                    yield parts[-2], z.read(name).decode("utf-8", "replace")
    else:
        for path in sorted((source / "p8").glob("*/*.p8")):
            if path.name == f"{path.parent.name}.p8":
                yield path.parent.name, path.read_text(encoding="utf-8", errors="replace")


def build_table(source: Path) -> Table:
    table: Table = {}
    for tid, text in _carts(source):
        used = {str(k): v for k, v in sfx_modes(text).items() if v & 0xFE}
        if used:
            table[tid] = used
    return table


def load_table(path: Optional[Path] = None) -> Optional[Table]:
    path = Path(path) if path else DEFAULT_TABLE
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def attach(song: Dict[str, Any], table: Optional[Table], tid: Optional[str] = None) -> Dict[str, Any]:
    """Copy of the song with `filters` in each block (and channel) that has one; generated songs pass through."""
    tid = str(tid if tid is not None else song.get("tid", ""))
    modes = (table or {}).get(tid)
    if not modes:
        return song
    blocks = {}
    for key, block in (song.get("blocks_dictionary") or {}).items():
        mode = modes.get(str(block.get("original_sfx_id")))
        blocks[key] = dict(block, filters=mode) if mode else block
    patterns = []
    for pat in song.get("patterns", []):
        chans = {}
        for name, ch in pat.get("channels", {}).items():
            mode = modes.get(str(ch.get("original_sfx_id"))) if ch.get("active") else None
            chans[name] = dict(ch, filters=mode) if mode else ch
        patterns.append(dict(pat, channels=chans))
    return dict(song, blocks_dictionary=blocks, patterns=patterns)


def main(argv=None):
    p = argparse.ArgumentParser(description="Extract SFX filter bytes from the raw carts")
    p.add_argument("--source", type=Path, required=True, help="dataset zip or extracted folder with p8/<tid>/<tid>.p8")
    p.add_argument("--out", type=Path, default=DEFAULT_TABLE)
    args = p.parse_args(argv)
    table = build_table(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(table, f, separators=(",", ":"), sort_keys=True)
    n_sfx = sum(len(v) for v in table.values())
    print(f"{len(table)} carts with filtered SFX, {n_sfx} SFX -> {args.out}")


if __name__ == "__main__":
    main()
