"""Build the small sample of the dataset that ships in the repository (data/amostra/).

The full dataset_pico8_v2 (8,367 games, 2.8 GB of songs.json) is not distributed; this
script picks N games at random (fixed seed) and zips only their audio/<tid>/songs.json,
under the same dataset_pico8_v2/ prefix as the full zip, so the README commands run
unchanged after `unzip data/amostra/dataset_pico8_amostra.zip -d dataset_pico8_v2`.

    python3 -s -m preprocessing.make_sample --n 300 --seed 831
"""

import argparse
import random
import zipfile
from pathlib import Path

from .build_dataset import DEFAULT_ROOT, REPO_ROOT

DEFAULT_OUT = REPO_ROOT / "data" / "amostra" / "dataset_pico8_amostra.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description="Zip a random sample of audio/<tid>/songs.json")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="folder holding audio/<tid>/songs.json")
    parser.add_argument("--n", type=int, default=300, help="number of games")
    parser.add_argument("--seed", type=int, default=831)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    tids = sorted((p.parent.name for p in (args.root / "audio").glob("*/songs.json")), key=int)
    if not tids:
        raise SystemExit(f"no audio/<tid>/songs.json under {args.root}")
    picked = sorted(random.Random(args.seed).sample(tids, min(args.n, len(tids))), key=int)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for tid in picked:
            z.write(args.root / "audio" / tid / "songs.json", f"dataset_pico8_v2/audio/{tid}/songs.json")
    print(f"{len(picked)} games of {len(tids)} -> {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
