#!/usr/bin/env python3
"""
Exporta músicas de exemplo de cada sistema em dois formatos: JSON (o formato nativo do PICO-8
no projeto, o mesmo do `songs.json`) e WAV (`src/synth.py`, o sintetizador do PICO-8).

As músicas são as N primeiras de cada arquivo, na ordem em que foram geradas, que passam no
mesmo filtro `few_notes` do dataset (pelo menos 16 notas audíveis). Não há escolha à mão.

Como no PICO-8 a música de um jogo fica em loop, músicas curtas com loop têm o trecho do loop
repetido até passar de `--min-seconds`; músicas longas são cortadas em `--max-seconds`.

Uso (da raiz do repositório):
    python evaluation/export_samples.py \
        --system lstm-grid_block-k10=<predicoes-teste>/grid_block-k10.jsonl.gz \
        --system markov=<predicoes-teste>/markov.jsonl.gz \
        --n 5 --out-dir samples
"""

import argparse
import csv
import gzip
import json
import math
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.synth import PICO_RATE, PICO_SAMPLES_PER_TICK, synthesize_song

TICK_SECONDS = PICO_SAMPLES_PER_TICK / PICO_RATE  # um tick do PICO-8 (~8,3 ms)
MIN_AUDIBLE_NOTES = 16  # o filtro few_notes do preprocessing


def read_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def audible_notes(song) -> int:
    return sum(1 for p in song.get("patterns", []) for ch in (p.get("channels") or {}).values()
               if ch and ch.get("active") for n in (ch.get("notes_unrolled") or []) if n.get("volume", 0) > 0)


def pattern_ticks(patterns) -> int:
    return sum(max(0, int(p.get("pattern_ticks", 512))) for p in patterns)


def loop_repeats(song, min_seconds: float, max_repeats: int = 32) -> int:
    """Quantas repetições extras do loop (a mesma regra do synth) até a música passar de `min_seconds`."""
    loop_start = song.get("loop_start_pattern")
    if song.get("end_reason") != "loop_end" or loop_start is None:
        return 0
    loop = pattern_ticks([p for p in song.get("patterns", []) if p.get("pattern_id", 0) >= loop_start])
    if loop <= 0:
        return 0
    missing = min_seconds / TICK_SECONDS - pattern_ticks(song.get("patterns", []))
    return int(min(max_repeats, max(0, math.ceil(missing / loop))))


def write_wav(audio: np.ndarray, path: Path, rate: int) -> None:
    pcm = np.clip(np.round(audio * 32767.0), -32767, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())


def export_system(name: str, path: Path, out_dir: Path, n: int, min_seconds: float, max_seconds: float):
    rows = []
    sys_dir = out_dir / name
    sys_dir.mkdir(parents=True, exist_ok=True)
    for index, song in enumerate(read_jsonl(path)):
        if len(rows) >= n:
            break
        if audible_notes(song) < MIN_AUDIBLE_NOTES:
            continue
        stem = f"{name}_{len(rows) + 1:02d}"
        repeats = loop_repeats(song, min_seconds)
        audio = synthesize_song(song, loop_extra_repeats=repeats, seed=0, rate=PICO_RATE)
        full_seconds = len(audio) / PICO_RATE
        audio = audio[: int(round(max_seconds * PICO_RATE))]
        (sys_dir / f"{stem}.json").write_text(json.dumps(song, ensure_ascii=False), encoding="utf-8")
        write_wav(audio, sys_dir / f"{stem}.wav", PICO_RATE)

        rows.append({
            "system": name,
            "file": f"{name}/{stem}",
            "source_index": index,
            "song_id": song.get("song_id"),
            "patterns": len(song.get("patterns", [])),
            "audible_notes": audible_notes(song),
            "one_pass_seconds": round(pattern_ticks(song.get("patterns", [])) * TICK_SECONDS, 1),
            "loop_repeats": repeats,
            "seconds": round(len(audio) / PICO_RATE, 1),
            "cut": full_seconds > max_seconds,
        })
        print(f"[samples] {stem}: linha {index} do arquivo, {rows[-1]['seconds']} s, "
              f"{repeats} repetições do loop{', cortada' if rows[-1]['cut'] else ''}", flush=True)
    if len(rows) < n:
        print(f"[samples] aviso: {name} só tem {len(rows)} músicas com {MIN_AUDIBLE_NOTES}+ notas", flush=True)
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Exporta exemplos em JSON e WAV")
    parser.add_argument("--system", action="append", required=True,
                        help="nome=arquivo.jsonl[.gz], repetível (ex.: markov=predicoes/markov.jsonl.gz)")
    parser.add_argument("--n", type=int, default=5, help="músicas por sistema")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "samples")
    parser.add_argument("--min-seconds", type=float, default=20.0,
                        help="repete o loop das músicas mais curtas que isso")
    parser.add_argument("--max-seconds", type=float, default=60.0, help="corta as músicas mais longas que isso")
    args = parser.parse_args(argv)

    rows = []
    for spec in args.system:
        name, path = spec.split("=", 1)
        rows += export_system(name, Path(path), args.out_dir, args.n, args.min_seconds, args.max_seconds)
    with open(args.out_dir / "index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[samples] {len(rows)} exemplos em {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
