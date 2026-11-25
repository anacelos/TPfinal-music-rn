#!/usr/bin/env python3
"""
Sintetiza músicas em representação canônica JSONL para WAV usando src.synth.

O mesmo script pode ser usado para saídas do baseline de Markov e do modelo neural,
desde que sigam o formato canônico do projeto.

O songs.json não guarda os filtros de SFX (noiz, buzz, detune, reverb, dampen).
Para músicas reais, --sfx-filters src/sfx_filters.json anexa os filtros lidos dos
carts (ver src/sfx_filters.py), e o áudio fica mais perto do PICO-8. Músicas geradas
não têm filtros; não use a opção nelas, porque uma música que copia a estrutura de
uma música de treino (o baseline de Markov) herdaria os filtros desse jogo.
"""

import argparse
import csv
import json
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.sfx_filters import attach, load_table
from src.synth import synthesize_song_to_wav


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL canônico de entrada")
    parser.add_argument("--output-dir", required=True, help="Diretório dos WAVs")
    parser.add_argument("--prefix", default="song", help="Prefixo dos arquivos WAV")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Sintetiza apenas as primeiras N músicas",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recria WAVs que já existirem",
    )
    parser.add_argument(
        "--sfx-filters",
        default=None,
        help="Tabela de filtros de SFX (src/sfx_filters.json) para músicas reais; padrão: sem filtros",
    )
    args = parser.parse_args()

    filter_table = None
    if args.sfx_filters:
        filter_table = load_table(Path(args.sfx_filters))
        if filter_table is None:
            parser.error(f"tabela de filtros não encontrada: {args.sfx_filters}")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.csv"
    rows = []

    processed = 0
    synthesized = 0
    reused = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            if args.limit is not None and processed >= args.limit:
                break

            line = line.strip()
            if not line:
                continue

            song = json.loads(line)

            song_id = song.get("song_id", line_index)
            tid = song.get("tid")

            wav_name = f"{args.prefix}_{line_index:04d}.wav"
            wav_path = output_dir / wav_name

            if wav_path.exists() and not args.overwrite:
                duration = wav_duration(wav_path)
                reused += 1
            else:
                if filter_table is not None:
                    song = attach(song, filter_table)
                duration = synthesize_song_to_wav(song, str(wav_path))
                synthesized += 1

            rows.append(
                {
                    "index": line_index,
                    "song_id": song_id,
                    "tid": tid,
                    "wav_file": wav_name,
                    "duration_sec": f"{duration:.6f}",
                    "num_patterns": song.get("num_patterns"),
                }
            )

            processed += 1

            if processed % 100 == 0:
                print(
                    f"{processed} músicas processadas "
                    f"(sintetizadas={synthesized}, reutilizadas={reused})"
                )

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "song_id",
                "tid",
                "wav_file",
                "duration_sec",
                "num_patterns",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Músicas processadas: {processed}")
    print(f"WAVs sintetizados:   {synthesized}")
    print(f"WAVs reutilizados:   {reused}")
    print(f"Diretório:            {output_dir}")
    print(f"Manifest:             {manifest_path}")


if __name__ == "__main__":
    main()
