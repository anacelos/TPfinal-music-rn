#!/usr/bin/env python3
"""
Renderiza clipes de 10 s das músicas reais e geradas e calcula FAD, CLAP score e KLD.

É o script que produziu as métricas de áudio de docs/RESULTADOS.md. Os conjuntos são:

  real_test   400 músicas sorteadas do teste (semente 831): a referência;
  real_val    200 músicas sorteadas da validação: o piso, real contra real;
  <sistema>   as 200 primeiras músicas de cada samples.jsonl passado em --system.

Cada música é cortada em --seconds (10 s, a janela de entrada do CLAP) e
sintetizada pelo src/render.py. As reais recebem os filtros de SFX do jogo
(src/sfx_filters.json); as geradas não têm filtros. --no-sfx-filters sintetiza
as reais também sem filtros, como controle. Depois o evaluation/audio_metrics.py
compara cada conjunto com real_test.

Uso, a partir da raiz do repositório (pede evaluation/requirements.txt e torch):

    python evaluation/audio_clips_eval.py \
      --test data/splits/test.jsonl --val data/splits/val.jsonl \
      --system lstm-grid_block=outputs/grid_block/samples/samples.jsonl \
      --system markov=results/markov/samples.jsonl \
      --out-dir results/audio

Saídas em --out-dir: audio_metrics.csv, audio_metrics.json, embeddings.npz e
render/<conjunto>.json (um registro por clipe). Os WAVs ficam em --wav-dir
(padrão: <out-dir>/wav). --quick usa 24 e 12 músicas, só para testar o caminho.
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "evaluation"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import audio_metrics  # noqa: E402
import render  # noqa: E402

SEED = 831


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", default="data/splits/test.jsonl")
    ap.add_argument("--val", default="data/splits/val.jsonl")
    ap.add_argument("--system", action="append", default=[], help="nome=caminho do samples.jsonl; pode repetir")
    ap.add_argument("--n-real", type=int, default=400, help="clipes do teste real (a referência)")
    ap.add_argument("--n", type=int, default=200, help="clipes de cada outro conjunto")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out-dir", type=Path, default=Path("results/audio"))
    ap.add_argument("--wav-dir", type=Path, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-sfx-filters", action="store_true", help="sintetiza as músicas reais sem os filtros de SFX")
    ap.add_argument("--render-only", action="store_true", help="só renderiza os clipes, sem calcular as métricas")
    args = ap.parse_args(argv)
    if args.quick:
        args.n_real, args.n = 24, 12
    out = args.out_dir
    wav = args.wav_dir or out / "wav"
    rng = random.Random(SEED)

    sets = {}
    for name, path, n in (("real_test", args.test, args.n_real), ("real_val", args.val, args.n)):
        songs = render.read_jsonl(path)
        sets[name] = rng.sample(songs, min(n, len(songs)))  # os arquivos de split vêm ordenados por jogo
    for s in args.system:
        name, path = s.split("=", 1)
        sets[name] = render.read_jsonl(path)[: args.n]

    (out / "render").mkdir(parents=True, exist_ok=True)
    for name, songs in sets.items():
        rows = render.render(songs, wav / name, max_seconds=args.seconds,
                             sfx_filters=None if args.no_sfx_filters else render.DEFAULT_TABLE)
        (out / "render" / f"{name}.json").write_text(json.dumps(rows))
        print(f"[render] {name}: {sum(not r['empty'] for r in rows)} clipes, {sum(r['empty'] for r in rows)} vazios",
              flush=True)
    if args.render_only:
        return

    audio_metrics.main(["--reference", f"real_test={wav / 'real_test'}", "--out-dir", str(out),
                        "--boot-size", str(min(100, args.n)), "--boot-rounds", "5" if args.quick else "20"]
                       + [a for name in sets if name != "real_test" for a in ("--set", f"{name}={wav / name}")])


if __name__ == "__main__":
    main()
