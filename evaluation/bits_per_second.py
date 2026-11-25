#!/usr/bin/env python3
"""
Bits por segundo de um checkpoint do src/train.py em músicas inteiras.

A perda do treino é por janela e por token: o estado da LSTM recomeça a cada
janela, o fim das músicas longas entra em duas janelas e cada tokenizador gasta
um número diferente de tokens por segundo de música. Para comparar
tokenizadores, este script lê cada música do split inteira (o estado da LSTM
passa de um trecho para o seguinte), soma a NLL de todos os alvos e divide pela
duração da música, tirada dos ticks decodificados. O cálculo está em src/nll.py.

    bits/s = (soma da NLL em nats) / ln 2 / (soma das durações em segundos)

Uso, a partir da raiz do repositório:

    python evaluation/bits_per_second.py \
      --checkpoint outputs/grid_block/best.pt \
      --data-dir data/processed/grid_block \
      --split val

O JSON impresso (e gravado com --output) traz o número de músicas, de alvos e
de segundos, nats por token, perplexidade por token, bits por segundo e bits
por música.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from model import ModelMusic  # noqa: E402
from nll import song_seconds, songs_nll, summarize_nll  # noqa: E402
from tokenizer import Pico8Tokenizer  # noqa: E402


def score(checkpoint: Path, data_dir: Path, split: str, device: torch.device) -> dict:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    vocab = json.loads((data_dir / "vocab.json").read_text(encoding="utf-8"))
    model = ModelMusic(
        vocab_size=cfg["vocab_size"],
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        padding_idx=cfg["pad_id"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    songs = json.loads((data_dir / f"{split}.json").read_text(encoding="utf-8"))
    tokenizer = Pico8Tokenizer(mode=vocab["mode"], unroll_song_loop=vocab.get("unroll_song_loop", False))
    per_song = songs_nll(model, [s["token_ids"] for s in songs], device, pad_id=cfg["pad_id"])
    seconds = [song_seconds(tokenizer.decode_ids(s["token_ids"])) for s in songs]
    result = summarize_nll(per_song, seconds)
    return {
        "checkpoint": str(checkpoint),
        "split": split,
        "mode": vocab["mode"],
        "best_epoch": ckpt.get("epoch"),
        "best_val_window_loss": ckpt.get("val_loss"),
        **result,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True, help="best.pt gravado pelo src/train.py")
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="pasta do tokenizador gerada pelo src/data.py (vocab.json e <split>.json)")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--output", type=Path, default=None, help="grava o JSON neste arquivo")
    ap.add_argument("--device", default=None, help="cuda ou cpu (padrão: cuda se houver)")
    args = ap.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    result = score(args.checkpoint, args.data_dir, args.split, device)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
