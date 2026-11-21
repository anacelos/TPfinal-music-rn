"""
Pipeline de dados e Datasets PyTorch para geração musical simbólica no
PICO-8 (DCC831 - UFMG). A divisão sem vazamento vem de `preprocessing/build_dataset.py`.
"""

import argparse
import json
import os
from itertools import islice
from typing import Dict, List, Any, Optional, Tuple

try:  # imported as part of the src package (repo root on sys.path)
    from .tokenizer import Pico8Tokenizer
    from .synth import synthesize_song_to_wav
except ImportError:  # run as `python3 src/data.py`, src/ on sys.path
    from tokenizer import Pico8Tokenizer
    from synth import synthesize_song_to_wav

SPLIT_NAMES = ("train", "val", "test")


def load_splits(
    split_dir: str = "data/splits",
    limit_songs: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    splits: Dict[str, List[Dict[str, Any]]] = {}
    for name in SPLIT_NAMES:
        with open(os.path.join(split_dir, f"{name}.jsonl"), "r", encoding="utf-8") as f:
            splits[name] = [json.loads(line) for line in islice(f, limit_songs)]
    return splits


def compute_split_statistics(
    splits: Dict[str, List[Dict[str, Any]]],
    tokenizer: Pico8Tokenizer,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Tokeniza todas as músicas de `train`, `val` e `test` e calcula estatísticas
    descritivas (comprimento médio, mediana, p90, total de tokens, tamanho do vocabulário).
    """
    tokenized_splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    stats: Dict[str, Any] = {
        "mode": tokenizer.mode,
        "unroll_song_loop": tokenizer.unroll_song_loop,
        "vocab_size": tokenizer.vocab_size,
        "splits": {},
    }
    used_tokens = set()

    for split_name, songs in splits.items():
        lengths: List[int] = []
        tids_in_split = set()

        for s in songs:
            ids = tokenizer.encode_song_to_ids(s)
            lengths.append(len(ids))
            used_tokens.update(ids)
            tids_in_split.add(s["tid"])
            tokenized_splits[split_name].append({
                "tid": s["tid"],
                "song_id": s["song_id"],
                "num_patterns": s["num_patterns"],
                "duration_seconds": s.get("duration_sec", 0.0),
                "token_ids": ids,
            })

        lengths_sorted = sorted(lengths)
        n = len(lengths_sorted)
        total_toks = sum(lengths_sorted)
        stats["splits"][split_name] = {
            "num_games": len(tids_in_split),
            "num_songs": n,
            "total_tokens": total_toks,
            "mean_tokens_per_song": round(total_toks / n, 1) if n else 0.0,
            "median_tokens_per_song": lengths_sorted[n // 2] if n else 0,
            "p90_tokens_per_song": lengths_sorted[int(n * 0.9)] if n else 0,
            "max_tokens_per_song": lengths_sorted[-1] if n else 0,
        }

    stats["active_vocab_size"] = len(used_tokens)
    return tokenized_splits, stats


def save_processed_dataset(
    out_dir: str,
    tokenizer: Pico8Tokenizer,
    tokenized_splits: Dict[str, List[Dict[str, Any]]],
    stats: Dict[str, Any],
) -> None:
    """Salva vocabulário, estatísticas e splits tokenizados em disco."""
    os.makedirs(out_dir, exist_ok=True)

    vocab_path = os.path.join(out_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": tokenizer.mode,
                "unroll_song_loop": tokenizer.unroll_song_loop,
                "id_to_token": tokenizer.id_to_token,
                "token_to_id": tokenizer.token_to_id,
            },
            f,
            indent=2,
        )

    stats_path = os.path.join(out_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    for split_name, items in tokenized_splits.items():
        split_path = os.path.join(out_dir, f"{split_name}.json")
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(items, f)


# -----------------------------------------------------------------------------
# Dataset PyTorch para Treinamento Autoregressivo (Next-Token Prediction)
# -----------------------------------------------------------------------------
def build_windows(
    tokenized_songs: List[Dict[str, Any]],
    seq_len: int = 512,
    stride: int = 256,
    pad_id: int = 0,
) -> List[List[int]]:
    """
    Divide cada música tokenizada em janelas de tamanho `seq_len + 1` (para formar
    pares `(x, y)` de tamanho `seq_len`). Todas as janelas permanecem no mesmo split
    da música original.
    """
    target_len = seq_len + 1
    windows: List[List[int]] = []

    for item in tokenized_songs:
        ids = item["token_ids"]
        if len(ids) <= target_len:
            padded = ids + [pad_id] * (target_len - len(ids))
            windows.append(padded)
        else:
            for start in range(0, len(ids) - target_len + 1, stride):
                windows.append(ids[start : start + target_len])
            # Garante cobertura do final da música (incluindo EOS)
            if (len(ids) - target_len) % stride != 0:
                windows.append(ids[-target_len:])

    return windows


def create_pytorch_datasets(
    processed_dir: str,
    seq_len: int = 512,
    stride: int = 256,
):
    """
    Cria objetos `torch.utils.data.Dataset` para `train`, `val` e `test`.
    Cada amostra retorna `(x, y)` de shape `(seq_len,)` com tipo `torch.long`.
    """
    import torch
    from torch.utils.data import Dataset

    with open(os.path.join(processed_dir, "vocab.json"), "r", encoding="utf-8") as f:
        vocab_info = json.load(f)
    pad_id = vocab_info["token_to_id"]["PAD"]

    class Pico8SymbolDataset(Dataset):
        def __init__(self, split_file: str, is_train: bool):
            with open(split_file, "r", encoding="utf-8") as f:
                songs = json.load(f)
            win_stride = stride if is_train else seq_len
            raw_windows = build_windows(songs, seq_len=seq_len, stride=win_stride, pad_id=pad_id)
            self.data = torch.tensor(raw_windows, dtype=torch.long)

        def __len__(self) -> int:
            return self.data.size(0)

        def __getitem__(self, idx: int):
            row = self.data[idx]
            return row[:-1], row[1:]

    datasets = {
        split: Pico8SymbolDataset(
            os.path.join(processed_dir, f"{split}.json"),
            is_train=(split == "train"),
        )
        for split in ("train", "val", "test")
    }
    return datasets, vocab_info


def run_roundtrip_verification(
    song: Dict[str, Any],
    out_dir: str,
) -> None:
    """
    Testa a codificação e decodificação da mesma música nos 3 modos (`nlm`,
    `grid_unrolled`, `grid_block`) e renderiza os arquivos `.wav` para inspeção.
    """
    os.makedirs(out_dir, exist_ok=True)
    tid = song.get("tid", "sample")
    song_id = song.get("song_id", 0)

    orig_wav = os.path.join(out_dir, f"tid{tid}_song{song_id}_0_original.wav")
    synthesize_song_to_wav(song, orig_wav, loop_extra_repeats=1)
    print(f"[Round-Trip] Original salvo em: {orig_wav}")

    for mode in ("nlm", "grid_unrolled", "grid_block"):
        tok = Pico8Tokenizer(mode=mode, unroll_song_loop=False)
        tokens = tok.encode_song(song)
        ids = tok.tokens_to_ids(tokens)
        decoded_song = tok.decode_ids(ids)

        wav_path = os.path.join(out_dir, f"tid{tid}_song{song_id}_{mode}.wav")
        dur = synthesize_song_to_wav(decoded_song, wav_path, loop_extra_repeats=1)
        print(
            f"  - Modo '{mode:13s}': {len(ids):5d} tokens | "
            f"vocab={tok.vocab_size:3d} | patterns decodificados={decoded_song['num_patterns']} | "
            f"duração={dur:.2f}s -> {wav_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pré-processamento e Tokenização do Dataset PICO-8 (TP2)")
    parser.add_argument("--split-dir", type=str, default="data/splits")
    parser.add_argument("--out-dir", type=str, default="data/processed")
    parser.add_argument("--mode", type=str, default="all", choices=["nlm", "grid_unrolled", "grid_block", "all"])
    parser.add_argument("--unroll-song-loop", action="store_true")
    parser.add_argument("--limit-games", type=int, default=None, help="Lê só as N primeiras músicas de cada split")
    parser.add_argument("--test-roundtrip", action="store_true", help="Gera WAVs de verificação round-trip")
    args = parser.parse_args()

    print(f"Carregando splits de: {args.split_dir} ...")
    splits = load_splits(args.split_dir, limit_songs=args.limit_games)
    print(
        f"Train: {len(splits['train'])} músicas | "
        f"Val: {len(splits['val'])} músicas | Test: {len(splits['test'])} músicas\n"
    )

    if args.test_roundtrip and splits["train"]:
        # Escolhe a primeira música do TID 140614 se presente, ou a primeira música de treino com >= 4 patterns
        all_songs = [s for name in SPLIT_NAMES for s in splits[name]]
        sample_song = next((s for s in all_songs if s["tid"] == "140614"), None)
        if sample_song is None:
            sample_song = next((s for s in splits["train"] if s.get("num_patterns", 0) >= 4), splits["train"][0])
        run_roundtrip_verification(sample_song, os.path.join(args.out_dir, "roundtrip_check"))
        print()

    modes = ["nlm", "grid_unrolled", "grid_block"] if args.mode == "all" else [args.mode]
    for m in modes:
        tok = Pico8Tokenizer(mode=m, unroll_song_loop=args.unroll_song_loop)
        tokenized_splits, stats = compute_split_statistics(splits, tok)
        mode_out_dir = os.path.join(args.out_dir, m)
        save_processed_dataset(mode_out_dir, tok, tokenized_splits, stats)

        tr = stats["splits"]["train"]
        va = stats["splits"]["val"]
        te = stats["splits"]["test"]
        print(f"=== Modo: {m} (vocab={stats['vocab_size']}, ativos={stats['active_vocab_size']}) ===")
        print(
            f"  Train: {tr['num_songs']:5d} músicas ({tr['num_games']:4d} jogos) | "
            f"Média: {tr['mean_tokens_per_song']:6.1f} toks | Mediana: {tr['median_tokens_per_song']:4d} | "
            f"P90: {tr['p90_tokens_per_song']:4d} | Total: {tr['total_tokens']:,}"
        )
        print(
            f"  Val  : {va['num_songs']:5d} músicas ({va['num_games']:4d} jogos) | "
            f"Média: {va['mean_tokens_per_song']:6.1f} toks | Mediana: {va['median_tokens_per_song']:4d} | "
            f"P90: {va['p90_tokens_per_song']:4d} | Total: {va['total_tokens']:,}"
        )
        print(
            f"  Test : {te['num_songs']:5d} músicas ({te['num_games']:4d} jogos) | "
            f"Média: {te['mean_tokens_per_song']:6.1f} toks | Mediana: {te['median_tokens_per_song']:4d} | "
            f"P90: {te['p90_tokens_per_song']:4d} | Total: {te['total_tokens']:,}"
        )
        print(f"  Salvo em: {mode_out_dir}\n")


if __name__ == "__main__":
    main()
