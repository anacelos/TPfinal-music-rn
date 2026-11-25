"""
Song-level negative log-likelihood, comparable across tokenizers.

Window losses (what training optimizes) restart the LSTM at every window, count
the overlapped tail of long songs twice and depend on how many tokens a
tokenizer spends per second of music. For the paper and the tokenizer
ablation we score each full song instead: the model reads the whole sequence
in chunks, carrying its hidden state, and we sum the NLL of every target.
Dividing by the song's duration (from the decoded ticks) gives bits per second
of music, which does not depend on the vocabulary or on the sequence length.
"""

import math
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

SAMPLE_RATE = 44100
SAMPLES_PER_TICK = 366


def song_seconds(song: Dict[str, Any]) -> float:
    """Duration of a decoded (or canonical) song, one pass, no loop repeat."""
    ticks = sum(int(p.get("pattern_ticks", 0)) for p in song.get("patterns", []))
    return ticks * SAMPLES_PER_TICK / SAMPLE_RATE


@torch.no_grad()
def songs_nll(
    model: torch.nn.Module,
    songs_ids: Sequence[Sequence[int]],
    device: torch.device,
    pad_id: int = 0,
    batch_size: int = 32,
    chunk: int = 512,
) -> List[Dict[str, float]]:
    """NLL (nats) of every target of every song, with the hidden state carried across chunks.

    Returns one dict per song, in input order: {"nll": total nats, "targets": n}.
    Songs are batched by length so padding stays small; padded targets are ignored.
    """
    model.eval()
    order = sorted(range(len(songs_ids)), key=lambda i: len(songs_ids[i]))
    out: List[Dict[str, float]] = [{} for _ in songs_ids]
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        seqs = [list(songs_ids[i]) for i in idx]
        longest = max(len(s) for s in seqs)
        x = torch.full((len(seqs), longest), pad_id, dtype=torch.long)
        for j, s in enumerate(seqs):
            x[j, : len(s)] = torch.tensor(s, dtype=torch.long)
        x = x.to(device)
        inputs, targets = x[:, :-1], x[:, 1:]
        total = torch.zeros(len(seqs), device=device)
        hidden = None
        for t0 in range(0, inputs.size(1), chunk):
            logits, hidden = model(inputs[:, t0 : t0 + chunk], hidden)
            nll = F.cross_entropy(
                logits.transpose(1, 2), targets[:, t0 : t0 + chunk], ignore_index=pad_id, reduction="none"
            )
            total += nll.sum(dim=1)
        for j, i in enumerate(idx):
            out[i] = {"nll": float(total[j]), "targets": len(seqs[j]) - 1}
    return out


def summarize_nll(per_song: List[Dict[str, float]], seconds: List[float]) -> Dict[str, float]:
    """Aggregate per-song NLL into per-token and per-second figures (bits)."""
    nats = sum(r["nll"] for r in per_song)
    targets = sum(r["targets"] for r in per_song)
    secs = sum(seconds)
    return {
        "songs": len(per_song),
        "targets": targets,
        "seconds": round(secs, 1),
        "nats_per_token": nats / max(targets, 1),
        "perplexity_per_token": math.exp(nats / max(targets, 1)),
        "bits_per_second": nats / math.log(2) / max(secs, 1e-9),
        "bits_per_song": nats / math.log(2) / max(len(per_song), 1),
    }
