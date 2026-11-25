"""NLL por música inteira e bits por segundo (src/nll.py)."""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import ModelMusic  # noqa: E402
from nll import SAMPLE_RATE, SAMPLES_PER_TICK, song_seconds, songs_nll, summarize_nll  # noqa: E402


def tiny_model():
    torch.manual_seed(0)
    return ModelMusic(vocab_size=11, embedding_dim=8, hidden_dim=16, num_layers=2, padding_idx=0).eval()


def test_song_nll_equals_one_forward_pass_whatever_the_chunk_and_the_batch():
    model = tiny_model()
    songs = [[1, 2, 3, 4, 5, 6, 7], [3, 4, 5], [9, 8, 7, 6, 5, 4, 3, 2, 1, 10]]
    expected = []
    with torch.no_grad():
        for s in songs:
            x = torch.tensor([s])
            logits, _ = model(x[:, :-1])
            expected.append(float(F.cross_entropy(logits[0], x[0, 1:], reduction="sum")))
    for chunk, batch in ((2, 32), (3, 1), (512, 2)):
        got = songs_nll(model, songs, torch.device("cpu"), pad_id=0, batch_size=batch, chunk=chunk)
        assert [r["targets"] for r in got] == [len(s) - 1 for s in songs]
        for r, ref in zip(got, expected):
            assert r["nll"] == pytest.approx(ref, rel=1e-5)


def test_bits_per_second_is_total_bits_over_total_seconds():
    per_song = [{"nll": 2.0 * math.log(2), "targets": 4}, {"nll": 6.0 * math.log(2), "targets": 4}]
    out = summarize_nll(per_song, [1.0, 3.0])
    assert out["bits_per_second"] == pytest.approx(2.0)  # 8 bits in 4 seconds
    assert out["bits_per_song"] == pytest.approx(4.0)
    assert out["nats_per_token"] == pytest.approx(math.log(2))
    assert out["perplexity_per_token"] == pytest.approx(2.0)
    assert out["songs"] == 2 and out["targets"] == 8


def test_song_seconds_counts_every_pattern_once():
    song = {"patterns": [{"pattern_ticks": 512}, {"pattern_ticks": 256}], "loop_start": 0}
    assert song_seconds(song) == pytest.approx(768 * SAMPLES_PER_TICK / SAMPLE_RATE)
    assert song_seconds({"patterns": []}) == 0.0
