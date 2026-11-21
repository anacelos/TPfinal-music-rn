"""How close our synth (src/synth.py) is to the real PICO-8, on recordings of the official player.

A recording of a song playing in the BBS player (an excerpt, any length) is aligned
with our rendering of the same song (the excerpt is searched inside it) and compared:

* chroma similarity: are the same notes sounding (1 = same pitch classes);
* onset and envelope correlation: do notes start and decay at the same moments;
* spectral distance: mean |dB| difference per frequency bin, after matching the overall level;
* spectral centroid and energy per band: is the timbre as bright as the real one;
* level ratio: our RMS over the recording's.

    python evaluation/synth_fidelity.py --real rec.wav --synth antigo=old.wav --synth novo=new.wav \
        --out work/synth-check/celeste --max-lag 50

How to record (Chrome, BBS cart page, window visible in front the whole time, or the
tab is throttled and the audio gets gaps): before pressing play, run in the console a
hook that taps everything connected to the AudioContext destination into a
ScriptProcessor and keeps the Float32 buffers; play, then write them as a WAV blob.
Check `pico8_state.sound_volume` (the player can be muted) and cut out gaps of exact
zeros. The web player renders at 22050 Hz and upsamples to 44100 Hz by linear interpolation.
"""

import argparse
import wave
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import correlate, stft

HOP, NFFT = 512, 4096
BANDS = ((0, 250), (250, 1000), (1000, 4000), (4000, 11025))


def load_wav(path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(axis=1)
    return x, sr


def save_wav(path, x: np.ndarray, sr: int):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def spectrogram(x: np.ndarray, sr: int):
    f, _, z = stft(x, sr, nperseg=NFFT, noverlap=NFFT - HOP, boundary=None, padded=False)
    return f, np.abs(z)


def onset(mag: np.ndarray) -> np.ndarray:
    d = np.diff(np.log1p(100 * mag), axis=1)
    return np.concatenate([[0], np.maximum(d, 0).sum(axis=0)])


def chroma(f: np.ndarray, mag: np.ndarray) -> np.ndarray:
    keep = (f > 55) & (f < 4200)
    pc = np.round(12 * np.log2(f[keep] / 440.0)).astype(int) % 12
    c = np.zeros((12, mag.shape[1]))
    for k in range(12):
        c[k] = (mag[keep][pc == k] ** 2).sum(axis=0)
    return c / (np.linalg.norm(c, axis=0, keepdims=True) + 1e-9)


def _onset_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> int:
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    cc = correlate(a, b, mode="full")
    lags = np.arange(-len(b) + 1, len(a))
    ok = (lags >= 0) & (lags <= max_lag)
    return int(lags[ok][np.argmax(cc[ok])])


def band_shares(f: np.ndarray, mag: np.ndarray) -> List[float]:
    e = np.array([(mag[(f >= a) & (f < b)] ** 2).sum() for a, b in BANDS])
    return list(100 * e / e.sum())


def centroid(f: np.ndarray, mag: np.ndarray, mask: np.ndarray) -> float:
    c = (f[:, None] * mag).sum(0) / (mag.sum(0) + 1e-9)
    return float(np.median(c[mask[:mag.shape[1]]]))


def compare(real: np.ndarray, ours: Dict[str, np.ndarray], sr: int, max_lag_s: float = 25.0):
    """Scores of each rendering against the recording, plus the aligned excerpts for listening and plots."""
    fps = sr / HOP
    f, mr = spectrogram(real, sr)
    loud = mr.sum(axis=0) > np.percentile(mr.sum(axis=0), 20)
    cr = chroma(f, mr)
    rows = [{"versão": "PICO-8 real", "centroide (Hz)": centroid(f, mr, loud), "volume": 1.0,
             **{f"{a}-{b} Hz (%)": v for (a, b), v in zip(BANDS, band_shares(f, mr))}}]
    aligned = {}
    n_r = mr.shape[1]
    for name, x in ours.items():
        _, mo = spectrogram(x, sr)
        co = chroma(f, mo)
        # where the recorded excerpt sits in our rendering: best chroma agreement, refined by onsets
        cand = range(0, max(0, min(int(max_lag_s * fps), mo.shape[1] - n_r)) + 1)
        scores = [(cr[:, loud] * co[:, k:k + n_r][:, loud]).sum(axis=0).mean() for k in cand]
        coarse = int(np.argmax(scores))
        lo = max(0, coarse - 4)
        lag = lo + _onset_lag(onset(mo)[lo:coarse + 4 + n_r], onset(mr), 8)
        mo, co = mo[:, lag:lag + n_r], co[:, lag:lag + n_r]
        n = mo.shape[1]
        xs = x[lag * HOP:lag * HOP + len(real)]
        aligned[name] = (xs, mo)
        lr, lo_db = 20 * np.log10(mr[:, :n] + 1e-5), 20 * np.log10(mo + 1e-5)
        lo_db = lo_db + (lr[:, loud[:n]].mean() - lo_db[:, loud[:n]].mean())
        rows.append({
            "versão": name,
            "início no nosso (s)": lag / fps,
            "chroma": float((cr[:, :n] * co).sum(axis=0)[loud[:n]].mean()),
            "onsets": float(np.corrcoef(onset(mr)[:n], onset(mo))[0, 1]),
            "envelope": float(np.corrcoef(np.log1p(mr[:, :n].sum(0)), np.log1p(mo.sum(0)))[0, 1]),
            "dist. espectral (dB)": float(np.abs(lr - lo_db)[:, loud[:n]].mean()),
            "centroide (Hz)": centroid(f, mo, loud),
            "volume": float(np.sqrt(np.mean(xs ** 2)) / np.sqrt(np.mean(real[:len(xs)] ** 2))),
            **{f"{a}-{b} Hz (%)": v for (a, b), v in zip(BANDS, band_shares(f, mo))},
        })
    return rows, aligned, (f, mr)


def figure(real_spec, aligned, sr: int, title: str, seconds: float = 12.0):
    import matplotlib.pyplot as plt
    f, mr = real_spec
    fps = sr / HOP
    m = min(int(seconds * fps), mr.shape[1])
    t = np.arange(m) / fps
    fmax = np.searchsorted(f, 8000)
    panels = [("PICO-8 de verdade (player oficial, gravado)", mr)] + \
             [(f"nosso sintetizador: {name}", mag) for name, (_, mag) in aligned.items()]
    fig, ax = plt.subplots(len(panels), 1, figsize=(13, 3.0 * len(panels)), sharex=True)
    top = 20 * np.log10(mr[:fmax, :m].max() + 1e-6)
    for a, (label, mag) in zip(ax, panels):
        db = 20 * np.log10(mag[:fmax, :m] + 1e-6)
        db = db - db.max() + top
        a.pcolormesh(t[:db.shape[1]], f[:fmax], db, vmin=top - 70, vmax=top, shading="auto", cmap="magma")
        a.set_yscale("symlog", linthresh=300)
        a.set_ylim(80, 8000)
        a.set_ylabel("Hz")
        a.set_title(label, fontsize=10, loc="left")
        a.grid(False)
    ax[-1].set_xlabel("segundos (alinhados)")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def ab_track(real: np.ndarray, ours: np.ndarray, sr: int) -> np.ndarray:
    """Recording, 1 s of silence, then ours at the same loudness."""
    rms = lambda v: np.sqrt(np.mean(v ** 2)) + 1e-9
    r = real[:len(ours)]
    return np.concatenate([r, np.zeros(sr, dtype=np.float32), ours * rms(r) / rms(ours)])


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    p = argparse.ArgumentParser(description="Compare a PICO-8 recording with our renderings of the same song")
    p.add_argument("--real", required=True)
    p.add_argument("--synth", action="append", required=True, help="name=path.wav (repeatable)")
    p.add_argument("--out", required=True, help="prefix for <out>_comparacao.png and <out>_AB_<name>.wav")
    p.add_argument("--max-lag", type=float, default=25.0, help="how far into our rendering to search (s)")
    p.add_argument("--fig-seconds", type=float, default=12.0)
    args = p.parse_args(argv)
    real, sr = load_wav(args.real)
    ours = {}
    for item in args.synth:
        name, path = item.split("=", 1)
        x, sr2 = load_wav(path)
        if sr2 != sr:
            raise SystemExit(f"{path}: {sr2} Hz, the recording is {sr} Hz")
        ours[name] = x
    rows, aligned, real_spec = compare(real, ours, sr, args.max_lag)
    keys = list(rows[1].keys())
    print(" | ".join(keys))
    for row in rows:
        print(" | ".join(f"{row[k]:.2f}" if isinstance(row.get(k), float) else str(row.get(k, "")) for k in keys))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure(real_spec, aligned, sr, out.name, args.fig_seconds).savefig(f"{out}_comparacao.png", dpi=100)
    for name, (xs, _) in aligned.items():
        save_wav(f"{out}_AB_{name}.wav", ab_track(real, xs, sr), sr)


if __name__ == "__main__":
    main()
