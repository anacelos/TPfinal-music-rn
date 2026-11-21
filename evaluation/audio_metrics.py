"""
Audio-domain evaluation: compare sets of rendered clips with pretrained audio models.

The symbolic metrics (pico8_metrics.py) look at notes; these look at the sound, as a
listener-like model hears it. Every set (real test, real val, each system) is a folder
of WAV clips (e.g. from evaluation/synthesize_jsonl.py), compared against a reference set
(the real test). Our runs cut every song at 60 s before rendering, so long songs do not
dominate the loading time; the clips do not need to have the same length.

  FAD   Frechet Audio Distance (Kilgour et al., 2019): fit a Gaussian to the embeddings
        of each set and measure the Frechet distance between the two Gaussians,
            FAD = |mu_r - mu_g|^2 + Tr(S_r + S_g - 2 (S_r S_g)^(1/2)).
        Embeddings come from CLAP (laion/clap-htsat-unfused). The music-only checkpoint
        laion/larger_clap_music was tried first, but its transformers port returns
        collapsed embeddings (cosine 0.99 between any two clips, uniform zero-shot
        scores, on CPU and GPU, transformers 4.57 and 5.23). FAD grows when the set is small, so every set is
        scored with bootstrap resamples of the same size (mean and std reported), and a
        second real set (val) gives the floor: what two sets of real songs score.
  CLAP  cosine similarity between each clip and a text prompt in CLAP's joint space
        ("8-bit chiptune video game music"); higher = sounds more like the prompt. Two
        tempo prompts give an audio-side check of the tempo conditioning.
  KLD   AudioSet tag distribution from AST (MIT/ast-finetuned-audioset-10-10-0.4593):
        mean of the per-clip sigmoid outputs, normalized to sum 1, and KL(reference || set).
        Also reports the top tags of each set, which says in words how each set sounds.

Outputs, in --out-dir: audio_metrics.csv / .json (one row per set), embeddings.npz
(CLAP embeddings, AST probabilities and set labels, for plots).

Example:
  python3 evaluation/audio_metrics.py --reference real_test=wav/real_test \\
      --set real_val=wav/real_val --set lstm=wav/lstm --set markov=wav/markov --out-dir results/audio
"""

import argparse
import csv
import json
import time
import wave
from fractions import Fraction
from pathlib import Path
from typing import Dict, List

import numpy as np

CLAP_MODEL = "laion/clap-htsat-unfused"
AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
PROMPT = "8-bit chiptune video game music"
TEMPO_PROMPTS = ("slow chiptune music with long notes", "fast chiptune music with quick notes")


def load_wav(path, rate: int) -> np.ndarray:
    """Mono float32 in [-1, 1], resampled to `rate`."""
    from scipy.signal import resample_poly

    with wave.open(str(path)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != rate:
        f = Fraction(rate, sr).limit_denominator(1000)
        x = resample_poly(x, f.numerator, f.denominator).astype(np.float32)
    return x


def _features(out):
    """get_*_features returns a tensor in some transformers versions and a model output in others."""
    import torch

    return out if torch.is_tensor(out) else out.pooler_output


class Clap:
    rate = 48_000

    def __init__(self, device, model=CLAP_MODEL):
        from transformers import ClapModel, ClapProcessor

        self.device = device
        self.model = ClapModel.from_pretrained(model).to(device).eval()
        self.processor = ClapProcessor.from_pretrained(model)

    def audio(self, clips: List[np.ndarray], batch=16) -> np.ndarray:
        import torch

        out = []
        for i in range(0, len(clips), batch):
            feats = self.processor.feature_extractor(clips[i : i + batch], sampling_rate=self.rate,
                                                     return_tensors="pt")
            feats = {k: v.to(self.device) for k, v in feats.items()}
            with torch.no_grad():
                e = _features(self.model.get_audio_features(**feats))
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.concatenate(out)

    def text(self, prompts: List[str]) -> np.ndarray:
        import torch

        tok = self.processor.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            e = _features(self.model.get_text_features(**tok))
        return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()


class AudioSetTagger:
    rate = 16_000

    def __init__(self, device, model=AST_MODEL):
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        self.device = device
        self.model = ASTForAudioClassification.from_pretrained(model).to(device).eval()
        self.extractor = ASTFeatureExtractor.from_pretrained(model)
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    def probs(self, clips: List[np.ndarray], batch=16) -> np.ndarray:
        import torch

        out = []
        for i in range(0, len(clips), batch):
            feats = self.extractor(clips[i : i + batch], sampling_rate=self.rate, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**{k: v.to(self.device) for k, v in feats.items()}).logits
            out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out)


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.linalg import sqrtm

    mu_a, mu_b = a.mean(0), b.mean(0)
    s_a, s_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = sqrtm(s_a @ s_b)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(s_a + s_b - 2 * covmean))


def fad_bootstrap(ref: np.ndarray, emb: np.ndarray, size: int, rounds: int, rng) -> Dict[str, float]:
    vals = [frechet_distance(ref, emb[rng.integers(0, len(emb), size)]) for _ in range(rounds)]
    return {"fad": float(np.mean(vals)), "fad_std": float(np.std(vals))}


def kl_divergence(p: np.ndarray, q: np.ndarray, eps=1e-8) -> float:
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log((p + eps) / (q + eps))))


def wav_files(folder) -> List[Path]:
    return sorted(Path(folder).glob("*.wav"))


def evaluate(sets: Dict[str, Path], reference: str, out_dir: Path, device="cpu", boot_size=100, boot_rounds=20,
             seed=831, limit=None) -> List[Dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    clap, tagger = Clap(device), AudioSetTagger(device)
    prompts = clap.text([PROMPT, *TEMPO_PROMPTS])
    emb, probs, files = {}, {}, {}
    for name, folder in sets.items():
        files[name] = wav_files(folder)[:limit]
        emb[name] = clap.audio([load_wav(f, clap.rate) for f in files[name]])
        probs[name] = tagger.probs([load_wav(f, tagger.rate) for f in files[name]])
        print(f"[audio] {name}: {len(files[name])} clips embedded ({time.time() - t0:.0f} s)", flush=True)

    ref_emb, ref_tags = emb[reference], probs[reference].mean(0)
    vgm = tagger.labels.index("Video game music") if "Video game music" in tagger.labels else None
    rows = []
    for name in sets:
        e, p = emb[name], probs[name]
        sim = e @ prompts.T
        row = {"set": name, "clips": len(e), **fad_bootstrap(ref_emb, e, boot_size, boot_rounds, rng),
               "clap_score": float(sim[:, 0].mean()),
               "clap_fast_minus_slow": float((sim[:, 2] - sim[:, 1]).mean()),
               "kld": kl_divergence(ref_tags, p.mean(0)),
               "video_game_music_prob": float(p[:, vgm].mean()) if vgm is not None else None,
               "top_tags": [[tagger.labels[i], round(float(p.mean(0)[i]), 3)] for i in np.argsort(-p.mean(0))[:5]]}
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)

    keys = ["set", "clips", "fad", "fad_std", "clap_score", "clap_fast_minus_slow", "kld", "video_game_music_prob"]
    with open(out_dir / "audio_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    meta = {"reference": reference, "clap_model": CLAP_MODEL, "ast_model": AST_MODEL, "prompt": PROMPT,
            "tempo_prompts": TEMPO_PROMPTS, "boot_size": boot_size, "boot_rounds": boot_rounds,
            "seconds": round(time.time() - t0, 1), "rows": rows}
    (out_dir / "audio_metrics.json").write_text(json.dumps(meta, indent=2))
    names = list(sets)
    np.savez_compressed(out_dir / "embeddings.npz", clap=np.concatenate([emb[n] for n in names]),
                        ast=np.concatenate([probs[n] for n in names]).astype(np.float16),
                        set=np.array([n for n in names for _ in emb[n]]), prompts=prompts,
                        files=np.array([f.name for n in names for f in files[n]]), ast_labels=np.array(tagger.labels))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description="FAD / CLAP score / AudioSet KLD between sets of WAV clips")
    p.add_argument("--reference", required=True, help="name=folder of the reference set (real test)")
    p.add_argument("--set", action="append", default=[], help="name=folder, repeatable")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--boot-size", type=int, default=100, help="clips per bootstrap resample (same for every set)")
    p.add_argument("--boot-rounds", type=int, default=20)
    p.add_argument("--limit", type=int, default=None, help="use only the first N clips of each set")
    args = p.parse_args(argv)
    if args.device == "auto":
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    ref_name, ref_dir = args.reference.split("=", 1)
    sets = {ref_name: Path(ref_dir), **{n: Path(d) for n, d in (s.split("=", 1) for s in args.set)}}
    evaluate(sets, ref_name, Path(args.out_dir), args.device, args.boot_size, args.boot_rounds, limit=args.limit)


if __name__ == "__main__":
    main()
