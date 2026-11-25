"""
Compare per-song metric distributions of several systems against the real test set,
with the protocol of Yang & Lerch (2020), "On the evaluation of generative models in music".

For each metric and each system G (real R is the reference):
  * intra-set distances: |a - b| for all pairs inside R, and inside G;
  * inter-set distances: |g - r| for every g in G and r in R;
  * a Gaussian KDE of each distance set; overlap area OA(intra_R, inter_G) and
    KL(intra_R || inter_G). A generator that matches R has inter distances that look
    like R's own intra distances: OA near 1 and KL near 0.
  * also KS and Wasserstein-1 between the raw values, the mean and std, and how many
    songs had no value (empty or silent generations are reported, not hidden).

Inputs are the per-song CSVs of evaluation/evaluate_split.py. The primary metrics are the
course ones that evaluate_split.py also marks as primary (Pitch Entropy, Scale Consistency,
Groove Consistency, all computed on note onsets); the rest are diagnostics.

Example:
  python3 evaluation/distribution_compare.py --real real.csv --system markov=markov.csv \\
      --system neural=neural.csv --out-dir results/compare
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

PRIMARY = ["pitch_entropy", "scale_consistency", "groove_consistency"]
DIAGNOSTIC = ["pitch_range", "pitch_class_entropy", "audible_step_rate", "audible_channel_utilization",
              "duration_seconds", "pattern_count"]
METRICS = PRIMARY + DIAGNOSTIC
MAX_PAIRS = 400_000
GRID = 512


def read_column(path, metric):
    vals, missing = [], 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            v = row.get(metric, "")
            try:
                x = float(v)
                if math.isfinite(x):
                    vals.append(x)
                    continue
            except (TypeError, ValueError):
                pass
            missing += 1
    return np.asarray(vals, dtype=float), missing


def pair_distances(a, b=None, rng=None):
    """|x - y| over all pairs (inside a when b is None), subsampled to MAX_PAIRS."""
    if b is None:
        i, j = np.triu_indices(len(a), k=1)
        d = np.abs(a[i] - a[j])
    else:
        d = np.abs(a[:, None] - b[None, :]).ravel()
    if len(d) > MAX_PAIRS:
        d = rng.choice(d, MAX_PAIRS, replace=False)
    return d


def kde(samples, grid):
    """Gaussian KDE with Scott's bandwidth, evaluated on grid (density, sums to ~1 * dx)."""
    n = len(samples)
    std = samples.std()
    if n < 2 or std == 0:
        bw = max(1e-3, (grid[-1] - grid[0]) / GRID * 2)
    else:
        bw = 1.06 * std * n ** (-1 / 5)
    sub = samples if n <= 20_000 else np.random.default_rng(0).choice(samples, 20_000, replace=False)
    dens = np.zeros_like(grid)
    for chunk in np.array_split(sub, max(1, len(sub) // 2000)):
        dens += np.exp(-0.5 * ((grid[:, None] - chunk[None, :]) / bw) ** 2).sum(axis=1)
    dens /= len(sub) * bw * math.sqrt(2 * math.pi)
    return dens


def overlap_and_kl(p_samples, q_samples):
    lo = 0.0
    hi = max(p_samples.max(initial=0), q_samples.max(initial=0)) * 1.1 + 1e-9
    grid = np.linspace(lo, hi, GRID)
    dx = grid[1] - grid[0]
    p, q = kde(p_samples, grid), kde(q_samples, grid)
    eps = 1e-12
    # Distances are >= 0, so part of each kernel falls below 0, off the grid. Renormalize
    # both densities on the grid: identical sets then overlap fully (OA = 1, KL = 0).
    pn, qn = p / (p.sum() * dx + eps), q / (q.sum() * dx + eps)
    oa = float(np.minimum(pn, qn).sum() * dx)
    kl = float(np.sum(pn * np.log((pn + eps) / (qn + eps))) * dx)
    return oa, kl


def ks_w1(a, b):
    a, b = np.sort(a), np.sort(b)
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side="right") / len(a)
    cdf_b = np.searchsorted(b, allv, side="right") / len(b)
    ks = float(np.max(np.abs(cdf_a - cdf_b)))
    grid = np.sort(allv)
    ca = np.searchsorted(a, grid[:-1], side="right") / len(a)
    cb = np.searchsorted(b, grid[:-1], side="right") / len(b)
    w1 = float(np.sum(np.abs(ca - cb) * np.diff(grid)))
    return ks, w1


def compare(real_csv, systems, metrics=METRICS, seed=831):
    rng = np.random.default_rng(seed)
    rows = []
    for metric in metrics:
        r, r_missing = read_column(real_csv, metric)
        intra_r = pair_distances(r, rng=rng)
        rows.append({"metric": metric, "primary": metric in PRIMARY, "system": "real", "n": len(r),
                     "missing": r_missing,
                     "mean": float(r.mean()) if len(r) else None, "std": float(r.std()) if len(r) else None,
                     "ks": 0.0, "w1": 0.0, "oa": None, "kl": None})
        for name, path in systems.items():
            g, g_missing = read_column(path, metric)
            row = {"metric": metric, "primary": metric in PRIMARY, "system": name, "n": len(g),
                   "missing": g_missing,
                   "mean": float(g.mean()) if len(g) else None, "std": float(g.std()) if len(g) else None}
            if len(g) >= 2 and len(r) >= 2:
                row["ks"], row["w1"] = ks_w1(g, r)
                inter = pair_distances(g, r, rng=rng)
                row["oa"], row["kl"] = overlap_and_kl(intra_r, inter)
                row["intra_oa"], row["intra_kl"] = overlap_and_kl(intra_r, pair_distances(g, rng=rng))
            rows.append(row)
    return rows


def plot(real_csv, systems, out_dir, metrics=METRICS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = math.ceil(len(metrics) / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(12.6, 3.25 * rows), squeeze=False)
    for ax, metric in zip(axes.ravel(), metrics):
        for name, path in {"real": real_csv, **systems}.items():
            v, _ = read_column(path, metric)
            if len(v):
                v = np.sort(v)
                ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", label=name)
        ax.set_title(metric + (" (primária)" if metric in PRIMARY else ""), fontsize=9)
        ax.set_ylabel("ECDF")
        if metric in ("audible_step_rate", "duration_seconds"):
            ax.set_xscale("symlog")
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "ecdf.png", dpi=130)


def main(argv=None):
    p = argparse.ArgumentParser(description="Yang & Lerch style distribution comparison")
    p.add_argument("--real", required=True)
    p.add_argument("--system", action="append", required=True, help="name=path.csv, repeatable")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)
    systems = dict(s.split("=", 1) for s in args.system)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = compare(args.real, systems)
    keys = ["metric", "primary", "system", "n", "missing", "mean", "std", "ks", "w1", "oa", "kl", "intra_oa", "intra_kl"]
    with open(out / "distribution_compare.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    (out / "distribution_compare.json").write_text(json.dumps(rows, indent=2))
    plot(args.real, systems, out)
    for r in rows:
        if r["system"] != "real":
            print(f"{r['metric']:<28} {r['system']:<22} n={r['n']:<5} miss={r['missing']:<4} "
                  f"KS={r.get('ks', float('nan')):.3f} W1={r.get('w1', float('nan')):.3f} "
                  f"OA={r.get('oa') or float('nan'):.3f} KL={r.get('kl') or float('nan'):.3f}")


if __name__ == "__main__":
    main()
