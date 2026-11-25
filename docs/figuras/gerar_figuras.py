"""Gera as figuras de docs/ETAPAS.md a partir dos arquivos do Drive do projeto.

    python docs/figuras/gerar_figuras.py --artefatos <pasta do Drive baixada> --splits data/splits

--artefatos: a pasta do Drive (checkpoints/, predicoes-teste/, resultados/).
--splits:    a pasta com build_stats.json e test.jsonl (ou test.jsonl.gz), saída de
             `python -m preprocessing.build_dataset`.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AZUL, LARANJA, VERDE, CINZA = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
TINTA, TINTA2, FUNDO, GRADE = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
TICKS_POR_SEGUNDO = 44100 / 366  # o PICO-8 toca ~120,5 ticks por segundo
CORES_CANAL = [AZUL, LARANJA, VERDE, "#6250d6"]
NOMES = {"lstm-grid_block-k10": "LSTM grid_block\ntop-k 10", "lstm-grid_block-t1": "LSTM grid_block\nT = 1",
         "lstm-nlm-k10": "LSTM nlm\ntop-k 10", "lstm-nlm-t1": "LSTM nlm\nT = 1", "markov": "Markov"}
ORDEM = list(NOMES)


def cor(sistema):
    return AZUL if "grid_block" in sistema else LARANJA if "nlm" in sistema else CINZA


def estilo():
    plt.rcParams.update({
        "figure.facecolor": FUNDO, "axes.facecolor": FUNDO, "savefig.facecolor": FUNDO,
        "axes.edgecolor": GRADE, "axes.labelcolor": TINTA2, "text.color": TINTA,
        "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.grid": True, "grid.color": GRADE,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.axisbelow": True,
    })


def abrir(path):
    path = Path(path)
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def ler_jsonl(path, limite=None):
    out = []
    with abrir(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
                if limite and len(out) >= limite:
                    break
    return out


def notas(song, segundos):
    """(canal, início em s, duração em s, pitch) das notas audíveis dos primeiros `segundos`."""
    out, t0 = [], 0.0
    for pat in song["patterns"]:
        for c in range(4):
            ch = pat["channels"].get(f"ch{c + 1}") or {}
            if not ch.get("active"):
                continue
            passo = int(ch.get("speed") or pat.get("ref_speed") or 16) / TICKS_POR_SEGUNDO
            for i, n in enumerate(ch.get("notes_unrolled") or []):
                ini = t0 + i * passo
                if n.get("volume", 0) > 0 and ini < segundos:
                    out.append((c, ini, passo, n["pitch"]))
        t0 += int(pat.get("pattern_ticks", 512)) / TICKS_POR_SEGUNDO
        if t0 >= segundos:
            break
    return out


def piano_roll(ax, song, segundos=12, titulo=""):
    for c, ini, dur, pitch in notas(song, segundos):
        ax.barh(pitch, dur * 0.92, left=ini, height=0.8, color=CORES_CANAL[c], linewidth=0)
    ax.set_xlim(0, segundos)
    ax.set_title(titulo, loc="left")
    ax.set_xlabel("tempo (s)")
    ax.grid(axis="y", visible=False)


def salvar(fig, out, nome):
    fig.savefig(out / nome, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("salvo", out / nome)


def fig_funil(stats, out):
    d = stats["dropped"]
    etapas = [("músicas extraídas", stats["songs_in"], CINZA)]
    resto = stats["songs_in"]
    for chave, rotulo in (("few_notes", "tira: menos de 16 notas"), ("custom_instruments", "tira: instrumentos customizados"),
                          ("incompatible_speeds", "tira: velocidades incompatíveis"), ("duplicate", "tira: duplicadas")):
        resto -= d[chave]
        etapas.append((f"{rotulo} (−{d[chave]:,})".replace(",", "."), resto, CINZA))
    etapas[-1] = (etapas[-1][0], resto, AZUL)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [3, 2]})
    for i, (rotulo, n, c) in enumerate(etapas):
        a.barh(i, n, color=c, height=0.62)
        a.text(n + 300, i, f"{n:,}".replace(",", "."), va="center", color=TINTA)
    a.set_yticks(range(len(etapas)), [e[0] for e in etapas])
    a.invert_yaxis()
    a.set_xlim(0, stats["songs_in"] * 1.15)
    a.set_title("Dos jogos às músicas usadas", loc="left")
    a.grid(axis="y", visible=False)
    a.set_xlabel("músicas que sobram")
    sp = stats["splits"]
    nomes = [("train", "treino"), ("val", "validação"), ("test", "teste")]
    for i, (k, rotulo) in enumerate(nomes):
        b.barh(i, sp[k]["songs"], color=AZUL, height=0.62)
        b.text(sp[k]["songs"] + 150, i, f"{sp[k]['songs']:,} músicas\n{sp[k]['games']:,} jogos".replace(",", "."), va="center")
    b.set_yticks(range(3), [n[1] for n in nomes])
    b.invert_yaxis()
    b.set_xlim(0, sp["train"]["songs"] * 1.4)
    b.set_title("Split 70/15/15 (um jogo nunca fica em dois lados)", loc="left")
    b.grid(axis="y", visible=False)
    b.set_xlabel("músicas")
    fig.tight_layout()
    salvar(fig, out, "funil_e_split.png")


def fig_treino(art, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    for ax, modo in zip(axes, ("nlm", "grid_block")):
        h = json.loads((art / "checkpoints" / modo / "history.json").read_text())
        ep = np.arange(1, len(h["train"]) + 1)
        ax.plot(ep, h["train"], color=CINZA, lw=2, marker="o", ms=4, label="treino")
        ax.plot(ep, h["val"], color=AZUL, lw=2, marker="o", ms=4, label="validação")
        melhor = int(np.argmin(h["val"]))
        ax.scatter([ep[melhor]], [h["val"][melhor]], s=140, facecolor="none", edgecolor=TINTA, lw=1.6, zorder=5)
        ax.annotate(f"best.pt: época {ep[melhor]}", (ep[melhor], h["val"][melhor]), xytext=(8, 14),
                    textcoords="offset points", color=TINTA)
        ax.set_title(f"{modo}: loss por época", loc="left")
        ax.set_xlabel("época")
        ax.set_ylabel("loss por token")
        ax.legend(frameon=False)
    fig.tight_layout()
    salvar(fig, out, "treino.png")


def tabelas(art):
    simb = {}
    with open(art / "resultados" / "simbolico" / "distribution_compare.csv") as f:
        for r in csv.DictReader(f):
            if r["primary"] == "True" and r["system"] in NOMES:
                simb.setdefault(r["system"], []).append((float(r["oa"]), float(r["ks"])))
    simb = {s: (np.mean([v[0] for v in vs]), np.mean([v[1] for v in vs])) for s, vs in simb.items()}
    with open(art / "resultados" / "audio" / "audio_metrics.csv") as f:
        audio = {r["set"]: (float(r["fad"]), float(r["fad_std"])) for r in csv.DictReader(f)}
    return simb, audio


def fig_resultados(art, out):
    simb, audio = tabelas(art)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    paineis = [("OA (maior é melhor)", [simb[s][0] for s in ORDEM], None),
               ("KS (menor é melhor)", [simb[s][1] for s in ORDEM], None),
               ("FAD (menor é melhor)", [audio[s][0] for s in ORDEM], [audio[s][1] for s in ORDEM])]
    for ax, (titulo, vals, err) in zip(axes, paineis):
        y = np.arange(len(ORDEM))
        ax.barh(y, vals, xerr=err, color=[cor(s) for s in ORDEM], height=0.62, ecolor=TINTA2)
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * 0.03 + (err[yi] if err else 0), yi, f"{v + 1e-9:.2f}".replace(".", ",") if not err else f"{v:.3f}".replace(".", ","), va="center")
        ax.set_yticks(y, [NOMES[s] for s in ORDEM] if ax is axes[0] else [""] * len(ORDEM))
        ax.invert_yaxis()
        ax.set_xlim(0, max(vals) * 1.25)
        ax.set_title(titulo, loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
    piso = audio["real_val"][0]
    axes[2].axvline(piso, color=TINTA, lw=1.4, ls="--")
    axes[2].text(piso, len(ORDEM) - 0.42, f" piso real × real: {piso:.3f}".replace(".", ","), va="center", color=TINTA)
    axes[2].set_ylim(len(ORDEM) - 0.1, -0.6)
    fig.tight_layout()
    salvar(fig, out, "resultados.png")


def fig_distribuicoes(art, out):
    pasta = art / "resultados" / "simbolico" / "por-musica"
    series = [("real_test", "real (teste)", TINTA), ("lstm-grid_block-k10", "LSTM grid_block top-k 10", AZUL), ("markov", "Markov", LARANJA)]
    metricas = [("pitch_entropy", "Pitch entropy (bits)"), ("scale_consistency", "Scale consistency"), ("groove_consistency", "Groove consistency")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, (m, titulo) in zip(axes, metricas):
        for arq, rotulo, c in series:
            vals = []
            with open(pasta / f"{arq}.csv") as f:
                for r in csv.DictReader(f):
                    try:
                        vals.append(float(r[m]))
                    except (ValueError, KeyError):
                        pass
            lo, hi = (0, 3.6) if m == "pitch_entropy" else (0, 1)
            h, bordas = np.histogram(vals, bins=40, range=(lo, hi), density=True)
            ax.plot((bordas[:-1] + bordas[1:]) / 2, h, color=c, lw=2, label=rotulo)
        ax.set_title(titulo, loc="left")
        ax.set_yticks([])
        ax.set_ylabel("quantas músicas")
        ax.grid(axis="y", visible=False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    salvar(fig, out, "distribuicoes.png")


def fig_piano_rolls(art, test_path, out):
    real = ler_jsonl(test_path, 400)
    gerada = ler_jsonl(art / "predicoes-teste" / "grid_block-k10.jsonl.gz", 60)
    markov = ler_jsonl(art / "predicoes-teste" / "markov.jsonl.gz", 60)

    def escolher(songs):  # a primeira com 3+ canais soando ao longo dos 12 s
        for s in songs:
            ns = notas(s, 12)
            graves = sum(n[3] < 5 for n in ns)
            if len({n[0] for n in ns}) >= 3 and len(ns) > 80 and graves < 0.1 * len(ns) and max(n[1] for n in ns) > 11:
                return s
        return songs[0]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    for ax, (songs, titulo) in zip(axes, ((real, "Real (teste)"), (gerada, "Gerada: LSTM grid_block top-k 10"), (markov, "Gerada: Markov"))):
        piano_roll(ax, escolher(songs), 12, titulo)
    axes[0].set_ylabel("altura (pitch do PICO-8)")
    for c in range(4):
        axes[2].barh([np.nan], [0], color=CORES_CANAL[c], label=f"canal {c + 1}")
    axes[2].legend(frameon=False, fontsize=8, ncol=4, loc="upper right", bbox_to_anchor=(1, 1.22))
    fig.tight_layout()
    salvar(fig, out, "piano_rolls.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artefatos", required=True, type=Path)
    p.add_argument("--splits", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args()
    estilo()
    test = args.splits / "test.jsonl"
    test = test if test.exists() else args.splits / "test.jsonl.gz"
    fig_funil(json.loads((args.splits / "build_stats.json").read_text()), args.out)
    fig_treino(args.artefatos, args.out)
    fig_resultados(args.artefatos, args.out)
    fig_distribuicoes(args.artefatos, args.out)
    fig_piano_rolls(args.artefatos, test, args.out)


if __name__ == "__main__":
    main()
