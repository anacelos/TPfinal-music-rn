"""Apoio dos notebooks: caminhos, leitura dos arquivos, figuras, áudio e o modelo.

Os notebooks rodam no computador de cada um, sem cluster. Eles usam o código da `main`
(`src/`, `evaluation/`) e a pasta do Drive do projeto, baixada para `artefatos/` na raiz do
repositório (ou para o caminho em NB_ARTEFATOS). O `notebooks/README.md` diz como baixar.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import os
import sys
import wave
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ARTEFATOS = Path(os.environ.get("NB_ARTEFATOS", ROOT / "artefatos"))
if not (ARTEFATOS / "checkpoints").exists():
    raise FileNotFoundError(
        f"Não achei a pasta do Drive em {ARTEFATOS}. Baixe-a como descrito em notebooks/README.md "
        "ou aponte NB_ARTEFATOS para ela."
    )

AZUL, LARANJA, VERDE, VIOLETA, CINZA = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#8a8984"
TINTA, TINTA2, FUNDO, GRADE = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
CORES_CANAL = [AZUL, LARANJA, VERDE, VIOLETA]
TICKS_POR_SEGUNDO = 44100 / 366  # o PICO-8 toca ~120,5 ticks por segundo
NOTAS = ["dó", "dó#", "ré", "ré#", "mi", "fá", "fá#", "sol", "sol#", "lá", "lá#", "si"]
INSTRUMENTOS = ["triângulo", "serra inclinada", "serra", "quadrada", "pulso", "órgão", "ruído", "phaser"]
EFEITOS = ["nenhum", "slide", "vibrato", "drop", "fade in", "fade out", "arpejo rápido", "arpejo lento"]

SISTEMAS = {
    "real_test": ("real (teste)", TINTA),
    "lstm-grid_block-k10": ("LSTM grid_block, top-k 10", AZUL),
    "lstm-grid_block-t1": ("LSTM grid_block, T = 1", "#86b6ef"),
    "lstm-nlm-k10": ("LSTM nlm, top-k 10", LARANJA),
    "lstm-nlm-t1": ("LSTM nlm, T = 1", "#f2a586"),
    "markov": ("Markov", CINZA),
}


def estilo():
    """Aparência comum das figuras."""
    plt.rcParams.update({
        "figure.facecolor": FUNDO, "axes.facecolor": FUNDO, "savefig.facecolor": FUNDO,
        "axes.edgecolor": GRADE, "axes.labelcolor": TINTA2, "text.color": TINTA,
        "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.grid": True, "grid.color": GRADE,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.axisbelow": True,
        "figure.dpi": 110,
    })


def nome(sistema: str) -> str:
    return SISTEMAS.get(sistema, (sistema, CINZA))[0]


def cor(sistema: str) -> str:
    return SISTEMAS.get(sistema, (sistema, CINZA))[1]


# ---------------------------------------------------------------------------
# Arquivos
# ---------------------------------------------------------------------------
def ler_json(path):
    return json.loads(Path(path).read_text())


def ler_jsonl(path, limite=None) -> list[dict]:
    path = Path(path)
    abrir = gzip.open if path.suffix == ".gz" else open
    out = []
    with abrir(path, "rt") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
                if limite and len(out) >= limite:
                    break
    return out


def ler_csv(path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def musicas_reais(limite=None) -> list[dict]:
    """Músicas reais do teste (artefatos/dados/test.jsonl.gz)."""
    return ler_jsonl(ARTEFATOS / "dados" / "test.jsonl.gz", limite)


def musicas_geradas(sistema: str, limite=None) -> list[dict]:
    """Músicas de artefatos/predicoes-teste: 'grid_block-k10', 'nlm-t1', 'markov', ..."""
    return ler_jsonl(ARTEFATOS / "predicoes-teste" / f"{sistema}.jsonl.gz", limite)


def metricas_por_musica(sistema: str) -> list[dict]:
    """PE, SC, GC etc. por música: 'real_test', 'lstm-grid_block-k10', 'markov', ..."""
    return ler_csv(ARTEFATOS / "resultados" / "simbolico" / "por-musica" / f"{sistema}.csv")


def coluna(linhas: list[dict], chave: str) -> np.ndarray:
    """Valores numéricos de uma coluna; músicas sem valor ficam de fora."""
    out = []
    for r in linhas:
        try:
            out.append(float(r[chave]))
        except (KeyError, ValueError, TypeError):
            pass
    return np.array(out)


# ---------------------------------------------------------------------------
# Música: notas, figuras, áudio
# ---------------------------------------------------------------------------
def nome_nota(pitch: int) -> str:
    """Pitch do PICO-8 (0 = dó da oitava 0) por extenso."""
    return f"{NOTAS[pitch % 12]}{pitch // 12}"


def duracao(song) -> float:
    return sum(int(p.get("pattern_ticks", 512)) for p in song["patterns"]) / TICKS_POR_SEGUNDO


def notas(song, segundos=None) -> list[tuple]:
    """(canal, início em s, duração em s, pitch, instrumento) das notas audíveis."""
    out, t0 = [], 0.0
    for pat in song["patterns"]:
        for c in range(4):
            ch = pat["channels"].get(f"ch{c + 1}") or {}
            if not ch.get("active"):
                continue
            passo = int(ch.get("speed") or pat.get("ref_speed") or 16) / TICKS_POR_SEGUNDO
            for i, n in enumerate(ch.get("notes_unrolled") or []):
                ini = t0 + i * passo
                if n.get("volume", 0) > 0 and (segundos is None or ini < segundos):
                    out.append((c, ini, passo, n["pitch"], n.get("instrument", 0)))
        t0 += int(pat.get("pattern_ticks", 512)) / TICKS_POR_SEGUNDO
        if segundos is not None and t0 >= segundos:
            break
    return out


def piano_roll(song, ax=None, segundos=12, titulo="", legenda=True):
    """Cada traço é uma nota: posição = tempo, altura = pitch, cor = canal."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    for c, ini, dur, pitch, _ in notas(song, segundos):
        ax.barh(pitch, dur * 0.92, left=ini, height=0.8, color=CORES_CANAL[c], linewidth=0)
    ax.set_xlim(0, segundos)
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("altura (pitch)")
    ax.set_title(titulo, loc="left")
    ax.grid(axis="y", visible=False)
    if legenda:
        for c in range(4):
            ax.barh([np.nan], [0], color=CORES_CANAL[c], label=f"canal {c + 1}")
        ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower right", bbox_to_anchor=(1, 1))
    return ax


def tracker(song, pattern=0, passos=16) -> str:
    """Um pattern como o editor do PICO-8 mostra: uma linha por passo, uma coluna por canal."""
    pat = song["patterns"][pattern]
    linhas = ["passo | " + " | ".join(f"canal {c}      " for c in range(1, 5)), "-" * 66]
    for i in range(passos):
        celulas = []
        for c in range(1, 5):
            ch = pat["channels"].get(f"ch{c}") or {}
            ns = ch.get("notes_unrolled") or []
            if ch.get("active") and i < len(ns) and ns[i].get("volume", 0) > 0:
                n = ns[i]
                celulas.append(f"{nome_nota(n['pitch']):<5} i{n['instrument']} v{n['volume']} f{n['effect']}")
            else:
                celulas.append("·" + " " * 13)
        linhas.append(f"{i:>5} | " + " | ".join(celulas))
    return "\n".join(linhas)


def audio(x, taxa_origem, taxa=16000):
    """Player para um sinal mono; reamostra para `taxa` para o notebook ficar leve."""
    from IPython.display import Audio

    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        print("(sem som)")
        return None
    idx = np.arange(0, len(x), taxa_origem / taxa)
    y = np.interp(idx, np.arange(len(x)), x)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(taxa)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
    return Audio(buf.getvalue(), rate=taxa)


def tocar(song, segundos=8.0, taxa=16000):
    """Player com os primeiros `segundos` da música, sintetizada por src/synth.py."""
    from synth import PICO_RATE, synthesize_song

    x = synthesize_song(song, rate=PICO_RATE)[: int(segundos * PICO_RATE)]
    return audio(x, PICO_RATE, taxa)


def tabela(linhas: list[dict]):
    """Lista de dicionários como tabela (Markdown)."""
    from IPython.display import Markdown

    cols = list(linhas[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in linhas]
    return Markdown("\n".join(out))


def br(x, casas=2) -> str:
    """Número com vírgula decimal."""
    return f"{x:.{casas}f}".replace(".", ",")


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
def tokenizador(modo: str):
    from tokenizer import Pico8Tokenizer

    return Pico8Tokenizer(mode=modo)


def mostrar_tokens(tokens, n=48, por_linha=8):
    for i in range(0, min(n, len(tokens)), por_linha):
        print("  ".join(f"{t:<16}" for t in tokens[i:i + por_linha]))
    if len(tokens) > n:
        print(f"... ({len(tokens)} tokens no total)")


def categoria(token: str) -> str:
    """Família do token, para contar o que há no vocabulário."""
    t = token.split("_", 1)[1] if token.startswith("CH") and "_" in token else token
    for prefixo, rotulo in (("NOTEON", "nota liga"), ("NOTEOFF", "nota desliga"), ("PITCH", "altura"),
                            ("INST", "instrumento"), ("VOL", "volume"), ("FX", "efeito"), ("WT", "espera"),
                            ("STEP", "passo"), ("SPEED", "velocidade"), ("REF_SPEED", "velocidade"), ("REF_STEPS", "passo"),
                            ("DEF_BLOCK", "bloco"), ("USE_BLOCK", "bloco"), ("END_BLOCK", "bloco"),
                            ("BLK_", "bloco")):
        if t.startswith(prefixo):
            return rotulo
    return "estrutura"


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def carregar_modelo(modo: str, device="cpu"):
    """(modelo em modo de avaliação, tokenizador, checkpoint) de artefatos/checkpoints/<modo>."""
    import torch
    from model import ModelMusic

    ckpt = torch.load(ARTEFATOS / "checkpoints" / modo / "best.pt", map_location="cpu")
    cfg = ckpt["config"]
    modelo = ModelMusic(vocab_size=cfg["vocab_size"], embedding_dim=cfg["embedding_dim"],
                        hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"],
                        padding_idx=cfg["pad_id"], dropout=cfg["dropout"])
    modelo.load_state_dict(ckpt["model_state"])
    modelo.to(device).eval()
    tok = tokenizador(modo)
    assert len(tok.token_to_id) == cfg["vocab_size"]
    return modelo, tok, ckpt


def gerar(modelo, tok, n=1, temperatura=1.0, top_k=0, max_len=1828, seed=42, device="cpu") -> list[dict]:
    """Gera `n` músicas com a função `gerar` de src/sample.py e as devolve já decodificadas."""
    import torch
    import sample

    gerador = torch.Generator(device=device).manual_seed(seed)
    musicas = []
    for i in range(n):
        ids = sample.gerar(modelo, tok.bos_id, tok.eos_id, max_len, temperatura, top_k, gerador, device)
        m = tok.decode_ids(ids)
        m["song_id"], m["token_ids"] = i, ids
        musicas.append(m)
    return musicas
