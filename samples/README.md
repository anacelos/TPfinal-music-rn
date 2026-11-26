# Músicas geradas

Cinco músicas do modelo (`lstm-grid_block-k10`: LSTM com o tokenizador grid_block, amostragem
top-k 10) e cinco do baseline (`markov`), cada uma em dois formatos:

| Arquivo | O que é |
|---|---|
| `.json` | A música no formato nativo do PICO-8 usado no projeto (o mesmo do `songs.json` do dataset), como saiu do `src/sample.py` ou do `evaluation/markov_baseline.py`. É o que o tokenizador e o synth leem. |
| `.wav` | O áudio, sintetizado por `src/synth.py` (o sintetizador do PICO-8, 22050 Hz, mono, sem filtros de SFX). |

O projeto trabalha no formato do PICO-8, não em MIDI: cada canal é uma sequência de passos com
pitch (0 a 63), instrumento (8 formas de onda), volume (0 a 7) e efeito, e cada passo dura
`speed` ticks de 183/22050 s.

## Como as músicas foram escolhidas

Sem escolha à mão: são as 5 primeiras de cada arquivo de predições (as 1.854 músicas de cada
sistema, na ordem em que foram geradas, `--seed 42`) que têm pelo menos 16 notas audíveis, o
mesmo filtro `few_notes` do dataset. Nos dois sistemas, as 5 primeiras já passam no filtro.

No PICO-8 a música de um jogo toca em loop. Por isso, no `.wav`, as músicas com loop e mais
curtas que 20 s têm o trecho do loop repetido até passar de 20 s, e as mais longas que 60 s são
cortadas em 60 s. O `.json` guarda a música inteira, uma passada só. `index.csv` lista, para cada
exemplo, a linha de origem, a duração, as repetições do loop e se houve corte.

| Exemplo | Duração do WAV | Observação |
|---|---|---|
| `lstm-grid_block-k10_01` | 21,2 s | 1 pattern de 4,2 s, loop repetido 4 vezes |
| `lstm-grid_block-k10_02` | 60,0 s | 111 s numa passada, cortada em 60 s |
| `lstm-grid_block-k10_03` | 59,5 s | |
| `lstm-grid_block-k10_04` | 38,2 s | |
| `lstm-grid_block-k10_05` | 8,5 s | sem loop |
| `markov_01` | 25,5 s | 12,7 s numa passada, loop repetido 1 vez |
| `markov_02` | 53,1 s | |
| `markov_03` | 25,5 s | |
| `markov_04` | 23,9 s | 1 pattern de 4,8 s, loop repetido 4 vezes |
| `markov_05` | 3,2 s | termina com `stop`, sem loop |

## Como refazer

Com a pasta `predicoes-teste/` do Drive (ver `docs/RESULTADOS.md`):

```bash
python evaluation/export_samples.py \
    --system lstm-grid_block-k10=<predicoes-teste>/grid_block-k10.jsonl.gz \
    --system markov=<predicoes-teste>/markov.jsonl.gz \
    --n 5 --out-dir samples
```
