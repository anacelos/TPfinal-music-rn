# Resultados: o que já foi rodado e como ler

Estado em 25/11/2026. Tudo abaixo usa o split congelado 70/15/15
(`data/splits/split_manifest.csv`: 8.669 músicas de treino, 1.861 de validação, 1.854 de teste)
e o código da `main`: `src/train.py`, `src/sample.py`, `src/test_perplexity.py`, `src/synth.py`.

Os arquivos (checkpoints, predições e tabelas completas) estão no Drive, com um README que mapeia
cada um: https://drive.google.com/drive/folders/1wirkz0ZRidwRh7db2YXRPaPeTJ1zr4cO

## 1. Glossário

### Tokenizadores (`src/tokenizer.py`)

O modelo é o mesmo (LSTM de 2 camadas); muda a forma de escrever a música como sequência de tokens.

- **nlm**: lista de eventos, como no LakhNES. Cada token diz que algo mudou num canal (nota
  ligada, nota desligada, instrumento, volume, efeito) ou que o tempo avançou. Vocabulário de
  709 tokens; 711 tokens por música em média.
- **grid_block**: a grade do tracker do PICO-8, pattern por pattern, passo por passo. Cada bloco
  de 32 passos é escrito uma vez só; quando ele se repete na música, um único token aponta para
  ele. Vocabulário de 1.250 tokens; 615 tokens por música em média.

### Sistemas comparados

| Nome | O que é |
|---|---|
| `lstm-nlm-t1` | LSTM com nlm, amostragem com temperatura 1, sem filtro |
| `lstm-nlm-k10` | LSTM com nlm, temperatura 1 e top-k 10 |
| `lstm-grid_block-t1` | LSTM com grid_block, temperatura 1, sem filtro |
| `lstm-grid_block-k10` | LSTM com grid_block, temperatura 1 e top-k 10 |
| `markov` | baseline de Markov do grupo (`evaluation/markov_baseline.py`) |
| `real_test` | as 1.854 músicas reais do teste (a referência) |
| `real_val` | músicas reais da validação; mostram quanto duas amostras de música real diferem entre si |

Cada sistema gerado tem 1.854 músicas, o tamanho do teste. As músicas são geradas do zero: a
música `i` de um sistema não corresponde à música `i` do teste. Toda comparação é entre
distribuições.

### Métricas do modelo

- **Loss**: entropia cruzada média por token (em nats). Menor é melhor.
- **Perplexidade**: `exp(loss)`. Entre quantos tokens, em média, o modelo fica em dúvida. Só
  compara modelos com o mesmo vocabulário.
- **Bits por segundo (bits/s)**: bits que o modelo gasta para codificar uma música, divididos
  pela duração dela. Não depende do vocabulário, então compara nlm com grid_block. Menor é melhor.

### Métricas simbólicas (calculadas nas notas)

Uma por música, calculadas nos inícios de nota (`evaluation/pico8_metrics.py`):

- **Pitch entropy (PE)**: entropia das 12 classes de altura, de 0 a 3,58 bits. Baixa = poucas
  notas diferentes; alta = notas espalhadas, sem tonalidade.
- **Scale consistency (SC)**: fração das notas que cabem na escala maior que melhor se ajusta,
  de 0 a 1.
- **Groove consistency (GC)**: quanto o padrão rítmico de um trecho se parece com o do trecho
  seguinte, de 0 a 1. Fica indefinida em músicas com menos de dois trechos.

O objetivo não é ter valor alto, é ter a mesma distribuição das músicas reais. Para isso
(`evaluation/distribution_compare.py`, protocolo de Yang & Lerch, 2020):

- **OA (overlap area)**: sobreposição entre as distâncias dentro do conjunto real e as distâncias
  entre gerado e real. De 0 a 1; maior é melhor.
- **KS**: maior diferença entre as distribuições acumuladas do gerado e do real. De 0 a 1; menor
  é melhor.

### Métricas de áudio (calculadas no som)

As músicas são sintetizadas (`src/synth.py`) e comparadas por modelos de áudio pré-treinados
(`evaluation/audio_metrics.py`):

- **FAD (Fréchet Audio Distance)**: distância entre a distribuição dos embeddings CLAP do
  conjunto gerado e a do teste real. Menor é melhor. O valor de `real_val` é o piso: o que duas
  amostras de música real marcam entre si.
- **KLD**: divergência KL entre as distribuições de tags do AudioSet (modelo AST) do gerado e do
  real. Menor é melhor.
- **CLAP score**: similaridade entre cada clipe e o texto "8-bit chiptune video game music". Diz
  se o som lembra o texto, não se parece com o dataset.

## 2. O que já foi rodado

| Etapa | Estado | Jobs no cluster | Onde está |
|---|---|---|---|
| Treino nlm e grid_block (`src/train.py`) | feito | 21901, 21902 | Drive `checkpoints/` |
| Loss e perplexidade no teste (`src/test_perplexity.py`) | feito | local | Drive `checkpoints/<modo>/test_metrics.json` |
| Geração de 1.854 músicas por sistema (`src/sample.py`) | feito | 22979 a 22982 | Drive `predicoes-teste/` |
| Métricas simbólicas (PE, SC, GC, OA, KS) | feito | 22984, 22985 | `docs/resultados/`, Drive `resultados/simbolico/` |
| Métricas de áudio (FAD, KLD, CLAP score) | feito | 22986 | `docs/resultados/`, Drive `resultados/audio/` |
| Curvas de treino (`src/plot_history.py`) | feito | local | Drive `checkpoints/<modo>/history.json` |
| Escolha do top-k pela validação | não feito | | |
| Teste de escuta | no ar, coletando respostas | | [ETAPAS.md, etapa 11](ETAPAS.md#11-teste-de-escuta), <https://pico8-bh.giow.dev> |

## 3. Resultados

### Modelo

| Tokenizador | Melhor época | Val loss | Val bits/s | Test loss | Test perplexidade |
|---|---|---|---|---|---|
| nlm | 4 de 10 | 2,0429 | 85,6 | 2,0607 | 7,85 |
| grid_block | 8 de 10 | 0,7723 | 27,9 | 0,7668 | 2,15 |

O nlm passa a piorar na validação depois da época 4 (overfitting); o `best.pt` é o da época 4.

### Simbólico

Referência: `real_test`. Média real: PE 2,27, SC 0,93, GC 0,88.

| Sistema | OA médio | KS médio | PE (média / KS) | SC (média / KS) | GC (média / KS) |
|---|---|---|---|---|---|
| `lstm-grid_block-k10` | **0,89** | **0,22** | 2,23 / 0,06 | 0,95 / 0,15 | 0,75 / 0,45 |
| `lstm-grid_block-t1` | 0,86 | 0,37 | 2,62 / 0,31 | 0,88 / 0,32 | 0,73 / 0,49 |
| `lstm-nlm-t1` | 0,83 | 0,35 | 2,37 / 0,12 | 0,91 / 0,34 | 0,66 / 0,60 |
| `lstm-nlm-k10` | 0,81 | 0,38 | 1,81 / 0,30 | 0,97 / 0,26 | 0,68 / 0,57 |
| `markov` | 0,79 | 0,53 | 3,27 / 0,75 | 0,76 / 0,69 | 0,89 / 0,16 |

Tabela completa, com as métricas de diagnóstico: `docs/resultados/simbolico_distribution_compare.csv`.

### Áudio

Referência: 400 músicas de `real_test`. 200 clipes de 10 s por sistema.

| Sistema | FAD (± desvio) | FAD ÷ piso | KLD | CLAP score |
|---|---|---|---|---|
| `real_val` (piso) | 0,096 ± 0,006 | 1,00 | 0,019 | 0,236 |
| `lstm-grid_block-k10` | **0,118** ± 0,009 | 1,22 | 0,040 | 0,269 |
| `lstm-grid_block-t1` | 0,125 ± 0,011 | 1,30 | **0,038** | 0,293 |
| `lstm-nlm-k10` | 0,145 ± 0,008 | 1,51 | 0,072 | 0,279 |
| `lstm-nlm-t1` | 0,171 ± 0,010 | 1,78 | 0,085 | 0,284 |
| `markov` | 0,323 ± 0,012 | 3,35 | 0,213 | 0,336 |

Tabela completa: `docs/resultados/audio_metrics.csv`.

### Leitura

- O grid_block vence o nlm em tudo: bits/s, métricas simbólicas, FAD e KLD.
- As quatro LSTMs ficam à frente do Markov nas métricas simbólicas (na média) e no áudio.
- O top-k 10 corrige a pitch entropy do grid_block (2,62 → 2,23; real 2,27). No nlm ele
  empobrece as notas (1,81).
- O ponto fraco das LSTMs é a groove consistency (KS de 0,45 a 0,60). O Markov tem o melhor
  groove porque copia a estrutura de uma música de treino, e é o pior em altura e escala.
- O Markov tem o maior CLAP score e o pior FAD: soar como "chiptune" para o CLAP não é o mesmo
  que soar como o dataset.

## 4. Como reproduzir

```bash
# dados
python3 -s -m preprocessing.build_dataset
python3 src/data.py --out-dir data/processed

# treino, teste e geração (de dentro de src/)
python train.py --data-dir ../data/processed/grid_block --run-name grid_block
python test_perplexity.py --checkpoint outputs/grid_block/best.pt --data-dir ../data/processed/grid_block
python sample.py --checkpoint outputs/grid_block/best.pt --mode grid_block --n-samples 1854 \
    --top-k 10 --out ../results/lstm-grid_block-k10.jsonl

# métricas simbólicas (da raiz)
python evaluation/evaluate_split.py --input data/splits/test.jsonl --output results/real_test.csv
python evaluation/evaluate_split.py --input results/lstm-grid_block-k10.jsonl --output results/lstm-grid_block-k10.csv
python evaluation/distribution_compare.py --real results/real_test.csv \
    --system lstm-grid_block-k10=results/lstm-grid_block-k10.csv --out-dir results/compare

# áudio
python evaluation/synthesize_jsonl.py --input data/splits/test.jsonl --output-dir wav/real_test \
    --sfx-filters src/sfx_filters.json
python evaluation/synthesize_jsonl.py --input results/lstm-grid_block-k10.jsonl --output-dir wav/lstm-grid_block-k10
python evaluation/audio_metrics.py --reference real_test=wav/real_test \
    --set lstm-grid_block-k10=wav/lstm-grid_block-k10 --out-dir results/audio
```

Para o nlm, troque `grid_block` por `nlm`. Para a amostragem sem filtro, tire `--top-k 10`.
A geração usou `--max-len 1828` e `--seed 42`, os padrões do `sample.py`.

## 5. Limitações

- O top-k 10 não foi ajustado na validação.
- Os números de áudio vieram de clipes cortados em 10 s por um script do cluster (as 200
  primeiras músicas de cada sistema; 400 sorteadas do teste real). O `synthesize_jsonl.py`
  sintetiza a música inteira, então refazer pelo caminho acima dá valores próximos, não idênticos.
- O bits/s foi calculado só na validação, por um script do cluster.
- Filtros de SFX (noiz, buzz, detune, reverb, dampen) não estão nos tokens. As músicas reais são
  sintetizadas com eles e as geradas sem. Tirar os filtros das reais reduz a distância das LSTMs
  ao piso do FAD em cerca de 13% e não muda a ordem dos sistemas.
- A groove consistency fica indefinida em parte das músicas (673 das 1.854 reais; de 245 a 426
  das geradas pelas LSTMs; 630 do Markov) e é calculada só nas demais.
- O `test_perplexity.py` conta duas vezes os tokens da última janela de cada música longa.
