# Avaliação do TP2

Esta pasta contém a avaliação simbólica/estrutural do projeto de geração musical autoregressiva em PICO-8.

O objetivo é comparar, com o mesmo pipeline de métricas:

- músicas reais do test set;
- músicas geradas pelo baseline;
- músicas geradas pelo modelo neural principal.

A avaliação é feita na representação simbólica canônica de música utilizada pelo projeto, sem necessidade de conversão para MIDI.

## Pipeline atual

O fluxo oficial é:

```text
songs.json
    ↓
preprocessing
    ↓
data/splits/{train,val,test}.jsonl
    ↓
música canônica / saída decodificada
    ↓
evaluate_song_data()
    ↓
CSV por música
    ↓
compare_results.py
    ↓
estatísticas + KS + Wasserstein-1 + plots
```

Para músicas geradas, os tokens devem primeiro ser decodificados pelo tokenizer.

A avaliação é executada sobre a música canônica reconstruída, e não diretamente sobre os tokens.

## Test set real

O preprocessing final produz:

- train: 9.907 músicas;
- validation: 1.242 músicas;
- test: 1.235 músicas.

O split evita vazamento entre jogos relacionados por material musical compartilhado, conforme definido pelo pipeline de preprocessing.

A avaliação canônica foi validada sobre as 1.235 músicas do test set.

## Métricas primárias

As quatro métricas abaixo podem ser calculadas após o decode das diferentes representações e são usadas na comparação principal entre sistemas.

### Pitch Range

Diferença entre o maior e o menor pitch audível da música.

Somente steps com `volume > 0` são considerados.

### Normalized Pitch-Class Entropy

Entropia de Shannon das classes de pitch (`pitch % 12`), normalizada por `log2(12)`.

Valores maiores indicam maior diversidade de classes de pitch.

### Audible Step Rate

Quantidade de steps simbólicos audíveis por segundo.

Um step é considerado audível quando possui `volume > 0`.

O relógio temporal segue o sintetizador atual:

```text
SAMPLE_RATE = 44100
SAMPLES_PER_TICK = 366

duration_seconds = total_ticks * 366 / 44100
```

Portanto:

```text
audible_step_rate = audible_steps / duration_seconds
```

O nome "step rate" é usado porque os elementos de `notes_unrolled` representam steps simbólicos, e não necessariamente ataques musicais independentes.

### Audible Channel Utilization

Mede a ocupação audível da capacidade total dos quatro canais do PICO-8 ao longo do tempo.

Para cada step audível, sua duração é determinada por `speed`, limitada pelo fim do pattern.

```text
audible_channel_utilization =
    audible_channel_ticks / (4 * total_ticks)
```

O resultado fica entre 0 e 1.

## Métricas estruturais auxiliares

Estas métricas dependem da preservação explícita da estrutura de blocos/SFX.

### Structural Note Density

Proporção de slots audíveis entre os slots estruturais disponíveis nos blocos referenciados.

Cada bloco canônico possui 32 slots. Reutilizações de um bloco contam novamente porque representam novas ocorrências durante a execução da música.

Esta métrica é calculada quando `blocks_dictionary` e `block_id` estão disponíveis.

### Block Repetition Ratio

Proporção de referências ativas que reutilizam um bloco já definido:

```text
block_repetition_ratio =
    repeated_block_references / active_block_references
```

No formato `grid_block`, a primeira ocorrência é representada como definição do bloco e ocorrências posteriores como reutilizações.

As representações `nlm` e `grid_unrolled` não preservam necessariamente a identidade dos blocos após o decode. Nesses casos, as métricas dependentes dessa informação retornam `None` e não fazem parte da comparação principal entre representações.

## Distribuições auxiliares

O evaluator também coleta:

- instrumentos;
- efeitos;
- velocidades (`speed`);
- número de patterns;
- duração;
- número de referências ativas;
- número de steps audíveis;
- utilização de referências de canal.

Essas informações podem ser usadas para análises adicionais, mas não fazem parte do conjunto padrão de quatro métricas de comparação.

## Comparação entre sistemas

Para cada métrica primária, `compare_results.py` calcula:

- número de amostras;
- média;
- mediana;
- desvio padrão;
- primeiro quartil;
- terceiro quartil;
- mínimo;
- máximo;
- estatística de Kolmogorov-Smirnov contra o conjunto real;
- distância Wasserstein-1 contra o conjunto real.

Menores valores de KS e Wasserstein indicam distribuições mais próximas da distribuição real.

## Estrutura da pasta

```text
evaluation/
├── README_AVALIACAO.md
├── METRICS.md
├── pico8_metrics.py
├── evaluate_split.py
├── compare_results.py
├── plot_distributions.py
├── bits_per_second.py         # NLL por música inteira, em bits por segundo
├── audio_clips_eval.py        # clipes de 10 s + audio_metrics.py
├── requirements.txt
├── evaluate_manifest.py        # pipeline legado
├── pieces.example.json         # exemplo legado
└── tests/
    ├── test_song_metrics.py
    ├── test_metrics.py
    ├── test_manifest.py
    ├── test_compare_results.py
    └── test_synth_alignment.py
```

`evaluate_manifest.py` e `pieces.example.json` pertencem ao protótipo anterior baseado em `audio.json` e pattern IDs. Eles são mantidos temporariamente para compatibilidade e testes, mas não constituem o pipeline oficial da avaliação final.

## Ambiente

A partir da raiz do repositório:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r evaluation/requirements.txt
```

## Rodando os testes

```bash
cd evaluation
python -m unittest discover -s tests -v
```

Estado validado:

```text
Ran 24 tests
OK
```

## Avaliando o test set real

A partir da raiz do repositório:

```bash
python evaluation/evaluate_split.py \
  --input data/splits/test.jsonl \
  --output results/real_test_metrics.csv
```

O test set final contém 1.235 músicas, portanto o CSV possui 1.236 linhas incluindo o cabeçalho.

## Comparando Real, Baseline e Neural

Quando os três conjuntos estiverem disponíveis:

```bash
python evaluation/compare_results.py \
  --real results/real_test_metrics.csv \
  --baseline results/baseline_metrics.csv \
  --neural results/neural_metrics.csv \
  --output results/comparison_summary.csv
```

A comparação padrão usa:

- `pitch_range`;
- `pitch_class_entropy`;
- `audible_step_rate`;
- `audible_channel_utilization`.

## Gerando os gráficos

```bash
python evaluation/plot_distributions.py \
  --real results/real_test_metrics.csv \
  --baseline results/baseline_metrics.csv \
  --neural results/neural_metrics.csv \
  --output-dir results/plots
```

São gerados boxplots das quatro métricas primárias.

## Bits por segundo

Compara tokenizadores: a perda e a perplexidade são por token, e cada tokenizador gasta um
número diferente de tokens por segundo de música. O script lê cada música do split inteira,
soma a NLL de todos os tokens e divide pela duração da música (`src/nll.py`).

```bash
python evaluation/bits_per_second.py \
  --checkpoint outputs/grid_block/best.pt \
  --data-dir data/processed/grid_block \
  --split val
```

Com os checkpoints dos jobs 21901 (nlm) e 21902 (grid_block), dá 85,59 e 27,88 bits/s na
validação, os valores de `docs/RESULTADOS.md`.

## Métricas de áudio em clipes de 10 s

Sorteia 400 músicas do teste e 200 da validação (semente 831), pega as 200 primeiras de cada
sistema, corta cada música em 10 s, sintetiza (`src/render.py`) e chama o `audio_metrics.py`.
Pede `torch` além do `requirements.txt`.

```bash
python evaluation/audio_clips_eval.py \
  --test data/splits/test.jsonl --val data/splits/val.jsonl \
  --system lstm-grid_block=outputs/grid_block/samples/samples.jsonl \
  --system markov=results/markov/samples.jsonl \
  --out-dir results/audio
```

`--quick` roda com 24 e 12 músicas, só para testar o caminho. `--render-only` para nos WAVs.

## Resultados gerados

A pasta `results/` contém artefatos produzidos localmente e está ignorada pelo Git.

Os resultados atuais do test set real servem como referência para as futuras comparações com baseline e modelo neural.

Os smoke tests que reutilizam o mesmo CSV nos três sistemas são apenas validação da infraestrutura e não constituem resultados experimentais do TP2.
