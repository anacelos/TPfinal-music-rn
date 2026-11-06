# Avaliação — TP2 IA Generativa para Música

Esta pasta contém a infraestrutura da **Parte 4 — Avaliação/Métricas** do TP2.

O objetivo é usar o mesmo pipeline para comparar:

- músicas reais do conjunto de teste;
- músicas geradas pelo baseline;
- músicas geradas pelo modelo autoregressivo principal.

A avaliação é feita diretamente na representação simbólica do **PICO-8**, sem conversão para MIDI.

## Status

A parte independente do restante do projeto já está implementada.

**Estado atual: 20 testes automatizados passando.**

Já estão disponíveis:

- parser da representação `audio.json`;
- avaliação de conjuntos arbitrários de music patterns;
- métricas estruturais PICO-8;
- avaliação em lote via manifesto;
- exportação dos resultados para CSV;
- estatísticas descritivas;
- comparação Real × Baseline × Neural;
- KS e Wasserstein-1 como medidas candidatas de distância entre distribuições;
- geração automática de boxplots;
- testes para edge cases da representação;
- alinhamento do parser de avaliação com o sintetizador original do dataset.

Mais detalhes metodológicos estão em `METRICS.md`.

## Métricas implementadas

### Pitch Range

Diferença entre o maior e o menor pitch das notas audíveis.

### Normalized Pitch-Class Entropy

Entropia das classes de pitch (`pitch % 12`), normalizada pelo máximo possível para 12 classes.

### Structural Note Density

Proporção de slots com `volume > 0` entre os slots disponíveis nos SFX utilizados pela peça.

É uma métrica estrutural do PICO-8, não uma medida de notas por segundo.

### Audible Channel Utilization

Proporção dos quatro canais PICO-8 que realmente produzem som ao longo dos patterns avaliados.

### SFX Repetition Ratio

Mede a reutilização de SFX dentro da peça.

Além dessas métricas, o pipeline coleta distribuições de:

- instrumentos;
- efeitos;
- `speed`;
- flags;
- canais habilitados;
- SFX explícitos ou implicitamente vazios.

## Regras da representação usadas atualmente

A análise do `extract_audio.py` e do `synthesize_audio.py` originais do dataset confirmou que:

- valores de canal `0..63` referenciam diretamente SFX `0..63`;
- valores de canal `>= 64` não geram áudio;
- `volume > 0` define um slot audível;
- cada SFX possui 32 slots;
- um SFX referenciado mas ausente é tratado como vazio;
- apenas SFX referenciados pela peça entram nas métricas;
- quando um mesmo SFX é usado mais de uma vez, cada ocorrência é contabilizada;
- `speed` influencia a duração dos slots na síntese;
- `flags`, `loop_start` e `loop_end` são preservados pelo extrator, mas não são executados pelo sintetizador atual.

O sintetizador fornecido gera cada `music pattern` isoladamente.

Portanto:

`music_XX.wav` **não deve ser assumido como uma música completa**.

## Split do dataset

O split deve ser realizado por **TID inteiro**.

TID identifica um jogo, e um jogo pode conter várias músicas que reutilizam os mesmos SFX.

Assim, músicas do mesmo TID não devem aparecer em conjuntos diferentes de treino, validação e teste, evitando vazamento de informação.

## Unidade de avaliação

O avaliador recebe explicitamente os patterns que compõem uma peça:

```python
evaluate_piece_data(
    data,
    pattern_ids=[0, 1, 2, 3]
)
```

A definição final de como os patterns serão agrupados em músicas completas ainda depende da tokenização/decodificação final do projeto.

Essa decisão foi mantida separada do cálculo das métricas para que a infraestrutura não precise ser reescrita depois.

## Estrutura prevista

```text
evaluation/
├── pico8_metrics.py
├── evaluate_manifest.py
├── compare_results.py
├── plot_distributions.py
├── pieces.example.json
├── METRICS.md
├── README_AVALIACAO.md
├── requirements.txt
└── tests/
    ├── test_metrics.py
    ├── test_manifest.py
    ├── test_compare_results.py
    └── test_synth_alignment.py
```

## Ambiente

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Testes

```bash
python -m unittest -v
```

Estado validado atualmente:

```text
Ran 20 tests
OK
```

## Avaliação de peças

O manifesto define quais patterns formam cada peça.

Exemplo:

```json
{
  "pieces": [
    {
      "piece_id": "example_01",
      "tid": "12345",
      "patterns": [0, 1, 2, 3]
    }
  ]
}
```

Execução:

```bash
python evaluate_manifest.py \
  --dataset /caminho/para/dataset_pico8 \
  --manifest pieces.example.json \
  --output results.csv
```

## Comparação Real × Baseline × Neural

Quando os três conjuntos estiverem disponíveis:

```bash
python compare_results.py \
  --real real_test.csv \
  --baseline baseline.csv \
  --neural neural.csv \
  --output comparison_summary.csv
```

O pipeline calcula, para cada métrica:

- N;
- média;
- mediana;
- desvio padrão;
- Q1 e Q3;
- mínimo e máximo;
- KS em relação ao conjunto real;
- Wasserstein-1 em relação ao conjunto real.

KS e Wasserstein-1 estão implementados como candidatos; a decisão de reportá-los no paper será feita após os experimentos reais.

## Gráficos

```bash
python plot_distributions.py \
  --real real_test.csv \
  --baseline baseline.csv \
  --neural neural.csv \
  --output-dir plots
```

São gerados boxplots das principais métricas.

## Dependências das outras partes

Para concluir os experimentos reais ainda precisamos de:

1. definição final de como uma música completa é representada/segmentada;
2. tokenização final;
3. split definitivo por TID;
4. decoder do baseline;
5. decoder do modelo neural;
6. outputs gerados pelo baseline;
7. outputs gerados pelo modelo principal.

Métricas temporais de música completa permanecem pendentes até a definição da reconstrução de loops, stop e sequência de patterns.

## Importante

Os manifests, CSVs e resultados usados durante o desenvolvimento são apenas exemplos para validação da infraestrutura.

Eles **não são resultados experimentais finais do TP2**.
