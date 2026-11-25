# Mapa do projeto: etapas, o que foi feito e o que falta

Estado em 25/11/2026. Entrega: 25/11, 23:59. Apresentação: 26/11.

**A tarefa.** Ensinar uma rede neural a compor música de PICO-8 (chiptune de 4 canais) nota por
nota, e mostrar que ela faz isso melhor que um baseline simples (cadeia de Markov), comparando
as músicas geradas com músicas reais.

**Onde estão as coisas.**
- Código: este repositório.
- Checkpoints, músicas geradas e tabelas: [Drive](https://drive.google.com/drive/folders/1wirkz0ZRidwRh7db2YXRPaPeTJ1zr4cO) (o `README.md` de lá mapeia cada arquivo).
- Números e significado de cada métrica: [RESULTADOS.md](RESULTADOS.md).

## O caminho, de ponta a ponta

```
jogos de PICO-8 ─► filtros e split ─► tokens ─► treino da LSTM ─► músicas geradas ─┬─► métricas nas notas
     (1)              (2)             (3)          (4, 5)             (6)          └─► síntese ─► métricas no som
                                                                                        (8)          (9, 10)
                          baseline de Markov (7) ─────────────────────────────────────► mesmas métricas
```

## Etapas

| # | Etapa | Em uma frase | Código | Estado |
|---|---|---|---|---|
| 1 | Dataset | 24.077 músicas extraídas de 8.367 jogos de PICO-8 (`songs.json`). | Drive do João | feito |
| 2 | Filtros e split | Tira músicas curtas, duplicadas ou fora do escopo (sobram 12.384) e divide em treino / validação / teste sem que um jogo vaze entre eles. | `preprocessing/` | feito, congelado |
| 3 | Tokenização | Escreve cada música como uma sequência de tokens, de três formas: `nlm`, `grid_unrolled`, `grid_block`. | `src/tokenizer.py`, `src/data.py` | feito |
| 4 | Modelo | LSTM de 2 camadas que, dada a sequência até aqui, prevê o próximo token. | `src/model.py` | feito |
| 5 | Treino e teste | Treina no split de treino, guarda a época com menor loss de validação (`best.pt`) e mede a perplexidade no teste. | `src/train.py`, `src/test_perplexity.py`, `src/plot_history.py` | feito para `nlm` e `grid_block` |
| 6 | Geração | Sorteia um token por vez a partir do modelo até ele encerrar a música; gera 1.854 músicas por sistema (o tamanho do teste). | `src/sample.py` | feito (4 conjuntos) |
| 7 | Baseline | Cadeia de Markov de 1ª ordem por canal; a temperatura (1,6) foi escolhida na validação. | `evaluation/markov_baseline.py`, `evaluation/markov_selection.md` | feito |
| 8 | Síntese | Transforma a música simbólica em áudio, imitando o som do PICO-8. | `src/synth.py`, `src/sfx_filters.py`, `evaluation/synthesize_jsonl.py` | feito, fechado |
| 9 | Avaliação simbólica | Mede PE, SC e GC em cada música e compara a distribuição de cada sistema com a do teste real (OA, KS). | `evaluation/evaluate_split.py`, `evaluation/distribution_compare.py` | feito |
| 10 | Avaliação de áudio | Compara o som gerado com o real usando modelos de áudio pré-treinados (FAD, KLD, CLAP score). | `evaluation/audio_metrics.py` | feito |
| 11 | Teste de escuta | Pessoas ouvem músicas reais e geradas e tentam distinguir umas das outras. | branch `yuri/subjective-evaluation` | confirmar com o grupo |
| 12 | Artigo e slides | Artigo no formato ISMIR e apresentação. | latex.dcc.ufmg.br | confirmar com o grupo |

## O que cada etapa produziu

**2. Split 70/15/15.** 8.669 músicas de treino, 1.861 de validação, 1.854 de teste. A lista
exata está em `data/splits/split_manifest.csv`; ninguém deve gerar outro split.

**3. Tokenizadores.** `nlm` descreve a música como eventos (nota ligou, tempo passou).
`grid_block` descreve a grade do tracker e escreve cada bloco repetido uma vez só.
`grid_unrolled` é a grade sem esse reaproveitamento (sequências 2,5 vezes mais longas).

**5. Treino.** Dois modelos, com os hiperparâmetros padrão do `train.py`:

| Tokenizador | Melhor época | Val bits/s | Perplexidade no teste |
|---|---|---|---|
| `nlm` | 4 de 10 | 85,6 | 7,85 |
| `grid_block` | 8 de 10 | 27,9 | 2,15 |

**6. Geração.** Para cada modelo, dois jeitos de sortear: temperatura 1 sem filtro (`t1`) e
top-k 10 (`k10`, só os 10 tokens mais prováveis entram no sorteio). São os quatro arquivos de
`predicoes-teste/` no Drive.

**9 e 10. Avaliação.** Quanto mais perto do real, melhor: OA alto, KS baixo, FAD baixo.

| Sistema | OA | KS | FAD (piso real: 0,096) |
|---|---|---|---|
| `lstm-grid_block-k10` | 0,89 | 0,22 | 0,118 |
| `lstm-grid_block-t1` | 0,87 | 0,37 | 0,125 |
| `lstm-nlm-t1` | 0,83 | 0,35 | 0,171 |
| `lstm-nlm-k10` | 0,81 | 0,38 | 0,145 |
| `markov` | 0,79 | 0,54 | 0,323 |

Em resumo: a LSTM com `grid_block` é o melhor sistema, e todas as LSTMs ficam à frente do
Markov. O ponto fraco das LSTMs é o ritmo (groove consistency).

## O que falta

Precisa de decisão do grupo:

- [ ] **Escolher o sistema oficial** para o artigo. Sugestão: `lstm-grid_block-k10`.
- [ ] **Teste de escuta:** entra no artigo? Se entrar, os clipes precisam vir do sistema oficial.

Falta fazer:

- [ ] Artigo: escrever resultados e discussão com as tabelas acima.
- [ ] Slides da apresentação.
- [ ] `requirements.txt` na raiz do repositório.
- [ ] README: instruções de reprodução de ponta a ponta e contribuição de cada integrante.

Opcional, se sobrar tempo (nada disso muda a conclusão):

- [ ] Escolher o top-k pela validação (o valor 10 não foi ajustado).
- [ ] Treinar o `grid_unrolled` com o `train.py`, para a comparação ter os três tokenizadores.
- [ ] Calcular bits/s no teste (hoje só existe na validação).

## Limitações já conhecidas

- Os filtros de SFX do PICO-8 (reverb, dampen etc.) não estão nos tokens: as músicas reais são
  sintetizadas com eles, as geradas sem. O efeito no FAD é pequeno e não muda a ordem dos sistemas.
- As músicas geradas não têm par no teste: a comparação é sempre entre distribuições.
- Detalhes e demais ressalvas: [RESULTADOS.md](RESULTADOS.md), seção 5.
