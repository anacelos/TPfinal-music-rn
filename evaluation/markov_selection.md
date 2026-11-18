# Markov Baseline — Hyperparameter Selection

## Objetivo

Selecionar a temperatura de amostragem do baseline de Markov utilizando
exclusivamente o conjunto de validação, mantendo o conjunto de teste isolado
até a avaliação final.

O baseline é baseado no gerador original de João Silveira:

https://github.com/jcssilveira/TP1-DCC831

A adaptação para o TP final utiliza o formato canônico atual do projeto e
treina as transições exclusivamente no split de treino.

## Splits

Split canônico 70/15/15:

- train: 8.669 músicas
- validation: 1.861 músicas
- test: 1.854 músicas

O conjunto de teste não foi utilizado durante a seleção da temperatura.

## Configuração do baseline

- cadeia de Markov de primeira ordem;
- modelo independente por canal (`ch1` a `ch4`);
- estado: `(pitch, instrument, volume, effect)`;
- treinamento: somente `data/splits/train.jsonl`;
- seed: `42`;
- probabilidade de reativação após silêncio: `0.25`;
- número de gerações por candidato no validation: `1.861`.

Foram avaliadas três temperaturas:

- `tau = 0.6`
- `tau = 1.0`
- `tau = 1.6`

As mesmas condições experimentais e a mesma seed foram utilizadas nos três
casos, variando somente a temperatura.

## Critério de seleção

A regra de seleção foi definida antes da análise dos resultados:

> escolher a temperatura com a menor média da estatística
> Kolmogorov-Smirnov (KS) entre as métricas primárias
> Pitch Entropy, Scale Consistency e Groove Consistency.

A distância Wasserstein-1 foi calculada como informação complementar, mas não
foi utilizada para determinar o vencedor.

## Resultados no validation

| Temperatura | Métrica | KS | Wasserstein-1 |
|---|---|---:|---:|
| 0.6 | Pitch Entropy | 0.618485 | 0.843124 |
| 0.6 | Scale Consistency | 0.626545 | 0.157630 |
| 0.6 | Groove Consistency | 0.514157 | 0.171656 |
| 1.0 | Pitch Entropy | 0.714132 | 0.960057 |
| 1.0 | Scale Consistency | 0.674906 | 0.166814 |
| 1.0 | Groove Consistency | 0.303655 | 0.077534 |
| 1.6 | Pitch Entropy | 0.760344 | 1.008634 |
| 1.6 | Scale Consistency | 0.692638 | 0.166419 |
| 1.6 | Groove Consistency | 0.184054 | 0.024905 |

### KS médio

| Temperatura | KS médio |
|---|---:|
| 0.6 | 0.586395 |
| 1.0 | 0.564231 |
| 1.6 | **0.545679** |

## Configuração selecionada

A temperatura selecionada foi:

`tau = 1.6`

Essa configuração foi congelada após a avaliação no validation e será usada
para gerar as 1.854 músicas do conjunto Markov final, que serão comparadas
somente então com o conjunto real de teste e com o modelo neural.
## Avaliação final no test

Após a seleção de `tau = 1.6` exclusivamente no conjunto de validação, a
configuração foi congelada e utilizada para gerar 1.854 novas músicas, o mesmo
número de músicas presentes no conjunto real de teste.

O conjunto de teste não participou da escolha da temperatura.

### Resultados

| Sistema | Métrica | n | Média | Mediana | KS | Wasserstein-1 |
|---|---|---:|---:|---:|---:|---:|
| Real | Pitch Entropy | 1854 | 2.2680 | 2.3757 | 0.0000 | 0.0000 |
| Markov | Pitch Entropy | 1854 | 3.2739 | 3.3816 | 0.7508 | 1.0060 |
| Real | Scale Consistency | 1854 | 0.9276 | 0.9811 | 0.0000 | 0.0000 |
| Markov | Scale Consistency | 1854 | 0.7577 | 0.7437 | 0.6926 | 0.1701 |
| Real | Groove Consistency | 1181 | 0.8772 | 0.9330 | 0.0000 | 0.0000 |
| Markov | Groove Consistency | 1224 | 0.8866 | 0.9286 | 0.1602 | 0.0198 |

O baseline reproduziu com maior proximidade a distribuição de
`Groove Consistency`, enquanto apresentou diferenças substanciais em
`Pitch Entropy` e `Scale Consistency`.

A diferença no número de observações de `Groove Consistency` ocorre porque
essa métrica é indefinida para músicas com apenas um pattern.
