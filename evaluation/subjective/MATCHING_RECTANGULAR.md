# Matching Neural retangular — symbolic-balanced-rectangular-v1

Extensão separada para `lstm-grid_block-k10`. O matcher e o matching Real↔Markov
`symbolic-balanced-v1` permanecem intactos. A avaliação objetiva continua usando
as **1.854 gerações originais**; a exclusão abaixo só define elegibilidade para
matching subjetivo. Não há regeneração, imputação ou seleção por custo.

## Decisão e única exceção

Input oficial: `artefatos/predicoes-teste/grid_block-k10.jsonl.gz`, do
[Drive do projeto](https://drive.google.com/drive/folders/1dkd81qXwoqyRP2Jksyxo9qrtzjC7-uZz).

```text
SHA256 gzip: a2e5fbfc1e338e9583a029f09bc00ea7f16b9aeaf6d926e91680f87533d09bfc
SHA256 JSONL: dc59f03188e76703d3895561a3b580ee201035042005b39fe80b6f8d27603ae8
```

Somente `song_id=1416`, com SHA256 canônico completo
`5bcc9482c42f5d53bb0ee7a62d99ebb145073a0ba6f2c73716408c53a2964225`,
é excluído. O record termina em EOS, tem um pattern de 512 ticks (4,249252 s),
nenhum onset audível e 32 slots de volume zero. PE, SC e pitch range são
indefinidos. O loader verifica identidade **e** hash **e** o motivo musical;
qualquer divergência ou outro record inválido aborta, sem publicar resultados.

O protocolo exige 8.669 TRAIN, 1.854 Real, 1.854 Neural originais, uma exclusão,
1.853 Neural elegíveis, 1.853 pares e um Real unmatched. Nenhum Neural elegível
é descartado. O Real sem Neural é consequência da otimização global, não uma
escolha manual. Isso altera a cobertura e a política de missing do v1 quadrado,
por isso recebe um identificador de protocolo diferente.

## Lógica congelada reutilizada

`match_hungarian_rectangular.py` importa `match_hungarian.py`, sem refatorá-lo.
Reutiliza diretamente `extract_features`, `load_dataset` (TRAIN/Real),
`fit_scaler`, `normalize`, `scalar_distance`, `cost_components` e
`build_cost_matrix`.

- Pitch: média das distâncias de PE, pitch range e centro `(min + max)/2`.
- Rhythm: média da distância de onset rate e da distância L1 média dos perfis
  de groove de 32 bins. O perfil é a média sem ponderação dos patterns com
  duração positiva; o onset rate é `note_onsets/duration_seconds`.
- Texture: distância de `audible_channel_utilization`.
- Scaling: somente TRAIN; `log1p` no onset rate, mediana/IQR por escalar,
  fallback para desvio populacional e depois 1. O groove não é reescalado.
- Distância escalar: `u/(1+u)`, com `u=abs(z_real-z_generated)`.
- Custo total: `(pitch + rhythm + texture)/3`.

SC, GC escalar, duração independente e identidade de blocos não entram no
custo. Ausência de EOS, GC indefinida e uso do instrumento noise não introduzem
filtros novos. Não há threshold nem remoção posterior de pares por custo.

O solver é o mesmo `scipy.optimize.linear_sum_assignment`, agora com matriz
float64 **1.854 × 1.853**. Todas as colunas são cobertas uma vez; uma linha fica
livre. Real é ordenado por `(str(tid), str(song_id))`; Generated pelo hash
canônico completo, incluindo metadados. Não há epsilon para desempates.
Os índices físicos base zero são apenas proveniência, nunca chaves de join.

## Executar e testar

Da raiz do repositório, com as versões de `subjective/requirements.txt`:

```bash
.venv/bin/python -B -m evaluation.subjective.match_hungarian_rectangular \
  --train data/splits/train.jsonl \
  --real data/splits/test.jsonl \
  --generated artefatos/predicoes-teste/grid_block-k10.jsonl.gz \
  --output-dir outputs/subjective/symbolic-balanced-rectangular-v1/neural-grid_block-k10

PYTHONPATH=.:evaluation .venv/bin/python -B -m unittest discover \
  -s evaluation/tests -p 'test_subjective_matching*.py' -v
```

Não sobrescreve diretório existente. Para conferir determinismo, repetir com
os mesmos inputs em outro diretório novo e comparar os cinco arquivos byte a
byte. As identidades dos pares permanecem estáveis se a ordem dos inputs mudar;
hashes de arquivo e índices físicos mudam nesse caso, como devem.

O CLI fixa sistema, contagens e exclusão oficial; não permite override de
features, pesos ou regras de elegibilidade. A API aceita contagens menores e
`ExclusionSpec` explícito somente para fixtures, registrando ambos no config.

## Saídas e auditoria

As cinco saídas ficam em um diretório novo sob `outputs/`, ignorado pelo Git:

- `matching.csv`: schema do matcher congelado, com 1.853 pares e decomposição
  completa dos custos; ordenado pelo índice físico Real.
- `matching_config.json`: protocolo, sistema, hashes dos três inputs (gzip e
  conteúdo descompactado do Neural), contagens, exclusão aprovada, descritores,
  pesos, normalização, solver, versões e hashes dos códigos novo/congelado e
  evaluator; inclui checksums dos outros quatro arquivos.
- `matching_summary.json`: contagens, cobertura, shape da matriz e min, mean,
  median, p90, p95 e max dos custos. Quantis usam interpolação linear.
- `excluded_generated.csv`: somente o record aprovado, com identidade, hash,
  índice físico e motivo da ausência de features.
- `unmatched_real.csv`: identidade completa, hash e índice do único Real não
  associado; motivo explícito de não atribuição pelo solver retangular.

Timestamps, tempos de execução, RSS e caminho de saída não são serializados.
Com os mesmos inputs, códigos e versões, os cinco arquivos são determinísticos.
O gzip original não é regravado e não é criada cópia descompactada. A leitura
verifica a integridade gzip ao consumir o arquivo até o fim.

Todas as validações ocorrem antes da publicação. O lote é escrito em temporário
no mesmo filesystem e publicado por rename; assume um único escritor por
destino, como o matcher congelado. Uma falha não publica um diretório de sucesso.

Nenhum input ou artefato Markov é alterado. Não seleciona os ~48 cenários,
não altera contextos/serve order/permutação/scenario manifest, não sintetiza,
não gera Attention e não publica mídia. A futura subjetiva deverá declarar
elegibilidade condicionada à existência dos descritores e manter a falha de
geração visível nos resultados objetivos.

## Execução oficial validada

Com os inputs acima: 1.853 pares, 1.853 Real únicos e 1.853 Neural únicos;
todos os custos finitos e nenhum Neural elegível sem par. Os parâmetros de
normalização, descritores e pesos conferem exatamente com o config Markov.

O único Real unmatched foi `real:31396:0`, índice físico 383 (linha 384 do TEST),
com hash canônico
`0a00259795d4d2b02dfac35bbd0b77d070dfd56528df883834989a3066a432f7`.
A única exclusão foi o `song_id=1416` aprovado, no índice físico 1416.

| Estatística de custo | Valor |
| --- | ---: |
| min | 0.013047983056 |
| mean | 0.157091493587 |
| median | 0.143121727450 |
| p90 | 0.255927417626 |
| p95 | 0.297017523933 |
| max | 0.482809050542 |

Duas execuções completas produziram os cinco arquivos idênticos byte a byte.
SHA256 de `matching.csv`:
`f5d644a2cd49d3eb037aae401be5409ba4d04506b713757c9fad03cd61b2834a`.
Foram aprovados 78 testes: matcher congelado/extensão (54), manifest e métricas
simbólicas (24), além da auditoria independente dos índices, identidades,
hashes e estatísticas dos pares oficiais. Inputs, matching Markov e scenario
manifest mantiveram seus hashes.
