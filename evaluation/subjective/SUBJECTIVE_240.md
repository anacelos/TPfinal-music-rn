# subjective-240-v1 e context-description-ptbr-v1

Expansão autorizada para 240 cenários e redação contextual em português brasileiro.
O subset de 48 e seu handoff permanecem históricos e não são sobrescritos. Este
artefato é interno: contém origens, custos e fontes de auditoria; não é um DTO
para o browser e não constitui um novo pacote de coleta com áudio.

## Seleção e prioridade

Importa as funções de validação dos contratos e sanitização de imagem de
`select_subjective_subset.py`, sem modificar esse arquivo ou outros protocolos
congelados. Percorre `serve_order` crescente e aceita cada candidato tecnicamente
elegível enquanto a permutação original não tiver dez cenários. Para ao completar
24 × 10 = 240. Cada origem ocupa cada posição A/B/C/D sessenta vezes.

A elegibilidade preserva presença no pool, matching Real↔Markov e Real↔Neural,
contexto válido e visual utilizável. Mantém as exclusões `real:31396:0` e Neural
`song_id=1416`. Fallback textual permanece elegível. Falhas de imagem são motivos
técnicos, nunca julgamentos de aparência. Não refaz matching nem atribui novas
permutações. A imagem mantém a escolha congelada e passa pelo mesmo `rgb6-png-v1`.

A fronteira de decisão recebe apenas `Candidate`: ID, ordem, permutação,
inelegibilidades técnicas e referência de visual. Custos, durações, métricas,
resultados e qualidade das músicas não participam. Custos são copiados para
proveniência somente depois da seleção; nenhuma duração ou feature musical é
calculada nesta etapa. Descrições e flags de revisão também só entram depois.

A ordem dos selecionados permanece crescente em `serve_order`, inclusive para
priorizar uma coleta parcial. O desenho novo prevê **dois cenários por
participante**. Não há redistribuição por resultados ou rebalanceamento de um
prefixo. A alocação do servidor continua fora desta etapa; não se modifica o app.

Os 48 IDs históricos devem estar contidos, preservando associações, hashes de
registros, custos, imagens, tiers e permutações. O índice novo 0–239 é apenas a
posição no novo arquivo: `previous_48_selection_index` registra o índice anterior
quando aplicável. `external_id` e `serve_order` não mudam. O manifest canônico de
1.854 drafts também permanece intacto.

## Curadoria semântica das descrições

`context_descriptions_ptbr_v1.json` contém redações estáticas produzidas por
curadoria semântica assistida por Codex, não por tradução literal automática.
Não registra revisão humana independente. O construtor não chama tradutor,
modelo, serviço externo ou gerador de texto; apenas valida e aplica essa redação.

As evidências permitidas são o texto contextual do pool congelado
(`description_core` e `description`), suas tags e a imagem do cartucho. Nos casos
que usam inspeção visual, isso é declarado em `evidence_fields`. Não se usa código
de chamada de música, classificação musical, duração, matching cost ou resultado
Neural para redigir. Créditos, comandos, menus, instalações e referências musicais
presentes nas fontes são descartados da redação final.

Cada entrada traz explicitamente seus IDs, TID e SHA256 do conjunto contextual
de evidências. O fingerprint inclui textos originais, tags e hashes do visual,
mas não inclui métricas ou propriedades das músicas. Uma mudança das fontes
interrompe a aplicação da descrição até revisão explícita. Contextos idênticos
do mesmo cartucho reutilizam a mesma redação, sem inventar diferenças entre
faixas. Existem 215 contextos distintos para os 240 cenários.

Textos usam uma ou duas frases, procurando 25–60 palavras quando há conteúdo
suficiente. Casos limitados ficam mais curtos, com justificativa privada, sem
preencher lacunas ou inserir avisos artificiais só para atingir uma contagem.
Nomes próprios de personagens e jogos apoiados nas fontes não são inglês
residual. Não se usam emoções ou recomendações para indicar uma escolha musical.

`public_description` e `description_ptbr` contêm o texto novo no artefato derivado.
`public_description_original` preserva exatamente o texto anterior, incluindo
fallback. O CSV de descrições e o relatório guardam textos, palavras, tiers,
fallback, evidências completas/referências, hashes, justificativas e flags.

`manual_review_recommended` sinaliza evidências muito fracas ou ambíguas. Essa
flag não exclui o cenário nem altera sua prioridade. Os casos incluem imagens de
apresentação sem atividade clara, demonstração de interface e coletânea em que
não se identifica uma cena específica. Não há associação automática entre
fallback anterior e necessidade de revisão: o texto bruto pode fornecer contexto
suficiente mesmo quando o extrator anterior havia produzido fallback.

## Auditoria e reprodução

```bash
.venv/bin/python -B -m evaluation.subjective.select_subjective_240
```

Exige destino novo; recusa sobrescrita. Saída em
`outputs/subjective/subjective-240-v1/`:

- `scenarios.csv` e `scenarios.jsonl`: seleção ordenada, associações e contextos;
- `descriptions_ptbr.csv`: original, redação nova e auditoria por registro;
- `selection_config.json`: protocolos, hashes, quota, regras e implementação;
- `selection_summary.json`: cobertura, tiers, balanço e estatísticas textuais;
- `description_audit.json`: checagens e fontes completas dos 240 registros;
- `selection_scan.csv` e `ineligible_scenarios.csv`: trilha de seleção;
- `visuals/`: os 240 PNGs sanitizados, com hashes e referências congeladas.

Verificações lexicais detectam inglês significativo, URLs, referências musicais,
teclas/controles, instruções operacionais, texto vazio e número de frases. Elas
não substituem a avaliação semântica das fontes. Identificar instalação de
software não deve confundir a instalação de canos como mecânica de um jogo.
Uma falha não é corrigida automaticamente: o construtor exige revisão da redação.

A execução é determinística para os mesmos inputs, curadoria, código, imagens e
versão do Pillow. Config e outputs não contêm timestamp ou destino físico. A
verificação integral deve preparar duas execuções independentes e comparar os
bytes de todos os arquivos, incluindo imagens e config.

```bash
PYTHONPATH=.:evaluation:evaluation/tests .venv/bin/python -B -m unittest \
  test_subjective_240 test_subjective_subset test_scenario_context_export \
  test_scenario_manifest
```

Nenhum áudio novo, Attention, arquivo do paper, manifest canônico ou artefato
histórico de 48 é alterado. Não se gera tar de 240 nem se substitui o handoff
anterior. Os textos/flags devem ser considerados na revisão antes da futura
preparação da coleta, mantendo as decisões de seleção já estabelecidas.

## Resultado desta expansão

- 446 candidatos examinados: 240 aceitos e 206 pulados por quota já completa;
- tiers A=74, B=166, C=0, D=0; último `serve_order` examinado=446;
- 24 permutações ×10, com sessenta ocorrências de cada origem em cada slot;
- 240 Reais, 240 Markov e 240 Neural únicos; exclusões preservadas;
- 48/48 IDs históricos contidos, com associações, permutações e imagens idênticas;
- 240 registros de descrição, com 215 redações para 215 contextos/cartuchos;
- palavras: mínimo=7, mediana=33, média=32,1, máximo=39;
- 15 textos abaixo de 25 palavras, todos com justificativa de evidência limitada;
- 35 descrições originais em fallback, preservadas integralmente para auditoria;
- 12 cenários com revisão recomendada, mantidos na seleção e na ordem;
- zero ocorrências detectadas de URLs, referências musicais, comandos, instruções
  operacionais, inglês significativo, vazio ou número inválido de frases;
- duas preparações independentes: 248 arquivos idênticos byte a byte;
- 82 testes aprovados; 559 arquivos históricos preservados por SHA256, incluindo
  o handoff anterior e seu tar no HDD; nenhum áudio ou Attention novo.

Índices novos com revisão: 31, 41, 62, 99, 171, 181, 188, 194, 198, 211, 236, 239.
Os índices 31/198 compartilham o mesmo contexto de tiro pouco documentado. Os
demais envolvem demonstração de caminhada, bolhas com regras não descritas,
combate genérico, recursos em desenvolvimento, história sem missão identificada,
coletânea, quebra-cabeça de sapo, interface gráfica, grade sem regras ou apenas
imagem. Textos e motivos completos estão em `description_audit.json`.

SHA256 dos principais outputs:

```
scenarios.jsonl
7ae970d8df22a55ed11391c8e928502bae642cdc17526b362d97dd1473ac7e04

descriptions_ptbr.csv
6a320504d2a08a0b21bc1b6949cf961bc701dd3f811e643845ad839253b6d0b0

selection_config.json
2a36b0d6b1cea3fac6b545363cb1f4841eea7561d2ce58e09944808e25212c1e

description_audit.json
89d3812127da2f00e15dfacb5b7b415e9731c18a20b7de7a3bec716221f3de60
```
