# subjective-48-v1 — seleção congelada para a coleta

Artefato **interno** separado do manifest canônico de 1.854 drafts. Não é DTO
do navegador: contém identidades, origens, custos e proveniência privados.
Não reassocia músicas, não recalcula matching, não gera permutações e não altera
o manifest, os contextos, a pool ou a avaliação objetiva.

## Regra exata

1. Conferir hashes e joins dos contratos congelados de pool, contextos,
   manifest, matching Markov e matching Neural retangular `lstm-grid_block-k10`.
2. Percorrer exclusivamente o `serve_order` crescente, sem renumerar a origem.
3. Exigir os dois matchings, contexto válido e imagem utilizável da escolha
   visual congelada. `real:31396:0` não tem Neural. `none` é inelegível porque
   a interface exige imagem; fallback textual continua elegível.
4. Ler a permutação A/B/C/D já persistida no manifest; quota **2** por cada uma
   das 24 permutações. Aceitar o primeiro elegível enquanto sua quota tiver vaga.
5. Parar ao preencher as 24 quotas: exatamente 48 cenários, índices 0–47.

`scan_candidates` recebe apenas ID, serve_order, permutação, motivos técnicos
e referência visual. Não recebe matching cost, tier para ranking, PE/SC/GC,
duração, conteúdo de músicas ou resultados Neural. Custos são lidos e
serializados **depois** de a seleção estar fechada, exclusivamente como
proveniência. Os JSONLs musicais também só são abertos depois do scan.

Descrição e imagem são as escolhas congeladas: sem reclassificar tiers,
inventar cenas, substituir visuais ou preferir músicas. Uma corrupção de
hash/identidade aborta; uma imagem ausente/inválida é registrada como impedimento
técnico e o scan continua. Não há threshold nem filtro posterior por custo.

## Imagens utilizáveis localmente

Os contextos anteriores contêm fontes internas, não imagens já publicadas.
Este subset materializa PNGs locais `visuals/<external_id>.png` para que
`public_visual_source` aponte a um arquivo utilizável na futura interface.
Não publica nada na web, não altera readiness do manifest e não serve áudios.

Protocolo técnico `rgb6-png-v1`: decodificar a escolha PNG congelada, preservar
dimensões, converter para RGB, limpar os dois bits baixos de cada componente
(`& 0xFC`), descartar alpha/metadata e escrever uma imagem RGB nova. A aparência
RGB varia no máximo 3 níveis por componente; não há crop ou escolha estética.

Isso remove o carrier usado pelo cartucho PICO-8, cujo byte por pixel é
`(B&3) | ((G&3)<<2) | ((R&3)<<4) | ((A&3)<<6)`: nos derivados RGB ele é constante,
sem os bytes ROM/Lua/SFX do original. O decoder do dataset em
`../pico8-dataset-scraper-main/pico8-dataset-scraper-main/src/tools/shrinko8/pico_cart.py`
implementa esse layout. Copiar/renomear `.p8.png` ou remover só metadata não basta.
Spritesheets passam pela mesma camada, sem expor paths/TIDs em locators públicos.

A imagem é validada por decode, formato e dimensões (cartridge 160×205,
spritesheet 128×128); não por qualidade percebida ou representatividade.
As fontes originais são somente lidas. O inventário guarda seus hashes e os
hashes dos derivados. Visuais repetidos entre cenários têm locators distintos.
Pillow deve estar disponível na `.venv`; a versão utilizada fica no config.

## Execução e saídas

Da raiz do repositório:

```bash
.venv/bin/python -B -m evaluation.subjective.select_subjective_subset
PYTHONPATH=.:evaluation .venv/bin/python -B -m unittest discover \
  -s evaluation/tests -p 'test_subjective_subset.py' -v
```

O CLI usa os inputs congelados descritos em `REFS`; `--dataset-root` permite
resolver as fontes visuais no HDD. `--output-dir` só aceita destino novo.
A API aceita fixtures com menos contextos, mas mantém quota 2×24 e alvo 48.

Em `outputs/subjective/subjective-48-v1/`, ignorado pelo Git:

- `scenarios.csv` / `scenarios.jsonl`: 48 cenários em serve_order, com IDs,
  contexto público, imagem, permutação original, associações e custos auditáveis,
  hashes e durações simbólicas individuais. JSONL preserva `presentation` do
  manifest; o CSV a representa em `origin_A` a `origin_D`.
- `selection_config.json`: hashes de todos os inputs, regras, barreira entre
  seleção e diagnóstico, protocolos de contexto/visual, versões e checksums
  das demais saídas. Não serializa paths físicos do dataset ou do output.
- `selection_summary.json`: candidatos examinados, decisões, tiers, quotas,
  balanço por posição e diagnóstico de duração, inclusive todos os casos <30 s.
- `selection_scan.csv`: decisões do prefixo examinado; sem custos ou durações.
- `ineligible_scenarios.csv`: impedimentos conhecidos nos metadados de todos
  os contextos, mais falhas de imagem encontradas no prefixo examinado.
- `duration_diagnostics.csv`: durações Real/Markov/Neural e origens <30 s.
- `visual_assets_manifest.csv` e `visuals/`: proveniência e 48 PNGs opacos.

Os bytes de imagem são checados nos candidatos tecnicamente elegíveis do prefixo
examinado até a parada. Não se declara validação dos arquivos de imagem que
ficaram depois desse prefixo. Os dois `none` são conhecidos independentemente
disso e continuam registrados como inelegíveis, mesmo que não examinados.

Todos os joins usam identidade/hash completos, nunca song_id isolado nem linha
física. As associações selecionadas são conferidas com os records originais e
com os matchings congelados. Os custos e durações não mudam a decisão.

## Durações: diagnóstico, não política de excerpt

Somar `max(0, pattern_ticks)` e multiplicar por `366/44100`, o mesmo relógio
simbólico do evaluator. Sem repeats extras de loop, padding, truncamento ou
síntese. A duração simbólica não declara que um WAV existe.

Por origem: min, mediana, média, max, contagens estritas <10, <20, <30 e ≥30 s.
Nenhum cenário curto é removido. A política futura de excerpt de 30 s permanece
uma decisão separada; essa etapa não a antecipa.

## Auditorias e preservação

Exigir 48 IDs Real/Markov/Neural únicos, nenhuma identidade excluída, 2× cada
permutação, 12 aparições de cada origem em cada posição, serve_order preservado
e imagem sanitizada utilizável para todos. Fallback textual é testado como
elegível. Testes perturbam custos e diagnósticos sem mudar os selecionados e
impedem chamadas ao gerador de permutações.

O lote inteiro, inclusive imagens, é determinístico com os mesmos inputs,
fontes, código e versão Pillow. Não há timestamps nem timings. Repetir para
outro diretório novo e comparar todos os arquivos byte a byte. A escrita usa
temporário no mesmo filesystem e rename final; um escritor por destino.
Não sobrescreve lote anterior e não publica um diretório de sucesso parcial.

Nenhum WAV, Attention, paper, app, scenario manifest ou arquivo canônico de
matching é modificado. Não há push nem publicação de mídia na web.

## Execução oficial

Foram examinados 199 candidatos: 48 aceitos e 151 pulados por quota preenchida.
Tiers selecionados: A=43, B=5, C=0, D=0; oito descrições fallback preservadas.
Cada permutação aparece duas vezes; cada origem ocupa cada posição 12 vezes.
São 48 Real, 48 Markov e 48 Neural únicos, com 48 imagens locais sanitizadas.

Impedimentos conhecidos, todos posteriores ao prefixo selecionado:

- `real:31396:0`, serve_order 1396: ausência de matching Neural.
- `real:4002:0`, serve_order 1853: visual none, incompatível com a interface.
- `real:44019:0`, serve_order 1854: visual none, incompatível com a interface.

| Origem | min (s) | mediana (s) | média (s) | max (s) | <10 | <20 | <30 | ≥30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Real | 0.265578 | 12.614966 | 25.418050 | 163.596190 | 22 | 31 | 34 | 14 |
| Markov | 0.265578 | 11.951020 | 23.285125 | 153.769796 | 21 | 30 | 38 | 10 |
| Neural | 0.265578 | 21.246259 | 37.550272 | 154.035374 | 17 | 22 | 30 | 18 |

Há alguma faixa <30 s em 47 cenários: índices 0 e 2–47. Somente o índice 1,
`real:45995:0`, tem as três faixas ≥30 s. No índice 42, `real:32614:4`, as três
durações são 0.265578 s. Esses cenários foram mantidos; duração não é filtro.
A relação por external_id/Real e origem curta está em `duration_diagnostics.csv`
e `selection_summary.json`.

Duas execuções completas produziram os **56 arquivos idênticos byte a byte**,
incluindo os 48 PNGs. SHA256 de `scenarios.jsonl`:
`6c8c966b99c51d4eef8fd2a24a93b061ee6f7954d803f051f454aa238649fae5`.
Passaram 140 testes, incluindo 19 novos, e a auditoria independente confirmou
ordem, quotas, identidades, associações, hashes e ausência de payload nos PNGs.
