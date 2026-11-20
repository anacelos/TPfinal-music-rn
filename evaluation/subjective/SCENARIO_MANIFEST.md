# Manifest de cenários — DRAFT interno v1

`final-scenario-manifest-v1` nomeia o protocolo futuro. **Todos os registros
produzidos aqui têm `status="draft"` e `ready_for_import=false`.** O nome do
protocolo/diretório não certifica finalização. Não há importação de backend.

```bash
python3 -B -m evaluation.subjective.build_scenario_manifest
python3 -B -m unittest discover -s evaluation/tests -p 'test_scenario_manifest.py' -v
git diff --check
git status --short
```

Python >=3.11, somente biblioteca padrão. O CLI exige 1.854 cenários; a API
`prepare_manifest(..., expected_count=...)` permite fixtures menores. As opções
`--scenario-context`, `--scenario-context-config`, `--real-test`,
`--markov-generated`, `--markov-matching`, `--markov-matching-config` substituem
os seis arquivos de entrada. `--output-dir` substitui o diretório de saída.
Overrides representam os mesmos papéis lógicos: paths físicos nunca são
serializados. Não usar um override para promover outro conjunto silenciosamente;
os hashes do config identificam exatamente os bytes utilizados.

## Entradas e joins congelados

| Papel | Artefato relativo ao repositório |
| --- | --- |
| Contextos | `outputs/subjective/scenario-context-v1/scenario_contexts_internal.json` |
| Contrato dos contextos | `outputs/subjective/scenario-context-v1/scenario_context_config.json` |
| Real TEST | `data/splits/test.jsonl` |
| Markov simbólico | `results/markov_test_final.jsonl` |
| Matching | `outputs/subjective/symbolic-balanced-v1/markov/matching.csv` |
| Contrato do matching | `outputs/subjective/symbolic-balanced-v1/markov/matching_config.json` |

Cadeia: `external_id → scenario-context → (tid,song_id) → Real TEST → matching
Markov → generated_id → record Markov`. As três bijeções são verificadas, sem
reutilizar ou descartar registros. `song_id` isolado e posições físicas nunca
são chaves de join. O `external_id` deve ser exatamente o UUIDv5 já definido
em `scenario-context-v1`; o builder não emite identidades alternativas.

O hash canônico de cada record completo segue o matching: JSON UTF-8,
`sort_keys=True`, `ensure_ascii=False`, `separators=(",", ":")`,
`allow_nan=False`. `generated_id = "sha256:" + generated_record_sha256`.
Os hashes de records Real e Markov são comparados ao CSV; metadados de identidade
e índices são conferidos depois do join. Índices são zero-based da linha física
do JSONL, contando linhas vazias, exclusivamente como proveniência.

Os hashes dos arquivos TEST e Markov devem corresponder ao matching config.
O JSON dos contextos deve corresponder ao seu config. O builder registra SHA256
dos seis arquivos (incluindo o CSV do matching e os dois configs), além do código
utilizado. Ele não segue os paths físicos existentes nos configs. Não refaz
matching, métricas, classificação de contexto, escolha de visual ou síntese.

## Saídas e schema

Em `outputs/subjective/final-scenario-manifest-v1/`, já coberto pela regra
`outputs/` do `.gitignore`:

- `scenario_manifest_draft.jsonl`: uma linha interna por cenário;
- `scenario_manifest_config.json`: hashes, protocolos, políticas e auditoria.

A sequência física preserva `serve_order` dos contextos. Saídas são determinísticas,
sem timestamp, paths absolutos, aleatoriedade ou segredo. O CLI valida tudo em
memória antes de criar/escrever a pasta de saída. Os dois arquivos não são uma
transação atômica; usar um único escritor e verificar o hash do JSONL no config.

| Grupo | Conteúdo |
| --- | --- |
| `identity` | external_id, tid, song_id, serve_order, context_priority, context_tier |
| `protocol` | manifest_protocol, status draft, ready_for_import false |
| `context` | referência/ID do contexto, projeção textual/visual congelada, source_relative_path interno e estado de publicação visual |
| `provenance` | hashes TEST/record Real, protocolo/hash dos contextos e matching, hashes dos configs e JSONL Markov |
| `presentation` | protocolo e mapping A/B/C/D → origin, exclusivamente interno |
| `audios` | slots real, markov, neural, attention |

Não replica evidências Lua/musicais, scores, patterns ou justificativas dos tiers.
O campo `context.source_relative_path` permanece uma referência interna ao dataset,
nunca um locator público. Visuais com imagem têm estado `pending`; os dois sem
imagem têm estado explícito `none`. Nenhum asset é aberto, copiado ou publicado.
Aplicam-se as políticas de sanitização/publicação de [SCENARIO_CONTEXTS.md](SCENARIO_CONTEXTS.md),
inclusive a proibição de copiar diretamente cartuchos `.p8.png` para publicação.

Cada slot tem `origin`, `internal_audio_id`, `source_record_id`,
`source_artifact_ref`, `source_record_sha256`, `source_line_index`,
`symbolic_ready`, `matching_ready`, `render_protocol`, `wav_source_path`,
`wav_ready`, `duration_sec`, `sample_rate`, `channels`, `sample_width_bytes`,
`wav_sha256`, `wav_validation`, `published_audio_id`, `published_locator`,
`published_ready`. Identificador interno `<external_id>/<origin>` serve somente
para relações privadas; jamais deve virar ID/URL público.

- **Real:** `source_record_id="real:<tid>:<song_id>"`, simbólico pronto;
  `matching_ready=null` (não aplicável).
- **Markov:** source_record_id é generated_id, hash e índices do JSONL final;
  symbolic/matching prontos. Acrescenta `generated_id`,
  `generated_record_sha256`, `generated_index` e `matching` com referência/hash
  do CSV, protocolo, identidade/hash Real e identidade/hash gerado.
- **Neural:** identidade/artefato/hash/linha e `matching` null;
  symbolic/matching false. Nenhum artefato Neural é procurado, lido ou escolhido.
  Não há ainda um artefato final comprovado no split 70/15/15.
- **Attention:** `parent_origin=real`, `parent_audio_id` aponta ao ID interno
  Real, `protocol=attention-check-v1`, `global_seed=831`, `target_snr_db=0`.
  symbolic/matching null (derivado de áudio), `generation_ready=false`.
  `parent_wav_sha256`, `derived_seed`, `effective_snr_db`, `gain` null.
  A implementação existente só poderá derivar a seed após o freeze do WAV Real;
  o builder não importa/chama o gerador de attention.

Todos os campos WAV, render e publicação começam null/false. `wav_validation`
fica reservado a `{protocol: "wav-validation-v1", valid: true, wav_sha256: ...}`
emitido por um verificador futuro que leia os bytes do WAV. O builder não fornece
esse verificador nem cria tal evidência nos drafts.
**Duração simbólica e `wav_file` dos records nunca alimentam campos WAV.**
Nenhum WAV existente de Real/Markov/Neural é considerado definitivo.

## Permutação offline

Protocolo `abcd-balanced-rotation-v1`. Ordem canônica congelada dos origins:
`[real, markov, neural, attention]`, representados por índices `[0,1,2,3]`.

1. Ordenar os IDs por SHA256 do JSON compacto UTF-8
   `["abcd-balanced-rotation-v1", external_id]`, com external_id como desempate.
2. Percorrer lexicograficamente as seis permutações de índices `[1,2,3]`.
   Para cada uma, prefixar `0` e emitir as quatro rotações à esquerda, de 0 a 3.
3. Repetir o ciclo de 24 elementos e atribuir A/B/C/D na ordem dos IDs ordenados.
4. Persistir o mapping por external_id, mantendo no JSONL a ordem dos contextos.

Cada bloco de quatro usa cada origin uma vez por posição. Os seis blocos são
disjuntos e cobrem todas as 24 permutações. Para 1.854: 77 ciclos completos
(1.848) + 6 entradas; cada origin aparece **463 ou 464** vezes em cada posição.
Cada permutação ocorre 77 ou 78 vezes. A mesma construção balanceia qualquer
prefixo, evitando o desbalanceamento do resto de uma lista lexical simples.

A função recebe somente external_ids. Não usa contexto, tier, prioridade,
serve_order/qualidade, performance Markov, Neural, métricas ou respostas humanas.
O mapping depende do **conjunto congelado** de IDs: acrescentar/remover cenários
pode realocar posições e exige novo freeze. Não recalcular no browser ou endpoint.

## Readiness e limite público

`readiness_errors` lista pendências; `validate_for_import` rejeita qualquer
pendência; `validate_scenario` rejeita declarar ready=true para um cenário
incompleto. Nenhum validador promove registros automaticamente ou muda o status.
O builder só produz draft e confere que todos continuam incompletos.

Para admitir readiness futura, o gate exige:

- quatro slots e uma permutação completa com protocolo conhecido;
- symbolic pronto e record/artefato/hash presentes em Real, Markov e Neural;
- matching pronto e evidência ligada aos hashes Real/gerado para Markov/Neural;
- quatro render protocols, WAVs prontos, locators relativos/lógicos, checksums,
  duração positiva finita e formato com inteiros positivos;
- evidência `wav-validation-v1` positiva ligada ao mesmo checksum, por slot;
- quatro publicações prontas, com IDs e locators presentes e distintos;
- Attention gerável, ligada ao checksum do WAV Real definitivo, seed presente,
  SNR efetivo dentro da tolerância v1 de 0,1 dB e ganho positivo ≤1;
- visual publicado com checksum ou estado explícito `none`.

É um gate de **metadados internos confiáveis**. Não prova sozinho que arquivos
existem, que um WAV é válido, que uma URL responde ou que um visual foi sanitizado.
Essas verificações devem produzir evidência na futura etapa de render/publicação.
Flags/editagem manual não substituem a execução dessa etapa.

`public_projection` é uma função de allowlist testável para o DTO futuro;
rejeita drafts incompletos. Só admite os campos públicos de contexto e uma lista
de áudios `{position, audio_id, locator}` ordenada A/B/C/D. Nenhum origin, mapping,
tid/song_id, índice, tier, prioridade, hash ou proveniência entra no DTO.
Não implementa endpoint nem importação. O browser nunca recebe/calcula origins.

`published_audio_id=null` até uma futura alocação opaca e persistente, independente
do WAV e sem derivação previsível de external_id+origin. Esta tarefa não gera
segredo/credencial nem escolhe uma estratégia de produção. No freeze futuro,
persistir os IDs alocados; não reutilizar os identificadores internos como IDs
públicos. Locators públicos também devem preservar esse limite.

## Storage e render ainda pendentes

Root físico futuro, documentado apenas aqui:
`/mnt/hd/Documentos/mestrado/IAGenerativa/TP2/subjective-audio/`.
Root lógico serializável: `subjective-audio://`.

```text
render-<id>/
  real/
  markov/
  neural/
  attention/
  excerpts/
  published/
  manifests/
```

Nenhuma dessas pastas de storage é criada. O manifest só usa referências
relativas ao repositório/dataset; paths de WAV ficam null. A validação de locators
futuros admite paths POSIX relativos ou `subjective-audio://<path-relativo>` e
rejeita absoluto, traversal, Windows e URIs arbitrárias. Um publicador futuro pode
resolver esses locators para URLs no servidor sem expor a árvore interna.

`render_protocol` permanece null. Falta confirmar synth commit/hash aprovado,
SHA256 do arquivo synth, `loop_extra_repeats`, política excerpt/full-track,
política de ganho e formato final. Nenhum synth, WAV antigo ou duração simbólica
é congelado como render final nesta implementação.

## Auditoria

O CLI imprime contagens de readiness e a matriz de posições. O config registra
esses valores, unicidade Real/Markov, 24 permutações e hashes das entradas/JSONL.
Para inspecionar cinco linhas completas, sem editar ou importar:

```bash
head -n 5 outputs/subjective/final-scenario-manifest-v1/scenario_manifest_draft.jsonl
cat outputs/subjective/final-scenario-manifest-v1/scenario_manifest_config.json
```

Os testes incluem fixtures com TIDs/song_ids repetidos em combinações distintas,
ordens divergentes entre artefatos, linhas vazias, adulteração de hashes,
reutilização, pendências, allowlist pública, determinismo e acesso restrito às
seis entradas declaradas. Quando os artefatos congelados existem localmente,
um teste de integração confere os 1.854 registros e repete a construção inteira;
em clones sem dados ignorados esse teste é explicitamente skipped.
