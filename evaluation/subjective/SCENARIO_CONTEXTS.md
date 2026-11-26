# Contextos de cenário v1 — projeção public-segments-2

Artefatos de integração **internos**, uma linha por Real TEST `(tid, song_id)`.
Não são payloads do navegador. A pool, seus tiers, prioridades e serve_order
permanecem congelados. Esta revisão altera exclusivamente a apresentação.

```bash
python3 -B -m evaluation.subjective.export_scenario_contexts
PYTHONPATH=evaluation .venv/bin/python -B -m unittest discover -s evaluation/tests -p 'test_scenario_context_export.py' -v
PYTHONPATH=evaluation .venv/bin/python -B -m unittest discover -s evaluation/tests -p 'test_context_pool.py' -v
PYTHONPATH=evaluation .venv/bin/python -B -m unittest discover -s evaluation/tests -p 'test_subjective_*.py' -v
git diff --check
(cd data/splits && sha256sum -c split_hashes.sha256)
```

Python >=3.11, biblioteca padrão, sem rede/modelos/LLM/NLP externo. CLI:
`--pool-csv`, `--pool-config`, `--dataset-root`, `--output-dir`.
Dataset padrão: `../dataset_pico8_v2/dataset_pico8_v2`.
O CLI exige 1.854 contextos e 873 TIDs; fixtures da API podem ser menores.

## Outputs e identidade

Em `outputs/subjective/scenario-context-v1/`, ignorado por `outputs/`:

* `scenario_contexts_internal.csv` e `scenario_contexts_internal.json`;
* `scenario_context_config.json`: regras, schemas, checks, contagens e hashes;
* `visual_assets_manifest.csv`: inventário interno, não autorização de cópia.

O CLI remove os dois nomes antigos `scenario_contexts.csv/json` somente após
escrever as novas saídas. Não há arquivo público de produção nesta etapa.
A escrita dos quatro arquivos não é uma transação atômica; usar um escritor.

Confere protocolo/hash do CSV congelado, contagens, identidades únicas e TEST,
ordem física exata 1..1854 e prioridades. Copia tiers, patterns e ordem sem
ordenar ou chamar a geração da pool. Tier D permanece, com fallback e sem imagem.
Não lê matching, resultados de sistemas ou áudios.

O `external_id` continua UUIDv5 com namespace UUIDv5 de `uuid.NAMESPACE_URL`
e `urn:tpfinal:subjective:scenario-context-v1`; nome é o JSON compacto
`["<tid decimal canônico>",<song_id inteiro>]`. Texto, paths, tier, prioridades e
áudio não entram no ID. A revisão textual **não muda namespace nem IDs**.
Novo protocolo deve escolher explicitamente seu namespace; o manifest final
deve carregar estes IDs. IDs não são segredos nem tokens de autorização.

## Texto público

Lê exclusivamente `description_core` quando a flag congelada é informativa.
Não usa texto bruto, metadados auxiliares ou evidências como fallback.

1. Normaliza entidades HTML e espaços, preservando limites de linha para headers.
2. Reconhece cabeçalhos com dois-pontos/linha inteira e expressões convencionais
   como How to Play. Não trata `controls`, `our`, `add` ou `credits` isolados
   dentro de frases como metatexto.
3. Remove o header útil, preservando seu conteúdo. Seções explicitamente técnicas,
   de créditos e changelog são ignoradas até o próximo header reconhecido.
4. Divide frases/linhas e algumas fronteiras explícitas de cláusulas; remove
   relatos autorais, pedidos de feedback, planos futuros e relatos de alterações.
   Normaliza URLs antes da separação para não publicar pedaços de domínios.
5. Descarta referências a theme/soundtrack/song/music de forma conservadora,
   sem alterar music evidence interna. Mantém apenas extratos com pistas de
   papel, objetivo, mecânica ou gênero, incluindo upgrade/restauração/plantio.
6. Junta extratos existentes com pontuação normalizada ou aplica fallback:
   **Descrição contextual não disponível para esta avaliação.**

As expressões e sua revisão estão no código/config. Reutiliza constantes lexicais
congeladas da pool, mas não chama `description_quality`, não recalcula seus scores
nem suas classificações. Não traduz, parafraseia ou inventa cenas musicais.
As guardas são heurísticas EN/PT: podem perder contexto ou deixar fragmentos
imperfeitos. O inventário não equivale a validação semântica universal.

Nenhuma confiança de título da pool está aprovada. `public_title=""` e
`title_fallback=True`; frontend deve omitir o título vazio. Heading mantido:
**Cartucho de origem e contexto visual**.

## Schemas

Campos internos exclusivos:
`tid`, `song_id`, `serve_order`, `context_priority`, `context_tier`,
`start_pattern`, `end_pattern`, `title_fallback`, `description_fallback`,
`source_relative_path`.

A tabela interna também contém `external_id`, `public_title`,
`public_description`, `public_visual_label`, `public_visual_type` e
`public_context_heading`. **Não contém `public_visual_source`.**
TID é string decimal; IDs, texto, tipo, heading e source são strings; patterns,
song_id e ordens são inteiros; fallback flags são booleanos JSON (True/False CSV).
Source interno é vazio para none; relativo ao root lógico dataset nos demais.

Allowlist final: `external_id`, `public_title`, `public_description`,
`public_visual_label`, `public_visual_type`, `public_visual_source`,
`public_context_heading`.

`public_projection` valida UUIDv5 canônico, tipos string, enum visual, labels e
heading. Para imagem exige exatamente o argumento explícito
`published_visual_source="visuals/<external_id>.png"`. Para none exige locator
null e devolve null. Não lê nem encaminha source interno. Campos internos extras
são ignorados pela allowlist; não se deve serializar a linha inteira nem usar
filtragem por prefixo como substituto de um DTO público.

Nunca enviar TID, song_id, patterns, tiers, ordens, estrelas, confiança de título,
scores, hashes, matching, Lua/music evidence ou origem de sistemas ao participante.
A validação da função não prova publicação/sanitização: é responsabilidade do
publicador futuro verificar isso antes de fornecer o locator.

## Visuais e publicação obrigatória

Não reavalia pixels/representatividade: usa exatamente a escolha da pool.
Confere existência e paths POSIX relativos, rejeitando absoluto/Windows/URI,
traversal e symlinks fora do dataset. Não copia nem converte assets.

| Tipo | Label |
| --- | --- |
| cartridge | Imagem do cartucho de origem |
| spritesheet | Elementos gráficos do cartucho de origem |
| none | Sem imagem |

Manifesto: `external_id,source_relative_path,visual_type,publication_policy`.
Uma linha por cenário com imagem, repetição de source permitida. Todas as linhas
possuem a política **INTERNAL SOURCE ONLY; publishing required; direct public copy forbidden**.
O config também registra `direct_copy_authorized=false`. Nenhum consumidor pode
interpretar esse inventário como uma fila autorizada de cópias públicas.

**Cartridge `.p8.png` contém Lua/ROM/dados musicais recuperáveis nos pixels.**
Antes da publicação, DEVE ser convertido para uma imagem visual limpa que preserve
a aparência do label, sem payload recuperável do cartucho. Renomear ou copiar o
original para um UUID é proibido. Remover apenas metadata PNG também não basta.
Spritesheet DEVE igualmente passar pela camada de asset publishing, mesmo que o
risco seja diferente. Nenhuma sanitização é executada nesta etapa.

A cadeia futura é: join privado com Real/Markov/Neural/Attention, publicador de
assets, manifest final/importação e DTO público do endpoint. O publicador converte
source interno em locator opaco, preferencialmente um endereço por external_id
para não expor compartilhamento entre cenários. Pode deduplicar armazenamento
internamente sem reutilizar URLs públicas. Não enviar artefatos internos ao browser.
