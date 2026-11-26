# Context pool v1 — revisão observable-evidence-2-text-guard-1

Artefato **interno**, uma linha por música Real do TEST, sem exclusões.
Requer Python >=3.11, apenas biblioteca padrão, sem rede, síntese ou emulação.
A revisão substitui as regras anteriores de v1; a versão exata das regras e o
hash do código são registrados na auditoria. Não compara os tiers antigos e
novos como se fossem a mesma medida.

```bash
python3 -B -m evaluation.subjective.context_pool
PYTHONPATH=evaluation .venv/bin/python -B -m unittest discover -s evaluation/tests -p 'test_context_pool.py' -v
PYTHONPATH=evaluation .venv/bin/python -B -m unittest discover -s evaluation/tests -p 'test_subjective_*.py' -v
(cd data/splits && sha256sum -c split_hashes.sha256)
```

Dataset padrão: `../dataset_pico8_v2/dataset_pico8_v2`. Flags: `--dataset-root`,
`--splits-dir`, `--output-dir`. O CLI exige 1.854 músicas e 873 TIDs. Outputs:
`outputs/subjective/context-pool-v1/context_pool.csv` e `context_pool_config.json`.
A publicação dos dois arquivos não é transação atômica: usar um escritor por pasta.

## Três sinais independentes

Independência aqui significa cálculo separado; não presume independência estatística.

1. `description_informative`: núcleo textual com pistas lexicais explicativas.
2. `visual_informative`: pixels não triviais de label ou assets, com PNG associado.
3. `music_context_confidence` strong/moderate: associação estática ao início musical.

`context_evidence_count` soma os três booleanos: **A=3, B=2, C=1, D=0**.
A significa alta evidência contextual disponível; B moderada; C limitada;
D mínima/ausente. Nenhuma categoria confirma execução, cena, qualidade estética
ou representatividade de gameplay. Um tier pode ficar vazio.

## Texto: núcleo observável, sem NLP sofisticado

Preserva descrição original, normaliza entidades HTML e espaços. Remove URLs,
IDs `Cart #...` numéricos/alfanuméricos e licenças; remove cabeçalhos BBS até o
cartucho. Seções de controles/créditos iniciadas no começo ou após pontuação são
removidas até About/Goals/Gameplay/Description/Objetivo/Sobre ou fim do texto.
Divide em frases e marcadores changelog/update/version/added/fixed/changed.
Deduplica fragmentos por casefold. Só mantém fragmentos com pistas explícitas de:

* papel: you are/play as/control, você é/controla, player is/controls, jogador;
* objetivo: collect/catch/rescue/defeat/fight/destroy/survive/escape/reach/find/
  protect/solve/deliver/burn/slay, equivalentes PT explicitados no código,
  e help ... before;
* mecânica: one/single-button, um botão, jump/bounce/dash/shoot/rotate/build/stack/
  spin/shapeshift, platforming e equivalentes PT do código;
* gênero: fishing/pesca/puzzle/platformer/plataforma/roguelike/racing/corrida/
  shooter/shoot-em-up/tetris/snake/breakout/asteroids/simulation/simulator/simulador.

Discussão de Lua, tracks/channels, programação, código e tutorial é descartada
quando não contém também pistas de papel/objetivo. Palavras game/music/controls
sozinhas não bastam, e não são palavras proibidas em frases de gameplay.
A revisão textual exige corroboração local para copulas genéricas como `you are`,
`você é` e `player is`: outra família de gameplay no fragmento, personagem explícito
(robot/fisherman/raincloud/knight/hero/ghost/cat/slime/wizard/pilot/ninja/soldier/
detective/adventurer/farmer/teacher e equivalentes PT do código), ambiente
(maze/dungeon/forest/castle/world e equivalentes PT), ou pistas locais de ação/
objetos de jogo (character/farm/hive/nectar/bee/crosshairs/rocket/ball/boost/brake/
power-ups/bombs/detonate/enemy stats/seed/caverns). Play as/control e jogador
continuam pistas explícitas de agência. A lista lexical completa está no código.
Não se infere papel do jogador de `you are looking at this` sozinho.

Blocos de pedido a teacher/professor/instructor, pedidos de nota (give me a 100,
grade/mark/score, evaluate my etc.) e submissão escolar são ignorados. Se houver
uma cláusula explícita de papel do jogador no mesmo bloco, ela é separada para
continuar a avaliação. Teacher ou please isolados não causam rejeição global:
`You are a teacher. Rescue children` e `Please catch fish` permanecem elegíveis.
Fragmentos de planejamento técnico com map/cart loading, map data/memory,
cartdata, multi-cart ou token limit não sobrevivem somente por pista de gênero
ou nome de jogo como tetris. Isso não é um classificador geral de meta-texto.

Sem pontuação obrigatória. Três palavras, quinze letras e ao menos uma família
bastam para descrição informativa após essas guardas. Papel corroborado explícito
pode contar com dez letras, para preservar `You are a robot`; não relaxa os limites
para pistas genéricas. Não há exceções por TID. Qualidades:

* 0: sem núcleo informativo;
* 1: núcleo curto informativo;
* 2: >=20 palavras no núcleo e >=2 famílias;
* 3: >=75 palavras, >=3 famílias e boilerplate removido <50%.

A razão de boilerplate é `max(0,1-len(core)/len(text))`, arredondada a seis casas;
é proporção de caracteres descartados, não classificação semântica de cada trecho.
Métricas de comprimento/palavras referem-se explicitamente ao núcleo nos campos
`description_core_*`; `description_length` refere-se ao original normalizado.
O léxico é EN/PT e incompleto. Pode perder metáforas, textos em outras línguas,
frases sem essas pistas e conteúdo após seções achatadas de controles. Pode manter
créditos junto de uma frase explicativa ou relato técnico com pistas incidentais.
Não é inferência de LLM, descrição inventada ou rótulo de qualidade validado.

## Visual: informação de pixels, sem hipótese de gameplay

Analisa `__label__` do `.p8`: exige 128 linhas de 128 índices 0..v, duas cores
ou mais e pelo menos dezesseis pixels fora da cor dominante. Label ausente,
malformado, monocromático ou quase vazio não sinaliza informação. Cartucho
preferencial exige também assinatura PNG e tamanho 160x205 no IHDR.

Fallback: spritesheet PNG 128x128 associado a `__gfx__` não trivial. Gfx aceita
até 128 linhas, completando linhas/pixels omitidos com zero; aplica o mesmo
mínimo de dezesseis pixels minoritários e duas cores. `visual_evidence_level`:
`cartridge_pixels` (rank 2), `asset_pixels` (rank 1), `none` (rank 0).
`cartridge_visual_informative` fica separado do sinal visual final.

Mapas e URLs continuam inventariados; rendered_map não é automaticamente escolhido,
pois a memória pode armazenar dados. As categorias públicas futuras são cartridge,
spritesheet, rendered_map e none; nesta revisão rendered_map é somente inventário.
PNG é checado no cabeçalho; não há decodificação completa de pixels do PNG, OCR ou
verificação semântica. A análise usa pixels do texto do cartucho: informa conteúdo
não trivial, sem confirmar que a imagem contém gameplay. Textos, menus e padrões
podem satisfazer a regra. `visual_representativeness=unverified` sempre.

## Música: evidência específica por faixa

Mascara -- e //, strings e comentários longos [=[...]=]. Só reconhece chamadas
bare `music(...)`; não métodos. Inteiro literal decimal/hexadecimal, com sinal,
é constante. Expressões não são avaliadas. Detecta conservadoramente declarações
locais e atribuições múltiplas evidentes com music, parâmetros, redefinições e
_ENV; ambiguidades rebaixam a certificação. Não é parser Lua completo.

* strong: constante igual ao start_pattern no prefixo direto da global _init;
* moderate: mesma constante em prefixo direto de função nomeada plausível de estado;
* weak: constante no intervalo inclusivo start_pattern..end_pattern, sem certificação
  acima; inclui matches exatos condicionais/ambíguos e patterns internos;
* none: nenhuma constante relevante ao segmento encontrada.

Dinâmicas constam em `music_dynamic_call_count` e na evidência, mas **nunca promovem**
a faixa. Paradas negativas não associam segmento. song_id não é argumento de music.
Controle de fluxo encerra certificação de prefixo; shorthand if/escopo incerto pode
rebaixar funções posteriores. Nomes de estado são init/start/title/menu/ending/
gameover/state/level/room/scene delimitados, conforme regex do código. Certificação
não prova reachability, execução do callback ou correspondência a um quadro.
Evidências incluem linha física do .p8, lua_line, função sugerida e correspondências.

## Ranking e ordem de apresentação

`context_priority` começa em 1. Chave crescente: tier A/B/C/D; -description_quality;
-music confidence (strong=3, moderate=2, weak=1, none=0); -visual evidence rank;
-min(core_words,100); TID numérico; song_id. Contagem de evidências já determina
univocamente o tier. **Estrelas, tags, título, autor e comentários não entram.**

`serve_order` é uma permutação separada: processa tiers A/B/C/D; cria filas por
TID ordenadas por context_priority; em cada rodada ordena TIDs pela prioridade
da próxima música e serve uma por TID. Repete até esgotar o tier. Quando só resta
um TID, sequências são inevitáveis. Podem ocorrer adjacências nas fronteiras das
rodadas. Diversidade causa inversões de prioridade dentro do tier deliberadamente.
O CSV está fisicamente ordenado por **serve_order**, não context_priority.

## Auditoria, portabilidade e campos públicos

Confere hashes de train/val/test antes/depois, identidades únicas, conjunto exato
TEST, concordância com manifest, ausência de sobreposição com identidades reais
de train/val e igualdade integral com songs.json. Lê metadados/assets/Lua uma vez
por TID; compara cada música individualmente. Não lê resultados de sistemas,
matching, métricas, respostas humanas ou dados de avaliação subjetiva.

Auditoria registra regras, versão Python, hash de implementação/manifest/CSV,
hashes de fontes com paths relativos e roots lógicos dataset/splits/repository.
Nenhum root físico absoluto é serializado. A localização é fornecida pelo CLI;
a mesma árvore em outro path produz os mesmos bytes no mesmo Python/SO.
Não depende de relógio, rede, aleatoriedade ou ordem de filesystem. Versões de
Unicode/Python e finais de linha entre SOs podem alterar bytes.

Título/autor são somente auxiliares: generic credits made/music/code by não
viram nomes; mistura de papéis/jam deixa author vazio. Confiança header_only ou
unknown, sem validação para exibição pública. Não resolve créditos multilinha
universalmente. Stars são apenas contagem histórica informativa, zero ambíguo.

O CSV não é payload do frontend. Não enviar automaticamente evidência Lua, hashes,
tiers, paths internos, priority keys ou stars. A integração deverá construir uma
projeção pública explícita. Label sugerido: **Cartucho de origem e contexto visual**,
com tipo identificado como imagem do cartucho ou spritesheet de assets; sem cena.

Formulação metodológica: scenarios were prioritized using three separately computed,
deterministic proxies for available contextual evidence: informative description
cues, nontrivial cartridge/asset pixels, and sufficiently direct static references
to the segment start pattern. Runtime scenes and visual representativeness were
not verified. A separate tier-preserving serving order increases cartridge diversity.
