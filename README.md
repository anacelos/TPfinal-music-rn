# TPfinal-music-rn — Geração Musical Simbólica no PICO-8 (DCC831 - UFMG)

Repositório do Trabalho Prático Final de **IA Generativa para Música (DCC831 - UFMG)**, focado na geração simbólica nativa de músicas chiptune de 4 canais do console de fantasia **PICO-8** utilizando Redes Neurais Autoregressivas.

**Mapa do projeto, etapa por etapa, com o que falta fazer:** [docs/ETAPAS.md](docs/ETAPAS.md).

**Notebooks passo a passo (dados, tokenização, treino, geração, avaliação):** [notebooks/](notebooks/README.md).

**Resultados e glossário das métricas:** [docs/RESULTADOS.md](docs/RESULTADOS.md) (o que já foi rodado, o que cada métrica significa, tabelas e como reproduzir).

**Músicas geradas (5 do modelo e 5 do baseline, em JSON e WAV):** [samples/](samples/README.md).

---

## 1. Pré-processamento e Divisão sem Vazamento (`preprocessing/`)

O pré-processamento lê os arquivos `songs.json` do dataset (`dataset_pico8_v2/dataset_pico8_v2/audio/[TID]/songs.json`, 8.367 jogos, 24.077 músicas) e grava os splits em `data/splits/{train,val,test}.jsonl`: uma música por linha, no mesmo formato do `songs.json`, com o campo `tid` adicionado. O `src/data.py` só tokeniza esses arquivos.

### Filtros (nesta ordem; o primeiro que reprova é o motivo registrado)
- **`few_notes`**: menos de 16 notas audíveis (`volume > 0`). Remove também as músicas silenciosas e as de 0 ticks (`2.054` músicas).
- **`custom_instruments`**: `uses_custom_instruments == True`, porque o vocabulário e o sintetizador cobrem só os instrumentos `0..7` (`4.025`).
- **`incompatible_speeds`**: algum pattern com canais cujas velocidades não são múltiplas da menor (`1.356`).
- **`duplicate`**: música idêntica (mesmas notas, velocidades, ticks, loop e fim) a outra já mantida; fica a primeira na ordem do corpus (`tid` numérico, depois `song_id`) (`4.258`, quase todas cópias entre jogos diferentes).

Ficam **`12.384` músicas**.

### Divisão
Jogos que compartilham um bloco de SFX (mesmas 32 notas, com pelo menos 4 audíveis) ficam no mesmo grupo, e cada grupo vai inteiro para um split (`seed=831`, 70/15/15 por número de músicas). Os 2 grupos com mais de 1% das músicas vão para o treino. Resultado: **Treino `8.669` músicas (`4.092` jogos)**, **Validação `1.861` (`894` jogos)**, **Teste `1.854` (`873` jogos)**, sem nenhuma música repetida nem bloco de SFX compartilhado entre splits. Na divisão anterior do `data.py`, só por `TID`, 466 das 1.346 músicas de teste (34,6%) eram idênticas a músicas do treino.

```bash
python3 -s -m preprocessing.build_dataset   # grava data/splits/
python3 -s -m preprocessing.verify --roundtrip   # checagens de cobertura, vazamento, filtros, formato e round-trip
python3 -s -m preprocessing.analysis   # figuras em preprocessing/figures/
```

---

## 2. Representações e Tokenizadores (`src/tokenizer.py`)

O módulo [`src/tokenizer.py`](src/tokenizer.py) implementa três esquemas reversíveis (`encode_song` $\leftrightarrow$ `decode_tokens`) para investigar o impacto da representação e do nível de desenrolamento (*unrolling*):

1. **`nlm` (NES Music Language / LakhNES — Donahue et al., ISMIR 2019)**:
   - Linha do tempo orientada a eventos de mudança de estado por canal (`CHc_NOTEON_p`, `CHc_NOTEOFF`, `CHc_INST_i`, `CHc_VOL_v`, `CHc_FX_f`) intercalados por avanços temporais em ticks de hardware (`WT_1..64`).
   - **Vocabulário**: `709` tokens (`571` ativos no dataset).
   - **Comprimento no Treino**: Média `711,4` tokens/música | Mediana `270` | P90 `1.843` (`6,17M` tokens no total, **redução de 53,5%** vs. `grid_unrolled`).
   - **Fidelidade Round-Trip (`encode` $\rightarrow$ `decode`)**: **99,95%** (`12.378/12.384` músicas idênticas tick a tick, inclusive fim e início do loop; nas 6 restantes só o início do loop muda, porque ele cai num pattern de 0 ticks).

2. **`grid_unrolled` (REMI+ Tracker Grid Desenrolado — adaptado de Huang & Yang, 2020)**:
   - Grade polifônica explícita por pattern (`PATTERN_START`, `REF_SPEED_s`, `[REF_STEPS_k]`, `CHc_START`, `SPEED_s`, `STEP_k`, `PITCH_p`, `INST_i`, `VOL_v`, `[FX_f]`), totalmente desenrolada (repetindo todas as notas de blocos reutilizados).
   - **Vocabulário**: `1.151` tokens (`878` ativos no dataset).
   - **Comprimento no Treino**: Média `1.528,8` tokens/música | Mediana `552` | P90 `3.763` (`13,25M` tokens no total).
   - **Fidelidade Round-Trip (`encode` $\rightarrow$ `decode`)**: **99,91%** (`12.373/12.384` músicas idênticas tick a tick; 5 diferem nas notas e 6 no início do loop).

3. **`grid_block` (Tracker Grid Estruturado com Ponteiros de Bloco)**:
   - Mesma gramática local do `grid_unrolled`, mas preservando os ponteiros de blocos (`blocks_dictionary` / `is_repeat`): define os 32 steps canônicos de cada bloco inédito com `CHc_DEF_BLOCK_b ... END_BLOCK` (incluindo `BLK_LSTART_ls` e `BLK_LEND_le`) e referencia repetições futuras na música com um único token `CHc_USE_BLOCK_b`.
   - **Vocabulário**: `1.250` tokens (`947` ativos no dataset).
   - **Comprimento no Treino**: Média `615,2` tokens/música | Mediana `366` | P90 `1.324` (`5,33M` tokens no total, **redução de 59,8%** em relação ao `grid_unrolled`).
   - **Fidelidade Round-Trip (`encode` $\rightarrow$ `decode`)**: **99,95%** (`12.378/12.384` músicas idênticas tick a tick, inclusive fim e início do loop; nas 6 restantes só o início do loop muda, porque ele cai num pattern de 0 ticks).

---

## 3. Como Executar

### Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dataset

O `dataset_pico8_v2` (8.367 jogos, 24.077 músicas, 2,8 GB de `songs.json`) não é distribuído
neste repositório. Para obter o `dataset_pico8_v2.zip` completo, escreva para o João
(joao.c.silveira@ufv.br). Descompacte-o numa pasta `dataset_pico8_v2/` na raiz do repositório,
de modo que existam os arquivos `dataset_pico8_v2/dataset_pico8_v2/audio/<tid>/songs.json`:

```bash
unzip dataset_pico8_v2.zip -d dataset_pico8_v2
```

Sem o dataset completo, use a amostra de 300 jogos que vem no repositório
([data/amostra/](data/amostra/README.md), 1,4 MB). Ela segue o mesmo layout, então todos os
comandos abaixo rodam sem mudança; só os números (splits, métricas) saem diferentes dos do
artigo:

```bash
unzip data/amostra/dataset_pico8_amostra.zip -d dataset_pico8_v2
```

### Dados

```bash
# Gerar os splits e depois processar todos os modos (nlm, grid_unrolled, grid_block);
# --test-roundtrip também gera WAVs de verificação round-trip
python3 -s -m preprocessing.build_dataset
python3 src/data.py --out-dir data/processed --test-roundtrip
```

### Treino, geração e avaliação

Os comandos de treino, teste, geração, métricas simbólicas, áudio e exemplos estão em
[docs/RESULTADOS.md, seção 4](docs/RESULTADOS.md#4-como-reproduzir). Os checkpoints e as músicas
geradas que deram os números do artigo estão no
[Drive](https://drive.google.com/drive/folders/1wirkz0ZRidwRh7db2YXRPaPeTJ1zr4cO), para quem
quiser pular o treino.

---

## 4. Contribuição de cada integrante

Cada integrante escreve a sua parte, na própria subseção.

### Ana Clara Barcelos

- Modelo (`src/model.py`): classe `ModelMusic` (`nn.Module`), modelo de linguagem autoregressivo sobre os tokens do `src/tokenizer.py`, com Embedding (`padding_idx` = PAD, id 0) → LSTM (`batch_first=True`) → Linear (`hidden_dim` → `vocab_size`). O `forward(x, hidden=None)` recebe `(batch, seq_len)` e devolve `(logits, hidden)`, com logits `(batch, seq_len, vocab_size)`; devolver o `hidden` permite gerar token a token na amostragem. Baseado no tutorial oficial do PyTorch ([Build the Neural Network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)).

### Carlos Henrique Brito Malta Leão

- Análise exploratória do dataset do PICO-8 e figuras do corpus.
- Pré-processamento das músicas: filtros por notas audíveis, instrumentos personalizados e
  velocidades incompatíveis, e remoção de duplicatas pelo conteúdo musical.
- Divisão em treino, validação e teste por grupos de jogos que compartilham SFX, para que nenhuma
  música ou SFX apareça em mais de um subconjunto.
- Integração do pré-processamento com a tokenização, que passou a ler os splits gerados.
- Redação das seções "Dataset e pré-processamento" e "Trabalhos relacionados" do artigo.

### Giovanni Martinelli

- Execução de todos os treinos, gerações e avaliações no cluster de GPUs do laboratório (jobs,
  checkpoints e predições publicados no Drive do grupo).
- Reescrita do sintetizador do PICO-8 a partir do zepto8 (`src/synth.py`), com os filtros de SFX
  lidos dos cartuchos (`src/sfx_filters.py`), conferida com gravações do player oficial.
- Métricas de áudio: FAD sobre embeddings do CLAP, CLAP score e KLD (`evaluation/audio_metrics.py`),
  clipes de 10 s (`src/render.py`, `evaluation/audio_clips_eval.py`).
- Bits por segundo por música inteira, comparável entre tokenizadores (`src/nll.py`,
  `evaluation/bits_per_second.py`).
- Correções no decodificador do tokenizador e no pipeline de dados; vocabulário gravado no
  checkpoint do `train.py`.
- Documentação: `docs/RESULTADOS.md`, `docs/ETAPAS.md` com as figuras, os notebooks de
  `notebooks/` e os exemplos de `samples/` (`evaluation/export_samples.py`).
- Site e servidor do teste de escuta ([GiowGiow/pico8-listening-test](https://github.com/GiowGiow/pico8-listening-test)).

### Igor Varejão

_(a preencher)_

### João

_(a preencher)_

### Yuri Dantas

- Desenvolvimento e calibração do baseline generativo baseado em cadeias de Markov de primeira ordem.
- Implementação das métricas simbólicas PE, SC e GC e comparação das distribuições por Kolmogorov–Smirnov e Wasserstein-1.
- Desenvolvimento do pareamento global entre músicas reais e geradas pelo algoritmo Húngaro, utilizando descritores de altura, ritmo e textura.
- Planejamento da avaliação subjetiva, seleção e contextualização dos cenários, balanceamento das condições experimentais e elaboração das checagens de atenção.
- Preparação do pacote de 240 cenários para integração com a aplicação web e desenvolvimento do pipeline de análise estatística.
- Análise estatística das avaliações subjetivas de 71 participantes, incluindo comparações pareadas, correção para múltiplas comparações, análise de sensibilidade e análise exploratória por perfil.
- Redação e revisão das seções de avaliação objetiva e subjetiva do artigo e elaboração dos respectivos slides.

---

## 5. Uso de IA

O enunciado permite usar ferramentas de IA como apoio na implementação e na escrita, desde que o
uso seja declarado aqui: o que foi usado, com qual objetivo e os links das conversas. Quem não
usou escreve "Não usei IA".

### Ana Clara Barcelos

- **Ferramentas:** Claude Code (modelo Claude Opus 5.5, Anthropic).
- **Para quê:** tutoria. A IA não escreveu o código do modelo: explicou conceitos (Embedding, LSTM, `padding_idx`, shapes de `output`, `h_n` e `c_n`), mostrou exemplos numéricos para rodar no terminal, revisou o código que eu escrevi e apontou erros. Também foi usada para organizar o cronograma do trabalho (27/10) e para redigir esta descrição a partir do meu código (25/11).
- **Conversas:** sessões do Claude Code (Code tab) de 27/10, 16–22/11 e 25/11/2026.

### Carlos Henrique Brito Malta Leão

- **Ferramentas:** Claude Code (Anthropic), no terminal.
- **Para quê:** entender a estrutura dos dados de música dos cartuchos do PICO-8; revisão do pré-processamento, e das análises do corpus; conferir os números do artigo com o código; pesquisar referências e montar o `.bib`; revisar o português das seções "Dataset e pré-processamento" e "Trabalhos
  relacionados".
- **Conversas:** sessões do Claude Code no terminal, que não geram link público. As transcrições
  ficam guardadas localmente e podem ser enviadas ao professor sob pedido.

### Giovanni Martinelli

- **Ferramentas:** Claude Code (Anthropic), no terminal e no app de desktop.
- **Para quê:** apoio de implementação, revisão e documentação: correções no tokenizador e no
  pipeline de dados (PR #11); reescrita do sintetizador e tabela de filtros de SFX (PR #12);
  métricas de áudio (PR #13); gravação do vocabulário no checkpoint do `train.py` (PR #14, mudança
  pequena no código da Ana); documentação de resultados, mapa do projeto, figuras e notebooks
  (PRs #17 a #21); bits/s por música, clipes de áudio e exemplos de `samples/` (PRs #23 e #25);
  scripts para rodar os jobs no cluster (fora do repositório); site do teste de escuta. Não usei
  IA na classe do modelo, no laço de treinamento nem na amostragem (`src/model.py`,
  `src/train.py`, `src/sample.py`), exceto a mudança do PR #14.
- **Conversas:** sessões do Claude Code no terminal, que não geram link público. As transcrições
  ficam guardadas localmente e podem ser enviadas ao professor sob pedido.

### Igor Varejão

- **Ferramentas:** _(a preencher)_
- **Para quê:** _(a preencher)_
- **Conversas:** _(links)_

### João

- **Ferramentas:** _(a preencher)_
- **Para quê:** _(a preencher)_
- **Conversas:** _(links)_

### Yuri Dantas

- **Ferramentas:** OpenAI ChatGPT e OpenAI Codex.
- **Para quê:** apoio ao planejamento metodológico, estudo das métricas musicais, definição do protocolo experimental, implementação e depuração de código, elaboração de testes automatizados, interpretação dos resultados, documentação, revisão científica e preparação do artigo e da apresentação. As implementações e decisões metodológicas foram supervisionadas, revisadas e validadas pelo integrante.
- **Conversas:** [Relatório consolidado do uso de ChatGPT e Codex durante o desenvolvimento do trabalho](https://chatgpt.com/s/t_6b076c3e437481918dbccb26c0f424b5). O registro apresenta uma síntese retrospectiva das atividades realizadas com apoio de IA.
