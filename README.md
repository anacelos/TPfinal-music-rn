# TPfinal-music-rn — Geração Musical Simbólica no PICO-8 (DCC831 - UFMG)

Repositório do Trabalho Prático Final de **IA Generativa para Música (DCC831 - UFMG)**, focado na geração simbólica nativa de músicas chiptune de 4 canais do console de fantasia **PICO-8** utilizando Redes Neurais Autoregressivas.

**Mapa do projeto, etapa por etapa, com o que falta fazer:** [docs/ETAPAS.md](docs/ETAPAS.md).

**Notebooks passo a passo (dados, tokenização, treino, geração, avaliação):** [notebooks/](notebooks/README.md).

**Resultados e glossário das métricas:** [docs/RESULTADOS.md](docs/RESULTADOS.md) (o que já foi rodado, o que cada métrica significa, tabelas e como reproduzir).

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

```bash
# Gerar os splits e depois processar todos os modos (nlm, grid_unrolled, grid_block) e gerar WAVs de verificação round-trip
python3 -s -m preprocessing.build_dataset
python3 src/data.py --out-dir data/processed --test-roundtrip
```
