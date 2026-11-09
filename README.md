# TPfinal-music-rn — Geração Musical Simbólica no PICO-8 (DCC831 - UFMG)

Repositório do Trabalho Prático Final de **IA Generativa para Música (DCC831 - UFMG)**, focado na geração simbólica nativa de músicas chiptune de 4 canais do console de fantasia **PICO-8** utilizando Redes Neurais Autoregressivas.

---

## 1. Pipeline de Dados e Divisão sem Vazamento (`src/data.py`)

O pré-processamento consome os arquivos `songs.json` extraídos pelo repositório do dataset (`/home/jcss/projetos/dataset_pico8/dataset_pico8/audio/[TID]/songs.json`).

### Critérios de Filtragem Padrão
- **Apenas instrumentos nativos padrão (`0..7`)**: `uses_custom_instruments == False`, garantindo 100% de fidelidade entre os tokens simbólicos e o sintetizador de formas de onda PICO-8.
- **Progressão estrutural mínima**: `num_patterns >= 2` e `total_ticks > 0` (descarta loops triviais de um único bloco ou silenciosos).
- **Divisão por Jogo (`TID`)**: Os `5.742` cartuchos válidos (`12.951` músicas) são divididos por `TID` (`seed=42`) em **80% Treino (`4.593` jogos / `10.362` músicas)**, **10% Validação (`574` jogos / `1.243` músicas)** e **10% Teste (`575` jogos / `1.346` músicas)**, garantindo **vazamento zero (zero data leakage)** entre splits.

---

## 2. Representações e Tokenizadores (`src/tokenizer.py`)

O módulo [`src/tokenizer.py`](src/tokenizer.py) implementa três esquemas reversíveis (`encode_song` $\leftrightarrow$ `decode_tokens`) para investigar o impacto da representação e do nível de desenrolamento (*unrolling*):

1. **`nlm` (NES Music Language / LakhNES — Donahue et al., ISMIR 2019)**:
   - Linha do tempo orientada a eventos de mudança de estado por canal (`CHc_NOTEON_p`, `CHc_NOTEOFF`, `CHc_INST_i`, `CHc_VOL_v`, `CHc_FX_f`) intercalados por avanços temporais em ticks de hardware (`WT_1..64`).
   - **Vocabulário**: `709` tokens (`565` ativos no dataset).
   - **Comprimento no Treino**: Média `1.118,2` tokens/música | Mediana `641` | P90 `2.506` (`11,59M` tokens no total — **redução de 51,4%** vs. `grid_unrolled`).
   - **Fidelidade Round-Trip (`encode` $\rightarrow$ `decode`)**: **100,00%** (`12.951/12.951` músicas idênticas tick a tick).

2. **`grid_unrolled` (REMI+ Tracker Grid Desenrolado — adaptado de Huang & Yang, 2020)**:
   - Grade polifônica explícita por pattern (`PATTERN_START`, `REF_SPEED_s`, `[REF_STEPS_k]`, `CHc_START`, `SPEED_s`, `STEP_k`, `PITCH_p`, `INST_i`, `VOL_v`, `[FX_f]`), totalmente desenrolada (repetindo todas as notas de blocos reutilizados).
   - **Vocabulário**: `1.151` tokens (`876` ativos no dataset).
   - **Comprimento no Treino**: Média `2.300,5` tokens/música | Mediana `1.298` | P90 `5.205` (`23,84M` tokens no total).
   - **Fidelidade Round-Trip (`encode` $\rightarrow$ `decode`)**: **99,95%** (`12.944/12.951` músicas idênticas; `99,9935%` dos ticks).

3. **`grid_block` (Tracker Grid Estruturado com Ponteiros de Bloco)**:
   - Mesma gramática local do `grid_unrolled`, mas preservando os ponteiros de blocos (`blocks_dictionary` / `is_repeat`): define os 32 steps canônicos de cada bloco inédito com `CHc_DEF_BLOCK_b ... END_BLOCK` (incluindo `BLK_LSTART_ls` e `BLK_LEND_le`) e referencia repetições futuras na música com um único token `CHc_USE_BLOCK_b`.
   - **Vocabulário**: `1.250` tokens (`947` ativos no dataset).
   - **Comprimento no Treino**: Média `853,1` tokens/música | Mediana `662` | P90 `1.777` (`8,84M` tokens no total — **redução de 62,9%** em relação ao `grid_unrolled`).
   - **Fidelidade Round-Trip (`encode` $\rightarrow$ `decode`)**: **100,00%** (`12.951/12.951` músicas idênticas tick a tick).

---

## 3. Como Executar

```bash
# Processar todos os modos (nlm, grid_unrolled, grid_block) e gerar WAVs de verificação round-trip
python3 src/data.py --out-dir data/processed --test-roundtrip
```
