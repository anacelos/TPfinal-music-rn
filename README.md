# TPfinal-music-rn — Geração Musical Simbólica no PICO-8 (DCC831 - UFMG)

Repositório do Trabalho Prático Final de **IA Generativa para Música (DCC831 - UFMG)**, focado na geração simbólica nativa de músicas chiptune de 4 canais do console de fantasia **PICO-8** utilizando Redes Neurais Autoregressivas.

---

## 1. Pipeline de Dados e Divisão sem Vazamento (`src/data.py`)

O pré-processamento consome os arquivos `songs.json` extraídos pelo repositório do dataset (`/home/jcss/projetos/dataset_pico8/dataset_pico8/audio/[TID]/songs.json`).

### Critérios de Filtragem Padrão
- **Apenas instrumentos nativos padrão (`0..7`)**: `uses_custom_instruments == False`, garantindo 100% de fidelidade entre os tokens simbólicos e o sintetizador de formas de onda PICO-8.
- **Progressão estrutural mínima**: `num_patterns >= 2` (descarta loops triviais de um único bloco).
- **Divisão por Jogo (`TID`)**: Os `5.753` cartuchos válidos (`13.010` músicas) são divididos por `TID` (`seed=42`) em **80% Treino (`4.602` jogos / `10.515` músicas)**, **10% Validação (`575` jogos / `1.247` músicas)** e **10% Teste (`576` jogos / `1.248` músicas)**, garantindo **vazamento zero (zero data leakage)** entre splits.

---

## 2. Representações e Tokenizadores (`src/tokenizer.py`)

O módulo [`src/tokenizer.py`](src/tokenizer.py) implementa três esquemas reversíveis (`encode_song` $\leftrightarrow$ `decode_tokens`) para investigar o impacto da representação e do nível de desenrolamento (*unrolling*):

1. **`nlm` (NES Music Language / LakhNES — Donahue et al., ISMIR 2019)**:
   - Linha do tempo orientada a eventos de mudança de estado por canal (`CHc_NOTEON_p`, `CHc_NOTEOFF`, `CHc_INST_i`, `CHc_VOL_v`, `CHc_FX_f`) intercalados por avanços temporais em ticks de hardware (`WT_1..64`).
   - **Vocabulário**: `677` tokens (`534` ativos no dataset).
   - **Comprimento no Treino**: Média `1.099,5` tokens/música | Mediana `627` | P90 `2.465` (`11,56M` tokens no total).

2. **`grid_unrolled` (REMI+ Tracker Grid Desenrolado — adaptado de Huang & Yang, 2020)**:
   - Grade polifônica explícita por pattern (`PATTERN_START`, `REF_SPEED_s`, `CHc_START`, `SPEED_s`, `STEP_k`, `PITCH_p`, `INST_i`, `VOL_v`, `[FX_f]`), totalmente desenrolada (repetindo todas as notas de blocos reutilizados).
   - **Vocabulário**: `735` tokens (`461` ativos no dataset).
   - **Comprimento no Treino**: Média `2.253,0` tokens/música | Mediana `1.298` | P90 `5.018` (`23,69M` tokens no total).

3. **`grid_block` (Tracker Grid Estruturado com Ponteiros de Bloco)**:
   - Mesma gramática local do `grid_unrolled`, mas preservando os ponteiros de blocos (`blocks_dictionary` / `is_repeat`): define cada bloco inédito com `CHc_DEF_BLOCK_b ... END_BLOCK` e referencia repetições futuras na música com um único token `CHc_USE_BLOCK_b`.
   - **Vocabulário**: `1.248` tokens (`949` ativos no dataset).
   - **Comprimento no Treino**: Média `941,7` tokens/música | Mediana `697` | P90 `1.878` (`9,90M` tokens no total — **redução de 58,2%** em relação ao `grid_unrolled`).

---

## 3. Como Executar

```bash
# Processar todos os modos (nlm, grid_unrolled, grid_block) e gerar WAVs de verificação round-trip
python3 src/data.py --out-dir data/processed --test-roundtrip
```
