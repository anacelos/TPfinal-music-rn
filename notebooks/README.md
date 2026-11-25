# Notebooks: o projeto passo a passo

Seis notebooks curtos que explicam o trabalho na ordem em que ele acontece: o que é o dado,
que decisões tomamos, por quê, e o que saiu. Já estão executados: dá para ler (e ouvir) direto
no GitHub ou no Jupyter, sem rodar nada.

| Notebook | Pergunta que responde |
|---|---|
| [01_pico8_e_o_dataset](01_pico8_e_o_dataset.ipynb) | O que é uma música de PICO-8 e como ela chega até nós? |
| [02_filtros_e_split](02_filtros_e_split.ipynb) | Quais músicas entram, e como dividir treino/validação/teste sem vazamento? |
| [03_tokenizacao](03_tokenizacao.ipynb) | Como a música vira uma sequência de tokens (nlm, grid_unrolled, grid_block)? |
| [04_treino](04_treino.ipynb) | O que o modelo aprende, e como lemos loss, perplexidade e bits/s? |
| [05_geracao](05_geracao.ipynb) | Como o modelo gera uma música, e o que são temperatura e top-k? |
| [06_avaliacao](06_avaliacao.ipynb) | As geradas se parecem com as reais? (PE, SC, GC, OA, KS, FAD, KLD) |

Cada um segue o mesmo roteiro: contexto, opções, decisão e porquê, mão na massa com os dados
reais, resultado e uma seção "Experimente".

## Para rodar

Roda no computador de cada um, em CPU, em poucos minutos. Nada é treinado de novo: os
notebooks leem os pesos, as músicas geradas e as tabelas da pasta do Drive do projeto.

1. Instale as dependências:

   ```
   pip install -r notebooks/requirements.txt
   ```

2. Baixe a [pasta do Drive](https://drive.google.com/drive/folders/1wirkz0ZRidwRh7db2YXRPaPeTJ1zr4cO)
   (cerca de 55 MB) para `artefatos/`, na raiz do repositório:

   ```
   gdown --folder https://drive.google.com/drive/folders/1wirkz0ZRidwRh7db2YXRPaPeTJ1zr4cO -O artefatos
   ```

   Também serve baixar pelo navegador e descompactar. Se a pasta ficar em outro lugar, aponte
   a variável `NB_ARTEFATOS` para ela.

3. Abra o Jupyter a partir desta pasta:

   ```
   cd notebooks
   jupyter lab
   ```

A pasta `artefatos/` deve ficar assim (o `README.md` do Drive descreve cada arquivo):

```
artefatos/
  checkpoints/{nlm,grid_block}/   best.pt, history.json, val_metrics.json, test_metrics.json, ...
  dados/                          test.jsonl.gz (músicas reais do teste), build_stats.json
  predicoes-teste/                1.854 músicas geradas por sistema
  resultados/                     métricas simbólicas e de áudio
```

## Como estão organizados

- `nbkit.py`: funções de apoio (ler arquivos, piano roll, player de áudio, carregar o modelo).
  Os notebooks usam o código da `main` sem cópias: `src/tokenizer.py`, `src/model.py`,
  `src/sample.py`, `src/synth.py` e `evaluation/pico8_metrics.py`.
- Os áudios embutidos têm 8 s e 16 kHz, para os arquivos ficarem leves.
- A geração do notebook 5 é sorteada com semente fixa; em outra máquina as músicas podem sair
  diferentes das que estão salvas.
