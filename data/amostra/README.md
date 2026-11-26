# Amostra do dataset

`dataset_pico8_amostra.zip` tem 300 jogos sorteados do `dataset_pico8_v2` (seed 831), só o
`audio/<tid>/songs.json` de cada um, 1,4 MB. Serve para rodar o pipeline inteiro sem o dataset
completo (8.367 jogos, 2,8 GB), que não é distribuído: peça ao João (joao.c.silveira@ufv.br).

Depois dos filtros e da divisão, a amostra dá 502 músicas: 351 de treino (163 jogos), 77 de
validação (37) e 74 de teste (40). Os números do `README.md`, do `docs/RESULTADOS.md` e do artigo
são do dataset completo e não se reproduzem com a amostra; ela serve para conferir que o código
roda de ponta a ponta.

O zip usa o mesmo prefixo `dataset_pico8_v2/` do zip completo, então os comandos do `README.md`
são os mesmos:

```bash
unzip data/amostra/dataset_pico8_amostra.zip -d dataset_pico8_v2
python3 -s -m preprocessing.build_dataset
python3 src/data.py --out-dir data/processed
```

`preprocessing.verify` compara os splits com o `data/splits/split_manifest.csv` do dataset
completo e por isso reprova na amostra.

Para refazer a amostra a partir do dataset completo:

```bash
python3 -s -m preprocessing.make_sample --n 300 --seed 831
```
