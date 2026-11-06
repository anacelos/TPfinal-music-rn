# Avaliação do TP2

Essa pasta é a parte de **avaliação/métricas** do projeto.

A ideia é simples: usar o mesmo pipeline para comparar três conjuntos:

- músicas reais do test set;
- músicas geradas pelo baseline;
- músicas geradas pelo modelo principal.

Tudo é avaliado direto na representação simbólica do PICO-8, sem precisar converter para MIDI.

## O que já funciona

Hoje essa parte já consegue:

- ler os `audio.json`;
- avaliar uma peça a partir de um conjunto de music patterns;
- calcular métricas estruturais;
- processar várias peças de uma vez;
- salvar os resultados em CSV;
- comparar Real × Baseline × Neural;
- calcular estatísticas e distâncias entre distribuições;
- gerar boxplots automaticamente.

Também tem **20 testes automatizados passando**.

## Métricas atuais

Por enquanto estão implementadas:

- **Pitch Range**  
  extensão entre o menor e o maior pitch usado;

- **Pitch-Class Entropy**  
  diversidade das classes de pitch;

- **Structural Note Density**  
  proporção de slots realmente audíveis;

- **Audible Channel Utilization**  
  quanto dos 4 canais do PICO-8 é efetivamente usado;

- **SFX Repetition Ratio**  
  quanto os mesmos SFX são reutilizados dentro da peça.

Além disso, o pipeline também coleta informações de instrumento, efeito, `speed`, flags e SFX vazios.

As definições mais detalhadas estão em [`METRICS.md`](METRICS.md).

## Como a representação está sendo interpretada

A implementação foi conferida com o `extract_audio.py` e o `synthesize_audio.py` usados para gerar o dataset.

As regras que estamos usando são:

- valores de canal `0..63` apontam diretamente para um SFX;
- valores `>= 64` não produzem áudio;
- uma nota só é considerada audível quando `volume > 0`;
- cada SFX possui 32 slots;
- um SFX referenciado mas ausente é tratado como vazio;
- reutilizações do mesmo SFX contam novamente, porque ele está sendo tocado outra vez.

O sintetizador atual renderiza cada `music pattern` separadamente, então um arquivo `music_XX.wav` não representa necessariamente uma música completa.

## Sobre o split

O split deve ser feito por **TID inteiro**.

TID é o ID do jogo, e um mesmo jogo pode ter várias músicas reaproveitando os mesmos SFX. Então não devemos colocar músicas do mesmo TID em train e test, por exemplo.

## O que ainda não está fechado

A parte estrutural da avaliação já funciona, mas algumas decisões ainda dependem das outras frentes do projeto:

- como os patterns serão agrupados em uma música completa;
- tokenização final;
- split definitivo de train/val/test;
- formato da saída/decoder do baseline;
- formato da saída/decoder do modelo principal;
- reconstrução de loops, stop e sequência completa de patterns.

Por isso, o avaliador recebe explicitamente quais patterns formam uma peça. Quando essa definição estiver fechada, basta trocar o manifesto, sem reescrever as métricas.

## Estrutura da pasta

```text
evaluation/
├── README_AVALIACAO.md
├── METRICS.md
├── pico8_metrics.py
├── evaluate_manifest.py
├── compare_results.py
├── plot_distributions.py
├── pieces.example.json
├── requirements.txt
└── tests/
```

## Rodando os testes

Dentro da pasta `evaluation`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Estado atual:

```text
Ran 20 tests
OK
```

## Avaliando peças

O manifesto diz quais patterns pertencem a cada peça.

Exemplo:

```json
{
  "pieces": [
    {
      "piece_id": "example_01",
      "tid": "12345",
      "patterns": [0, 1, 2, 3]
    }
  ]
}
```

Depois:

```bash
python evaluate_manifest.py \
  --dataset /caminho/para/dataset_pico8 \
  --manifest pieces.example.json \
  --output results.csv
```

## Comparando os sistemas

Quando tivermos os três conjuntos:

```bash
python compare_results.py \
  --real real_test.csv \
  --baseline baseline.csv \
  --neural neural.csv \
  --output comparison_summary.csv
```

Isso gera estatísticas como média, mediana, desvio padrão, quartis e também KS/Wasserstein em relação ao conjunto real.

## Gerando os gráficos

```bash
python plot_distributions.py \
  --real real_test.csv \
  --baseline baseline.csv \
  --neural neural.csv \
  --output-dir plots
```

Hoje os gráficos principais são boxplots das métricas estruturais.

## Observação

Os arquivos de exemplo servem só para testar a infraestrutura.

Eles **não são resultados finais do TP2**.
