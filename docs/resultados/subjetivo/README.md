# Avaliação subjetiva — snapshot definitivo 25/11/2026, 20h16

Foram registrados 83 participantes; 71 concluíram pelo menos um cenário, 68 concluíram dois. Dos 139 cenários completos, 138 passaram no controle de atenção (99,28%), mantendo 71 participantes e 414 notas musicais. Quatro participantes contribuíram com um cenário válido e 67 com dois. O snapshot integral foi adotado como oficial. A checagem reprovada exclui apenas o cenário correspondente. A coleta utilizou 240 cenários preparados, com duas avaliações previstas por participante, quatro faixas de 30 s e escala ordinal de 1 a 5. As notas foram habilitadas após pelo menos 10 s de reprodução por faixa.

| Condição | Média por cenário | Mediana | DP amostral |
|---|---:|---:|---:|
| Real | 3,391 | 4 | 1,275 |
| Neural (grid_block-k10) | 3,167 | 3 | 1,293 |
| Markov | 2,442 | 2 | 1,301 |

Teste exato bilateral de sinais sobre diferenças médias por participante, com correção de Holm para três comparações (alfa=0,05): Real–Markov p=0,00001445; Neural–Markov p=0,00013476; Real–Neural p=0,69888343. A inclusão do cenário reprovado não modifica as conclusões. Ausência de significância entre Real e Neural não demonstra equivalência. A amostra de conveniência, a priorização dos contextos e o pareamento pós-geração limitam a generalização. Os resultados por perfil são exploratórios (mínimo de cinco participantes distintos por categoria, sem testes de hipótese por subgrupo).

## Reprodução

A partir da raiz do repositório, com Python 3 (biblioteca padrão):

```bash
python3 evaluation/subjective/analyze_giovanni_snapshot.py --ratings /caminho/privado/ratings.csv --key /caminho/privado/scenario_key.json --out /caminho/privado/resultados
PYTHONPATH=evaluation/subjective python3 -m unittest discover -s evaluation/tests -p test_analyze_subjective_results.py
```

O adaptador converte nomes de colunas do aplicativo para o formato do analisador original; o arquivo intermediário é temporário. Os resultados agregados reproduzidos foram conferidos byte a byte com os quatro arquivos oficiais. A auditoria registra hashes dos insumos. Os CSVs brutos, identificadores dos participantes e a chave privada não são publicados; a reprodução integral requer acesso autorizado a esses arquivos. Os quatro testes do analisador passaram.

- `summary.json`: fluxo, estatísticas, comparações e balanceamento observado.
- `descriptive.csv`: notas por condição e análise de sensibilidade.
- `paired_comparisons.csv`: diferenças por participante, teste de sinais e Holm.
- `exploratory_profiles.csv`: médias por perfil elegível.
- `audit.json`: auditoria dos insumos e da coleta.
