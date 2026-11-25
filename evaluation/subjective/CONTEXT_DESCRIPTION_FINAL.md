# Finalização de context-description-ptbr-v1

Revisão documental dos 15 casos investigados: 31, 36, 41, 61, 62, 99, 171, 183,
188, 194, 198, 211, 220, 236 e 239. Os textos antes/depois, identidades, strength,
justificativas factuais e checksums das fontes locais estão versionados em
`context_description_final_review_v1.json`. A redação utiliza somente os fatos
seguros do relatório anterior, sem importar opiniões ou referências musicais.
Não houve jogo manual, nova tradução automática ou execução dos cartuchos.

Para não substituir silenciosamente o desenho congelado, a saída é derivada:
`outputs/subjective/subjective-240-v1-context-final/`. O `subjective-240-v1` original,
os artefatos de 48 e seus áudios/Attention permanecem históricos intactos.

```bash
.venv/bin/python -B -m evaluation.subjective.finalize_context_descriptions
```

O destino deve ser novo. O construtor verifica os checksums dos outputs de origem
e das fontes documentais, preserva índices/IDs, serve_order, matching, custos,
permutações, visuais e todas as descrições não investigadas. Atualiza o heading
dos 240 registros para **Contexto do jogo**. Os originais textuais e a auditoria
anterior permanecem disponíveis; a revisão documental é adicionada à auditoria.

## Ordem de coleta separada

`collection_order_v2` é zero-based (0–239). A ordem física de `scenarios.jsonl` e
o `serve_order` original continuam iguais aos do subset congelado. A futura coleta
deve usar a fila explícita `collection_queue_v2.csv`:

1. os 10 registros strong da investigação, por serve_order crescente;
2. os 4 moderate, por serve_order crescente;
3. os 225 restantes, por serve_order crescente;
4. o registro weak, índice 188, por último: ordem 239, posição 240.

Os restantes têm `evidence_strength=not_assessed` e grupo `normal`; isso não é uma
nova avaliação de qualidade nem remove flags anteriores. Em particular, o índice
181 mantém sua recomendação de revisão, pois não fez parte dos 15 investigados.
A categoria e o serve_order são os únicos dados usados para ordenar. Custos,
resultados, duração e permutações não entram nessa função. Esta camada altera
explicitamente a prioridade de uma coleta parcial em relação à ordem original;
não altera quais cenários foram selecionados ou seus pares/permutação. Não há
integração com app ou alteração de uma reserva de participante existente.

As flags documentais passam de 12 para 2: 181 e 188. Moderate não implica que seja
preciso jogar: Walker é uma demonstração, Tetris um protótipo e I'm Very Sad uma
animação. Esses escopos são documentados sem acrescentar missões. PewCo 31/198
mantém a mesma descrição; Collab16 mantém literalmente a redação genérica anterior.
Não se força um mínimo artificial de 25 palavras.

## Pergunta e escala

`study_protocol.json` registra a pergunta final:

**Considerando a imagem e a descrição do jogo, quanto esta música combina com esse contexto?**

1. Não combina nada
2. Combina pouco
3. Mais ou menos
4. Combina bem
5. Combina muito bem

## Attention futuro: attention-check-spoken-v2

Somente documentação, sem voz ou WAV gerado:

> Este áudio é uma checagem de atenção. Para confirmar que você está prestando atenção à avaliação, selecione a opção ‘Não combina nada’ para esta faixa.

A fala começa em t=0 e o restante é silêncio até completar exatamente 30 s.
A resposta solicitada é 1. Voz e parâmetros de produção não são decididos nesta
etapa. Os Attention Gaussian do handoff histórico não são convertidos ou
renomeados como spoken-v2. A troca é metodológica e explícita para uma futura
coleta; este artefato documental não é um pacote de áudio pronto para ela.

## Validação

```bash
PYTHONPATH=.:evaluation:evaluation/tests .venv/bin/python -B -m unittest \
  test_context_description_final test_subjective_240 test_subjective_subset
```

Conferir preservação de todos os campos congelados e dos 225 textos restantes,
240 IDs únicos, 48 históricos contidos, 24 permutações ×10, 60 origens/slot,
fila sem repetições e estável por grupo, índice 188 por último, guardas de texto,
fontes verificadas e ausência de novos arquivos de áudio. A comparação de duas
preparações verifica determinismo de todos os arquivos derivados.

Resultado verificado: 44 testes aprovados; 252 arquivos derivados idênticos em
duas preparações; os 248 arquivos do subset de 240 original e os 559 arquivos
históricos conferidos mantêm seus hashes. Permanecem 240 identidades únicas por
conjunto, 24 permutações ×10, 60 ocorrências de cada origem por slot e as mesmas
associações. Revisão recomendada: somente 181 e 188. Não há WAV novo no derivado.
