# Pacote definitivo — 240 cenários

Este arquivo de transferência contém apenas `public/` e estas instruções. Nenhum
frontend/backend foi alterado. Extrair o `.tar.zst` e servir somente o conteúdo de
`public/` como assets estáticos. Os caminhos são relativos à raiz desses assets.

## Integração mínima

- `scenarios.json`: lista de 240 registros já ordenados por `collection_order_v2`
  (zero-based, 0–239). Campos: external_id, descrição, heading, imagem, caminhos
  A/B/C/D e ordem de coleta. Usar essa ordem para reservar os próximos disponíveis.
- `study.json`: heading, pergunta, escala, dois cenários por participante e os
  seis campos da tela inicial. Coletar faixa etária, experiência musical,
  frequência com jogos, familiaridade com jogos retrô/8-bit, região do Brasil ou
  exterior e uso de fones. Os primeiros cinco campos podem ser armazenados como
  strings de categorias; fones como sim/não. O pacote não define novos cortes de
  idade ou escalas demográficas: manter as categorias adotadas pelo estudo/app e
  registrar seus rótulos e versão, sem inferir respostas a partir das avaliações.
- `scenarios/<external_id>/`: `context.png`, `A.wav`, `B.wav`, `C.wav`, `D.wav`.
  Todos os WAVs têm 30 segundos, mono PCM16, 44.100 Hz. Usar os arquivos sem
  conversão, normalização, repetição ou alteração de duração.

## Tela por cenário

Mostrar imagem, heading **Contexto do jogo**, descrição e quatro players A/B/C/D.
Para cada player, perguntar exatamente:

**Considerando a imagem e a descrição do jogo, quanto esta música combina com esse contexto?**

1. Não combina nada
2. Combina pouco
3. Mais ou menos
4. Combina bem
5. Combina muito bem

Exigir as quatro notas antes de avançar e validar também no backend: inteiros
entre 1 e 5. Cada participante recebe **dois cenários**. Não embaralhar novamente
A/B/C/D nem trocar arquivos de posição; a atribuição já está balanceada na ordem
de coleta. Os rótulos exibidos devem ser apenas A, B, C e D.

## Reserva e conclusão no backend

Reservar atomicamente os próximos cenários disponíveis da lista. Uma reserva
ativa impede que outro participante receba o mesmo cenário. Reserva abandonada
pode expirar após timeout configurável; impedir que seu antigo dono conclua
usando um token vencido após realocação. Uma conclusão deve ser idempotente e
retirar definitivamente o cenário do pool, sem reutilizá-lo para outra pessoa.

Guardar participante/reserva, respostas demográficas, `external_id`,
`collection_order_v2` e `ratings` A/B/C/D. Exemplo de avaliação:

```json
{"external_id":"<id recebido>","collection_order_v2":0,"ratings":{"A":3,"B":1,"C":4,"D":2}}
```

Não é necessário qualquer componente privado para alocar os cenários ou guardar
as avaliações. A chave de análise das origens fica com a equipe de pesquisa,
fora do arquivo de transferência; não solicitar essa chave ao browser nem
incluí-la em endpoints estáticos ou payloads de coleta. A associação posterior
para análise será feita por external_id e slot.

O balanceamento é garantido nos prefixos da fila preparada. Abandonos e respostas
faltantes podem mudar o balanço da amostra efetivamente concluída; não reordenar
por notas, atributos dos arquivos ou preferências do participante para compensar.
