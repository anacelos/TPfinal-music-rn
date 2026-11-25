# Handoff da avaliação subjetiva — 48 cenários

O arquivo de transferência contém este documento e `public/`. Não requer código
de síntese, datasets ou chave de análise. Nenhuma integração no app foi executada.

Disponibilize o conteúdo de `public/` como assets estáticos do frontend.
`scenarios.json` é uma lista com `external_id`, `description`, `context_heading`,
`context_image` e `audios` (A/B/C/D). Todos os caminhos são relativos à raiz desses
assets; acrescente o prefixo de hospedagem usado pelo app. São 48 imagens e 192
WAVs mono PCM16, 44.100 Hz, exatamente 30 segundos cada.

## Interface por cenário

Mostrar nesta ordem:

1. Imagem indicada por `context_image`.
2. Título contextual `context_heading` e descrição `description`, como texto.
3. Quatro players identificados apenas como A, B, C e D, nessa ordem.
4. Para **cada áudio**, a pergunta:

**“Essa música combina com o contexto descrito acima?”**

- 1 — Não combina nada
- 2 — Combina pouco
- 3 — Mais ou menos
- 4 — Combina bem
- 5 — Combina muito bem

O participante deve responder os quatro antes de concluir o cenário. Validar
também no backend que as quatro notas sejam inteiros entre 1 e 5. Não alterar a
ordem A/B/C/D nem sortear novas permutações. Não revelar a origem dos áudios.

## Alocação e registro

- Cada cenário pode ser concluído por **no máximo um participante**.
- Reservar o cenário atomicamente no backend: uma reserva ativa impede que outro
  participante receba o mesmo cenário. O frontend sozinho não garante isso.
- Reserva abandonada pode voltar ao pool após timeout configurável. Uma reserva
  expirada não pode concluir um cenário posteriormente reservado por outra pessoa.
- Após conclusão, o cenário sai definitivamente do pool. Reenvios da mesma
  conclusão devem ser idempotentes, sem criar uma segunda resposta.
- Cada participante deve receber aproximadamente **3–4 cenários**, conforme a
  disponibilidade. Usar os cenários ainda disponíveis da lista; não reordenar por
  características das imagens ou dos áudios.
- Armazenar `external_id` e notas A/B/C/D, associadas ao participante/reserva no
  mecanismo já usado pelo backend. Exemplo de conteúdo das respostas:

```json
{
  "external_id": "<identificador recebido no manifest>",
  "ratings": {"A": 1, "B": 3, "C": 4, "D": 2}
}
```

A coleta não precisa conhecer as origens dos slots. A associação posterior será
feita com `private/scenario_key.json`, mantido localmente pela equipe de análise e
**ausente deste arquivo de transferência**. Não solicitar nem disponibilizar essa
chave no frontend, em endpoints ou em diretórios estáticos.

## Conferência ao receber

Extrair o `.tar.zst` preservando a estrutura. Confirmar 48 subpastas em
`public/scenarios/`, cada uma com `context.png` e `A.wav`, `B.wav`, `C.wav`, `D.wav`.
O `scenarios.json` referencia somente esses arquivos. Não é necessário modificar,
renomear, converter ou normalizar os assets para integrar os players existentes.
