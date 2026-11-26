# Dados da avaliação subjetiva (teste de escuta)

Este documento explica o que há nos arquivos coletados pelo teste de escuta, coluna por coluna,
para quem for fazer a análise. Ele não traz resultados: a análise depende da chave que diz qual
faixa veio de qual sistema, e essa chave fica com a equipe de pesquisa, fora do app.

- App: <https://pico8-bh.giow.dev> (código em <https://github.com/GiowGiow/pico8-listening-test>)
- Dados: pasta do Drive do projeto, em `avaliacao-subjetiva/snapshot-<data>-<hora>/`
- Pacote de estímulos: `subjective-240-handoff-v1` (240 cenários, 960 áudios)

## 1. Como os dados foram gerados

1. A pessoa entra com um e-mail. O e-mail não é guardado: o servidor guarda só um código
   derivado dele (HMAC), que é o identificador anônimo.
2. Ela responde seis perguntas de múltipla escolha (perfil). Quem marca "Menos de 18 anos" não
   participa e não tem nada guardado.
3. Ela recebe 2 cenários. Cada cenário mostra a imagem e a descrição de um jogo de PICO-8 e
   quatro players, A, B, C e D, com 30 s de música cada.
4. Para cada player ela responde: "Considerando a imagem e a descrição do jogo, quanto esta
   música combina com esse contexto?", em cinco opções.

Regras que valem para todos os dados:

- Os cenários saem de uma fila única, na ordem `collection_order_v2` do pacote.
- Cada cenário é avaliado por uma pessoa só e não volta para a fila depois de avaliado.
- A, B, C e D são sempre os arquivos `A.wav` a `D.wav` do pacote. O app não embaralha nada e não
  sabe a origem de nenhuma faixa.
- Os botões de nota de uma faixa só liberam depois de 10 s ouvidos daquela faixa.

## 2. Arquivos de um snapshot

| Arquivo | O que é |
|---|---|
| `ratings.csv` | Uma linha por cenário avaliado. É o arquivo para a análise. |
| `participants.csv` | Uma linha por pessoa que entrou no app, com o questionário, inclusive quem não avaliou nada. |
| `reservations.csv` | Uma linha por entrega de cenário a uma pessoa: concluída, em andamento ou perdida. |
| `scenarios.csv` | Os 240 cenários, a posição na fila e quais já foram concluídos. |
| `study_texts_and_questionnaire.json` | A pergunta, as cinco opções da escala e o rótulo de cada código do questionário. |

Os identificadores de participante e de reserva nesses arquivos são códigos curtos de 12
caracteres, derivados dos identificadores internos do app. O mesmo participante tem o mesmo
código em todos os arquivos e em todos os snapshots, mas o código não serve para entrar no app
no lugar dele.

Um snapshot é uma foto do momento em que foi baixado. Enquanto a coleta estiver aberta, um
snapshot mais novo contém todas as linhas do anterior e mais as novas.

## 3. Colunas de `ratings.csv`

Cada linha é um cenário avaliado por um participante (quatro notas).

| Coluna | Significado |
|---|---|
| `participant_id` | Código anônimo do participante (12 caracteres). Aparece em até 2 linhas. |
| `presented_order` | 1 ou 2: se foi o primeiro ou o segundo cenário daquela pessoa. |
| `external_id` | Identificador do cenário no pacote. É a chave para cruzar com a tabela das origens. |
| `collection_order_v2` | Posição do cenário na fila do pacote (0 a 239). |
| `rating_A` a `rating_D` | Nota de cada player, inteiro de 1 a 5 (escala abaixo). |
| `time_spent_ms` | Tempo com o cenário na tela, em milissegundos, do carregamento ao envio. |
| `listened_A_s` a `listened_D_s` | Segundos de áudio efetivamente tocados de cada faixa. Passa de 30 se a pessoa repetiu. |
| `reserved_at` | Quando o cenário foi entregue à pessoa (UTC). |
| `completed_at` | Quando as quatro notas foram enviadas (UTC). |
| `age_range` a `headphones` | As seis respostas do questionário, como códigos (seção 4). Repetem-se nas linhas do mesmo participante. |
| `questionnaire_version` | Versão das categorias do questionário (`2026-11-25`). |
| `ui_locale` | Idioma da interface (`pt-BR` ou `en`). A pergunta, a escala e as descrições aparecem sempre em português. |
| `participant_created_at` | Quando a pessoa entrou no app (UTC). |

Escala das notas:

| Valor | Rótulo mostrado |
|---|---|
| 1 | Não combina nada |
| 2 | Combina pouco |
| 3 | Mais ou menos |
| 4 | Combina bem |
| 5 | Combina muito bem |

Os horários estão em UTC; o horário de Brasília é UTC menos 3 horas.

## 4. Códigos do questionário

| Coluna | Pergunta | Códigos |
|---|---|---|
| `age_range` | Qual é a sua faixa etária? | `18_24`, `25_34`, `35_44`, `45_54`, `55_plus` (`under_18` nunca aparece: menores não participam) |
| `musical_experience` | Qual é a sua experiência musical? | `none` (nenhuma), `hobby` (toca ou canta por hobby), `formal` (estudou ou estuda música), `professional` (trabalha com música) |
| `gaming_frequency` | Com que frequência você joga videogame? | `never` (nunca ou quase nunca), `monthly` (algumas vezes por mês), `weekly` (algumas vezes por semana), `daily` (todos os dias ou quase) |
| `retro_familiarity` | Quanto você conhece de jogos retrô ou de 8 bits? | `none` (nada), `little` (pouco), `some` (razoavelmente), `much` (muito) |
| `region` | Em que região você mora? | `norte`, `nordeste`, `centro_oeste`, `sudeste`, `sul`, `exterior` |
| `headphones` | Você está usando fones de ouvido agora? | `yes`, `no` |

## 5. O que os dados não dizem

- **Qual faixa é de qual sistema.** O CSV só tem o cenário (`external_id`) e a letra do player.
  A origem de cada faixa sai do cruzamento `external_id` + letra com a chave da equipe de
  pesquisa. Sem a chave, a média de uma letra não é a média de um sistema.
- **Quais cenários contêm a checagem de atenção.** Também está na chave. Regra combinada: sai da
  análise só o cenário reprovado na checagem, não o participante inteiro.
- **Quem é o participante.** Não há e-mail, nome nem endereço de rede nos arquivos.

## 6. Cuidados na análise

- A unidade é o cenário: as quatro notas de uma linha são da mesma pessoa, sobre o mesmo jogo.
  Comparações entre sistemas devem ser pareadas dentro da linha.
- Cada pessoa contribui com até 2 linhas; as linhas de um mesmo `participant_id` não são
  independentes.
- Quem parou depois do primeiro cenário tem só uma linha, e ela é válida.
- A ordem dos players na tela é sempre A, B, C, D. Efeito de posição só pode ser separado do
  efeito de sistema se a chave distribuir os sistemas entre as letras.
- `listened_*_s` permite filtrar por atenção: o mínimo possível é 10 s por faixa.
- Buracos em `collection_order_v2` são cenários reservados por alguém que não concluiu. Eles
  voltam para a fila depois de 45 minutos sem atividade.

## 7. Os outros três CSVs (para auditoria)

- `participants.csv`: `id` (o mesmo código de `participant_id`), `created_at`, `ui_locale`, as
  seis respostas do questionário, `questionnaire_version`, `profile_completed_at` (vazio se a
  pessoa não respondeu o questionário) e `finished_at` (preenchido quando ela concluiu os dois
  cenários).
- `scenarios.csv`: `external_id`, `position` (o `collection_order_v2`), `holder_id` (a reserva
  que está com o cenário) e `completed_at`.
- `reservations.csv`: `id`, `participant_id`, `scenario_id` (o `external_id`), `state`,
  `reserved_at`, `expires_at`, `completed_at`, as quatro notas, `time_spent_ms` e os segundos
  ouvidos. O estado é `active` (em andamento), `completed` (avaliado) ou `lost` (expirou e o
  cenário passou para outra pessoa).

O `ratings.csv` é exatamente a lista das reservas `completed`, com o questionário do participante
ao lado. No snapshot de 20:16 havia 83 cadastros, 160 reservas (139 concluídas, 11 em andamento
e 10 perdidas).

## 8. Snapshot de 25/11/2026, 20:16 (horário de Brasília)

Coleta ainda aberta nesse momento.

| Item | Valor |
|---|---|
| Cenários avaliados (linhas) | 139 de 240 |
| Notas | 556 |
| Participantes com avaliação | 71 (68 com dois cenários, 3 com um) |
| Primeira e última avaliação | 18:47 e 20:15 |
| Tempo mediano por cenário | 100 s |
| Escuta mediana por faixa | 15,3 s (mínimo 10,0 s) |

Perfil dos 71 participantes:

| Pergunta | Distribuição |
|---|---|
| Faixa etária | 25 a 34: 41; 18 a 24: 25; 35 a 44: 2; 45 a 54: 2; 55 ou mais: 1 |
| Experiência musical | nenhuma: 34; hobby: 26; estudo formal: 10; trabalha com música: 1 |
| Frequência de jogo | mensal: 24; nunca: 20; diária: 19; semanal: 8 |
| Jogos retrô | pouco: 30; razoavelmente: 18; nada: 15; muito: 8 |
| Região | Sudeste: 50; Nordeste: 11; Sul: 8; Centro-Oeste: 2 |
| Fones de ouvido | não: 50; sim: 21 |

As notas por player não estão resumidas aqui de propósito: sem a chave elas não dizem nada sobre
os sistemas, e a coleta ainda estava aberta.
