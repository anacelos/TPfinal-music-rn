# Pacote definitivo subjective-240-handoff-v1

Entradas: os 240 registros de `subjective-240-v1-context-final`, sem nova seleção,
matching ou revisão de contexto. Mantém external_ids, identidades, pares, imagens,
descrições, heading, pergunta, escala e `collection_order_v2`. Saída pesada no HDD:

```
/mnt/hd/Documentos/mestrado/IAGenerativa/TP2/subjective-240-handoff-v1/
  HANDOFF_GIOVANNI.md
  public/
    scenarios.json
    study.json
    scenarios/<external_id>/{context.png,A.wav,B.wav,C.wav,D.wav}
  private/
    scenario_key.json
    validation_summary.json
    attention-preview.mp3
    approved-spoken.wav
    audio/<external_id>/{real.wav,markov.wav,neural.wav}
```

Transferir somente o arquivo vizinho `subjective-240-handoff-v1.tar.zst`, contendo
`public/` e `HANDOFF_GIOVANNI.md`. Não servir o diretório completo. O backend usa
os IDs públicos e a ordem para reserva/coleta; dispensa a chave privada, mantida
localmente para associar posteriormente notas, condições e checagem de atenção.

## Áudio musical e reaproveitamento

Usa exatamente `render_subjective_audio.render_clip`, sem alteração do synth,
seed, filtros ou formato: `subjective-audio-30s-v1`, t=0, repetição simbólica
integral quando necessária, corte de 1.323.000 frames, mono PCM16 a 44.100 Hz.
Real recebe a tabela SFX congelada; Markov/Neural não. Neural é
`lstm-grid_block-k10`. Os inputs são confrontados com os hashes congelados.

Nos 48 IDs presentes no handoff anterior, reaproveita os 144 WAVs musicais após
verificar identidade do registro, hash do WAV/synth/tabela, protocolo, seed e
política de filtros. Renderiza os 576 restantes. Não reutiliza Attention Gaussian.
As cópias públicas são byte a byte das origens privadas, com nomes A/B/C/D.
Os novos renders têm cache privado no HDD em `.subjective-240-musical-cache-v1`,
indexado por origem e hash do registro. Reuso exige checksum, formato, synth,
protocolo, seed e filtros compatíveis. O cache não entra no tar. Os metadados
públicos são checados antes de renderizar; o adjetivo português "real" (como em
"múmia real") não é confundido com identificadores de origem ou mapping.

## Fala aprovada: attention-check-spoken-v2

Única origem autorizada: `/tmp/attention-preview.mp3`, fornecida e aprovada
auditivamente pelo usuário. Ferramenta/voz informadas: edge-tts,
pt-BR-FranciscaNeural. O arquivo original é preservado em `private/`.

SHA256 do MP3 aprovado:
`b790b6f21d476eb9e3305bc6033912ebab55ad5f44632c54bcbc2e428ffb1a6a`.

Texto informado pelo usuário:

> Este áudio é uma checagem de atenção. Para confirmar que você está prestando atenção à avaliação, selecione a opção Não combina nada para esta faixa.

Não executa TTS ou eSpeak e não acrescenta ruído. O ambiente não tinha o binário
ffmpeg; usa o decoder MP3 nativo disponível, libsndfile 1.2.2. O arquivo tem
235.584 frames a 24.000 Hz, mono: 9,816 s. Converte para 44.100 Hz usando
`scipy.signal.resample_poly`, janela Kaiser(5.0), pad constante e razão 147/80.
Usa a quantização PCM do projeto e acrescenta somente zeros até 30 s. Sem ganho,
normalização, fade, corte ou repetição da fala. Todos os 240 cenários usam cópias
desse mesmo WAV aprovado convertido. Proveniência distingue informações fornecidas
pelo usuário de propriedades medidas, registrando versões, frames e checksums.

## Nova atribuição de posições

`attention-middle-balanced-v1` substitui **somente no pacote definitivo** a antiga
atribuição de 24 permutações. O manifest privado conserva a apresentação histórica
e registra o novo mapping efetivo. Nunca modifica os artefatos anteriores.

Na ordem `collection_order_v2`, repete um ciclo determinístico de 12 atribuições,
20 vezes. Para cada orientação d=+1,-1 e início s=0,1,2:

- Attention em B; as condições musicais ocupam A/C/D em rotação s+i*d;
- Attention em C; as condições musicais ocupam A/B/D em rotação s+1+i*d.

As operações usam índices módulo 3 na lista Real/Markov/Neural. O resultado
contém exatamente as 12 permutações permitidas. Não usa custos, resultados,
duração ou qualidade. Não muda identidade ou associação entre faixas.

Cada par consecutivo tem um Attention em B e outro em C. A cada seis cenários,
cada condição musical ocupa A/D duas vezes e B/C uma vez. A cada bloco de doze,
todas as permutações aparecem uma vez. Globalmente: Attention B=120, C=120,
A=D=0; cada condição musical A=D=80, B=C=40. O balanço desigual das condições
musicais por slot decorre da restrição aprovada de Attention em B/C.

Não incluir o algoritmo, mapping ou origem dos slots nos payloads do browser.
A cegagem aqui é a ausência de mapping/proveniência nos arquivos e metadados
públicos; a fala se identifica intencionalmente como checagem, conforme aprovado.

## Construção e validação

```bash
.venv/bin/python -B -m evaluation.subjective.build_final_handoff_240
PYTHONPATH=.:evaluation:evaluation/tests .venv/bin/python -B -m unittest \
  test_final_handoff_240 test_context_description_final test_subjective_audio \
  test_subjective_handoff
```

Exige destinos novos e HDD montado. Usa staging no HDD; só publica após validar
todos os arquivos e comprimir/extrair o tar. Diretório e arquivo são dois renames,
não uma transação única contra falha de energia. Nunca sobrescreve os históricos.

Valida 240 IDs e pares únicos, 960 WAVs públicos, cópias byte a byte, todos com
30,000 s/mono/PCM16/44.100 Hz, sem tags adicionais; PNGs sanitizados e iguais aos
congelados; textos e ordem exatos; schema e árvore pública por allowlist; hashes
do MP3/PCM; silêncio digital após a fala; 12×20 e balanço em prefixos de 2, 4, 6,
12, 24, 48, 120 e 240. A conversão da fala é reproduzida e comparada byte a byte.

Tar normalizado, Zstd nível 9 com uma thread; permite só membros públicos e
instruções, sem links/PAX/paths privados. Faz extração efetiva em temporário no
HDD e compara o hash de cada arquivo extraído. São 1.203 arquivos regulares:
1.200 assets, dois manifests públicos e as instruções. Temporários são removidos.

Integração/reservas, coleta das seis informações iniciais e notas cabem ao Giovanni;
o contrato público está no documento incluído. Não modifica frontend/backend,
paper, experimentos objetivos ou arquivos de áudio históricos. Não faz push.

## Resultado definitivo

47 testes aprovados. Pacote extraído integralmente no HDD e todos os hashes
extraídos conferidos: 1.203 arquivos regulares, 1.445 membros incluindo diretórios,
nenhum arquivo privado. São 240 cenários/imagens e 960 WAVs públicos válidos, com
144 músicas reaproveitadas e 576 novas em relação ao histórico de 48. As 12
permutações aparecem 20 vezes e o balanço em prefixos foi validado.

Arquivo de transferência: **1.415.067.822 bytes**.

SHA256 `subjective-240-handoff-v1.tar.zst`:
`670ed220ef9b5160997e5506b6a161a3752bb64da8f907f337feea4c81d7f2b3`.

SHA256 do WAV falado convertido:
`71837b960c8da3e72a38638e57ce958bd028c647422af2b693855bba4a860091`.

Fala convertida: 432.886 frames; silêncio acrescentado: 890.114 frames. Total:
1.323.000 frames. O tar histórico de 48 mantém o SHA256
`aa44bf7d2e986e99e9b2529ca2e7ae4ad7fa2d7546f6968dd0267057169217a8`.

A primeira tentativa foi interrompida antes da publicação por um falso positivo
lexical em "múmia real". O texto congelado foi preservado, o filtro foi corrigido,
um teste de regressão foi acrescentado e o pacote final passou em todas as
validações. Não há saída parcial publicada dessa tentativa.
