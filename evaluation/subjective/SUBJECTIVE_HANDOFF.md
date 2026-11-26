# subjective-48-handoff-v1

Construtor local do pacote para Giovanni, sem acesso ao homelab, alteração do app,
publicação ou envio. Entradas: subset `subjective-48-v1`, fontes e hashes já
congelados, permutações `abcd-balanced-rotation-v1`, imagens RGB6 e textos públicos.
Não refaz matching, seleção ou permutações; custos só entram na chave privada.

## Construção

```bash
.venv/bin/python -B -m evaluation.subjective.build_subjective_handoff \
  --output-dir /mnt/hd/Documentos/mestrado/IAGenerativa/TP2/subjective-48-handoff-v1
```

Exige HDD `/mnt/hd` montado e destinos inexistentes. Staging e compressão são no
mesmo HDD. A validação completa ocorre antes da publicação por rename; erros de
processamento normais não publicam o pacote. Diretório e tar são dois renames,
portanto não constituem uma única transação perante falha de energia. Não há
sobrescrita automática nem cleanup de diretórios preexistentes.

O código importa, sem modificar, `render_subjective_audio.render_clip`: 48 Real
com tabela SFX, 48 Markov e 48 Neural sem a tabela. Exatamente o protocolo
`subjective-audio-30s-v1`: início em zero, sequência simbólica completa repetida
se curta, uma chamada contínua ao synth, corte de 1.323.000 frames, PCM16/44.100 Hz.
Não há normalização adicional nem ganho por sistema.

Attention é produzido **depois** a partir do WAV Real final escrito. Reutiliza
`read_pcm16`, `derive_seed`, `mix_noise` e `measure_effective_snr` do gerador
congelado `attention-check-v1`, global_seed=831, SNR=0 dB. TID e song_id são strings
decimais, como no CSV original do gerador. A seed inclui o SHA256 dos bytes desse
WAV Real. O ruído PCG64/standard_normal tem RMS calibrado; a eventual atenuação
conjunta de sinal e ruído, headroom e quantização são os do protocolo existente.
Não é aplicada essa atenuação ao Real separado, Markov ou Neural. A SNR efetiva
após quantização deve ficar a no máximo 0,1 dB do alvo.

## Estrutura e cegagem

```
subjective-48-handoff-v1/
  HANDOFF_GIOVANNI.md
  public/
    scenarios.json
    scenarios/<external_id>/{context.png,A.wav,B.wav,C.wav,D.wav}
  private/
    scenario_key.json
    validation_summary.json
    audio/<external_id>/{real.wav,markov.wav,neural.wav,attention.wav}
subjective-48-handoff-v1.tar.zst
```

Os 192 áudios privados são os originais finais para auditoria. Cada WAV público é
uma cópia byte a byte da origem definida na permutação congelada. As imagens são
cópias exatas dos PNGs sanitizados do subset. O manifest público permite somente
`external_id`, `description`, `context_heading`, `context_image` e `audios`, com
chaves A/B/C/D. URLs são relativas à raiz de `public/`; não há hashes, custos,
índices de seleção, nomes de sistemas ou IDs de músicas no DTO.

A chave privada registra origens, identidades e hashes de registros/WAVs, custos,
context tier/serve_order, filtros, plano de repetição, seed/SNR/gain e parentesco
Attention→Real, protocolos, hashes de inputs/código e versões. O diretório
completo é local e **não deve ser servido**. Transferir somente
o `.tar.zst`, que contém explicitamente `public/` e `HANDOFF_GIOVANNI.md`.

O arquivo tar normaliza mtime/UID/GID para zero e omite nomes de proprietário,
PAX, links e paths privados. Zstd usa nível 9 e uma thread. O arquivo é lido
integralmente depois da compressão, sem extração, conferindo a allowlist e o hash
de cada membro. São 242 arquivos regulares (241 públicos mais as instruções).
O hash e tamanho ficam em `private/validation_summary.json`.

## Auditorias

- 48 IDs, três conjuntos de músicas únicos, exclusões anteriores preservadas;
- 24 permutações exatamente duas vezes; cada origem 12 vezes em cada slot;
- 48 PNGs com somente IHDR/IDAT/IEND, RGB e dois bits inferiores zerados;
- 192 WAVs públicos com cabeçalho canônico fmt/data, sem tags de origem, todos
  mono PCM16/44.100 Hz e 30,000 s, finitos e não nulos;
- allowlist estrita de campos, paths, diretórios e arquivos públicos; igualdade
  com textos/imagens/permutação congelados, sem tokens de origem/proveniência;
- igualdade byte a byte dos WAVs públicos com suas origens privadas;
- recalcular os 48 Attention a partir do Real correspondente e confrontar PCM,
  seed, hashes, ganho e SNR medida;
- conferência integral dos membros do tar, sem qualquer `private/`;
- nenhuma modificação dos artefatos canônicos ou do app.

O áudio pode conter a saturação nativa do synth; não há tentativa de "corrigir"
conteúdo congelado. Os diagnósticos privados registram clipping nativo e devem
confirmar zero clipping novo na conversão Real/Markov/Neural.

## Testes

```bash
PYTHONPATH=.:evaluation:evaluation/tests .venv/bin/python -B -m unittest \
  test_subjective_handoff test_subjective_audio test_subjective_attention_checks \
  test_subjective_subset test_scenario_manifest
```

Fixtures pequenas verificam cegagem, troca de slots, vazamentos por schema/path/
tags, vínculo do Attention, cobertura e determinismo do tar e recusa de sobrescrita.
Nenhum WAV ou output pesado é adicionado ao Git. Não há implementação de app,
reserva, coleta de notas ou regra nova de exclusão de participantes neste código.

## Pacote construído e conferido

Diretório local: `/mnt/hd/Documentos/mestrado/IAGenerativa/TP2/subjective-48-handoff-v1/`.
Arquivo de transferência: mesmo caminho com extensão `.tar.zst`.

- 48 cenários, 48 PNGs e 192 WAVs públicos, todos mono PCM16/44.100 Hz/30,000 s;
- 48 arquivos de cada slot, cada origem exatamente 12 vezes em cada slot;
- 48 origens únicas em cada conjunto Real/Markov/Neural; 48 Attention conferidos;
- cópias públicas byte a byte; os nove áudios do smoke anterior conservam os hashes;
- 10 Reais com filtros efetivamente não nulos; nenhum filtro real anexado aos gerados;
- erro máximo da SNR efetiva de Attention: 0,0000036891391287 dB;
- 7.148 samples nos limites da saturação nativa, zero clipping novo na conversão;
- arquivo tar: 242 arquivos regulares, 292 membros contando diretórios, nenhum privado;
- `zstd -t` e listagem independente via GNU tar aprovados; sem staging remanescente;
- 110 testes relevantes aprovados; 107 arquivos preexistentes preservados por SHA256.

Tamanho do arquivo de transferência: **361.563.934 bytes** (344,814 MiB).

SHA256:

```
subjective-48-handoff-v1.tar.zst
aa44bf7d2e986e99e9b2529ca2e7ae4ad7fa2d7546f6968dd0267057169217a8

public/scenarios.json
8d9228a156d376d50295088e24335fa4c60dee3b503262170c3bcbd17144a2cd

private/scenario_key.json (somente local, fora do tar)
a707d456baa86885273350435f5fe2a785feca0ead64acad1e802a8a01d8ce2e
```
