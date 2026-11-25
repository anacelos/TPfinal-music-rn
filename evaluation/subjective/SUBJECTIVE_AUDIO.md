# subjective-audio-30s-v1

Protocolo de áudio da avaliação subjetiva derivado de `subjective-48-v1`, sem
alterar seleção, associações Real↔Markov/Neural, permutações, manifest canônico,
avaliação objetiva ou conteúdo musical. Neural: `lstm-grid_block-k10`.

## Auditoria do synth congelado

`src/synth.py` implementa o relógio de 183 frames/tick a 22.050 Hz, interpolado
linearmente para 366 frames/tick a 44.100 Hz. A duração original renderizada é
`sum(max(0, int(pattern_ticks))) * 366 / 44100`; o default nativo de ticks
ausentes é 512. Os nove inputs do smoke devem coincidir com as durações simbólicas
congeladas; uma discrepância interrompe o pipeline para revisão.

`_song_patterns(song, loop_extra_repeats)` só repete quando `end_reason` é
`loop_end` e `loop_start_pattern` não é nulo. Nesse caso acrescenta, no domínio
simbólico, apenas os patterns cujo `pattern_id >= loop_start_pattern`. Não repete
músicas terminadas por stop/EOS, nem necessariamente repete a introdução.
Portanto, usar esse parâmetro diretamente NÃO implementa o protocolo de repetir
a música inteira desde t=0.

O pipeline faz uma expansão simbólica temporária da lista **completa e ordenada**
de patterns. Para `F` frames originais e `T = 1.323.000` frames alvo:

```
full_song_play_count = max(1, (T + F - 1) // F)
extra_repeat_count = full_song_play_count - 1
expanded_patterns = original_patterns * full_song_play_count
```

Uma única chamada a `synthesize_song(expanded_song, loop_extra_repeats=0,
seed=0, rate=44100)` renderiza essa sequência. F=0 é erro; não há padding.
A expansão é só em memória: registros canônicos e seus hashes permanecem intactos.
`loop_start_pattern` não controla a repetição da música neste protocolo. Loops
internos dos SFX (`_sfx_indices`), notas e efeitos mantêm a interpretação nativa.

Os quatro objetos `_Channel` são criados uma vez por chamada; estado de fase,
ruído/PRNG e transições atravessa os patterns e as fronteiras entre ciclos.
Reverb com realimentação e dampen são aplicados ao buffer contínuo em `finish()`.
O synth já aplica seu crossfade de 1/130 s nas mudanças de estado. Isso é mais
fiel à continuidade do motor do que reiniciar cada música e concatenar WAVs,
que reiniciaria oscilador, ruído, filtros e transições em cada ciclo. Não se
promete identidade PCM entre ciclos. O histórico local de slide (`prev_key`,
`prev_vol`) reinicia em cada disparo de SFX, como no código existente.

Nenhum crossfade, fade inicial/final, normalização, ganho, dither, resampling
adicional ou processamento de qualidade foi acrescentado. `src/synth.py` e
`evaluation/synthesize_jsonl.py` permanecem intactos. O segundo é um wrapper
JSONL→WAV, sem política de 30 s, que opcionalmente anexa a tabela de filtros.

## Corte, filtros e PCM

- Começar sempre no frame zero; reter exatamente os primeiros 1.323.000 frames.
- Música original ≥30 s: uma execução completa, sem repetição; cortar o prefixo.
- Música original <30 s: repetir ciclos completos até cobrir 30 s; cortar o
  excedente. Se a duração coincidisse exatamente com o alvo, não haveria ciclo
  extra. A decisão é inteira em frames, sem erro de arredondamento em segundos.
- Real: `src.sfx_filters.attach` com a tabela congelada `src/sfx_filters.json`.
  Não ter entrada não nula para um SFX significa modo nativo zero, não falha da
  política. O manifest registra se a tabela foi aplicada e quantas entradas
  efetivamente têm filtros.
- Markov/Neural: nunca anexar essa tabela, mesmo com `tid` herdado de um cart
  real. Rejeitar filtros não nulos inesperados nos inputs gerados.
- Quantização exatamente como `synthesize_song_to_wav`:
  `np.clip(np.round(audio * 32767.0), -32767, 32767).astype('<i2')`, após o corte.
  Saída RIFF/WAV mono, PCM16 little endian, 44.100 Hz, 2 bytes/frame, 30,000 s.

O motor já limita cada canal a ±0,99 e satura a mistura de quatro canais. Samples
nos limites de PCM podem ser consequência desse comportamento canônico; não
motivam ganho adaptativo. A auditoria distingue samples nos limites nativos,
samples PCM em ±32767 e qualquer clipping **introduzido** na conversão final
(este deve ser zero). Verifica também samples finitos, frames e áudio não nulo.

As fronteiras entre ciclos recebem somente o tratamento nativo. O manifest
registra sua quantidade, maior salto PCM adjacente nessas fronteiras e maior
salto global. O corte em 30 s não recebe fade; o valor final também é registrado.
Essas medidas são diagnóstico, não critérios para substituir cenários ou sons.
Saltos adjacentes não provam ausência perceptiva de clicks; não foi feita seleção
auditiva nem correção para agradabilidade.

## Smoke autorizado e execução

Somente três cenários, com Real, Markov e Neural, sem Attention:

1. índice 42: stress de ~0,265578 s; 113 execuções totais (112 repetições),
   1.323.456 frames antes do corte, 30,010340 s. Preservar o conteúdo.
2. índice 1: as três músicas ≥30 s; nenhuma repetição.
3. menor índice congelado com pelo menos uma música <30 s e outra ≥30 s: 3.

O CLI só oferece esse smoke, não a renderização dos 48. Usar diretório novo no
HDD montado; recusar destino existente e paths fora de `/mnt/hd`:

```bash
.venv/bin/python -B -m evaluation.subjective.render_subjective_audio --smoke \
  --output-dir /mnt/hd/Documentos/mestrado/IAGenerativa/TP2/subjective-audio-30s-v1-smoke
```

Uma segunda execução completa dos mesmos nove renders ocorre automaticamente em
diretório temporário no mesmo HDD. Comparam-se bytes de todos os WAVs, manifest e
config; o temporário é removido e ficam exatamente nove WAVs. Renderização usa
staging no HDD e publicação por rename; falha antes da publicação não deixa
output parcial. Uma falha na segunda passagem deixa a primeira disponível para
auditoria, sem comprovação de determinismo. Nunca se sobrescreve o destino.

Arquivos no HDD:

- `scenario_{42,01,03}_{real,markov,neural}.wav`: nove arquivos finais;
- `manifest.jsonl`: nove registros com identidades internas, linha JSONL, hashes,
  durações originais, ciclos/repetições, frames antes/depois do corte, formato,
  filtros e diagnósticos;
- `render_config.json`: protocolo, hashes dos inputs/implementação, versões de
  Python/NumPy/SciPy, regra de repetição, identidade Neural e checksums dos outputs;
- `verification.json`: comprovação das duas sínteses independentes com bytes
  idênticos e hashes completos. Não contém timestamps ou caminhos temporários.

Os hashes do synth, shim/tabela SFX, subset e seu config estão fixados no código.
Os arquivos de origem e os hashes canônicos de cada música são confrontados com
o subset congelado antes de renderizar. Não recalcular matching/seleção. Alterar
um desses contratos requer revisão explícita, em vez de continuar silenciosamente
com o mesmo nome de protocolo. Determinismo foi definido para os mesmos inputs,
código e versões de dependências; não é garantia entre plataformas/versões.

Estes manifests expõem as origens cegas e são **internos**. WAVs e outputs no HDD
não são versionados no Git. Nenhum artefato canônico preexistente é reescrito.

## Attention posterior

Não gerar Attention nesta etapa. Sua entrada posterior será o PCM dos **bytes
exatos do WAV Real final de 30 s**, verificado pelo `wav_sha256` deste manifest,
com o protocolo congelado `attention-check-v1` (Gaussian white noise, 0 dB SNR).
Não sintetizar outro Real, repetir depois de adicionar ruído ou mudar filtros.

## Testes leves

```bash
PYTHONPATH=.:evaluation:evaluation/tests .venv/bin/python -B -m unittest \
  test_subjective_audio test_subjective_subset test_synth_alignment
```

Os testes de áudio usam fixtures em memória: relógio/repetição inteira, diferença
entre loop nativo e ciclo completo, continuidade versus tiling de waveform,
prefixo idêntico ao writer canônico, filtros por origem, imutabilidade,
determinismo, rejeição de samples inválidos, formato, destino novo e publicação
atômica. Não escrevem WAVs no SSD nem sintetizam os 48 cenários.

## Resultado congelado do smoke

Todos os nove WAVs têm 1.323.000 frames, 30,000 s, mono PCM16/44.100 Hz,
2.646.044 bytes, conteúdo não nulo e samples finitos. Duas sínteses independentes
produziram bytes idênticos para os nove WAVs, manifest e config. A segunda foi
removida; o HDD contém os nove WAVs e os três arquivos de proveniência.

| Índice | Origem | Original (s) | Repetições extras | Antes do corte (s) | SHA256 WAV (prefixo) |
| --- | --- | ---: | ---: | ---: | --- |
| 42 | Real | 0,265578 | 112 | 30,010340 | `fad42a9091a4` |
| 42 | Markov | 0,265578 | 112 | 30,010340 | `43f2e5789ca1` |
| 42 | Neural | 0,265578 | 112 | 30,010340 | `a3a2c5ad6ec0` |
| 1 | Real | 95,608163 | 0 | 95,608163 | `c30c1c58830c` |
| 1 | Markov | 31,869388 | 0 | 31,869388 | `206061129acf` |
| 1 | Neural | 88,238367 | 0 | 88,238367 | `714528405cd3` |
| 3 | Real | 7,967347 | 3 | 31,869388 | `c49b645ccbd9` |
| 3 | Markov | 21,246259 | 1 | 42,492517 | `d55ae7c459d6` |
| 3 | Neural | 145,138503 | 0 | 145,138503 | `ba72c19fecb8` |

Identidades Real: índice 42 `real:32614:4`; índice 1 `real:45995:0`;
índice 3 `real:148341:0`. IDs e hashes completos dos nove registros de origem e
WAVs constam no manifest. A tabela SFX foi aplicada somente aos Reais; os três
Reais deste smoke não possuem entradas de filtros não nulos na tabela. Testes
com uma entrada não nula verificam separadamente a aplicação do filtro.

Clipping acrescentado na conversão: zero em todos os arquivos. O Real do índice 3
tem 1.901 samples nos limites da saturação canônica da mistura (também 1.901 em
±32767 PCM); os demais têm zero. Não houve normalização para eliminar a saturação
do motor. Um teste com mistura saturada verifica a igualdade com a saída nativa.

Maiores saltos adjacentes nas fronteiras entre ciclos (unidades PCM): índice 42,
Real=59, Markov=1, Neural=2; índice 3, Real=0 e Markov=35. Todos são inferiores
aos maiores saltos internos dos respectivos arquivos. Não há fronteiras de
repetição nos demais. O corte final permanece sem fade conforme o protocolo;
esses diagnósticos não substituem avaliação perceptiva e não foram usados para
trocar conteúdo.

Hashes SHA256 dos arquivos internos no HDD:

```
manifest.jsonl     33aadb597ffb83430faa38da951409848699abc0e26c9475145fe153d5705461
render_config.json 58fd37a28f317987b6060da3491e3b8b0b3d559517ba9b0a711661c5acd6d0cf
verification.json  db1884dd50889c068e1358deff9d897a61a0866a6f96e62c7d48047bf8c9744b
```

Nenhum Attention foi gerado. A renderização completa dos 48 continua pendente de
uma etapa explicitamente autorizada.
