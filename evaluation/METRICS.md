# PICO-8 Evaluation Protocol

## Status

This document describes the current evaluation protocol for the PICO-8
autoregressive music generation project.

The protocol is provisional with respect to song segmentation and temporal
reconstruction, because the final tokenization and full-song synthesis rules
are still being defined by the data/model pipeline.

The structural metrics below are already implemented and covered by automated
tests.

---

## Evaluation principle

The goal is to compare three systems using the same representation and
evaluation pipeline:

1. Real music from the held-out test set
2. Baseline generations
3. Neural autoregressive model generations

Dataset splitting must be performed by TID (game ID), not by individual
patterns or songs, because different songs from the same game may reuse SFX.
This prevents leakage of musical material between train, validation and test.

Evaluation itself is performed at the piece/song level once the final
segmentation rule is defined.

---

## PICO-8 representation used by the evaluator

Each `audio.json` contains:

- `sfx`
- `music`

Each SFX contains:

- `id`
- `speed`
- `loop_start`
- `loop_end`
- 32 note slots

Each note contains:

- `pitch`
- `instrument`
- `volume`
- `effect`

Each music pattern contains:

- `id`
- `flags`
- `ch1`
- `ch2`
- `ch3`
- `ch4`

### Current parsing rules

For each music pattern:

- Bit 6 of a channel value indicates that the channel is disabled.
- Bits 0..5 identify the referenced SFX.
- A note slot is considered audible when `volume > 0`.
- A referenced but non-serialized SFX is treated as an implicit empty SFX.
- An explicitly serialized SFX containing no audible notes is treated as an
  explicit empty SFX.
- Only SFX referenced by the evaluated piece are included in the metrics.
- If an SFX is referenced multiple times, each occurrence is counted because
  the musical material is played multiple times.
- Music flags are currently interpreted semantically as `flags & 0x07`.
- Raw flags are not currently used as a metric.

---

## Primary structural metrics

### 1. Pitch Range

Difference between the highest and lowest audible pitch in the evaluated
piece:

`pitch_range = max(pitch) - min(pitch)`

Only note slots with `volume > 0` are considered.

This measures the register span used by the piece.

---

### 2. Normalized Pitch-Class Entropy

Each audible pitch is converted to a pitch class:

`pitch_class = pitch % 12`

Shannon entropy is calculated over the resulting pitch-class distribution and
normalized by the maximum entropy for 12 pitch classes:

`H_norm = H / log2(12)`

The resulting value is approximately bounded between 0 and 1.

Lower values indicate concentration on fewer pitch classes.
Higher values indicate greater pitch-class diversity.

---

### 3. Structural Note Density

Structural note density measures the proportion of active note slots among all
available note slots in enabled SFX references:

`density = active_note_slots / available_note_slots`

An active slot is a note with `volume > 0`.

A referenced implicit or explicit empty SFX contributes available slots but no
active slots. This prevents empty generated material from obtaining an
artificially high density score.

This is a structural PICO-8 metric and is not currently interpreted as notes
per second.

---

### 4. Audible Channel Utilization

PICO-8 music provides four possible channels per pattern.

Audible channel utilization is:

`audible_channel_utilization = audible_channel_refs / (4 * pattern_count)`

A channel is considered audible only when:

1. the channel is enabled;
2. its referenced SFX exists; and
3. the referenced SFX contains at least one note with `volume > 0`.

This differs from `enabled_channel_utilization`, which counts enabled channel
references even when they point to empty SFX.

---

### 5. SFX Repetition Ratio

For all enabled SFX references in a piece:

`sfx_repetition_ratio = repeated_references / total_references`

where repeated references correspond to occurrences beyond the first use of
each SFX ID.

Equivalently:

`repeated_references = total_references - number_of_unique_referenced_sfx`

This measures reuse of PICO-8 musical building blocks.

Its final status as a primary metric is still provisional.

---

## Additional distributions collected

The evaluator also records:

- instrument distribution
- effect distribution
- SFX speed distribution
- semantic music flags
- enabled channel utilization
- implicit empty SFX references
- explicit empty SFX references

Instrument and effect distributions may be used as additional domain-specific
evaluation measures.

---

## Distribution-level comparison

Metrics are computed per piece.

The final comparison will use distributions from:

- real test pieces
- baseline generations
- neural-model generations

For every scalar metric the current pipeline calculates:

- sample count
- mean
- median
- standard deviation
- first quartile
- third quartile
- minimum
- maximum

Two additional distribution distances are implemented:

### Kolmogorov-Smirnov statistic

Measures the maximum difference between the empirical cumulative distribution
functions of a generated system and the real test distribution.

`0` means identical empirical distributions.

### Wasserstein-1 distance

Measures the one-dimensional transportation distance between a generated
distribution and the real test distribution.

Lower values indicate a distribution closer to the real test set.

KS and Wasserstein-1 are currently implemented as candidate comparison tools;
their inclusion in the final paper will be decided after observing the actual
experimental results.

---

## Automated validation

Current implementation:

- `pico8_metrics.py`
- `evaluate_manifest.py`
- `compare_results.py`
- `plot_distributions.py`

Tests:

- `test_metrics.py`
- `test_manifest.py`
- `test_compare_results.py`

Current status:

`18 automated tests passing`

The tests cover:

- entropy
- normalized pitch-class entropy
- pitch statistics
- structural note density
- channel utilization
- disabled channels
- implicit empty SFX
- explicit empty SFX
- SFX repetition
- missing patterns
- manifest-to-CSV evaluation
- descriptive statistics
- KS statistic
- Wasserstein-1 distance
- system comparison

---

## Pending decisions

The following aspects must be aligned with the final data/model pipeline before
the real experiment:

1. Exact definition of a complete musical piece inside a TID.
2. How music patterns are grouped using loop-start, loop-end and stop semantics.
3. Whether the final tokenization explicitly represents these structural flags.
4. Exact held-out test manifest after the TID-level split.
5. Output format of the baseline generator.
6. Output format of the neural generator.
7. Exact temporal reconstruction using SFX `speed`.
8. Whether time-based/rhythmic metrics will be added.
9. Whether instrument/effect distributions will become formal reported metrics.
10. Which distribution distances will be reported in the final paper.

The structural metric implementation should remain independent of the final
piece-segmentation strategy.

---

## Alignment with the provided dataset synthesizer

Static inspection of the dataset's original `extract_audio.py` and
`synthesize_audio.py` established additional implementation details.

### Channel interpretation

The provided synthesizer does not decode channel values using a bit mask.

Its operational rule is:

- values `0..63`: direct reference to the SFX with the same ID;
- values `>= 64`: no audio for that channel.

The evaluator follows this rule in order to remain aligned with the dataset
pipeline.

### Pattern synthesis

The provided synthesizer renders each entry of `music` independently.

Therefore:

`music_XX.wav` represents one music pattern, not necessarily one complete
musical piece.

The synthesizer does not reconstruct a sequence of patterns into a complete
song.

### Flags and loops

The extractor preserves:

- music `flags`;
- SFX `loop_start`;
- SFX `loop_end`.

However, the current synthesizer does not use these values during rendering.

Consequently, full-song segmentation and loop reconstruction must not be
inferred from the current WAV generation pipeline.

### Timing

SFX `speed` is used by the synthesizer.

For each of the 32 note slots, nominal duration is:

`duration_seconds = max(1, speed) / 120`

The actual implementation converts this value to an integer number of audio
samples per note.

Temporal metrics based on complete songs remain pending because the current
synthesizer renders patterns independently and does not execute music-level
loop/stop semantics.

### Instruments

The symbolic representation preserves instrument values `0..15`.

The provided synthesizer implements distinct waveform behavior primarily for
values `0..7`; values outside that range fall back to a sine waveform.

Therefore, the evaluator currently treats instrument distribution as a
distribution of symbolic instrument codes, not as a distribution of distinct
rendered timbres.

### Consequence for the current protocol

The existing structural metrics remain independent of full-song synthesis:

- pitch range;
- normalized pitch-class entropy;
- structural note density;
- audible channel utilization;
- SFX repetition ratio.

Metrics involving complete-song duration, temporal structure, loop behavior or
rhythmic timing remain provisional until the group finalizes tokenization,
decoding and full-song reconstruction.
