# PICO-8 Evaluation Protocol

## Status

This document describes the symbolic evaluation protocol used in the PICO-8
autoregressive music generation project.

The final preprocessing and canonical song representation are now defined. The
main evaluation compares:

1. Real music from the held-out test set
2. Baseline generations
3. Neural autoregressive model generations

All systems are evaluated after conversion to the same canonical song
representation.

The evaluator does not compare raw token sequences directly.

---

## Evaluation pipeline

The canonical evaluation flow is:

```text
songs.json
    ↓
preprocessing
    ↓
data/splits/{train,val,test}.jsonl
    ↓
canonical song / decoded generation
    ↓
evaluate_song_data()
    ↓
per-song metrics
    ↓
distribution-level comparison
```

For generated samples, model tokens must first be decoded by the corresponding
tokenizer.

The evaluator consumes the reconstructed song representation rather than the
token sequence itself.

---

## Canonical song representation

A canonical song contains:

- `song_id`
- `tid`, for dataset songs
- `start_pattern`
- `end_pattern`
- `loop_start_pattern`
- `end_reason`
- `num_patterns`
- `total_ticks`
- `duration_sec`
- `patterns`
- optionally `blocks_dictionary`

Each pattern contains:

- `pattern_id`
- `ref_speed`
- `pattern_ticks`
- four channels

Each channel may contain:

- `active`
- `speed`
- `notes_unrolled`
- `block_id`, when block identity is preserved
- `is_repeat`, when block identity is preserved

Each symbolic note step contains:

- `pitch`
- `instrument`
- `volume`
- `effect`

A symbolic step is considered audible when:

```text
volume > 0
```

---

## Unrolled representation

The primary metrics operate on `notes_unrolled`.

This timeline is already reconstructed by the canonical dataset or by the
tokenizer decoder. The evaluator does not perform a second independent
unrolling pass.

This is important because the project supports tokenizations with different
internal structures:

- `nlm`
- `grid_unrolled`
- `grid_block`

After decoding, the primary metrics operate on the resulting canonical
timeline.

Block-dependent information is treated separately because not every
tokenization preserves block identity after decoding.

---

# Primary metrics

The following four metrics form the default comparison between Real, Baseline
and Neural systems.

They are intended to remain available independently of whether the original
token representation preserved explicit block identity.

---

## 1. Pitch Range

Pitch Range measures the register span of the song.

Only audible symbolic steps are considered.

```text
pitch_range = max(pitch) - min(pitch)
```

where:

```text
volume > 0
```

If no audible pitch exists, the metric is undefined.

Lower values indicate a narrower register.

Higher values indicate a broader register.

---

## 2. Normalized Pitch-Class Entropy

Each audible pitch is converted into one of 12 pitch classes:

```text
pitch_class = pitch % 12
```

Shannon entropy is then calculated:

```text
H = -sum(p_i * log2(p_i))
```

and normalized by the maximum entropy for 12 pitch classes:

```text
H_norm = H / log2(12)
```

The result is approximately bounded between 0 and 1.

Lower values indicate concentration on fewer pitch classes.

Higher values indicate greater pitch-class diversity.

Only audible symbolic steps are considered.

---

## 3. Audible Step Rate

Audible Step Rate measures how many audible symbolic steps occur per second of
the reconstructed song.

An audible step satisfies:

```text
volume > 0
```

The project synthesizer uses:

```text
SAMPLE_RATE = 44100
SAMPLES_PER_TICK = 366
```

Therefore the evaluator derives duration as:

```text
duration_seconds =
    total_ticks * SAMPLES_PER_TICK / SAMPLE_RATE
```

and computes:

```text
audible_step_rate =
    audible_step_count / duration_seconds
```

The term "step" is intentional.

Entries in `notes_unrolled` represent symbolic tracker steps and must not be
interpreted automatically as independent musical note attacks.

This metric captures symbolic event activity over time.

---

## 4. Audible Channel Utilization

PICO-8 provides a maximum of four simultaneous music channels.

Audible Channel Utilization measures how much of this four-channel temporal
capacity is occupied by audible symbolic material.

For each audible step, its duration is determined by the channel `speed`.

The occupied interval is clipped at the end of the corresponding pattern.

For one song:

```text
audible_channel_utilization =
    audible_channel_ticks / (4 * total_ticks)
```

The result lies between 0 and 1.

Examples:

- `0.25` means that, on average, the equivalent of one of four channels is
  sounding throughout the song;
- `1.0` means all four channels are continuously occupied by audible material.

This metric differs from simply counting channels marked `active`, because an
active channel may still contain silent symbolic steps.

---

# Auxiliary structural metrics

The following metrics describe PICO-8 block structure.

They are useful when that structure is available, but they are not included in
the default cross-representation comparison.

---

## 5. Structural Note Density

Canonical PICO-8 blocks contain 32 structural note slots.

For every active block reference whose block can be resolved through
`blocks_dictionary`, the evaluator counts:

- available structural slots;
- slots with `volume > 0`.

The metric is:

```text
structural_note_density =
    audible_structural_slots / available_structural_slots
```

Repeated uses of the same block are counted again because they correspond to
new occurrences of that material during playback.

This metric preserves the interpretation of the original structural evaluator.

However, it requires explicit block identity.

If the decoded representation does not preserve `blocks_dictionary` and
`block_id`, the metric is returned as unavailable (`None`).

---

## 6. Block Repetition Ratio

When block identity is preserved, each active block reference may indicate
whether it is the first definition of the block or a later reuse.

The metric is:

```text
block_repetition_ratio =
    repeated_block_references / active_block_references
```

A reference is considered repeated when:

```text
is_repeat == True
```

The final tokenizer semantics use:

- first occurrence: block definition;
- later occurrence: block reuse.

The test-set audit confirmed that these markers are consistent with block
occurrence order.

This metric is therefore suitable for representations such as `grid_block`.

Representations such as `nlm` and `grid_unrolled` do not necessarily preserve
block identity after decoding. In those cases, Block Repetition Ratio is
returned as unavailable and is excluded from the default comparison.

---

# Additional diagnostics

The evaluator also records several diagnostic quantities:

- minimum pitch;
- maximum pitch;
- total ticks;
- reconstructed duration;
- number of patterns;
- number of audible symbolic steps;
- number of audible channel-ticks;
- number of active channel references;
- active-channel reference utilization;
- instrument distribution;
- effect distribution;
- speed distribution;
- structural active slots, when block information is available;
- structural available slots, when block information is available.

These values may support additional analyses but are not part of the default
four-metric comparison.

---

# Distribution-level comparison

Metrics are computed independently for each song.

The main experiment compares distributions from:

- real held-out test songs;
- baseline-generated songs;
- neural-model-generated songs.

For each primary metric, the pipeline calculates:

- sample count;
- mean;
- median;
- standard deviation;
- first quartile;
- third quartile;
- minimum;
- maximum.

Two distribution distances are also implemented.

---

## Kolmogorov-Smirnov statistic

The two-sample Kolmogorov-Smirnov statistic measures the maximum absolute
difference between two empirical cumulative distribution functions.

```text
KS = max_x |F_real(x) - F_generated(x)|
```

The statistic lies between 0 and 1.

Lower values indicate a generated distribution closer to the real
distribution.

An identical empirical distribution gives:

```text
KS = 0
```

---

## Wasserstein-1 distance

Wasserstein-1 measures the one-dimensional transportation distance between two
empirical distributions.

Its unit is the same as the evaluated metric.

Lower values indicate a generated distribution closer to the real
distribution.

An identical empirical distribution gives:

```text
W1 = 0
```

Because different metrics have different numerical scales, Wasserstein values
should primarily be interpreted within each metric rather than compared
directly across different metrics.

---

# Real test-set validation

The final preprocessing pipeline produces:

```text
train: 9907 songs
validation: 1242 songs
test: 1235 songs
```

The canonical evaluator was executed on all 1,235 held-out test songs.

Validation results:

```text
songs evaluated: 1235
duration consistency: 1235 / 1235
missing primary metric values: 0
out-of-range normalized metric values: 0
```

The reconstructed duration uses:

```text
total_ticks * 366 / 44100
```

When rounded to two decimal places, it matched the dataset `duration_sec` field
for all 1,235 test songs.

---

# Real test-set reference statistics

The current real test set produced the following descriptive values.

```text
pitch_range
    median: 33.000000
    mean:   32.306883
    min:     0.000000
    max:    63.000000

pitch_class_entropy
    median: 0.661703
    mean:   0.633397
    min:   -0.000000
    max:    0.993393

audible_step_rate
    median: 11.001937
    mean:   17.467459
    min:     0.941342
    max:   481.967213

audible_channel_utilization
    median: 0.375000
    mean:   0.397856
    min:    0.014657
    max:    1.000000
```

The auxiliary block-based metrics for the real test set were:

```text
structural_note_density
    median: 0.687500
    mean:   0.684470

block_repetition_ratio
    median: 0.142857
    mean:   0.281292
```

These values are reference statistics for the real test distribution and are
not, by themselves, results about baseline or neural-model quality.

---

# Automated validation

The current evaluation code includes:

- `pico8_metrics.py`
- `evaluate_split.py`
- `compare_results.py`
- `plot_distributions.py`

Legacy compatibility code remains temporarily available:

- `evaluate_manifest.py`
- `pieces.example.json`

The automated test suite currently contains 24 passing tests.

The tests cover:

- Shannon entropy;
- normalized pitch-class entropy;
- pitch statistics;
- canonical song evaluation;
- symbolic timing;
- audible-step counting;
- temporal channel utilization;
- pattern-end clipping;
- structural note density;
- block repetition;
- evaluation without block identity;
- legacy `audio.json` compatibility;
- manifest evaluation;
- descriptive statistics;
- percentiles;
- KS statistic;
- Wasserstein-1 distance;
- comparison between systems;
- synthesizer alignment.

Current validated status:

```text
Ran 24 tests
OK
```

---

# Generated outputs

The canonical real test-set command is:

```bash
python evaluation/evaluate_split.py \
  --input data/splits/test.jsonl \
  --output results/real_test_metrics.csv
```

The generated CSV contains one row per song.

For the final experiment, equivalent metric CSV files should be produced for
the baseline and neural model after their outputs are decoded into canonical
songs.

The default comparison uses:

```text
pitch_range
pitch_class_entropy
audible_step_rate
audible_channel_utilization
```

Block-dependent metrics remain available for representation-specific or
structural analyses when their required metadata is preserved.
