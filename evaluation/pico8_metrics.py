import json
import math
from collections import Counter
from pathlib import Path

SYNTH_SAMPLE_RATE = 44100
SYNTH_SAMPLES_PER_TICK = 366

GROOVE_RESOLUTION = 32
MAJOR_SCALE_INTERVALS = frozenset({0, 2, 4, 5, 7, 9, 11})


def shannon_entropy(values):
    if not values:
        return None

    counts = Counter(values)
    total = len(values)

    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
    )


def normalized_pitch_class_entropy(pitches):
    """
    Normalized Shannon entropy of the 12 pitch classes.

    This is retained as a complementary metric from the original evaluation
    protocol. Values are approximately bounded between 0 and 1.
    """
    if not pitches:
        return None

    pitch_classes = [pitch % 12 for pitch in pitches]
    entropy = shannon_entropy(pitch_classes)

    return entropy / math.log2(12)


def pitch_entropy(pitches):
    """
    Shannon entropy, in bits, of the 12 pitch classes.

    This follows the definition used in the course evaluation material:
    minimum 0 and maximum log2(12).
    """
    if not pitches:
        return None

    pitch_classes = [pitch % 12 for pitch in pitches]
    return shannon_entropy(pitch_classes)


def scale_consistency(pitches):
    """
    Fraction of pitch occurrences belonging to the best-fitting major scale.

    All 12 possible major-scale tonics are evaluated, and the maximum
    fraction is returned.
    """
    if not pitches:
        return None

    pitch_classes = [pitch % 12 for pitch in pitches]

    return max(
        sum(
            1
            for pitch_class in pitch_classes
            if (pitch_class - tonic) % 12 in MAJOR_SCALE_INTERVALS
        ) / len(pitch_classes)
        for tonic in range(12)
    )


def _extract_onset_features(patterns, resolution=GROOVE_RESOLUTION):
    """
    Extract note-on pitches and one binary groove vector per positive-duration
    PICO-8 pattern.

    State is preserved across pattern boundaries, matching the NLM tokenizer.
    An identical consecutive pitch/instrument/volume/effect state is sustained,
    except effects 3, 4 and 5, which force a new NOTEON.
    """
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    onset_pitches = []
    groove_vectors = []
    channel_states = {}

    for pattern in patterns:
        pattern_ticks = max(0, int(pattern.get("pattern_ticks") or 0))
        vector = [0] * resolution

        channels = pattern.get("channels", {})

        # An inactive/missing channel terminates any state carried from the
        # preceding pattern, as in the NLM tokenizer.
        for channel_name in set(channel_states) | set(channels):
            channel = channels.get(channel_name, {})

            if not channel.get("active") or not channel.get("notes_unrolled"):
                channel_states[channel_name] = {
                    "active": False,
                    "pitch": None,
                    "instrument": None,
                    "volume": 0,
                    "effect": None,
                }
                continue

            state = channel_states.setdefault(
                channel_name,
                {
                    "active": False,
                    "pitch": None,
                    "instrument": None,
                    "volume": 0,
                    "effect": None,
                },
            )

            speed = max(
                1,
                int(
                    channel.get("speed")
                    or pattern.get("ref_speed")
                    or 1
                ),
            )

            notes = channel.get("notes_unrolled", [])

            for step_index, note in enumerate(notes):
                start_tick = step_index * speed

                if start_tick >= pattern_ticks:
                    break

                pitch = int(note.get("pitch", 0))
                instrument = int(note.get("instrument", 0))
                volume = int(note.get("volume", 0))
                effect = int(note.get("effect", 0))

                if volume <= 0:
                    state["active"] = False
                    state["pitch"] = None
                    state["volume"] = 0
                    continue

                is_sustained = (
                    state["active"]
                    and state["pitch"] == pitch
                    and state["instrument"] == instrument
                    and state["volume"] == volume
                    and state["effect"] == effect
                    and effect not in (3, 4, 5)
                )

                if not is_sustained:
                    onset_pitches.append(pitch)

                    if pattern_ticks > 0:
                        bin_index = min(
                            resolution - 1,
                            (start_tick * resolution) // pattern_ticks,
                        )
                        vector[bin_index] = 1

                state["active"] = True
                state["pitch"] = pitch
                state["instrument"] = instrument
                state["volume"] = volume
                state["effect"] = effect

            # Same NOTE-OFF rule used by the tokenizer when the unrolled list
            # ends before the end of the pattern.
            end_tick = len(notes) * speed
            if end_tick < pattern_ticks:
                state["active"] = False
                state["pitch"] = None
                state["volume"] = 0

        if pattern_ticks > 0:
            groove_vectors.append(vector)

    return onset_pitches, groove_vectors


def groove_consistency_from_vectors(vectors):
    """
    Mean similarity between consecutive binary groove vectors.

    GC = 1 - mean(normalized Hamming distance)
    """
    if len(vectors) < 2:
        return None

    resolution = len(vectors[0])

    if resolution == 0:
        return None

    normalized_distances = []

    for previous, current in zip(vectors, vectors[1:]):
        hamming_distance = sum(
            a != b
            for a, b in zip(previous, current)
        )
        normalized_distances.append(
            hamming_distance / resolution
        )

    return 1.0 - (
        sum(normalized_distances) / len(normalized_distances)
    )


def groove_consistency(patterns, resolution=GROOVE_RESOLUTION):
    """
    Groove Consistency adapted to PICO-8 patterns.

    Positive-duration music patterns are the temporal units analogous to
    measures. Their NOTEON events are projected onto a normalized binary grid.
    """
    _, vectors = _extract_onset_features(patterns, resolution)
    return groove_consistency_from_vectors(vectors)


def evaluate_piece_data(data, pattern_ids, tid=None):
    """
    Avalia uma peça PICO-8 definida por um conjunto ordenado de pattern IDs.

    Esta função é mantida para compatibilidade com o pipeline legado baseado
    em audio.json.

    Regras atualmente validadas:
    - valores de canal 0..63 referenciam diretamente SFX 0..63;
    - valores de canal >= 64 são tratados como canal sem áudio pelo sintetizador;
    - volume > 0 = slot audível;
    - SFX habilitado mas ausente = SFX implícito/vazio;
    - flags musicais são interpretados semanticamente com flags & 0x07;
    - somente SFX referenciados pelos patterns da peça entram nas métricas;
    - reutilizações do mesmo SFX contam novamente, pois representam
      novas ocorrências durante a execução da peça.
    """

    pattern_ids = list(pattern_ids)

    patterns_by_id = {
        pattern["id"]: pattern
        for pattern in data.get("music", [])
    }

    sfx_by_id = {
        sfx["id"]: sfx
        for sfx in data.get("sfx", [])
    }

    pitches = []
    instruments = Counter()
    effects = Counter()
    speeds = Counter()
    semantic_flags = Counter()
    sfx_refs = Counter()

    total_patterns = 0

    enabled_refs = 0
    audible_refs = 0

    implicit_empty_refs = 0
    explicit_empty_refs = 0

    active_note_slots = 0
    available_note_slots = 0

    missing_pattern_ids = []

    for pattern_id in pattern_ids:
        pattern = patterns_by_id.get(pattern_id)

        if pattern is None:
            missing_pattern_ids.append(pattern_id)
            continue

        total_patterns += 1
        semantic_flags[pattern.get("flags", 0) & 0x07] += 1

        for channel in range(1, 5):
            raw = pattern[f"ch{channel}"]

            # O sintetizador do dataset considera qualquer valor >= 64
            # como canal sem áudio. Valores 0..63 referenciam diretamente
            # o SFX de mesmo ID.
            if raw >= 64:
                continue

            enabled_refs += 1

            sfx_id = raw
            sfx_refs[sfx_id] += 1

            sfx = sfx_by_id.get(sfx_id)

            if sfx is None:
                # SFX não serializado: tratamos como bloco padrão vazio.
                implicit_empty_refs += 1
                available_note_slots += 32
                continue

            notes = sfx.get("notes", [])

            # Normalmente são 32 slots. Mantemos o tamanho real do JSON
            # para detectar anomalias sem mascará-las.
            available_note_slots += len(notes)

            active_notes = [
                note
                for note in notes
                if note.get("volume", 0) > 0
            ]

            if not active_notes:
                explicit_empty_refs += 1
                continue

            audible_refs += 1
            active_note_slots += len(active_notes)

            speeds[sfx.get("speed")] += 1

            for note in active_notes:
                pitch = note.get("pitch")

                if pitch is not None:
                    pitches.append(pitch)

                instruments[note.get("instrument")] += 1
                effects[note.get("effect")] += 1

    if pitches:
        pitch_min = min(pitches)
        pitch_max = max(pitches)
        pitch_range = pitch_max - pitch_min
    else:
        pitch_min = None
        pitch_max = None
        pitch_range = None

    structural_note_density = (
        active_note_slots / available_note_slots
        if available_note_slots
        else None
    )

    audible_channel_utilization = (
        audible_refs / (4 * total_patterns)
        if total_patterns
        else None
    )

    enabled_channel_utilization = (
        enabled_refs / (4 * total_patterns)
        if total_patterns
        else None
    )

    total_sfx_refs = sum(sfx_refs.values())

    repeated_sfx_refs = sum(
        count - 1
        for count in sfx_refs.values()
        if count > 1
    )

    sfx_repetition_ratio = (
        repeated_sfx_refs / total_sfx_refs
        if total_sfx_refs
        else None
    )

    return {
        "tid": tid,
        "pattern_ids": pattern_ids,
        "missing_pattern_ids": missing_pattern_ids,
        "pattern_count": total_patterns,

        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "pitch_range": pitch_range,
        "pitch_class_entropy": normalized_pitch_class_entropy(pitches),

        "structural_note_density": structural_note_density,

        "audible_channel_utilization": audible_channel_utilization,
        "enabled_channel_utilization": enabled_channel_utilization,

        "active_note_slots": active_note_slots,
        "available_note_slots": available_note_slots,

        "enabled_refs": enabled_refs,
        "audible_refs": audible_refs,
        "implicit_empty_refs": implicit_empty_refs,
        "explicit_empty_refs": explicit_empty_refs,

        "sfx_repetition_ratio": sfx_repetition_ratio,

        "instruments": dict(sorted(instruments.items())),
        "effects": dict(sorted(effects.items())),
        "speeds": dict(sorted(speeds.items())),
        "semantic_flags": dict(sorted(semantic_flags.items())),
    }


def evaluate_piece_file(audio_json, pattern_ids, tid=None):
    """
    Evaluate a legacy audio.json piece.
    """
    audio_json = Path(audio_json)

    with audio_json.open(encoding="utf-8") as f:
        data = json.load(f)

    return evaluate_piece_data(
        data=data,
        pattern_ids=pattern_ids,
        tid=tid,
    )


def evaluate_song_data(song):
    """
    Evaluate a canonical song dictionary produced by songs.json or by a
    tokenizer decoder.

    Metrics common to all representations are calculated from patterns /
    notes_unrolled. Block-dependent metrics are returned only when the
    representation preserves block metadata.

    Course-aligned symbolic metrics:
    - pitch_entropy
    - scale_consistency
    - groove_consistency

    Complementary PICO-8 metrics:
    - pitch_range
    - normalized pitch-class entropy
    - audible step rate
    - audible channel utilization
    - structural note density
    - block repetition ratio
    """

    step_pitches = []
    instruments = Counter()
    effects = Counter()
    speeds = Counter()

    patterns = song.get("patterns", [])
    blocks = song.get("blocks_dictionary", {})

    total_ticks = 0
    audible_ticks = 0
    audible_note_steps = 0
    active_refs = 0

    structural_active_slots = 0
    structural_available_slots = 0
    block_metadata_complete = True

    repeated_block_refs = 0
    repeat_metadata_complete = True

    for pattern in patterns:
        pattern_ticks = max(
            0,
            int(pattern.get("pattern_ticks") or 0),
        )
        total_ticks += pattern_ticks

        for channel in pattern.get("channels", {}).values():
            if not channel.get("active"):
                continue

            active_refs += 1

            speed = max(
                1,
                int(
                    channel.get("speed")
                    or pattern.get("ref_speed")
                    or 1
                ),
            )
            speeds[speed] += 1

            notes = channel.get("notes_unrolled", [])

            for step_index, note in enumerate(notes):
                if note.get("volume", 0) <= 0:
                    continue

                audible_note_steps += 1

                pitch = note.get("pitch")
                if pitch is not None:
                    step_pitches.append(pitch)

                instruments[note.get("instrument")] += 1
                effects[note.get("effect")] += 1

                # Each unrolled symbolic step occupies `speed` ticks, clipped
                # at the end of the pattern.
                start_tick = step_index * speed

                if start_tick < pattern_ticks:
                    audible_ticks += min(
                        speed,
                        pattern_ticks - start_tick,
                    )

            # Structural 32-slot metric is available only when block identity
            # was preserved by the representation.
            block_id = channel.get("block_id")
            block = (
                blocks.get(str(block_id))
                if block_id is not None
                else None
            )

            if block is None:
                block_metadata_complete = False
            else:
                block_notes = block.get("notes", [])
                structural_available_slots += len(block_notes)
                structural_active_slots += sum(
                    1
                    for note in block_notes
                    if note.get("volume", 0) > 0
                )

            if "is_repeat" not in channel:
                repeat_metadata_complete = False
            elif channel.get("is_repeat"):
                repeated_block_refs += 1

    onset_pitches, groove_vectors = _extract_onset_features(patterns)

    if onset_pitches:
        pitch_min = min(onset_pitches)
        pitch_max = max(onset_pitches)
        pitch_range = pitch_max - pitch_min
    else:
        pitch_min = None
        pitch_max = None
        pitch_range = None

    duration_seconds = (
        total_ticks
        * SYNTH_SAMPLES_PER_TICK
        / SYNTH_SAMPLE_RATE
        if total_ticks > 0
        else None
    )

    audible_step_rate = (
        audible_note_steps / duration_seconds
        if duration_seconds
        else None
    )

    audible_channel_utilization = (
        audible_ticks / (4 * total_ticks)
        if total_ticks > 0
        else None
    )

    active_channel_reference_utilization = (
        active_refs / (4 * len(patterns))
        if patterns
        else None
    )

    structural_note_density = (
        structural_active_slots / structural_available_slots
        if (
            block_metadata_complete
            and structural_available_slots > 0
        )
        else None
    )

    block_repetition_ratio = (
        repeated_block_refs / active_refs
        if (
            repeat_metadata_complete
            and active_refs > 0
        )
        else None
    )

    return {
        "tid": song.get("tid"),
        "song_id": song.get("song_id"),
        "pattern_count": len(patterns),

        "total_ticks": total_ticks,
        "duration_seconds": duration_seconds,

        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "pitch_range": pitch_range,

        # Course-aligned symbolic metrics.
        "pitch_entropy": pitch_entropy(onset_pitches),
        "scale_consistency": scale_consistency(onset_pitches),
        "groove_consistency": groove_consistency_from_vectors(
            groove_vectors
        ),

        # Complementary pitch metric retained from the previous protocol.
        "pitch_class_entropy": normalized_pitch_class_entropy(
            step_pitches
        ),

        # Common representation-independent PICO-8 metrics.
        "audible_step_rate": audible_step_rate,
        "audible_channel_utilization": audible_channel_utilization,

        # Diagnostics.
        "audible_note_steps": audible_note_steps,
        "note_onsets": len(onset_pitches),
        "audible_ticks": audible_ticks,
        "active_refs": active_refs,
        "active_channel_reference_utilization":
            active_channel_reference_utilization,

        # Available only when block structure is preserved.
        "structural_note_density": structural_note_density,
        "block_repetition_ratio": block_repetition_ratio,

        "structural_active_slots": (
            structural_active_slots
            if block_metadata_complete
            else None
        ),
        "structural_available_slots": (
            structural_available_slots
            if block_metadata_complete
            else None
        ),

        "instruments": dict(sorted(instruments.items())),
        "effects": dict(sorted(effects.items())),
        "speeds": dict(sorted(speeds.items())),
    }
