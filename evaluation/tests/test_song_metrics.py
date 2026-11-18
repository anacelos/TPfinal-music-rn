import math
import unittest

from pico8_metrics import (
    SYNTH_SAMPLE_RATE,
    SYNTH_SAMPLES_PER_TICK,
    evaluate_song_data,
    groove_consistency,
    pitch_entropy,
    scale_consistency,
)


def empty_note():
    return {
        "pitch": 0,
        "instrument": 0,
        "volume": 0,
        "effect": 0,
    }


def sounding_note(pitch, instrument=0, volume=7, effect=0):
    return {
        "pitch": pitch,
        "instrument": instrument,
        "volume": volume,
        "effect": effect,
    }


def block_with_notes(*notes):
    result = list(notes)
    while len(result) < 32:
        result.append(empty_note())

    return {
        "original_sfx_id": 0,
        "speed": 16,
        "loop_start": 0,
        "loop_end": 0,
        "notes": result[:32],
    }


def inactive_channel():
    return {
        "active": False,
        "speed": None,
        "notes_unrolled": [],
    }


def canonical_pattern(
    pattern_id,
    notes,
    *,
    speed=16,
    pattern_ticks=64,
    block_id=None,
    is_repeat=None,
):
    channel = {
        "active": True,
        "speed": speed,
        "notes_unrolled": list(notes),
    }

    if block_id is not None:
        channel["block_id"] = block_id

    if is_repeat is not None:
        channel["is_repeat"] = is_repeat

    return {
        "pattern_id": pattern_id,
        "ref_speed": speed,
        "pattern_ticks": pattern_ticks,
        "channels": {
            "ch1": channel,
            "ch2": inactive_channel(),
            "ch3": inactive_channel(),
            "ch4": inactive_channel(),
        },
    }


class TestCourseAlignedMetrics(unittest.TestCase):

    def test_pitch_entropy_matches_course_definition(self):
        # One pitch class only -> minimum entropy.
        self.assertEqual(
            pitch_entropy([60, 60, 72, 84]),
            0.0,
        )

        # Uniform use of all 12 pitch classes -> maximum entropy.
        self.assertAlmostEqual(
            pitch_entropy(list(range(12))),
            math.log2(12),
        )

        self.assertIsNone(pitch_entropy([]))

    def test_scale_consistency_major_scale(self):
        # C-major pitch classes: C D E F G A B.
        c_major = [60, 62, 64, 65, 67, 69, 71]

        self.assertEqual(
            scale_consistency(c_major),
            1.0,
        )

        # If all 12 pitch classes occur equally, the best major scale
        # contains seven of the twelve.
        self.assertAlmostEqual(
            scale_consistency(list(range(12))),
            7 / 12,
        )

        self.assertIsNone(scale_consistency([]))

    def test_groove_consistency_identical_patterns(self):
        n = sounding_note(60)

        patterns = [
            canonical_pattern(
                0,
                [n, empty_note(), empty_note(), empty_note()],
            ),
            canonical_pattern(
                1,
                [n, empty_note(), empty_note(), empty_note()],
            ),
        ]

        self.assertEqual(
            groove_consistency(patterns),
            1.0,
        )

    def test_groove_consistency_uses_onset_positions(self):
        n = sounding_note(60)

        patterns = [
            canonical_pattern(
                0,
                [n, empty_note(), empty_note(), empty_note()],
            ),
            canonical_pattern(
                1,
                [empty_note(), n, empty_note(), empty_note()],
            ),
        ]

        # Resolution is 32.
        # First pattern has onset in bin 0.
        # Second pattern has onset in bin 8.
        # Hamming distance = 2/32.
        self.assertAlmostEqual(
            groove_consistency(patterns),
            1.0 - (2 / 32),
        )

    def test_sustained_steps_are_not_multiple_note_onsets(self):
        c = sounding_note(60)
        e = sounding_note(64)

        song = {
            "patterns": [
                canonical_pattern(
                    0,
                    [
                        c,
                        c,  # sustained state, not another NOTEON
                        e,
                        empty_note(),
                    ],
                )
            ],
        }

        r = evaluate_song_data(song)

        self.assertEqual(r["audible_note_steps"], 3)
        self.assertEqual(r["note_onsets"], 2)

        # NOTEON pitch classes are C and E with equal frequency.
        self.assertEqual(r["pitch_entropy"], 1.0)
        self.assertEqual(r["scale_consistency"], 1.0)

        # The complementary normalized entropy still describes audible
        # unrolled steps: C, C, E -> probabilities 2/3 and 1/3.
        step_entropy = -(
            (2 / 3) * math.log2(2 / 3)
            + (1 / 3) * math.log2(1 / 3)
        )
        self.assertAlmostEqual(
            r["pitch_class_entropy"],
            step_entropy / math.log2(12),
        )

    def test_retrigger_effect_forces_new_onset(self):
        retrigger = sounding_note(60, effect=3)

        song = {
            "patterns": [
                canonical_pattern(
                    0,
                    [
                        retrigger,
                        retrigger,
                        empty_note(),
                        empty_note(),
                    ],
                )
            ],
        }

        r = evaluate_song_data(song)

        # Effect 3 is emitted as a new NOTEON by the tokenizer even when
        # the symbolic state otherwise remains identical.
        self.assertEqual(r["audible_note_steps"], 2)
        self.assertEqual(r["note_onsets"], 2)


class TestCanonicalSongMetrics(unittest.TestCase):

    def test_basic_canonical_song(self):
        n1 = sounding_note(60, instrument=1)
        n2 = sounding_note(64, instrument=2, volume=5, effect=3)

        song = {
            "tid": 123,
            "song_id": 0,
            "blocks_dictionary": {
                "0": block_with_notes(n1, n2),
            },
            "patterns": [
                canonical_pattern(
                    0,
                    [
                        n1,
                        n2,
                        empty_note(),
                        empty_note(),
                    ],
                    block_id=0,
                    is_repeat=False,
                )
            ],
        }

        r = evaluate_song_data(song)

        self.assertEqual(r["tid"], 123)
        self.assertEqual(r["song_id"], 0)
        self.assertEqual(r["pattern_count"], 1)

        self.assertEqual(r["pitch_min"], 60)
        self.assertEqual(r["pitch_max"], 64)
        self.assertEqual(r["pitch_range"], 4)

        # C and E occur once each as NOTEON events.
        self.assertEqual(r["pitch_entropy"], 1.0)
        self.assertEqual(r["scale_consistency"], 1.0)
        self.assertIsNone(r["groove_consistency"])
        self.assertEqual(r["note_onsets"], 2)

        # Previous normalized entropy metric remains available.
        self.assertAlmostEqual(
            r["pitch_class_entropy"],
            1.0 / math.log2(12),
        )

        self.assertEqual(r["total_ticks"], 64)

        expected_duration = (
            64
            * SYNTH_SAMPLES_PER_TICK
            / SYNTH_SAMPLE_RATE
        )

        self.assertAlmostEqual(
            r["duration_seconds"],
            expected_duration,
        )

        self.assertEqual(r["audible_note_steps"], 2)

        self.assertAlmostEqual(
            r["audible_step_rate"],
            2 / expected_duration,
        )

        # Two audible steps * 16 ticks = 32 occupied channel-ticks.
        # Capacity = four channels * 64 ticks.
        self.assertEqual(r["audible_ticks"], 32)

        self.assertAlmostEqual(
            r["audible_channel_utilization"],
            32 / (4 * 64),
        )

        self.assertAlmostEqual(
            r["active_channel_reference_utilization"],
            1 / 4,
        )

        # Canonical block contains two audible slots among 32.
        self.assertAlmostEqual(
            r["structural_note_density"],
            2 / 32,
        )

        self.assertEqual(
            r["block_repetition_ratio"],
            0.0,
        )

    def test_block_repetition_counts_later_occurrences(self):
        n = sounding_note(60)
        block = block_with_notes(n)

        song = {
            "blocks_dictionary": {
                "0": block,
            },
            "patterns": [
                canonical_pattern(
                    0,
                    [n, empty_note()],
                    pattern_ticks=32,
                    block_id=0,
                    is_repeat=False,
                ),
                canonical_pattern(
                    1,
                    [n, empty_note()],
                    pattern_ticks=32,
                    block_id=0,
                    is_repeat=True,
                ),
            ],
        }

        r = evaluate_song_data(song)

        self.assertEqual(r["active_refs"], 2)
        self.assertAlmostEqual(
            r["block_repetition_ratio"],
            1 / 2,
        )

        # Same onset position in both patterns.
        self.assertEqual(r["groove_consistency"], 1.0)

        # The block is played twice, so its structural capacity is
        # counted twice.
        self.assertEqual(
            r["structural_active_slots"],
            2,
        )
        self.assertEqual(
            r["structural_available_slots"],
            64,
        )

    def test_metrics_without_block_identity(self):
        n = sounding_note(60)

        song = {
            # NLM/grid_unrolled-like decoded representation:
            # no blocks_dictionary, block_id or is_repeat.
            "patterns": [
                canonical_pattern(
                    0,
                    [n, empty_note()],
                    pattern_ticks=32,
                )
            ],
        }

        r = evaluate_song_data(song)

        # Common metrics remain available.
        self.assertEqual(r["pitch_range"], 0)
        self.assertEqual(r["pitch_entropy"], 0.0)
        self.assertEqual(r["pitch_class_entropy"], 0.0)
        self.assertEqual(r["scale_consistency"], 1.0)
        self.assertEqual(r["note_onsets"], 1)

        self.assertIsNotNone(
            r["audible_step_rate"]
        )
        self.assertIsNotNone(
            r["audible_channel_utilization"]
        )

        # A single positive-duration pattern is insufficient for GC.
        self.assertIsNone(r["groove_consistency"])

        # Block-specific metrics are intentionally unavailable.
        self.assertIsNone(
            r["structural_note_density"]
        )
        self.assertIsNone(
            r["block_repetition_ratio"]
        )

    def test_audible_ticks_are_clipped_to_pattern_end(self):
        n = sounding_note(60)

        song = {
            "patterns": [
                canonical_pattern(
                    0,
                    [n, n],
                    pattern_ticks=20,
                )
            ],
        }

        r = evaluate_song_data(song)

        # First step occupies 16 ticks; second is clipped to the
        # remaining 4.
        self.assertEqual(r["audible_ticks"], 20)

        self.assertAlmostEqual(
            r["audible_channel_utilization"],
            20 / (4 * 20),
        )

        # The two identical adjacent steps are one sustained note.
        self.assertEqual(r["audible_note_steps"], 2)
        self.assertEqual(r["note_onsets"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
