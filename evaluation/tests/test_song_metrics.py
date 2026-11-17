import math
import unittest

from pico8_metrics import (
    SYNTH_SAMPLE_RATE,
    SYNTH_SAMPLES_PER_TICK,
    evaluate_song_data,
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
                {
                    "pattern_id": 0,
                    "ref_speed": 16,
                    "pattern_ticks": 64,
                    "channels": {
                        "ch1": {
                            "active": True,
                            "block_id": 0,
                            "is_repeat": False,
                            "speed": 16,
                            "notes_unrolled": [
                                n1,
                                n2,
                                empty_note(),
                                empty_note(),
                            ],
                        },
                        "ch2": inactive_channel(),
                        "ch3": inactive_channel(),
                        "ch4": inactive_channel(),
                    },
                }
            ],
        }

        r = evaluate_song_data(song)

        self.assertEqual(r["tid"], 123)
        self.assertEqual(r["song_id"], 0)
        self.assertEqual(r["pattern_count"], 1)

        self.assertEqual(r["pitch_min"], 60)
        self.assertEqual(r["pitch_max"], 64)
        self.assertEqual(r["pitch_range"], 4)

        self.assertAlmostEqual(
            r["pitch_class_entropy"],
            1.0 / math.log2(12),
        )

        self.assertEqual(r["total_ticks"], 64)
        expected_duration = (
            64 * SYNTH_SAMPLES_PER_TICK / SYNTH_SAMPLE_RATE
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

        self.assertEqual(r["block_repetition_ratio"], 0.0)

    def test_block_repetition_counts_later_occurrences(self):
        n = sounding_note(60)
        block = block_with_notes(n)

        def pattern(pid, repeat):
            return {
                "pattern_id": pid,
                "ref_speed": 16,
                "pattern_ticks": 32,
                "channels": {
                    "ch1": {
                        "active": True,
                        "block_id": 0,
                        "is_repeat": repeat,
                        "speed": 16,
                        "notes_unrolled": [
                            n,
                            empty_note(),
                        ],
                    },
                    "ch2": inactive_channel(),
                    "ch3": inactive_channel(),
                    "ch4": inactive_channel(),
                },
            }

        song = {
            "blocks_dictionary": {"0": block},
            "patterns": [
                pattern(0, False),
                pattern(1, True),
            ],
        }

        r = evaluate_song_data(song)

        self.assertEqual(r["active_refs"], 2)
        self.assertAlmostEqual(r["block_repetition_ratio"], 1 / 2)

        # The block is played twice, so its structural capacity is counted twice.
        self.assertEqual(r["structural_active_slots"], 2)
        self.assertEqual(r["structural_available_slots"], 64)

    def test_metrics_without_block_identity(self):
        n = sounding_note(60)

        song = {
            # NLM/grid_unrolled-like decoded representation:
            # no blocks_dictionary, block_id or is_repeat.
            "patterns": [
                {
                    "pattern_id": 0,
                    "ref_speed": 16,
                    "pattern_ticks": 32,
                    "channels": {
                        "ch1": {
                            "active": True,
                            "speed": 16,
                            "notes_unrolled": [
                                n,
                                empty_note(),
                            ],
                        },
                        "ch2": inactive_channel(),
                        "ch3": inactive_channel(),
                        "ch4": inactive_channel(),
                    },
                }
            ],
        }

        r = evaluate_song_data(song)

        # Common metrics remain available.
        self.assertEqual(r["pitch_range"], 0)
        self.assertEqual(r["pitch_class_entropy"], 0.0)
        self.assertIsNotNone(r["audible_step_rate"])
        self.assertIsNotNone(r["audible_channel_utilization"])

        # Block-specific metrics are intentionally unavailable.
        self.assertIsNone(r["structural_note_density"])
        self.assertIsNone(r["block_repetition_ratio"])

    def test_audible_ticks_are_clipped_to_pattern_end(self):
        n = sounding_note(60)

        song = {
            "patterns": [
                {
                    "pattern_id": 0,
                    "ref_speed": 16,
                    "pattern_ticks": 20,
                    "channels": {
                        "ch1": {
                            "active": True,
                            "speed": 16,
                            "notes_unrolled": [n, n],
                        },
                        "ch2": inactive_channel(),
                        "ch3": inactive_channel(),
                        "ch4": inactive_channel(),
                    },
                }
            ],
        }

        r = evaluate_song_data(song)

        # First step occupies 16 ticks; second is clipped to the remaining 4.
        self.assertEqual(r["audible_ticks"], 20)
        self.assertAlmostEqual(
            r["audible_channel_utilization"],
            20 / (4 * 20),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
