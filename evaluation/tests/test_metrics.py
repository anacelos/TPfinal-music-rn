import math
import unittest

from pico8_metrics import (
    evaluate_piece_data,
    normalized_pitch_class_entropy,
    shannon_entropy,
)


def empty_note():
    return {
        "pitch": 0,
        "instrument": 0,
        "volume": 0,
        "effect": 0,
    }


def make_sfx(sfx_id, active_notes=None, speed=16):
    notes = [empty_note() for _ in range(32)]

    for slot, note in (active_notes or {}).items():
        notes[slot] = {
            "pitch": note["pitch"],
            "instrument": note.get("instrument", 0),
            "volume": note.get("volume", 7),
            "effect": note.get("effect", 0),
        }

    return {
        "id": sfx_id,
        "speed": speed,
        "loop_start": 0,
        "loop_end": 0,
        "notes": notes,
    }


def pattern(pid, flags=0, ch1=64, ch2=66, ch3=67, ch4=68):
    return {
        "id": pid,
        "flags": flags,
        "ch1": ch1,
        "ch2": ch2,
        "ch3": ch3,
        "ch4": ch4,
    }


class TestEntropy(unittest.TestCase):

    def test_empty_entropy(self):
        self.assertIsNone(shannon_entropy([]))

    def test_single_value_entropy(self):
        self.assertEqual(shannon_entropy([1, 1, 1]), 0.0)

    def test_two_equiprobable_values(self):
        self.assertAlmostEqual(
            shannon_entropy([0, 1]),
            1.0,
        )

    def test_normalized_pitch_class_entropy(self):
        result = normalized_pitch_class_entropy([60, 64])

        self.assertAlmostEqual(
            result,
            1.0 / math.log2(12),
        )


class TestPieceMetrics(unittest.TestCase):

    def test_basic_audible_sfx(self):
        data = {
            "sfx": [
                make_sfx(
                    0,
                    {
                        0: {
                            "pitch": 60,
                            "instrument": 1,
                            "volume": 7,
                            "effect": 0,
                        },
                        1: {
                            "pitch": 64,
                            "instrument": 2,
                            "volume": 5,
                            "effect": 3,
                        },
                    },
                )
            ],
            "music": [
                # flags=9 -> semanticamente 1 após & 0x07.
                pattern(0, flags=9, ch1=0),
            ],
        }

        r = evaluate_piece_data(
            data,
            pattern_ids=[0],
            tid="test",
        )

        self.assertEqual(r["pattern_count"], 1)

        self.assertEqual(r["enabled_refs"], 1)
        self.assertEqual(r["audible_refs"], 1)

        self.assertEqual(r["active_note_slots"], 2)
        self.assertEqual(r["available_note_slots"], 32)

        self.assertAlmostEqual(
            r["structural_note_density"],
            2 / 32,
        )

        self.assertAlmostEqual(
            r["audible_channel_utilization"],
            1 / 4,
        )

        self.assertEqual(r["pitch_min"], 60)
        self.assertEqual(r["pitch_max"], 64)
        self.assertEqual(r["pitch_range"], 4)

        self.assertEqual(r["instruments"], {1: 1, 2: 1})
        self.assertEqual(r["effects"], {0: 1, 3: 1})

        self.assertEqual(r["semantic_flags"], {1: 1})

    def test_implicit_empty_sfx(self):
        data = {
            "sfx": [],
            "music": [
                # SFX 2 habilitado, mas não serializado.
                pattern(0, ch1=2),
            ],
        }

        r = evaluate_piece_data(data, [0])

        self.assertEqual(r["enabled_refs"], 1)
        self.assertEqual(r["audible_refs"], 0)

        self.assertEqual(r["implicit_empty_refs"], 1)
        self.assertEqual(r["explicit_empty_refs"], 0)

        self.assertEqual(r["available_note_slots"], 32)
        self.assertEqual(r["active_note_slots"], 0)

        self.assertEqual(r["structural_note_density"], 0.0)
        self.assertEqual(r["enabled_channel_utilization"], 0.25)
        self.assertEqual(r["audible_channel_utilization"], 0.0)

        self.assertIsNone(r["pitch_range"])
        self.assertIsNone(r["pitch_class_entropy"])

    def test_explicit_empty_sfx(self):
        data = {
            "sfx": [
                make_sfx(1)
            ],
            "music": [
                pattern(0, ch1=1)
            ],
        }

        r = evaluate_piece_data(data, [0])

        self.assertEqual(r["enabled_refs"], 1)
        self.assertEqual(r["audible_refs"], 0)

        self.assertEqual(r["implicit_empty_refs"], 0)
        self.assertEqual(r["explicit_empty_refs"], 1)

        self.assertEqual(r["available_note_slots"], 32)
        self.assertEqual(r["active_note_slots"], 0)

    def test_sfx_repetition(self):
        data = {
            "sfx": [
                make_sfx(
                    0,
                    {
                        0: {
                            "pitch": 60,
                            "volume": 7,
                        }
                    },
                )
            ],
            "music": [
                pattern(0, ch1=0),
                pattern(1, ch1=0),
            ],
        }

        r = evaluate_piece_data(data, [0, 1])

        # Duas referências, sendo a segunda uma repetição.
        self.assertAlmostEqual(
            r["sfx_repetition_ratio"],
            1 / 2,
        )

        # O mesmo SFX é executado duas vezes.
        self.assertEqual(r["active_note_slots"], 2)
        self.assertEqual(r["available_note_slots"], 64)

    def test_disabled_channel_is_ignored(self):
        data = {
            "sfx": [
                make_sfx(
                    2,
                    {
                        0: {
                            "pitch": 60,
                            "volume": 7,
                        }
                    },
                )
            ],
            "music": [
                # 66 = bit 6 ligado + SFX ID 2.
                pattern(0, ch1=66),
            ],
        }

        r = evaluate_piece_data(data, [0])

        self.assertEqual(r["enabled_refs"], 0)
        self.assertEqual(r["audible_refs"], 0)
        self.assertEqual(r["available_note_slots"], 0)

        self.assertEqual(r["enabled_channel_utilization"], 0.0)
        self.assertEqual(r["audible_channel_utilization"], 0.0)

    def test_missing_pattern_is_reported(self):
        data = {
            "sfx": [],
            "music": [
                pattern(0)
            ],
        }

        r = evaluate_piece_data(data, [0, 99])

        self.assertEqual(r["pattern_count"], 1)
        self.assertEqual(r["missing_pattern_ids"], [99])


if __name__ == "__main__":
    unittest.main(verbosity=2)
