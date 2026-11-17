import unittest

from pico8_metrics import evaluate_piece_data


def note(pitch=0, volume=0):
    return {
        "pitch": pitch,
        "instrument": 0,
        "volume": volume,
        "effect": 0,
    }


def make_sfx(sfx_id):
    notes = [note() for _ in range(32)]
    notes[0] = note(pitch=60, volume=7)

    return {
        "id": sfx_id,
        "speed": 16,
        "loop_start": 0,
        "loop_end": 0,
        "notes": notes,
    }


def make_pattern(ch1):
    return {
        "id": 0,
        "flags": 0,
        "ch1": ch1,
        "ch2": 66,
        "ch3": 67,
        "ch4": 68,
    }


class TestSynthesizerAlignment(unittest.TestCase):

    def test_channel_63_is_valid_sfx_reference(self):
        data = {
            "sfx": [make_sfx(63)],
            "music": [make_pattern(63)],
        }

        r = evaluate_piece_data(data, [0])

        self.assertEqual(r["enabled_refs"], 1)
        self.assertEqual(r["audible_refs"], 1)
        self.assertEqual(r["active_note_slots"], 1)

    def test_channel_128_is_disabled_by_dataset_synth(self):
        data = {
            "sfx": [make_sfx(0)],
            "music": [make_pattern(128)],
        }

        r = evaluate_piece_data(data, [0])

        self.assertEqual(r["enabled_refs"], 0)
        self.assertEqual(r["audible_refs"], 0)
        self.assertEqual(r["available_note_slots"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
