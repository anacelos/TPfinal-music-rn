import random
import unittest

from markov_baseline import (
    CHANNELS,
    SILENCE,
    ChannelMarkov,
    generate_song,
    note_to_state,
    state_to_note,
)


def note(pitch=0, instrument=0, volume=0, effect=0):
    return {
        "pitch": pitch,
        "instrument": instrument,
        "volume": volume,
        "effect": effect,
    }


class TestMarkovStateConversion(unittest.TestCase):
    def test_silent_note_becomes_silence_state(self):
        self.assertEqual(
            note_to_state(note(pitch=40, instrument=3, volume=0, effect=2)),
            SILENCE,
        )

    def test_audible_note_roundtrip(self):
        original = note(
            pitch=37,
            instrument=5,
            volume=6,
            effect=2,
        )

        state = note_to_state(original)

        self.assertEqual(state, (37, 5, 6, 2))
        self.assertEqual(state_to_note(state), original)


class TestChannelMarkov(unittest.TestCase):
    def test_observe_counts_first_order_transitions(self):
        model = ChannelMarkov()

        a = note(10, 1, 5, 0)
        b = note(12, 1, 5, 0)

        model.observe([a, b, a])

        state_a = note_to_state(a)
        state_b = note_to_state(b)

        self.assertEqual(
            model.transitions[state_a][state_b],
            1,
        )
        self.assertEqual(
            model.transitions[state_b][state_a],
            1,
        )
        self.assertEqual(model.transition_count, 2)

    def test_generation_is_reproducible_with_same_seed(self):
        model = ChannelMarkov()

        model.observe(
            [
                note(10, 0, 5, 0),
                note(12, 0, 5, 0),
                note(14, 0, 5, 0),
                note(10, 0, 5, 0),
            ]
        )

        generated_a = model.generate(
            length=16,
            temperature=1.0,
            reactivation_prob=0.25,
            rng=random.Random(42),
        )

        generated_b = model.generate(
            length=16,
            temperature=1.0,
            reactivation_prob=0.25,
            rng=random.Random(42),
        )

        self.assertEqual(generated_a, generated_b)
        self.assertEqual(len(generated_a), 16)


class TestCanonicalSongGeneration(unittest.TestCase):
    def test_generate_song_preserves_structure_but_removes_block_identity(self):
        template = {
            "tid": "real_game",
            "song_id": 9,
            "wav_file": "original.wav",
            "uses_custom_instruments": False,
            "blocks_dictionary": {
                "7": {
                    "notes": [note(20, 0, 5, 0)]
                }
            },
            "patterns": [
                {
                    "step_index": 0,
                    "pattern_id": 3,
                    "flags": 0,
                    "ref_speed": 16,
                    "pattern_ticks": 512,
                    "channels": {
                        "ch1": {
                            "active": True,
                            "original_sfx_id": 7,
                            "block_id": 7,
                            "is_repeat": True,
                            "speed": 16,
                            "notes_unrolled": [
                                note(20, 0, 5, 0),
                                note(21, 0, 5, 0),
                            ],
                        },
                        "ch2": {
                            "active": False,
                            "original_sfx_id": None,
                            "block_id": None,
                            "is_repeat": False,
                            "speed": None,
                            "notes_unrolled": [],
                        },
                        "ch3": {
                            "active": False,
                            "original_sfx_id": None,
                            "block_id": None,
                            "is_repeat": False,
                            "speed": None,
                            "notes_unrolled": [],
                        },
                        "ch4": {
                            "active": False,
                            "original_sfx_id": None,
                            "block_id": None,
                            "is_repeat": False,
                            "speed": None,
                            "notes_unrolled": [],
                        },
                    },
                }
            ],
        }

        models = {
            channel: ChannelMarkov()
            for channel in CHANNELS
        }

        models["ch1"].observe(
            [
                note(30, 2, 6, 0),
                note(32, 2, 6, 0),
            ]
        )

        generated = generate_song(
            template=template,
            models=models,
            generated_index=0,
            temperature=1.0,
            reactivation_prob=0.25,
            seed=42,
            rng=random.Random(42),
        )

        self.assertEqual(generated["tid"], "markov_seed42")
        self.assertEqual(generated["song_id"], 0)
        self.assertIsNone(generated["wav_file"])
        self.assertEqual(generated["blocks_dictionary"], {})

        pattern = generated["patterns"][0]

        self.assertEqual(pattern["pattern_ticks"], 512)
        self.assertEqual(pattern["step_index"], 0)

        ch1 = pattern["channels"]["ch1"]

        self.assertTrue(ch1["active"])
        self.assertEqual(ch1["speed"], 16)
        self.assertEqual(len(ch1["notes_unrolled"]), 2)
        self.assertIsNone(ch1["original_sfx_id"])
        self.assertIsNone(ch1["block_id"])
        self.assertFalse(ch1["is_repeat"])

        ch2 = pattern["channels"]["ch2"]

        self.assertFalse(ch2["active"])
        self.assertEqual(ch2["notes_unrolled"], [])
        self.assertIsNone(ch2["block_id"])


if __name__ == "__main__":
    unittest.main()
