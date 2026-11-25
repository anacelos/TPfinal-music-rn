import itertools
import json
import unittest
from pathlib import Path

import numpy as np

from evaluation.subjective import build_final_handoff_240 as final


class TestFinalHandoff240(unittest.TestCase):
    def test_cycle_is_exactly_all_twelve_allowed_permutations(self):
        cycle = final.permutation_cycle()
        expected = {p for p in itertools.permutations(final.shared.ORIGINS) if p.index("attention") in (1, 2)}
        self.assertEqual({tuple(m[s] for s in "ABCD") for m in cycle}, expected)

    def test_every_pair_has_one_middle_check_and_prefixes_balance(self):
        maps = final.permutation_cycle() * 20
        for i in range(0, 240, 2):
            self.assertEqual([list(m.values()).index("attention") for m in maps[i:i+2]], [1, 2])
        for n in (6, 12, 24, 48, 120, 240):
            for slot in "AD":
                self.assertEqual({o: sum(m[slot] == o for m in maps[:n]) for o in final.shared.audio.ORIGINS},
                                 {o: n // 3 for o in final.shared.audio.ORIGINS})
        for slot in "BC":
            self.assertEqual(sum(m[slot] == "attention" for m in maps), 120)

    def test_assignment_uses_only_collection_order_and_is_deterministic(self):
        rows = [{"external_id": str(i), "collection_order_v2": i} for i in range(240)]
        expected = final.assignments(rows)
        self.assertEqual(final.assignments(list(reversed(rows))), expected)
        for r in rows: r.update(matching_cost=-100, duration=9999, historical_permutation="irrelevant")
        self.assertEqual(final.assignments(rows), expected)
        self.assertEqual(rows[0]["external_id"], "0")

    def test_approved_speech_is_deterministic_and_tail_is_digital_silence(self):
        path = Path('/tmp/attention-preview.mp3')
        if not path.exists(): self.skipTest('approved user input unavailable')
        first, audit = final.decode_speech(path)
        second, repeated = final.decode_speech(path)
        self.assertEqual(first, second)
        self.assertEqual(audit, repeated)
        pcm = final.shared.validate_public_wav(first)
        self.assertTrue(np.any(pcm[:audit['speech_frames']]))
        self.assertTrue(np.all(pcm[audit['speech_frames']:] == 0))
        self.assertEqual(audit['source_mp3_sha256'], final.shared.file_sha(path))
        self.assertFalse(audit['repeated'])
        self.assertEqual(len(pcm), 1323000)

    def test_public_projection_contains_no_private_mapping(self):
        path = final.context.OUTPUT / 'scenarios.jsonl'
        row = json.loads(path.read_bytes().splitlines()[0])
        p = final.public_record(row)
        self.assertEqual(set(p), final.shared.PUBLIC_FIELDS | {'collection_order_v2'})
        self.assertEqual(set(p['audios']), set('ABCD'))
        self.assertNotIn('mapping', p)

    def test_royal_mummy_is_context_not_condition_leak(self):
        row = json.loads((final.context.OUTPUT / 'scenarios.jsonl').read_bytes().splitlines()[219])
        p = final.public_record(row)
        self.assertIn('múmia real', p['description'])
        final.check_public_metadata(p)
        p['mapping'] = {'A': 'real'}
        with self.assertRaises(ValueError): final.check_public_metadata(p)


if __name__ == '__main__': unittest.main()
