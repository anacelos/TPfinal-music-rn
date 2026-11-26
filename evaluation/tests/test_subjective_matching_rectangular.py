import copy
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from subjective import match_hungarian as frozen
from subjective import match_hungarian_rectangular as rectangular


def song(tid, sid, low=24, high=28):
    return {"tid": tid, "song_id": sid, "system": rectangular.SYSTEM_NAME, "patterns": [{
        "pattern_ticks": 32, "ref_speed": 8, "channels": {"ch1": {
            "active": True, "speed": 8, "notes_unrolled": [
                {"pitch": p, "instrument": 0, "volume": v, "effect": 0}
                for p, v in [(low, 7), (0, 0), (high, 7), (0, 0)]
            ],
        }},
    }]}


def official_silence():
    """The inspected complete record, including metadata and its eight tokens."""
    return {
        "tid": "lstm_seed42", "song_id": 1416, "system": rectangular.SYSTEM_NAME,
        "start_pattern": 0, "end_pattern": 0, "loop_start_pattern": 0,
        "end_reason": "stop", "num_patterns": 1,
        "token_ids": [1, 4, 3, 21, 549, 308, 1130, 2],
        "patterns": [{"pattern_id": 0, "ref_speed": 16, "pattern_ticks": 512, "channels": {
            "ch1": {"active": True, "block_id": "0", "is_repeat": False, "speed": 16,
                    "notes_unrolled": [{"pitch": 0, "instrument": 0, "volume": 0, "effect": 0}
                                       for _ in range(32)]},
            **{f"ch{i}": {"active": False, "speed": None, "notes_unrolled": []} for i in (2, 3, 4)},
        }}],
    }


class TestRectangularSolver(unittest.TestCase):
    def test_global_optimum_covers_all_generated_and_beats_greedy(self):
        matrix = np.array([[0.1, 0.2], [0.2, 1.0], [5.0, 5.0]])
        rows, columns = rectangular.solve_rectangular_assignment(matrix)
        brute_force = min(sum(matrix[r, c] for c, r in enumerate(choice))
                          for choice in itertools.permutations(range(3), 2))
        self.assertAlmostEqual(matrix[rows, columns].sum(), brute_force)
        self.assertAlmostEqual(brute_force, 0.4)
        self.assertEqual(set(rows), {0, 1})
        self.assertEqual(set(columns), {0, 1})

    def test_ties_are_repeatable_without_perturbing_matrix(self):
        matrix = np.ones((3, 2))
        before = matrix.copy()
        first = rectangular.solve_rectangular_assignment(matrix)
        second = rectangular.solve_rectangular_assignment(matrix)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(matrix, before)

    def test_invalid_shape_or_nonfinite_costs_fail(self):
        for matrix in (np.empty((0, 0)), np.ones((1, 2)), np.ones(3),
                       np.array([[0.0], [np.nan]]), np.array([[np.inf], [0.0]])):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                rectangular.solve_rectangular_assignment(matrix)

    def test_frozen_solver_still_rejects_rectangular_input(self):
        with self.assertRaises(ValueError):
            frozen.solve_assignment(np.ones((3, 2)))


class TestRectangularPipeline(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.train = self.root / "train.jsonl"
        self.real = self.root / "real.jsonl"
        self.generated = self.root / "generated.jsonl.gz"
        self.output = self.root / "matching"
        self.train_songs = [song("train", i, a, b) for i, (a, b) in enumerate(
            [(12, 19), (24, 28), (36, 43), (60, 62)])]
        self.real_songs = [song("real", 0, 24, 28), song("real", 1, 36, 43), song("real", 2, 60, 62)]
        self.generated_songs = [song("lstm_seed42", 0, 60, 62), official_silence(),
                                song("lstm_seed42", 1, 24, 28)]
        self.write_records(self.train, self.train_songs)
        self.write_records(self.real, self.real_songs)
        self.write_records(self.generated, self.generated_songs)
        self.counts = {"train": 4, "real": 3, "generated": 3}

    def write_records(self, path, records, *, leading_blank=False):
        raw = (b"\n" if leading_blank else b"") + b"".join(
            (json.dumps(s, ensure_ascii=False) + "\n").encode() for s in records)
        path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)

    def run_matching(self, output=None, **kwargs):
        return rectangular.run_matching(self.train, self.real, self.generated,
                                        output or self.output, expected_counts=self.counts, **kwargs)

    def read_rows(self, filename="matching.csv", output=None):
        with ((output or self.output) / filename).open(newline="") as handle:
            return list(csv.DictReader(handle))

    def test_official_exclusion_hash_is_the_inspected_complete_record(self):
        self.assertEqual(rectangular.canonical_record_sha256(official_silence()),
                         rectangular.APPROVED_EXCLUSION.record_sha256)

    def test_coverage_exclusion_unmatched_and_all_costs(self):
        summary = self.run_matching()
        rows = self.read_rows()
        self.assertEqual(summary["pairs"], 2)
        self.assertEqual(summary["cost_matrix"]["shape"], [3, 2])
        self.assertEqual(len({(r["real_tid"], r["real_song_id"]) for r in rows}), 2)
        self.assertEqual(len({r["generated_id"] for r in rows}), 2)
        self.assertEqual({(r["real_song_id"], r["generated_song_id"]) for r in rows}, {("0", "1"), ("2", "0")})
        for row in rows:
            self.assertEqual(set(row), set(frozen.CSV_FIELDS))
            self.assertTrue(all(np.isfinite(float(row[k])) for k in frozen.COST_FIELDS))
            self.assertEqual(float(row["total_cost"]), 0)
        excluded = self.read_rows("excluded_generated.csv")
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["generated_song_id"], "1416")
        self.assertEqual(excluded[0]["generated_index"], "1")
        self.assertEqual(excluded[0]["generated_record_sha256"], rectangular.APPROVED_EXCLUSION.record_sha256)
        self.assertEqual(excluded[0]["reason"], rectangular.EXCLUSION_REASON)
        unmatched = self.read_rows("unmatched_real.csv")
        self.assertEqual(len(unmatched), 1)
        self.assertEqual(unmatched[0]["real_record_id"], "real:real:1")
        self.assertEqual(unmatched[0]["reason"], rectangular.UNMATCHED_REASON)

    def test_frozen_math_scaler_and_inputs_are_unchanged(self):
        paths = [self.train, self.real, self.generated, Path(frozen.__file__)]
        before = {p: p.read_bytes() for p in paths}
        with patch.object(frozen, "fit_scaler", wraps=frozen.fit_scaler) as fit, \
                patch.object(frozen, "build_cost_matrix", wraps=frozen.build_cost_matrix) as build, \
                patch.object(frozen, "cost_components", wraps=frozen.cost_components) as components:
            self.run_matching()
        self.assertEqual(fit.call_count, 1)
        self.assertEqual({item.tid for item in fit.call_args.args[0]}, {"train"})
        self.assertEqual(len(fit.call_args.args[0]), 4)
        self.assertEqual(build.call_count, 1)
        self.assertGreater(components.call_count, 0)
        config = json.loads((self.output / "matching_config.json").read_text())
        self.assertEqual(config["normalization"]["parameters"],
                         frozen.fit_scaler(frozen.load_dataset(self.train, "train", 4).items))
        self.assertEqual(config["family_weights"], dict(pitch=1 / 3, rhythm=1 / 3, texture=1 / 3))
        self.assertEqual(config["features"]["scalars_in_order"], list(frozen.SCALAR_FEATURES))
        self.assertEqual(config["features"]["transforms"], frozen.TRANSFORMS)
        for p in paths:
            self.assertEqual(p.read_bytes(), before[p])

    def test_config_provenance_and_output_checksums(self):
        self.run_matching()
        config = json.loads((self.output / "matching_config.json").read_text())
        self.assertEqual(config["protocol"], "symbolic-balanced-rectangular-v1")
        self.assertEqual(config["system_name"], "lstm-grid_block-k10")
        self.assertEqual(config["counts"], dict(train=4, real=3, generated_original=3,
                                              generated_eligible=2, excluded_generated=1, unmatched_real=1))
        for role, path in (("train", self.train), ("real", self.real), ("generated", self.generated)):
            self.assertEqual(config["inputs"][role]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(config["inputs"]["generated"]["content_sha256"],
                         hashlib.sha256(gzip.decompress(self.generated.read_bytes())).hexdigest())
        self.assertEqual(config["frozen_implementation_sha256"], frozen.sha256_file(frozen.__file__))
        for name, digest in config["outputs_sha256"].items():
            self.assertEqual(hashlib.sha256((self.output / name).read_bytes()).hexdigest(), digest)
        self.assertEqual({p.name for p in self.output.iterdir()}, {
            "matching.csv", "matching_config.json", "matching_summary.json",
            "excluded_generated.csv", "unmatched_real.csv",
        })

    def test_all_five_outputs_are_byte_deterministic(self):
        self.run_matching()
        other = self.root / "second"
        self.run_matching(output=other)
        for path in self.output.iterdir():
            self.assertEqual(path.read_bytes(), (other / path.name).read_bytes())

    def test_reordered_inputs_preserve_assignment_and_unmatched_identity(self):
        self.run_matching()
        first = {(r["real_tid"], r["real_song_id"]): r["generated_id"] for r in self.read_rows()}
        self.write_records(self.real, list(reversed(self.real_songs)))
        self.write_records(self.generated, list(reversed(self.generated_songs)))
        other = self.root / "reordered"
        self.run_matching(output=other)
        second = {(r["real_tid"], r["real_song_id"]): r["generated_id"] for r in self.read_rows(output=other)}
        self.assertEqual(first, second)
        self.assertEqual(self.read_rows("unmatched_real.csv")[0]["real_record_id"],
                         self.read_rows("unmatched_real.csv", other)[0]["real_record_id"])

    def test_plain_and_gzip_preserve_pairs_and_decompressed_hash(self):
        self.run_matching()
        old = self.output
        self.generated = self.root / "generated.jsonl"
        self.write_records(self.generated, self.generated_songs)
        other = self.root / "plain"
        self.run_matching(output=other)
        for name in ("matching.csv", "excluded_generated.csv", "unmatched_real.csv"):
            self.assertEqual((old / name).read_bytes(), (other / name).read_bytes())
        a = json.loads((old / "matching_config.json").read_text())["inputs"]["generated"]
        b = json.loads((other / "matching_config.json").read_text())["inputs"]["generated"]
        self.assertEqual(a["content_sha256"], b["content_sha256"])
        self.assertNotEqual(a["sha256"], b["sha256"])

    def test_blank_line_is_only_physical_provenance(self):
        self.write_records(self.generated, self.generated_songs, leading_blank=True)
        self.run_matching()
        self.assertEqual(self.read_rows("excluded_generated.csv")[0]["generated_index"], "2")
        self.assertEqual({r["generated_index"] for r in self.read_rows()}, {"1", "3"})

    def test_changed_exclusion_hash_is_rejected(self):
        self.generated_songs[1]["extra_metadata"] = "changed"
        self.write_records(self.generated, self.generated_songs)
        with self.assertRaisesRegex(ValueError, "identity/hash mismatch"):
            self.run_matching()
        self.assertFalse(self.output.exists())

    def test_hash_with_wrong_exclusion_identity_is_rejected(self):
        wrong = rectangular.ExclusionSpec(1417, rectangular.APPROVED_EXCLUSION.record_sha256)
        with self.assertRaisesRegex(ValueError, "identity/hash mismatch"):
            self.run_matching(exclusion=wrong)

    def test_valid_record_cannot_be_excluded_even_with_matching_hash(self):
        valid = song("lstm_seed42", 1416)
        self.generated_songs[1] = valid
        self.write_records(self.generated, self.generated_songs)
        spec = rectangular.ExclusionSpec(1416, rectangular.canonical_record_sha256(valid))
        with self.assertRaisesRegex(ValueError, "no audible onsets"):
            self.run_matching(exclusion=spec)

    def test_missing_exclusion_is_rejected(self):
        self.generated_songs[1] = song("lstm_seed42", 2)
        self.write_records(self.generated, self.generated_songs)
        with self.assertRaisesRegex(ValueError, "one approved exclusion"):
            self.run_matching()

    def test_additional_silent_record_fails_without_imputation_or_discard(self):
        other_silence = official_silence()
        other_silence["song_id"] = 22
        self.generated_songs[0] = other_silence
        self.write_records(self.generated, self.generated_songs)
        with self.assertRaisesRegex(ValueError, "pitch ausente"):
            self.run_matching()
        self.assertFalse(self.output.exists())

    def test_wrong_system_is_rejected_including_on_excluded_record(self):
        for index in (0, 1):
            with self.subTest(index=index):
                songs = copy.deepcopy(self.generated_songs)
                songs[index]["system"] = "lstm-grid_block-t1"
                self.write_records(self.generated, songs)
                with self.assertRaisesRegex(ValueError, "unexpected Neural system"):
                    self.run_matching()

    def test_duplicate_records_and_duplicate_song_ids_are_rejected(self):
        for exact in (True, False):
            with self.subTest(exact=exact):
                songs = copy.deepcopy(self.generated_songs)
                songs[2] = dict(songs[0]) if exact else dict(songs[2], song_id=songs[0]["song_id"])
                self.write_records(self.generated, songs)
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    self.run_matching()

    def test_real_duplicates_are_rejected_by_frozen_loader(self):
        self.real_songs[2] = self.real_songs[0]
        self.write_records(self.real, self.real_songs)
        with self.assertRaisesRegex(ValueError, "duplicada"):
            self.run_matching()

    def test_counts_are_checked_for_every_input_and_cli_defaults_are_official(self):
        self.assertEqual(rectangular.EXPECTED_COUNTS, dict(train=8669, real=1854, generated=1854))
        for role in self.counts:
            with self.subTest(role=role):
                path = getattr(self, role)
                records = getattr(self, role + "_songs")
                self.write_records(path, records[:-1])
                with self.assertRaises(ValueError):
                    self.run_matching()
                self.write_records(path, records)
        self.assertFalse(self.output.exists())

    def test_truncated_gzip_fails_without_publication(self):
        self.generated.write_bytes(self.generated.read_bytes()[:-4])
        with self.assertRaises((EOFError, OSError)):
            self.run_matching()
        self.assertFalse(self.output.exists())

    def test_invalid_json_and_nonfinite_metadata_are_rejected(self):
        for raw in (b"not JSON\n", b'{"system":"lstm-grid_block-k10","song_id":0,"x":NaN}\n'):
            self.generated.write_bytes(gzip.compress(raw, mtime=0))
            with self.assertRaises(ValueError):
                self.run_matching()
        self.assertFalse(self.output.exists())

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            self.run_matching()
        self.assertEqual(sentinel.read_bytes(), b"keep")

    def test_publication_failure_leaves_no_success_or_staging_directory(self):
        with patch.object(rectangular.os, "replace", side_effect=OSError("publication failure")):
            with self.assertRaisesRegex(OSError, "publication failure"):
                self.run_matching()
        self.assertEqual(set(self.root.iterdir()), {self.train, self.real, self.generated})


if __name__ == "__main__":
    unittest.main(verbosity=2)
