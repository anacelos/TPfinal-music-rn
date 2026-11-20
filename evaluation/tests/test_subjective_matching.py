import copy
import csv
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import scipy

import subjective.match_hungarian as matching


def note(pitch=0, volume=0, effect=0):
    return {"pitch": pitch, "volume": volume, "instrument": 0, "effect": effect}


def pattern(notes, ticks=32, speed=8):
    return {"pattern_ticks": ticks, "ref_speed": speed, "channels": {
        "ch1": {"active": True, "speed": speed, "notes_unrolled": notes},
    }}


def song(tid, song_id, low=60, high=62):
    return {"tid": tid, "song_id": song_id, "patterns": [
        pattern([note(low, 7), note(), note(high, 7), note()]),
    ]}


def table_item(value, index=0, groove=None):
    # log1p(onset_rate) = value, making all five transformed columns known.
    return matching.SongFeatures(
        index, "game", str(index), f"{index:064x}",
        np.array([value, value, value, np.expm1(value), value], dtype=np.float64),
        np.zeros(32) if groove is None else np.asarray(groove, dtype=np.float64),
    )


class TestFrozenMatchingMath(unittest.TestCase):
    def test_scalar_distance_known_values_and_symmetry(self):
        np.testing.assert_allclose(
            matching.scalar_distance(0, np.array([0, 1, -1, 2, 3])),
            [0, 0.5, 0.5, 2 / 3, 3 / 4], rtol=0, atol=1e-15,
        )
        self.assertEqual(matching.scalar_distance(2, -1), matching.scalar_distance(-1, 2))

    def test_scaler_known_train_quantiles_and_log_before_fit(self):
        train = [table_item(v, i) for i, v in enumerate([0, 2, 4, 6])]
        scaler = matching.fit_scaler(train)
        for feature in matching.SCALAR_FEATURES:
            self.assertAlmostEqual(scaler[feature]["median_train"], 3)
            self.assertAlmostEqual(scaler[feature]["q25_train"], 1.5)
            self.assertAlmostEqual(scaler[feature]["q75_train"], 4.5)
            self.assertAlmostEqual(scaler[feature]["scale"], 3)
            self.assertEqual(scaler[feature]["scale_source"], "iqr")
        self.assertEqual(scaler["onset_rate"]["transform"], "log1p")
        np.testing.assert_allclose(matching.normalize([table_item(8)], scaler), np.full((1, 5), 5 / 3))

    def test_scaler_falls_back_to_population_std(self):
        train = [table_item(0, i) for i in range(7)] + [table_item(8, 7)]
        scaler = matching.fit_scaler(train)
        for feature in matching.SCALAR_FEATURES:
            self.assertEqual(scaler[feature]["iqr_train"], 0)
            self.assertEqual(scaler[feature]["scale_source"], "std")
            self.assertAlmostEqual(scaler[feature]["scale"], np.sqrt(7))
            self.assertFalse(scaler[feature]["constant"])

    def test_scaler_constant_feature_uses_one_and_records_flag(self):
        scaler = matching.fit_scaler([table_item(2, i) for i in range(4)])
        for parameters in scaler.values():
            self.assertEqual(parameters["scale"], 1)
            self.assertEqual(parameters["scale_source"], "one")
            self.assertTrue(parameters["constant"])
            self.assertEqual(parameters["n_train"], 4)

    def test_pitch_family(self):
        costs = matching.cost_components(np.zeros(5), [1, 2, 3, 0, 0], np.zeros(32), np.zeros(32))
        self.assertAlmostEqual(costs["pitch_entropy_cost"], 1 / 2)
        self.assertAlmostEqual(costs["pitch_range_cost"], 2 / 3)
        self.assertAlmostEqual(costs["pitch_center_cost"], 3 / 4)
        self.assertAlmostEqual(costs["pitch_cost"], 23 / 36)
        self.assertEqual(costs["rhythm_cost"], 0)
        self.assertEqual(costs["texture_cost"], 0)

    def test_rhythm_family_uses_mean_groove_not_gc(self):
        costs = matching.cost_components(np.zeros(5), [0, 0, 0, 3, 0], np.zeros(32), np.full(32, 0.5))
        self.assertEqual(costs["onset_rate_cost"], 0.75)
        self.assertEqual(costs["groove_profile_cost"], 0.5)
        self.assertEqual(costs["rhythm_cost"], 0.625)
        self.assertEqual(costs["pitch_cost"], 0)

    def test_texture_family(self):
        costs = matching.cost_components(np.zeros(5), [0, 0, 0, 0, 3], np.zeros(32), np.zeros(32))
        self.assertEqual(costs["texture_cost"], 0.75)
        self.assertEqual(costs["audible_channel_utilization_cost"], 0.75)

    def test_total_cost_known_fixed_family_weights(self):
        costs = matching.cost_components(np.zeros(5), [1, 2, 3, 4, 5], np.zeros(32), np.ones(32))
        self.assertAlmostEqual(costs["total_cost"], 427 / 540)
        self.assertAlmostEqual(costs["total_cost"], (23 / 36 + 0.9 + 5 / 6) / 3)

    def test_feature_extraction_single_pattern_and_pitch_center(self):
        features = matching.extract_features(song("r", 0))
        expected = np.zeros(32)
        expected[[0, 16]] = 1
        np.testing.assert_array_equal(features.groove, expected)
        np.testing.assert_allclose(features.scalars, [1, 2, 61, 2 / (32 * 366 / 44100), 0.125])

    def test_mean_groove_multiple_patterns_ignores_zero_duration(self):
        s = song("r", 0)
        s["patterns"].append(pattern([], ticks=0))
        s["patterns"].append(pattern([note(), note(64, 7), note(), note(65, 7)]))
        expected = np.zeros(32)
        expected[[0, 8, 16, 24]] = 0.5
        np.testing.assert_array_equal(matching.extract_features(s).groove, expected)

    def test_groove_preserves_sustained_state_across_patterns(self):
        s = {"patterns": [pattern([note(60, 7)], ticks=8), pattern([note(60, 7)], ticks=8)]}
        expected = np.zeros(32)
        expected[0] = 0.5  # Only the first pattern starts a new onset.
        np.testing.assert_array_equal(matching.extract_features(s).groove, expected)

    def test_retrigger_effect_uses_existing_onset_semantics(self):
        s = {"patterns": [pattern([note(60, 7, 3), note(60, 7, 3)], ticks=16)]}
        expected = np.zeros(32)
        expected[[0, 16]] = 1
        np.testing.assert_array_equal(matching.extract_features(s).groove, expected)

    def test_missing_or_nonpositive_duration_features_fail_without_imputation(self):
        for s in ({"patterns": []}, {"patterns": [pattern([note()], ticks=32)]}):
            with self.subTest(song=s), self.assertRaises(ValueError):
                matching.extract_features(s)

    def test_cost_matrix_blocking_preserves_component_values(self):
        train = [table_item(v, i) for i, v in enumerate([0, 2, 4, 6])]
        real = [table_item(1, 10), table_item(3, 11)]
        generated = [table_item(4, 12, np.ones(32)), table_item(2, 13)]
        scaler = matching.fit_scaler(train)
        a = matching.build_cost_matrix(real, generated, scaler, block_size=1)
        b = matching.build_cost_matrix(real, generated, scaler, block_size=64)
        np.testing.assert_array_equal(a, b)
        for i, r in enumerate(real):
            for j, g in enumerate(generated):
                expected = matching.cost_components(
                    matching.normalize([r], scaler)[0], matching.normalize([g], scaler)[0], r.groove, g.groove,
                )["total_cost"]
                self.assertEqual(a[i, j], expected)

    def test_hungarian_known_global_optimum(self):
        matrix = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]], dtype=float)
        rows, columns = matching.solve_assignment(matrix)
        self.assertEqual(matrix[rows, columns].sum(), 5)
        self.assertEqual(list(columns), [1, 0, 2])

    def test_hungarian_beats_row_greedy_and_is_one_to_one(self):
        matrix = np.array([[0.1, 0.2], [0.2, 1.0]])
        rows, columns = matching.solve_assignment(matrix)
        self.assertAlmostEqual(matrix[rows, columns].sum(), 0.4)
        self.assertGreater(matrix[0, 0] + matrix[1, 1], matrix[rows, columns].sum())
        self.assertEqual(set(rows), {0, 1})
        self.assertEqual(set(columns), {0, 1})

    def test_solver_rejects_incomplete_or_nonfinite_matrices(self):
        for matrix in (np.empty((0, 0)), np.ones((2, 3)), np.array([[np.nan]]), np.array([[np.inf]])):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                matching.solve_assignment(matrix)


class TestMatchingPipeline(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.train = self.root / "train.jsonl"
        self.real = self.root / "real.jsonl"
        self.generated = self.root / "generated.jsonl"
        self.output = self.root / "matching"
        self.train_songs = [song("train", i, low, high) for i, (low, high) in enumerate([(24, 28), (36, 43), (48, 55), (60, 62)])]
        self.real_songs = [song("real_game", 0, 24, 28), song("real_game", 1, 60, 62)]
        self.generated_songs = [song("markov_seed42", 0, 60, 62), song("markov_seed42", 1, 24, 28)]
        for path, songs in ((self.train, self.train_songs), (self.real, self.real_songs), (self.generated, self.generated_songs)):
            self.write_jsonl(path, songs)
        self.counts = {"train": 4, "real": 2, "generated": 2}

    def write_jsonl(self, path, songs):
        path.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in songs), encoding="utf-8")

    def run_matching(self, output=None, **kwargs):
        return matching.run_matching(
            self.train, self.real, self.generated, output or self.output, "synthetic",
            expected_counts=self.counts, **kwargs,
        )

    def read_rows(self, output=None):
        with ((output or self.output) / "matching.csv").open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_output_manifest_config_hashes_and_full_coverage(self):
        before = {p: p.read_bytes() for p in (self.train, self.real, self.generated)}
        summary = self.run_matching()
        rows = self.read_rows()
        config = json.loads((self.output / "matching_config.json").read_text())
        self.assertEqual(set(self.output.iterdir()), {
            self.output / "matching.csv", self.output / "matching_config.json", self.output / "matching_summary.json",
        })
        self.assertEqual(config["protocol"], "symbolic-balanced-v1")
        self.assertEqual(config["family_weights"], {"pitch": 1 / 3, "rhythm": 1 / 3, "texture": 1 / 3})
        self.assertEqual(config["features"]["transforms"]["onset_rate"], "log1p")
        self.assertNotIn("duration_seconds", config["features"]["scalars_in_order"])
        self.assertNotIn("groove_consistency", config["features"]["scalars_in_order"])
        self.assertEqual(config["versions"], {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__})
        for role, path in (("train", self.train), ("real", self.real), ("generated", self.generated)):
            self.assertEqual(config["inputs"][role]["sha256"], hashlib.sha256(before[path]).hexdigest())
            self.assertEqual(config["inputs"][role]["path"], str(path))
            self.assertEqual(config["inputs"][role]["count"], self.counts[role])
            self.assertEqual(path.read_bytes(), before[path])
        self.assertEqual(config["implementation_sha256"], hashlib.sha256(Path(matching.__file__).read_bytes()).hexdigest())
        self.assertEqual(config["metrics_source_sha256"], hashlib.sha256((matching.EVALUATION_DIR / "pico8_metrics.py").read_bytes()).hexdigest())
        self.assertEqual(summary["pairs"], 2)
        self.assertEqual(summary["unique_real"], 2)
        self.assertEqual(summary["unique_generated"], 2)
        self.assertEqual(summary["discarded_real"], 0)
        self.assertEqual(summary["discarded_generated"], 0)
        self.assertEqual({r["real_index"] for r in rows}, {"0", "1"})
        self.assertEqual({r["generated_index"] for r in rows}, {"0", "1"})
        self.assertEqual(len({r["generated_id"] for r in rows}), 2)
        self.assertEqual(rows[0]["generated_index"], "1")  # Never match merely by line position.
        self.assertEqual(rows[1]["generated_index"], "0")
        for row in rows:
            self.assertEqual(set(row), set(matching.CSV_FIELDS))
            self.assertEqual(row["system_name"], "synthetic")
            self.assertEqual(float(row["total_cost"]), 0)
            self.assertEqual(row["generated_id"], "sha256:" + row["generated_record_sha256"])

    def test_scaler_uses_only_train_even_when_real_generated_change(self):
        self.run_matching()
        first = json.loads((self.output / "matching_config.json").read_text())["normalization"]
        self.write_jsonl(self.real, [song("r", 0, 0, 1), song("r", 1, 62, 63)])
        self.write_jsonl(self.generated, [song("g", 0, 1, 2), song("g", 1, 2, 3)])
        output = self.root / "second"
        fit = matching.fit_scaler
        with patch.object(matching, "fit_scaler", wraps=fit) as spy:
            self.run_matching(output=output)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(len(spy.call_args.args[0]), 4)
        self.assertEqual({item.tid for item in spy.call_args.args[0]}, {"train"})
        second = json.loads((output / "matching_config.json").read_text())["normalization"]
        self.assertEqual(first, second)

    def test_generated_order_does_not_change_assignment_by_identity(self):
        self.run_matching()
        first = {r["real_song_id"]: (r["generated_id"], r["total_cost"]) for r in self.read_rows()}
        self.write_jsonl(self.generated, list(reversed(self.generated_songs)))
        output = self.root / "reordered"
        self.run_matching(output=output)
        second = {r["real_song_id"]: (r["generated_id"], r["total_cost"]) for r in self.read_rows(output)}
        self.assertEqual(first, second)

    def test_generated_reordering_in_exact_ties_is_deterministic(self):
        self.write_jsonl(self.real, [song("r", 0), song("r", 1)])
        self.write_jsonl(self.generated, [song("g", 0), song("g", 1)])
        self.run_matching()
        first = {r["real_song_id"]: r["generated_id"] for r in self.read_rows()}
        self.write_jsonl(self.generated, [song("g", 1), song("g", 0)])
        output = self.root / "reordered"
        self.run_matching(output=output)
        self.assertEqual(first, {r["real_song_id"]: r["generated_id"] for r in self.read_rows(output)})

    def test_real_logical_identity_duplicates_are_rejected(self):
        duplicate = {**self.real_songs[1], "song_id": 0}
        self.write_jsonl(self.real, [self.real_songs[0], duplicate])
        with self.assertRaisesRegex(ValueError, "identidade real duplicada"):
            self.run_matching()
        self.assertFalse(self.output.exists())

    def test_missing_real_identity_is_rejected(self):
        for key in ("tid", "song_id"):
            with self.subTest(key=key):
                songs = copy.deepcopy(self.real_songs)
                del songs[0][key]
                self.write_jsonl(self.real, songs)
                with self.assertRaisesRegex(ValueError, "tid e song_id não vazios"):
                    self.run_matching()

    def test_canonical_duplicate_generated_records_are_rejected(self):
        s = self.generated_songs[0]
        self.generated.write_text(json.dumps(s) + "\n" + json.dumps(s, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "identidade generated duplicada"):
            self.run_matching()

    def test_generated_missing_or_nonunique_song_id_is_not_invented(self):
        for mode in ("missing", "repeated"):
            with self.subTest(mode=mode):
                songs = copy.deepcopy(self.generated_songs)
                for s in songs:
                    if mode == "missing":
                        s.pop("song_id")
                        s.pop("tid")
                    else:
                        s["song_id"] = 99
                self.write_jsonl(self.generated, songs)
                output = self.root / mode
                self.run_matching(output=output)
                rows = self.read_rows(output)
                self.assertEqual(len({r["generated_id"] for r in rows}), 2)
                self.assertEqual({r["generated_song_id"] for r in rows}, {"" if mode == "missing" else "99"})

    def test_count_validation_for_every_input_and_official_defaults(self):
        for role, path in (("train", self.train), ("real", self.real), ("generated", self.generated)):
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "esperadas.*encontradas"):
                matching.load_dataset(path, role, self.counts[role] + 1)
        with self.assertRaisesRegex(ValueError, "esperadas 8669"):
            matching.run_matching(self.train, self.real, self.generated, self.output, "synthetic")
        self.assertFalse(self.output.exists())

    def test_physical_index_is_only_provenance_and_blank_lines_are_hashed(self):
        text = self.generated.read_text()
        self.generated.write_text("\n" + text, encoding="utf-8")
        self.run_matching()
        rows = self.read_rows()
        self.assertEqual({row["generated_index"] for row in rows}, {"1", "2"})
        config = json.loads((self.output / "matching_config.json").read_text())
        self.assertEqual(config["inputs"]["generated"]["sha256"], hashlib.sha256(self.generated.read_bytes()).hexdigest())

    def test_output_directory_is_never_overwritten(self):
        self.output.mkdir()
        original = self.output / "keep.txt"
        original.write_bytes(b"keep")
        with self.assertRaisesRegex(FileExistsError, "já existe"):
            self.run_matching()
        self.assertEqual(original.read_bytes(), b"keep")

    def test_failed_publication_leaves_no_success_directory_or_temporaries(self):
        with patch.object(matching.os, "replace", side_effect=OSError("simulated publication error")):
            with self.assertRaisesRegex(OSError, "publication error"):
                self.run_matching()
        self.assertFalse(self.output.exists())
        self.assertEqual(set(self.root.iterdir()), {self.train, self.real, self.generated})


if __name__ == "__main__":
    unittest.main(verbosity=2)
