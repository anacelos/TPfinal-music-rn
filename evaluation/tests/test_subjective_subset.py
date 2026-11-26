import copy
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, PngImagePlugin

from evaluation.subjective import select_subjective_subset as subset
from evaluation.subjective import build_scenario_manifest as manifest
from evaluation.subjective import export_scenario_contexts as contexts
from test_subjective_matching_rectangular import official_silence


def png_bytes():
    pixels = bytes(v for i in range(160 * 205) for v in (100 + i % 4, 120 + (i // 4) % 4, 80 + (i // 16) % 4, 252 + i % 4))
    image = Image.frombytes("RGBA", (160, 205), pixels)
    meta = PngImagePlugin.PngInfo()
    meta.add_text("private", "cartridge payload and metadata must not survive")
    output = io.BytesIO()
    image.save(output, format="PNG", pnginfo=meta)
    return output.getvalue()


class TestSubsetDecision(unittest.TestCase):
    def candidates(self):
        permutations = list(subset.PERMUTATIONS) + [subset.PERMUTATIONS[0]] * 2 + list(subset.PERMUTATIONS[1:])
        return [subset.Candidate(str(i), i + 1, p, (), "cartridge", f"images/{i}.png")
                for i, p in enumerate(permutations)]

    def test_exact_quota_prefix_and_no_extra_fields_available(self):
        candidates = self.candidates()
        selected, audit, _ = subset.scan_candidates(list(reversed(candidates)), lambda c: (b"image", "hash", ""))
        self.assertEqual(len(selected), 48)
        self.assertEqual(len(audit), 49)
        self.assertEqual([c.serve_order for c in selected], [i for i in range(1, 50) if i != 26])
        self.assertEqual(audit[25]["decision"], "quota_full")
        self.assertEqual(set(subset.Candidate.__dataclass_fields__), {
            "external_id", "serve_order", "permutation", "reasons", "visual_type", "source_relative_path"})

    def test_ineligible_candidate_is_skipped_before_quota_and_insufficient_fails(self):
        candidates = self.candidates()
        first = candidates[0]
        candidates[0] = subset.Candidate(first.external_id, first.serve_order, first.permutation,
                                          ("missing_neural",), first.visual_type, first.source_relative_path)
        selected, audit, _ = subset.scan_candidates(candidates, lambda c: (b"image", "hash", ""))
        self.assertNotIn(first.external_id, {c.external_id for c in selected})
        self.assertEqual(audit[0]["decision"], "ineligible")
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            subset.scan_candidates(candidates[:-1], lambda c: (b"image", "hash", ""))

    def test_duplicate_order_is_rejected(self):
        candidates = self.candidates()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            subset.scan_candidates(candidates + [candidates[0]], lambda c: (b"image", "hash", ""))


class TestSubsetFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.paths = {role: self.root / f"{role}{'.gz' if role == 'neural' else '.data'}" for role in subset.REFS}
        self.count = 52
        self.orders = (list(subset.PERMUTATIONS) + [subset.PERMUTATIONS[0]] * 4
                       + list(subset.PERMUTATIONS[1:]) + [subset.PERMUTATIONS[0]])
        self.real, self.markov, self.neural = [], [], []
        self.pool, self.contexts, self.drafts = [], [], []
        source_png = png_bytes()
        for i in range(self.count):
            tid = "31396" if i == 26 else str(10000 + i)
            identifier = contexts.external_id(tid, 0)
            kind = "none" if i in (27, 51) else "cartridge"
            source = "" if kind == "none" else f"cartridges/{tid}/cart.p8.png"
            if source:
                asset = self.dataset / source
                asset.parent.mkdir(parents=True)
                asset.write_bytes(source_png)
            real = dict(tid=tid, song_id=0, patterns=[{"pattern_ticks": (i + 1) * 64}])
            markov = dict(tid="markov_seed42", song_id=i, patterns=[{"pattern_ticks": (i + 1) * 128}])
            neural = (official_silence() if i == 26 else
                      dict(tid="lstm_seed42", song_id=200 + i, system=subset.NEURAL_SYSTEM,
                           patterns=[{"pattern_ticks": (i + 1) * 96}]))
            self.real.append(real)
            self.markov.append(markov)
            self.neural.append(neural)
            tier = "D" if kind == "none" else "A" if i < 30 else "B"
            description = contexts.DESCRIPTION_FALLBACK if i == 2 or kind == "none" else "Collect gems in a maze."
            context = dict(external_id=identifier, tid=tid, song_id=0, serve_order=i + 1,
                           context_priority=i + 1, context_tier=tier, public_title="",
                           public_description=description, description_fallback=description == contexts.DESCRIPTION_FALLBACK,
                           public_visual_label=contexts.VISUAL_LABELS[kind], public_visual_type=kind,
                           source_relative_path=source, public_context_heading=contexts.CONTEXT_HEADING)
            self.contexts.append(context)
            self.pool.append(dict(tid=tid, song_id=0, serve_order=i + 1, context_tier=tier))
            self.drafts.append(dict(
                identity={k: context[k] for k in ("external_id", "tid", "song_id", "serve_order", "context_tier")},
                protocol=dict(manifest_protocol=manifest.PROTOCOL, status="draft", ready_for_import=False),
                context={k: context[k] for k in contexts.PUBLIC_FIELDS if k not in ("external_id", "public_visual_source")}
                        | {"source_relative_path": source},
                presentation=dict(permutation_protocol=manifest.PERMUTATION_PROTOCOL,
                                  assignment=dict(zip(manifest.POSITIONS, self.orders[i]))),
                audios={"real": {"source_record_sha256": subset.sha(subset.canonical(real))},
                        "markov": {"generated_id": "sha256:" + subset.sha(subset.canonical(markov)),
                                   "source_record_sha256": subset.sha(subset.canonical(markov))}}))
        self.markov_costs = [0.1] * self.count
        self.neural_costs = [0.2] * self.count
        self.refresh()

    def save_json(self, role, value):
        self.paths[role].write_bytes(subset.json_bytes(value))

    def save_csv(self, role, rows):
        self.paths[role].write_bytes(subset.csv_bytes(tuple(rows[0]), rows))

    def refresh(self):
        for role, rows in (("real", self.real), ("markov", self.markov), ("neural", self.neural)):
            raw = b"".join(subset.canonical(r) + b"\n" for r in rows)
            self.paths[role].write_bytes(gzip.compress(raw, mtime=0) if role == "neural" else raw)
        self.save_csv("pool", self.pool)
        self.save_json("contexts", self.contexts)
        self.paths["manifest"].write_bytes(b"".join(subset.canonical(r) + b"\n" for r in self.drafts))
        self.save_csv("visual_inventory", [dict(external_id=c["external_id"], source_relative_path=c["source_relative_path"],
                                                 visual_type=c["public_visual_type"])
                                            for c in self.contexts if c["public_visual_type"] != "none"])
        matches = {}
        for role, records, costs in (("markov", self.markov, self.markov_costs), ("neural", self.neural, self.neural_costs)):
            rows = []
            for i, (real, generated) in enumerate(zip(self.real, records)):
                if role == "neural" and i == 26:
                    continue
                digest = subset.sha(subset.canonical(generated))
                rows.append(dict(real_index=i, real_tid=real["tid"], real_song_id=real["song_id"],
                                 real_record_sha256=subset.sha(subset.canonical(real)), generated_index=i,
                                 generated_id="sha256:" + digest, generated_record_sha256=digest,
                                 generated_song_id=generated["song_id"], total_cost=costs[i],
                                 system_name="markov" if role == "markov" else subset.NEURAL_SYSTEM))
            matches[role] = rows
            self.save_csv(role + "_matching", rows)
        self.save_csv("neural_unmatched", [dict(real_record_id=subset.UNMATCHED_REAL, real_tid="31396", real_song_id=0)])
        self.save_csv("neural_excluded", [dict(generated_song_id=1416,
                                             generated_record_sha256=subset.sha(subset.canonical(official_silence())))])
        self.rehash_configs()

    def rehash_configs(self):
        digest = lambda role: subset.sha(self.paths[role].read_bytes())
        self.save_json("pool_config", dict(protocol=contexts.POOL_PROTOCOL, context_pool_csv_sha256=digest("pool")))
        self.save_json("context_config", dict(protocol=contexts.PROTOCOL, source_pool_config_sha256=digest("pool_config"),
            source_pool_csv_sha256=digest("pool"), outputs_sha256={
                "scenario_contexts_internal.json": digest("contexts"), "visual_assets_manifest.csv": digest("visual_inventory")}))
        self.save_json("markov_config", dict(protocol=manifest.MATCHING_PROTOCOL,
                                             inputs={"generated": {"sha256": digest("markov")}}))
        self.save_json("neural_config", dict(protocol=subset.NEURAL_PROTOCOL, system_name=subset.NEURAL_SYSTEM,
            inputs={"real": {"sha256": digest("real")}, "generated": {"sha256": digest("neural"),
                "content_sha256": subset.sha(gzip.decompress(self.paths["neural"].read_bytes()))}},
            eligibility={"approved_exclusion": {"record_sha256": subset.sha(subset.canonical(official_silence()))}},
            outputs_sha256={"matching.csv": digest("neural_matching"), "excluded_generated.csv": digest("neural_excluded"),
                            "unmatched_real.csv": digest("neural_unmatched")}))
        self.save_json("manifest_config", dict(manifest_protocol=manifest.PROTOCOL,
            output_sha256={"scenario_manifest_draft.jsonl": digest("manifest")}, inputs={
                "markov_matching": {"sha256": digest("markov_matching")},
                "markov_matching_config": {"sha256": digest("markov_config")},
                "scenario_context": {"sha256": digest("contexts")},
                "scenario_context_config": {"sha256": digest("context_config")},
                "real_test": {"sha256": digest("real")}, "markov_generated": {"sha256": digest("markov")}}))

    def prepare(self):
        return subset.prepare_subset(self.paths, self.dataset, expected_count=self.count)

    def records(self, files):
        return [json.loads(line) for line in files["scenarios.jsonl"].splitlines()]

    def test_48_unique_quota_order_and_frozen_associations(self):
        files, summary = self.prepare()
        rows = self.records(files)
        self.assertEqual(summary["selected"], 48)
        self.assertEqual(summary["candidates_examined"], 51)
        self.assertEqual([summary[k] for k in ("unique_real", "unique_markov", "unique_neural")], [48, 48, 48])
        self.assertEqual(set(summary["permutation_counts"].values()), {2})
        self.assertEqual(len(summary["permutation_counts"]), 24)
        for counts in summary["position_counts"].values():
            self.assertEqual(set(counts.values()), {12})
        self.assertEqual([r["selection_index"] for r in rows], list(range(48)))
        self.assertEqual([r["serve_order"] for r in rows], sorted(r["serve_order"] for r in rows))
        for row in rows:
            original = self.drafts[row["serve_order"] - 1]
            self.assertEqual(row["presentation"], original["presentation"])
            self.assertEqual(row["markov_generated_id"], original["audios"]["markov"]["generated_id"])
            self.assertNotEqual(row["real_record_id"], subset.UNMATCHED_REAL)
            self.assertNotEqual(row["neural_song_id"], 1416)

    def test_fallback_description_is_eligible_and_copied_unchanged(self):
        rows = self.records(self.prepare()[0])
        row = next(r for r in rows if r["serve_order"] == 3)
        self.assertTrue(row["description_fallback"])
        self.assertEqual(row["public_description"], contexts.DESCRIPTION_FALLBACK)

    def test_never_calls_permutation_assignment(self):
        with patch.object(manifest, "permutation_assignments", side_effect=AssertionError("do not reassign")):
            self.assertEqual(self.prepare()[1]["selected"], 48)

    def test_cost_changes_cannot_change_selected_ids(self):
        first = self.records(self.prepare()[0])
        self.markov_costs = [i * 987.1 for i in range(self.count)]
        self.neural_costs = list(reversed(self.markov_costs))
        self.refresh()
        second = self.records(self.prepare()[0])
        self.assertEqual([r["external_id"] for r in first], [r["external_id"] for r in second])
        self.assertNotEqual([r["neural_matching_cost"] for r in first], [r["neural_matching_cost"] for r in second])

    def test_duration_only_runs_after_scan_and_cannot_change_selection(self):
        first = self.records(self.prepare()[0])
        original_scan = subset.scan_candidates
        finished = False

        def scan(*args):
            nonlocal finished
            result = original_scan(*args)
            finished = True
            return result

        def duration(song):
            self.assertTrue(finished)
            return 0.125

        with patch.object(subset, "scan_candidates", side_effect=scan), patch.object(subset, "symbolic_duration", side_effect=duration):
            second = self.records(self.prepare()[0])
        self.assertEqual([r["external_id"] for r in first], [r["external_id"] for r in second])
        self.assertTrue(all(r["real_duration_sec"] == 0.125 for r in second))

    def test_images_are_local_opaque_sanitized_and_not_original_carts(self):
        files, summary = self.prepare()
        rows = self.records(files)
        self.assertEqual(summary["usable_public_visuals"], 48)
        for row in rows:
            name = row["public_visual_source"]
            self.assertEqual(name, f"visuals/{row['external_id']}.png")
            self.assertTrue(row["visual_usable"])
            self.assertEqual(subset.sha(files[name]), row["public_visual_sha256"])
            self.assertNotEqual(files[name], (self.dataset / row["visual_source_relative_path"]).read_bytes())
            with Image.open(io.BytesIO(files[name])) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.info, {})
                rgba = image.convert("RGBA").tobytes()
                carrier = {((rgba[i + 2] & 3) | ((rgba[i + 1] & 3) << 2)
                            | ((rgba[i] & 3) << 4) | ((rgba[i + 3] & 3) << 6)) for i in range(0, len(rgba), 4)}
                self.assertEqual(carrier, {192})

    def test_missing_image_is_ineligible_and_next_same_permutation_fills_quota(self):
        (self.dataset / self.contexts[0]["source_relative_path"]).unlink()
        files, summary = self.prepare()
        self.assertNotIn(self.contexts[0]["external_id"], {r["external_id"] for r in self.records(files)})
        self.assertEqual(summary["selected"], 48)
        self.assertTrue(any("missing_or_unusable_image" in r["reason"] for r in summary["known_ineligible"]))

    def test_none_visual_is_excluded_for_image_requirement_not_text_fallback(self):
        files, summary = self.prepare()
        issues = {r["external_id"]: r["reason"] for r in summary["known_ineligible"]}
        for i in (27, 51):
            self.assertEqual(issues[self.contexts[i]["external_id"]], "interface_requires_image_but_frozen_context_has_none")
        self.assertTrue(all(r["public_visual_type"] != "none" for r in self.records(files)))

    def test_full_outputs_are_byte_deterministic_and_inputs_unchanged(self):
        protected = list(self.paths.values()) + list(self.dataset.rglob("*.png"))
        before = {p: p.read_bytes() for p in protected}
        first, _ = self.prepare()
        second, _ = self.prepare()
        self.assertEqual(first, second)
        for path, raw in before.items():
            self.assertEqual(path.read_bytes(), raw)
        config = json.loads(first["selection_config.json"])
        for name, digest in config["outputs_sha256"].items():
            self.assertEqual(subset.sha(first[name]), digest)
        for name in ("scenarios.csv", "scenarios.jsonl", "selection_config.json", "selection_summary.json"):
            self.assertIn(name, first)

    def test_matching_csv_order_is_not_a_join_key(self):
        first = self.records(self.prepare()[0])
        rows = subset.csv_rows(self.paths["neural_matching"].read_bytes())
        self.save_csv("neural_matching", list(reversed(rows)))
        self.rehash_configs()
        second = self.records(self.prepare()[0])
        self.assertEqual(first, second)

    def test_tampered_frozen_artifact_fails(self):
        self.paths["manifest"].write_bytes(self.paths["manifest"].read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.prepare()

    def test_reassociation_and_invalid_permutation_are_rejected(self):
        for mode in ("association", "permutation"):
            with self.subTest(mode=mode):
                old = copy.deepcopy(self.drafts)
                if mode == "association":
                    self.drafts[0]["audios"]["markov"]["generated_id"] = self.drafts[1]["audios"]["markov"]["generated_id"]
                else:
                    self.drafts[0]["presentation"]["assignment"]["A"] = "attention"
                    self.drafts[0]["presentation"]["assignment"]["B"] = "attention"
                self.refresh()
                with self.assertRaises(ValueError):
                    self.prepare()
                self.drafts = old
                self.refresh()

    def test_serve_order_cannot_be_reassigned(self):
        self.pool[0]["serve_order"] = 2
        self.refresh()
        with self.assertRaisesRegex(ValueError, "serve_order"):
            self.prepare()

    def test_publication_never_overwrites_and_failure_leaves_no_success(self):
        files, _ = self.prepare()
        output = self.root / "output"
        subset.write_subset(output, files)
        with self.assertRaises(FileExistsError):
            subset.write_subset(output, files)
        failed = self.root / "failed"
        with patch.object(subset.os, "replace", side_effect=OSError("publication failed")):
            with self.assertRaises(OSError):
                subset.write_subset(failed, files)
        self.assertFalse(failed.exists())


class TestDurationDiagnostic(unittest.TestCase):
    def test_symbolic_clock_never_loops_or_pads_or_truncates(self):
        song = dict(end_reason="loop_end", loop_start_pattern=0,
                    patterns=[{"pattern_ticks": 512}, {"pattern_ticks": 256}])
        self.assertEqual(subset.symbolic_duration(song), 768 * 366 / 44100)

    def test_thresholds_are_strict(self):
        rows = [{f"{origin}_duration_sec": value for origin in ("real", "markov", "neural")}
                for value in (5, 10, 20, 30, 31)]
        for stats in subset.duration_statistics(rows).values():
            self.assertEqual(stats, dict(min=5, median=20, mean=19.2, max=31,
                                         lt_10s=1, lt_20s=2, lt_30s=3, ge_30s=2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
