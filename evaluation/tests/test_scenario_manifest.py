import copy
import csv
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.subjective import build_scenario_manifest as manifest
from evaluation.subjective import export_scenario_contexts as contexts


def rows_from(files):
    return [json.loads(line) for line in files["scenario_manifest_draft.jsonl"].splitlines()]


class TestPermutation(unittest.TestCase):
    def test_1854_all_24_balanced_deterministic_and_no_other_inputs(self):
        identifiers = [contexts.external_id(i + 1, 0) for i in range(1854)]
        assignments = manifest.permutation_assignments(identifiers)
        self.assertEqual(assignments, manifest.permutation_assignments(reversed(identifiers)))
        counts = {}
        for value in assignments.values():
            permutation = tuple(value[p] for p in manifest.POSITIONS)
            counts[permutation] = counts.get(permutation, 0) + 1
        self.assertEqual(set(counts), set(itertools.permutations(manifest.ORIGINS)))
        self.assertEqual(set(counts.values()), {77, 78})
        for position in manifest.POSITIONS:
            for origin in manifest.ORIGINS:
                self.assertIn(sum(a[position] == origin for a in assignments.values()), (463, 464))

    def test_every_partial_block_balances_and_duplicate_id_fails(self):
        for n in range(1, 50):
            values = manifest.permutation_assignments([str(i) for i in range(n)]).values()
            for position in manifest.POSITIONS:
                for origin in manifest.ORIGINS:
                    self.assertIn(sum(v[position] == origin for v in values), (n // 4, (n + 3) // 4))
        with self.assertRaisesRegex(ValueError, "Duplicate external_id"):
            manifest.permutation_assignments(["same", "same"])


class TestScenarioManifest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.paths = {role: self.root / Path(ref).name for role, ref in manifest.REFS.items()}
        # Shared song_ids and shared TIDs, intentionally different physical order.
        self.real = [{"tid": tid, "song_id": song, "duration_sec": 999,
                      "wav_file": "/old/DO-NOT-READ.wav", "patterns": [{"example": i}]}
                     for i, (tid, song) in enumerate([("2", 0), ("1", 0), ("1", 1), ("3", 0)])]
        self.generated = [{"tid": "markov_seed42", "song_id": i,
                           "wav_file": "/old/markov.wav", "duration_sec": 888,
                           "patterns": [{"generated": i}]} for i in range(4)]
        self.write_records("real_test", self.real)
        self.write_records("markov_generated", self.generated)
        self.matches = []
        for i, generated_index in enumerate([2, 0, 3, 1]):
            real_hash = manifest.sha256(manifest.canonical(self.real[i]))
            generated_hash = manifest.sha256(manifest.canonical(self.generated[generated_index]))
            self.matches.append({
                "real_index": i, "real_tid": self.real[i]["tid"], "real_song_id": self.real[i]["song_id"],
                "real_record_sha256": real_hash, "generated_index": generated_index,
                "generated_id": "sha256:" + generated_hash, "generated_tid": "markov_seed42",
                "generated_song_id": generated_index, "generated_record_sha256": generated_hash,
                "system_name": "markov",
            })
        self.write_matching()
        self.source_contexts = []
        for order, real_index in enumerate([2, 3, 0, 1], 1):
            real = self.real[real_index]
            self.source_contexts.append({
                "external_id": contexts.external_id(real["tid"], real["song_id"]),
                "tid": real["tid"], "song_id": real["song_id"], "serve_order": order,
                "context_priority": 5 - order, "context_tier": "ABCD"[order - 1],
                "public_title": "", "public_description": "A contextual description.",
                "public_visual_type": "none" if order == 4 else "cartridge",
                "public_visual_label": contexts.VISUAL_LABELS["none" if order == 4 else "cartridge"],
                "source_relative_path": "" if order == 4 else f"cartridges/{real['tid']}/cart.p8.png",
                "public_context_heading": contexts.CONTEXT_HEADING,
                "internal_evidence_not_needed": "must not replicate",
            })
        self.write_contexts()
        self.write_matching_config()

    def write_records(self, role, rows):
        # A blank physical line tests zero-based provenance rather than record position.
        self.paths[role].write_bytes(b"\n" + b"\n".join(manifest.canonical(r) for r in rows) + b"\n")

    def write_matching(self):
        text = io.StringIO(newline="")
        writer = csv.DictWriter(text, fieldnames=self.matches[0])
        writer.writeheader()
        for match in reversed(self.matches):
            row = dict(match)
            row["real_index"] += 1
            row["generated_index"] += 1
            writer.writerow(row)
        self.paths["markov_matching"].write_text(text.getvalue())

    def write_contexts(self):
        self.paths["scenario_context"].write_bytes(manifest.canonical(self.source_contexts))
        self.paths["scenario_context_config"].write_text(json.dumps({
            "protocol": contexts.PROTOCOL, "summary": {"total_contexts": 4},
            "outputs_sha256": {"scenario_contexts_internal.json":
                               manifest.sha256(self.paths["scenario_context"].read_bytes())},
        }))

    def write_matching_config(self):
        self.paths["markov_matching_config"].write_text(json.dumps({
            "protocol": manifest.MATCHING_PROTOCOL, "system_name": "markov",
            "inputs": {input_name: {"sha256": manifest.sha256(self.paths[role].read_bytes()),
                                    "count": 4, "path": "/do/not/follow/this/path"}
                       for input_name, role in (("real", "real_test"), ("generated", "markov_generated"))},
        }))

    def prepare(self):
        return manifest.prepare_manifest(**self.paths, expected_count=4)

    def test_join_uses_full_identities_and_generated_hash_not_position(self):
        files, audit = self.prepare()
        rows = rows_from(files)
        self.assertEqual([r["identity"]["external_id"] for r in rows],
                         [c["external_id"] for c in self.source_contexts])
        by_key = {(r["tid"], r["song_id"]): (i, r) for i, r in enumerate(self.real)}
        for row in rows:
            key = (row["identity"]["tid"], row["identity"]["song_id"])
            i, real = by_key[key]
            a, m = row["audios"]["real"], row["audios"]["markov"]
            self.assertEqual(a["source_record_id"], f"real:{key[0]}:{key[1]}")
            self.assertEqual(a["source_record_sha256"], manifest.sha256(manifest.canonical(real)))
            self.assertEqual(a["source_line_index"], i + 1)
            self.assertEqual(m["generated_id"], self.matches[i]["generated_id"])
            self.assertEqual(m["generated_record_sha256"], self.matches[i]["generated_record_sha256"])
            self.assertEqual(m["source_record_sha256"], m["generated_record_sha256"])
            self.assertEqual(m["source_line_index"], self.matches[i]["generated_index"] + 1)
            self.assertEqual(m["matching"]["real_record_id"], a["source_record_id"])
        self.assertEqual(audit["summary"]["unique_real"], 4)
        self.assertEqual(audit["summary"]["unique_markov_generated_id"], 4)
        self.assertTrue(all(audit["checks"].values()))

    def test_all_pending_fields_no_symbolic_duration_or_old_wav_or_evidence(self):
        files, audit = self.prepare()
        for row in rows_from(files):
            for slot in row["audios"].values():
                for field in manifest.WAV_FIELDS:
                    self.assertIsNone(slot[field], field)
                self.assertFalse(slot["wav_ready"])
                self.assertFalse(slot["published_ready"])
            self.assertTrue(row["audios"]["real"]["symbolic_ready"])
            self.assertTrue(row["audios"]["markov"]["symbolic_ready"])
            self.assertTrue(row["audios"]["markov"]["matching_ready"])
            neural = row["audios"]["neural"]
            for field in ("source_record_id", "source_artifact_ref", "source_record_sha256", "matching"):
                self.assertIsNone(neural[field])
            self.assertFalse(neural["symbolic_ready"])
            self.assertFalse(neural["matching_ready"])
            attention = row["audios"]["attention"]
            self.assertEqual(attention["parent_audio_id"], row["audios"]["real"]["internal_audio_id"])
            self.assertEqual((attention["protocol"], attention["global_seed"], attention["target_snr_db"]),
                             ("attention-check-v1", 831, 0))
            self.assertFalse(attention["generation_ready"])
            for field in ("parent_wav_sha256", "derived_seed", "effective_snr_db", "gain"):
                self.assertIsNone(attention[field])
            self.assertEqual(row["protocol"]["status"], "draft")
            self.assertFalse(row["protocol"]["ready_for_import"])
            self.assertTrue(manifest.readiness_errors(row))
        self.assertNotIn("DO-NOT-READ", files["scenario_manifest_draft.jsonl"])
        self.assertNotIn("must not replicate", files["scenario_manifest_draft.jsonl"])
        self.assertEqual(audit["summary"]["ready_for_import"], 0)

    def test_only_declared_symbolic_inputs_and_implementation_files_opened(self):
        opened, original_open = [], Path.open

        def tracked_open(path, *args, **kwargs):
            opened.append(path.resolve())
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", tracked_open):
            self.prepare()
        expected = {p.resolve() for p in self.paths.values()} | {
            Path(manifest.__file__).resolve(), Path(contexts.__file__).resolve()}
        self.assertEqual(set(opened), expected)

    def test_determinism_hashes_relative_paths_and_metadata_independence(self):
        first, audit = self.prepare()
        self.assertEqual(first, self.prepare()[0])
        for role, entry in audit["inputs"].items():
            self.assertEqual(entry["sha256"], manifest.sha256(self.paths[role].read_bytes()))
            manifest.portable_ref(entry["artifact_ref"])
        self.assertEqual(audit["output_sha256"]["scenario_manifest_draft.jsonl"],
                         manifest.sha256(first["scenario_manifest_draft.jsonl"].encode()))
        for content in first.values():
            self.assertNotIn(str(self.root), content)
            self.assertNotIn("/do/not/follow", content)
        before = {r["identity"]["external_id"]: r["presentation"] for r in rows_from(first)}
        for c in self.source_contexts:
            c.update(context_priority=99, context_tier="D", public_description="Changed context")
        self.write_contexts()
        after = {r["identity"]["external_id"]: r["presentation"] for r in rows_from(self.prepare()[0])}
        self.assertEqual(before, after)

    def test_duplicate_real_and_generated_rejected(self):
        for role, records in (("real_test", self.real), ("markov_generated", self.generated)):
            original = self.paths[role].read_bytes()
            self.write_records(role, records[:3] + [records[0]])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                self.prepare()
            self.paths[role].write_bytes(original)

    def test_duplicate_context_external_id_and_wrong_identity_rejected(self):
        original = copy.deepcopy(self.source_contexts)
        self.source_contexts[1] = copy.deepcopy(self.source_contexts[0])
        self.write_contexts()
        with self.assertRaisesRegex(ValueError, "Duplicate context"):
            self.prepare()
        self.source_contexts = original
        self.source_contexts[0]["song_id"] = 88
        self.write_contexts()
        with self.assertRaisesRegex(ValueError, "external_id"):
            self.prepare()

    def test_matching_reuse_wrong_hash_wrong_id_and_line_rejected(self):
        original = copy.deepcopy(self.matches)
        cases = [("real_record_sha256", "0" * 64, "Real record hash"),
                 ("generated_record_sha256", "0" * 64, "Markov generated_id/hash"),
                 ("generated_id", "sha256:" + "0" * 64, "Matching identity missing"),
                 ("real_index", 999, "line provenance"),
                 ("generated_index", 999, "line provenance")]
        for field, value, message in cases:
            with self.subTest(field=field):
                self.matches = copy.deepcopy(original)
                self.matches[0][field] = value
                self.write_matching()
                with self.assertRaisesRegex(ValueError, message):
                    self.prepare()
        self.matches = copy.deepcopy(original)
        self.matches[0] = dict(self.matches[1])
        self.write_matching()
        with self.assertRaisesRegex(ValueError, "reused"):
            self.prepare()
        self.matches = copy.deepcopy(original)
        for k in ("generated_id", "generated_index", "generated_tid", "generated_song_id", "generated_record_sha256"):
            self.matches[0][k] = self.matches[1][k]
        self.write_matching()
        with self.assertRaisesRegex(ValueError, "Markov reused"):
            self.prepare()

    def test_input_file_hash_mismatch_and_missing_coverage_rejected(self):
        for role in ("real_test", "markov_generated", "scenario_context"):
            original = self.paths[role].read_bytes()
            self.paths[role].write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.prepare()
            self.paths[role].write_bytes(original)
        self.source_contexts.pop()
        self.write_contexts()
        with self.assertRaisesRegex(ValueError, "Context count"):
            self.prepare()

    def test_incomplete_validator_and_public_projection_reject_false_flags(self):
        row = rows_from(self.prepare()[0])[0]
        with self.assertRaisesRegex(ValueError, "Scenario incomplete"):
            manifest.validate_for_import(row)
        row["protocol"]["ready_for_import"] = True
        for slot in row["audios"].values():
            slot.update(symbolic_ready=True, matching_ready=True, wav_ready=True, published_ready=True)
        with self.assertRaisesRegex(ValueError, "Scenario incomplete"):
            manifest.validate_scenario(row)
        with self.assertRaisesRegex(ValueError, "Scenario incomplete"):
            manifest.public_projection(row)

    def complete_future_fixture(self, row):
        """Synthetic metadata only; no WAV generation and no invented Neural input."""
        row = copy.deepcopy(row)
        for i, origin in enumerate(manifest.ORIGINS):
            slot = row["audios"][origin]
            slot.update(symbolic_ready=True if origin != "attention" else None,
                        source_record_id=slot["source_record_id"] or "fixture-record",
                        source_artifact_ref=slot["source_artifact_ref"] or "fixtures/symbolic.jsonl",
                        source_record_sha256=slot["source_record_sha256"] or "f" * 64,
                        render_protocol="fixture-render-v1", wav_source_path=f"subjective-audio://render-fixture/{origin}/clip.wav",
                        duration_sec=1.5, sample_rate=44100, channels=1, sample_width_bytes=2,
                        wav_sha256=str(i) * 64, wav_ready=True,
                        wav_validation={"protocol": "wav-validation-v1", "valid": True, "wav_sha256": str(i) * 64},
                        published_audio_id=f"fixture-opaque-{i}", published_locator=f"published/fixture-{i}.wav",
                        published_ready=True)
            if origin == "neural":
                slot.update(matching_ready=True, matching={
                    "protocol": "fixture-matching", "artifact_ref": "fixtures/matching.csv",
                    "artifact_sha256": "f" * 64,
                    "real_record_sha256": row["audios"]["real"]["source_record_sha256"],
                    "generated_record_sha256": slot["source_record_sha256"],
                    "generated_id": slot["source_record_id"],
                })
        row["audios"]["attention"].update(
            generation_ready=True, parent_wav_sha256=row["audios"]["real"]["wav_sha256"],
            derived_seed=123, effective_snr_db=0.02, gain=0.5)
        visual = row["context"]["visual"]
        if row["context"]["public_visual_type"] != "none":
            visual.update(state="published", published_ready=True, published_sha256="f" * 64,
                          published_locator=f"visuals/{row['identity']['external_id']}.png")
        row["protocol"]["ready_for_import"] = True
        return row

    def test_future_gate_requires_all_slots_evidence_and_public_dto_has_no_origins(self):
        for initial in rows_from(self.prepare()[0]):
            row = self.complete_future_fixture(initial)
            manifest.validate_for_import(row)
            public = manifest.public_projection(row)
            self.assertEqual(set(public), set(contexts.PUBLIC_FIELDS) | {"audios"})
            for audio in public["audios"]:
                self.assertEqual(set(audio), set(manifest.PUBLIC_AUDIO_FIELDS))
            serialized = json.dumps(public)
            for forbidden in (*manifest.ORIGINS, "origin", "assignment", "source_record", "provenance", "context_priority"):
                self.assertNotIn(forbidden, serialized)
        valid = self.complete_future_fixture(rows_from(self.prepare()[0])[0])
        for origin in manifest.ORIGINS:
            for field in ("wav_sha256", "wav_validation", "duration_sec", "sample_rate", "channels",
                          "sample_width_bytes", "published_audio_id", "published_locator", "render_protocol"):
                row = copy.deepcopy(valid)
                row["audios"][origin][field] = None
                with self.subTest(origin=origin, field=field), self.assertRaises(ValueError):
                    manifest.validate_for_import(row)
        for mutate in (
            lambda r: r["presentation"]["assignment"].update(A="real", B="real"),
            lambda r: r["context"]["visual"].update(state="pending"),
            lambda r: r["audios"]["attention"].update(parent_wav_sha256="e" * 64),
            lambda r: r["audios"]["attention"].update(derived_seed=None),
            lambda r: r["audios"]["attention"].update(effective_snr_db=0.2),
            lambda r: r["audios"]["neural"].update(matching=None),
        ):
            row = copy.deepcopy(valid)
            mutate(row)
            with self.assertRaises(ValueError):
                manifest.validate_for_import(row)

    def test_paths_are_relative_or_logical_and_unsafe_paths_fail(self):
        manifest.portable_ref("subjective-audio://render-id/real/file.wav")
        for bad in ("/absolute/file.wav", "../escape", "a/../escape", "C:\\file.wav",
                    "https://host/file.wav", "subjective-audio:///absolute", "subjective-audio://../escape"):
            with self.subTest(path=bad), self.assertRaises(ValueError):
                manifest.portable_ref(bad)
        self.source_contexts[0]["source_relative_path"] = "/absolute/cart.png"
        self.write_contexts()
        with self.assertRaises(ValueError):
            self.prepare()


@unittest.skipUnless(all((manifest.REPO / ref).is_file() for ref in manifest.REFS.values()),
                     "Frozen local inputs are not distributed in git")
class TestFrozenScenarioManifest(unittest.TestCase):
    def test_actual_1854_records_and_deterministic_output(self):
        paths = {role: manifest.REPO / ref for role, ref in manifest.REFS.items()}
        files, audit = manifest.prepare_manifest(**paths)
        self.assertEqual(files, manifest.prepare_manifest(**paths)[0])
        rows = rows_from(files)
        self.assertEqual(len(rows), 1854)
        self.assertEqual(len({r["identity"]["external_id"] for r in rows}), 1854)
        self.assertEqual(len({r["audios"]["real"]["source_record_id"] for r in rows}), 1854)
        self.assertEqual(len({r["audios"]["markov"]["generated_id"] for r in rows}), 1854)
        summary = audit["summary"]
        self.assertEqual(summary["symbolic_ready"], dict(real=1854, markov=1854, neural=0, attention=0))
        self.assertEqual(summary["matching_ready"], dict(real=0, markov=1854, neural=0, attention=0))
        self.assertEqual(summary["distinct_permutations"], 24)
        for counts in summary["position_counts"].values():
            self.assertTrue(all(count in (463, 464) for count in counts.values()))
        self.assertEqual(summary["ready_for_import"], 0)
        for row in rows:
            manifest.validate_scenario(row)
            self.assertFalse(row["protocol"]["ready_for_import"])
            for slot in row["audios"].values():
                self.assertFalse(slot["wav_ready"])
                self.assertFalse(slot["published_ready"])
                self.assertIsNone(slot["duration_sec"])
                self.assertIsNone(slot["wav_source_path"])


if __name__ == "__main__":
    unittest.main()
