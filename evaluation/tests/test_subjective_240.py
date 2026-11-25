"""Selection and static semantic curation contracts; no audio or matching runs."""

import copy
from dataclasses import fields, replace
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.subjective import select_subjective_240 as selection


def candidates(rounds=12, clustered=False):
    pairs = [(p, r) for r in range(rounds) for p in selection.frozen.PERMUTATIONS]
    if clustered:
        pairs.sort(key=lambda pair: (pair[0], pair[1]))
    return [selection.frozen.Candidate(f"scenario-{i}", i + 1, p, (), "cartridge", f"source/{i}.png")
            for i, (p, _) in enumerate(pairs)]


def image(_candidate):
    return b"sanitized-fixture", "source-sha", ""


def description_fixture():
    evidence = {"one": {"description_core": "A cat looks for a friend.", "description": "A cat looks for a friend.",
                         "original_public_description": "Original preserved.", "tags": [], "visual": {}}}
    record = {"selection_index": 0, "external_id": "one", "real_tid": "1", "context_tier": "A", "serve_order": 1,
              "public_description_original": "Original preserved.", "original_description_fallback": False}
    item = {"tid": "1", "external_ids": ["one"], "evidence_sha256": selection.sha(selection.canonical(evidence["one"])),
            "description_ptbr": "Um gato procura seu amigo.", "evidence_fields": ["description_core"],
            "manual_review_recommended": False, "review_reason": "", "short_description_rationale": "Fonte limitada à premissa."}
    return [record], evidence, {"protocol": selection.DESCRIPTION_PROTOCOL, "entries": [item]}


class TestSubjective240Selection(unittest.TestCase):
    def test_exact_240_and_ten_of_each_existing_permutation(self):
        selected, scan, assets = selection.scan_candidates(candidates(), image)
        self.assertEqual((len(selected), len(scan), len(assets)), (240, 240, 240))
        counts = {p: sum(c.permutation == p for c in selected) for p in selection.frozen.PERMUTATIONS}
        self.assertEqual(set(counts.values()), {10})
        for slot in range(4):
            self.assertEqual({o: sum(c.permutation[slot] == o for c in selected) for o in selection.frozen.manifest.ORIGINS},
                             {o: 60 for o in selection.frozen.manifest.ORIGINS})

    def test_old_48_are_contained_with_unchanged_candidates(self):
        source = candidates()
        old, _, _ = selection.frozen.scan_candidates(source, image)
        new, _, _ = selection.scan_candidates(source, image)
        self.assertEqual(len(old), 48)
        self.assertTrue(set(old) <= set(new))

    def test_quota_scan_skips_full_permutations_without_reranking(self):
        source = candidates(clustered=True)
        selected, scan, _ = selection.scan_candidates(source, image)
        self.assertEqual(len(selected), 240)
        self.assertEqual(len(scan), 286)
        self.assertEqual(sum(a["decision"] == "quota_full" for a in scan), 46)
        self.assertEqual([c.serve_order for c in selected], sorted(c.serve_order for c in selected))
        self.assertEqual(selected[:10], source[:10])
        self.assertNotIn(source[10], selected)

    def test_input_physical_order_does_not_replace_serve_order(self):
        source = candidates()
        first = selection.scan_candidates(source, image)
        second = selection.scan_candidates(list(reversed(source)), image)
        self.assertEqual(first, second)

    def test_unmatched_or_missing_image_is_ineligible(self):
        source = candidates()
        source[0] = replace(source[0], reasons=("unmatched_real_no_neural",))
        source[1] = replace(source[1], reasons=("interface_requires_image_but_frozen_context_has_none",))

        def load(candidate):
            return (None, None, "missing_or_unusable_image") if candidate == source[2] else image(candidate)

        selected, scan, _ = selection.scan_candidates(source, load)
        self.assertEqual(len(selected), 240)
        self.assertFalse(set(source[:3]) & set(selected))
        self.assertTrue(all(a["decision"] == "ineligible" for a in scan[:3]))

    def test_selection_input_schema_cannot_carry_cost_or_duration(self):
        names = {f.name for f in fields(selection.frozen.Candidate)}
        self.assertEqual(names, {"external_id", "serve_order", "permutation", "reasons", "visual_type", "source_relative_path"})

    def test_insufficient_pool_and_duplicate_priority_fail(self):
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            selection.scan_candidates(candidates(rounds=9), image)
        source = candidates()
        source[1] = replace(source[1], serve_order=source[0].serve_order)
        with self.assertRaisesRegex(ValueError, "Duplicate serve_order"):
            selection.scan_candidates(source, image)

    def test_canonical_writer_refuses_overwriting_an_existing_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            marker = path / "unchanged.txt"
            marker.write_text("preserved")
            with self.assertRaises(FileExistsError):
                selection.frozen.write_subset(path, {"new.txt": b"new"})
            self.assertEqual(marker.read_text(), "preserved")


class TestContextDescriptionPtbr(unittest.TestCase):
    def test_original_is_preserved_separately_and_source_record_is_immutable(self):
        records, evidence, curation = description_fixture()
        before = copy.deepcopy(records)
        output, audits = selection.apply_descriptions(records, evidence, curation)
        self.assertEqual(records, before)
        self.assertEqual(output[0]["public_description_original"], "Original preserved.")
        self.assertEqual(output[0]["public_description"], "Um gato procura seu amigo.")
        self.assertEqual(audits[0]["description_original"], "Original preserved.")
        self.assertEqual(audits[0]["evidence_source"]["description_core"], "A cat looks for a friend.")

    def test_ambiguous_case_is_marked_and_retained_without_generation(self):
        records, evidence, curation = description_fixture()
        curation["entries"][0].update(manual_review_recommended=True, review_reason="Objetivo incompleto nas fontes.")
        output, audits = selection.apply_descriptions(records, evidence, curation)
        self.assertEqual(len(output), 1)
        self.assertTrue(output[0]["manual_review_recommended"])
        self.assertEqual(output[0]["selection_index"], records[0]["selection_index"])
        self.assertEqual(audits[0]["description_ptbr"], curation["entries"][0]["description_ptbr"])

    def test_source_drift_cannot_reuse_curated_text_silently(self):
        records, evidence, curation = description_fixture()
        evidence["one"]["description_core"] = "Different contextual evidence."
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            selection.apply_descriptions(records, evidence, curation)

    def test_missing_extra_or_duplicate_curation_fails(self):
        records, evidence, curation = description_fixture()
        for ids in ([], ["one", "extra"], ["one", "one"]):
            changed = copy.deepcopy(curation)
            changed["entries"][0]["external_ids"] = ids
            with self.assertRaises(ValueError):
                selection.apply_descriptions(records, evidence, changed)

    def test_short_text_requires_explicit_evidence_rationale(self):
        records, evidence, curation = description_fixture()
        curation["entries"][0]["short_description_rationale"] = ""
        with self.assertRaisesRegex(ValueError, "Short description"):
            selection.apply_descriptions(records, evidence, curation)

    def test_review_flag_requires_a_reason(self):
        records, evidence, curation = description_fixture()
        curation["entries"][0]["manual_review_recommended"] = True
        with self.assertRaisesRegex(ValueError, "Review rationale"):
            selection.apply_descriptions(records, evidence, curation)

    def test_guards_detect_empty_music_controls_urls_and_operational_text(self):
        texts = {"": "empty", "Uma música alegre acompanha a trilha sonora.": "music_audio",
                 "Pressione a tecla X para continuar.": "keys_controls", "Mais detalhes em https://example.org.": "url",
                 "Instale o jogo no computador.": "operational", "Abra o menu para ver o score.": "operational",
                 "You are the player and you can move with the keys.": "english_phrase"}
        for text, expected in texts.items():
            with self.subTest(text=text):
                self.assertIn(expected, selection.check_description(text)["issues"])

    def test_contextual_control_and_installing_pipes_are_not_software_instructions(self):
        for text in ("O jogador controla uma plataforma para rebater a bola.",
                     "Dois encanadores instalam novos trechos de tubulação para resolver problemas.",
                     "O jogador experimenta configurações de projéteis para defender as torres."):
            self.assertEqual(selection.check_description(text)["issues"], [])

    def test_three_sentences_are_reported(self):
        self.assertIn("sentence_count", selection.check_description("Uma ilha. Um personagem. Uma busca.")["issues"])

    def test_same_static_curation_produces_identical_serialized_records(self):
        records, evidence, curation = description_fixture()
        first = selection.apply_descriptions(records, evidence, curation)
        second = selection.apply_descriptions(records, evidence, curation)
        self.assertEqual(selection.json_bytes(first), selection.json_bytes(second))

    def test_versioned_curation_covers_240_and_contains_all_historical_ids(self):
        curation = json.loads(selection.CURATION.read_bytes())
        ids = [identifier for entry in curation["entries"] for identifier in entry["external_ids"]]
        self.assertEqual(len(ids), 240)
        self.assertEqual(len(set(ids)), 240)
        old = [json.loads(line) for line in (selection.REPO / selection.OLD_SUBSET / "scenarios.jsonl").read_bytes().splitlines()]
        self.assertTrue({r["external_id"] for r in old} <= set(ids))
        for entry in curation["entries"]:
            self.assertEqual(selection.check_description(entry["description_ptbr"])["issues"], [])
            self.assertEqual(len(entry["evidence_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
