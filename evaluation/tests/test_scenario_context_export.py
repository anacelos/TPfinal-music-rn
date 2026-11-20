import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import uuid

from subjective import export_scenario_contexts as export


def pool_row(**overrides):
    return {
        "tid": "1", "song_id": "0", "split": "test", "serve_order": "1",
        "context_priority": "1", "context_tier": "B", "start_pattern": "3",
        "end_pattern": "5", "game_title": "Unverified Title",
        "game_title_confidence": "header_only",
        "description_informative": "True",
        "description_core": "You are a robot. Collect keys to escape the maze",
        "preferred_visual_type": "none", "preferred_visual_path": "",
        **overrides,
    }


def project_one(row, root):
    return export.project_rows([row], root, expected_count=1, expected_tids=1)[0]


class TestIdentity(unittest.TestCase):
    def test_external_id_deterministic_canonical_and_unique(self):
        first = export.external_id("123", "0")
        self.assertEqual(first, export.external_id(123, 0))
        self.assertEqual(first, "2f4d2585-3128-502c-9cf3-06fc0086cdae")
        ids = {export.external_id(tid, song) for tid in range(1, 874) for song in range(3)}
        self.assertEqual(len(ids), 873 * 3)
        for identifier in ids:
            parsed = uuid.UUID(identifier)
            self.assertEqual(parsed.version, 5)
            self.assertEqual(str(parsed), identifier)

    def test_external_id_has_no_origin_tier_system_or_ranking(self):
        with tempfile.TemporaryDirectory() as root:
            original = project_one(pool_row(), root)
            changed = project_one(pool_row(context_tier="D", total_stars="9999",
                                           game_title="Attention Markov Neural Real",
                                           audio_origin="real.wav", music_context_evidence="secret"), root)
        self.assertEqual(original["external_id"], changed["external_id"])
        self.assertRegex(original["external_id"], r"^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$")
        self.assertNotIn("123-0", export.external_id(123, 0))

    def test_ambiguous_identities_rejected(self):
        for tid, song in [("01", 0), (1, "00"), ("1-2", 0), (0, 1), (1, -1), (True, 0)]:
            with self.subTest(tid=tid, song=song), self.assertRaises(ValueError):
                export.external_id(tid, song)


class TestPublicText(unittest.TestCase):
    def test_all_unapproved_title_confidences_use_neutral_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            for confidence in ["header_only", "unknown", "strong", "verified", ""]:
                with self.subTest(confidence=confidence):
                    row = project_one(pool_row(game_title_confidence=confidence), root)
                    self.assertEqual(row["public_title"], export.TITLE_FALLBACK)
                    self.assertTrue(row["title_fallback"])

    def test_missing_title_uses_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            row = project_one(pool_row(game_title=""), root)
        self.assertEqual(row["public_title"], export.TITLE_FALLBACK)

    def test_description_uses_frozen_core_not_raw_or_evidence(self):
        row = pool_row(description="BBS header: secret URL", music_context_evidence="music(3)",
                       description_core="You are a robot. Collect keys to escape the maze")
        self.assertEqual(export.public_description(row), row["description_core"])

    def test_description_fallback_for_missing_uninformative_and_technical(self):
        for fields in [
            {"description_core": ""}, {"description_informative": "False"},
            {"description_core": "Credits: music by Ada"},
            {"description_core": "I learned Lua to build a puzzle game"},
            {"description_core": "BBS > Cart #123: A fishing game"},
            {"description_core": "A puzzle game made by Ada"},
            {"description_core": "A platformer tutorial with jumps"},
            {"description_core": "Updated puzzle mechanics fixed collision bugs"},
            {"description_core": "A puzzle game at www.example.com"},
            {"description_core": "To save on space, use generic sprites to build furniture"},
            {"description_core": "I will probably add apples for the guy to collect"},
            {"description_core": "We rendered the asteroids in Photoshop"},
            {"description_core": "I tried to find interesting puzzle mechanics"},
        ]:
            with self.subTest(fields=fields):
                self.assertEqual(export.public_description(pool_row(**fields)), export.DESCRIPTION_FALLBACK)

    def test_filters_credits_urls_development_without_inventing_text(self):
        core = ("You are a robot. Music by Ada. I learned Lua to build a game. "
                "Collect keys to escape the maze. Visit https://example.com. "
                "Thanks to our playtesters")
        self.assertEqual(export.public_description(pool_row(description_core=core)),
                         "You are a robot. Collect keys to escape the maze")

    def test_section_tails_are_removed(self):
        for marker in ["Controls", "HOW TO PLAY", "Credits", "Technical Details", "Behind The Scenes"]:
            core = f"You are a robot. Collect keys. {marker}: Z to jump updated sprites"
            with self.subTest(marker=marker):
                self.assertEqual(export.public_description(pool_row(description_core=core)),
                                 "You are a robot. Collect keys")

    def test_legitimate_by_catching_is_kept(self):
        core = "Build burgers to order, by catching food items as they fall"
        self.assertEqual(export.public_description(pool_row(description_core=core)), core)


    def test_contextual_lexical_regressions(self):
        for text in [
            "The player controls a nameless farmer who must survive the winter.",
            "Enter into our universe and defeat the invading creatures.",
            "Upgrade its pips, pick a stronger skill, or add unique effects.",
            "Earn Carbon Credits by restoring the forest.",
        ]:
            with self.subTest(text=text):
                self.assertEqual(export.public_description(pool_row(description_core=text)), text.rstrip('.'))

    def test_useful_headers_preserve_following_gameplay(self):
        text = "Collect five keys and reach the exit before time runs out."
        for header in ["How to Play", "Instructions", "Controls:"]:
            with self.subTest(header=header):
                self.assertEqual(export.public_description(pool_row(description_core=header+"\n"+text)), text[:-1])

    def test_meta_and_musical_associations_regressions(self):
        for text in [
            "Wanted to challenge myself to a one day build.",
            "I need to find a better way to respawn these lines.",
            "Hopefully you find some enjoyment with it.",
            "Added new levels, fixed the jumping bugs.",
            "I will probably add apples for the guy to collect.",
            "This is Flowey's theme for his boss fight.",
            "flowies theme to his little fight",
            "music, tweaked the game loop, and allowed you to collect garbage",
            "in case anyone wanted an example of a spin-the-wheel mechanic",
            "Gotta find a more consistent way to respawn the lines",
            "This is a work in progress puzzle game",
            "To-do Add buildings Collect resources from trees.",
            "Look forward to: Find more enemies in future levels.",
            "The particle effect refers to a platformer example",
        ]:
            with self.subTest(text=text):
                self.assertEqual(export.public_description(pool_row(description_core=text)), export.DESCRIPTION_FALLBACK)

    def test_mixed_clauses_keep_context_without_meta(self):
        self.assertEqual(export.public_description(pool_row(description_core=
            "You are a robot. Collect keys to escape. I need to fix bugs. Hopefully you enjoy it.")),
            "You are a robot. Collect keys to escape")



class TestVisualAndBrowserBoundary(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "dataset"
        self.root.mkdir()

    def put(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        # Existence only: the exporter must not re-evaluate frozen pixels.
        path.write_bytes(b"already selected by frozen pool")

    def test_cartridge_and_spritesheet_labels_and_original_relative_paths(self):
        for kind, source, label in [
            ("cartridge", "cartridges/1/cart.p8.png", "Imagem do cartucho de origem"),
            ("spritesheet", "sprites/1/spritesheet.png", "Elementos gráficos do cartucho de origem"),
        ]:
            self.put(source)
            row = project_one(pool_row(preferred_visual_type=kind, preferred_visual_path=source), self.root)
            self.assertEqual(row["public_visual_type"], kind)
            self.assertEqual(row["public_visual_label"], label)
            self.assertEqual(row["source_relative_path"], source)

    def test_none_has_no_asset(self):
        row = project_one(pool_row(), self.root)
        self.assertEqual(row["public_visual_label"], "Sem imagem")
        self.assertEqual(row["source_relative_path"], "")
        self.assertIsNone(export.public_projection(row)["public_visual_source"])
        with self.assertRaises(ValueError):
            project_one(pool_row(preferred_visual_path="sprites/1/spritesheet.png"), self.root)

    def test_nonportable_paths_rejected(self):
        for source in ["/home/yuri/cart.png", "../cart.png", "sprites/../cart.png",
                       "C:/data/cart.png", "C:cart.png", r"\\server\cart.png",
                       r"sprites\cart.png", "https://host/cart.png", "./cart.png",
                       "sprites//cart.png", "sprites/cart.png\n", ""]:
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "relative and portable"):
                project_one(pool_row(preferred_visual_type="cartridge", preferred_visual_path=source), self.root)

    def test_missing_asset_and_symlink_escape_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing visual asset"):
            project_one(pool_row(preferred_visual_type="cartridge", preferred_visual_path="missing.png"), self.root)
        outside = self.root.parent / "outside.png"
        outside.write_bytes(b"outside")
        (self.root / "escape.png").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes dataset root"):
            project_one(pool_row(preferred_visual_type="cartridge", preferred_visual_path="escape.png"), self.root)

    def test_unknown_visual_rejected_without_selecting_alternative(self):
        with self.assertRaisesRegex(ValueError, "Unsupported frozen visual"):
            project_one(pool_row(preferred_visual_type="rendered_map"), self.root)

    def test_browser_allowlist_excludes_forbidden_fields_and_source_identity(self):
        self.put("cartridges/1/cart.p8.png")
        row = project_one(pool_row(preferred_visual_type="cartridge",
                                   preferred_visual_path="cartridges/1/cart.p8.png"), self.root)
        row.update(total_stars=99, source_song_sha256="secret", matching="secret",
                   music_evidence="secret", lua_evidence="secret", audio_origin="secret")
        path = f"visuals/{row['external_id']}.png"
        public = export.public_projection(row, published_visual_source=path)
        self.assertEqual(set(public), set(export.PUBLIC_FIELDS))
        self.assertFalse(set(export.INTERNAL_ONLY_FIELDS) & public.keys())
        self.assertNotIn("secret", json.dumps(public))
        self.assertNotIn("cartridges/1/", json.dumps(public))
        self.assertEqual(public["public_visual_source"], path)
        self.assertEqual(public["public_context_heading"], export.CONTEXT_HEADING)
        for forbidden in ["total_stars", "source_song_sha256", "matching", "music_evidence", "lua_evidence", "audio_origin"]:
            self.assertNotIn(forbidden, public)

    def test_original_visual_source_cannot_be_forwarded_to_browser(self):
        self.put("sprites/1/spritesheet.png")
        row = project_one(pool_row(preferred_visual_type="spritesheet",
                                   preferred_visual_path="sprites/1/spritesheet.png"), self.root)
        for source in [None, row["source_relative_path"], "/home/yuri/sheet.png", "https://example.com/1.png"]:
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "opaque visual path required"):
                export.public_projection(row, published_visual_source=source)


    def test_strict_public_contract(self):
        row = project_one(pool_row(), self.root)
        for fields in [
            {"external_id": "34636-song0-tierA-Real"},
            {"external_id": str(uuid.uuid4())},
            {"external_id": row["external_id"].upper()},
            {"public_visual_type": "neural"},
            {"public_description": {"tid": "leak"}},
            {"public_title": None},
            {"public_visual_label": "Gameplay"},
        ]:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                export.public_projection({**row, **fields})
        self.assertIsNone(export.public_projection(row)["public_visual_source"])
        self.assertNotIn("public_visual_source", row)
        with self.assertRaises(ValueError):
            export.public_projection(row, published_visual_source="visuals/fake.png")

    def test_id_independent_of_public_content_and_internal_paths(self):
        first = project_one(pool_row(), self.root)
        self.put("sprites/1/sheet.png")
        second = project_one(pool_row(description_core="Catch fish using one button", game_title="Other",
            preferred_visual_type="spritesheet", preferred_visual_path="sprites/1/sheet.png"), self.root)
        self.assertEqual(first["external_id"], second["external_id"])



class TestExportIntegration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "first"
        self.dataset = self.root / "dataset"
        self.dataset.mkdir(parents=True)
        for relative in ["cartridges/1/cart.png", "sprites/2/spritesheet.png"]:
            path = self.dataset / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(b"frozen visual")
        self.originals = [
            pool_row(context_priority="2", preferred_visual_type="cartridge",
                     preferred_visual_path="cartridges/1/cart.png"),
            pool_row(tid="2", context_priority="1", serve_order="2", context_tier="A",
                     preferred_visual_type="spritesheet", preferred_visual_path="sprites/2/spritesheet.png"),
            pool_row(song_id="1", context_priority="3", serve_order="3", preferred_visual_type="cartridge",
                     preferred_visual_path="cartridges/1/cart.png"),
            pool_row(tid="3", context_priority="4", serve_order="4", context_tier="D",
                     description_informative="False", description_core=""),
        ]
        self.pool_csv = self.root / "context_pool.csv"
        self.pool_config = self.root / "context_pool_config.json"
        self.pool_csv.write_text(export.csv_text(self.originals, tuple(self.originals[0])), encoding="utf-8")
        self.pool_config.write_text(json.dumps({
            "protocol": "context-pool-v1",
            "context_pool_csv_sha256": hashlib.sha256(self.pool_csv.read_bytes()).hexdigest(),
            "summary": {"total": 4, "tids": 3, "tiers": {"A": 1, "B": 2, "D": 1}},
        }), encoding="utf-8")

    def prepare(self):
        return export.prepare_export(self.pool_csv, self.pool_config, self.dataset,
                                     expected_count=4, expected_tids=3)

    def test_1_to_1_exact_serve_order_manifest_and_d_fallback(self):
        before = copy.deepcopy(self.originals)
        files, audit = self.prepare()
        rows = json.loads(files["scenario_contexts_internal.json"])
        self.assertEqual([(r["tid"], r["song_id"]) for r in rows], [("1", 0), ("2", 0), ("1", 1), ("3", 0)])
        self.assertEqual([r["serve_order"] for r in rows], [1, 2, 3, 4])
        self.assertEqual([r["context_priority"] for r in rows], [2, 1, 3, 4])
        for field in ["context_tier", "start_pattern", "end_pattern"]:
            self.assertEqual([str(r[field]) for r in rows], [r[field] for r in before])
        self.assertEqual(self.originals, before)
        last = rows[-1]
        self.assertEqual(last["context_tier"], "D")
        self.assertEqual(last["public_description"], export.DESCRIPTION_FALLBACK)
        self.assertEqual(last["source_relative_path"], "")
        manifest = list(csv.DictReader(io.StringIO(files["visual_assets_manifest.csv"])))
        self.assertEqual(len(manifest), 3)
        self.assertTrue(all(r["publication_policy"] == export.ASSET_POLICY for r in manifest))
        self.assertFalse(audit["asset_publication"]["direct_copy_authorized"])
        self.assertEqual([r["external_id"] for r in manifest], [r["external_id"] for r in rows[:3]])
        self.assertEqual(manifest[0]["source_relative_path"], manifest[2]["source_relative_path"])
        self.assertTrue(all(audit["checks"].values()))
        self.assertEqual(audit["summary"]["unique_source_assets"], 2)
        self.assertEqual(audit["summary"]["text_fallbacks"], 1)
        self.assertEqual(audit["summary"]["title_fallbacks"], 4)
        self.assertEqual(audit["summary"]["public_titles"], 0)
        self.assertEqual(audit["summary"]["informative_public_descriptions"], 3)
        self.assertEqual(audit["summary"]["visual_types"], {"cartridge": 2, "spritesheet": 1, "none": 1})

    def test_csv_and_json_have_same_schema_and_values(self):
        files, _ = self.prepare()
        csv_rows = list(csv.DictReader(io.StringIO(files["scenario_contexts_internal.csv"])))
        json_rows = json.loads(files["scenario_contexts_internal.json"])
        self.assertEqual(len(csv_rows), len(json_rows))
        for csv_row, json_row in zip(csv_rows, json_rows):
            self.assertEqual(set(csv_row), set(export.FIELDS))
            self.assertEqual(set(json_row), set(export.FIELDS))
            self.assertEqual(csv_row, {key: str(value) for key, value in json_row.items()})

    def test_determinism_relocation_and_output_hashes(self):
        first, audit = self.prepare()
        second, _ = self.prepare()
        self.assertEqual(first, second)
        relocated = self.root.parent / "second"
        shutil.copytree(self.root, relocated)
        third, _ = export.prepare_export(relocated / self.pool_csv.name, relocated / self.pool_config.name,
                                         relocated / "dataset", expected_count=4, expected_tids=3)
        self.assertEqual(first, third)
        for content in first.values():
            self.assertNotIn(str(self.root), content)
        for name, digest in audit["outputs_sha256"].items():
            self.assertEqual(hashlib.sha256(first[name].encode()).hexdigest(), digest)

    def test_tampered_pool_csv_and_wrong_protocol_rejected(self):
        self.pool_csv.write_text(self.pool_csv.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.prepare()
        config = json.loads(self.pool_config.read_text())
        config["protocol"] = "context-pool-v2"
        self.pool_config.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "context-pool-v1"):
            self.prepare()

    def test_wrong_pool_summary_rejected(self):
        config = json.loads(self.pool_config.read_text())
        config["summary"]["tiers"] = {"A": 1, "B": 3}
        self.pool_config.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "summary does not match"):
            self.prepare()

    def test_reordered_rows_and_changed_serve_order_rejected(self):
        cases = [list(reversed(self.originals)), copy.deepcopy(self.originals)]
        cases[1][1]["serve_order"] = "1"
        for rows in cases:
            with self.assertRaisesRegex(ValueError, "exactly follow frozen serve_order"):
                export.project_rows(rows, self.dataset, expected_count=4, expected_tids=3)

    def test_missing_duplicate_or_non_test_identities_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expected 4 contexts"):
            export.project_rows(self.originals[:-1], self.dataset, expected_count=4, expected_tids=3)
        duplicate = copy.deepcopy(self.originals)
        duplicate[2]["song_id"] = "0"
        with self.assertRaisesRegex(ValueError, "Duplicate pool identity"):
            export.project_rows(duplicate, self.dataset, expected_count=4, expected_tids=3)
        wrong_split = copy.deepcopy(self.originals)
        wrong_split[0]["split"] = "train"
        with self.assertRaisesRegex(ValueError, "Only frozen TEST"):
            export.project_rows(wrong_split, self.dataset, expected_count=4, expected_tids=3)


if __name__ == "__main__":
    unittest.main()
