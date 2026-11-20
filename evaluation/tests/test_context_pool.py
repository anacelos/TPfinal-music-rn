import csv
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest

from subjective import context_pool as pool


def evidence(code, start=3, end=5):
    return pool.song_evidence(pool.analyze_lua(code), start, end)


def row(tid="1", song=0, text=True, visual=True, confidence="none", **extra):
    r = {"tid": tid, "song_id": song, "split": "test", "description_informative": text,
         "description_quality": int(text), "description_core_words": 10 if text else 0,
         "visual_informative": visual, "visual_evidence_level": "cartridge_pixels" if visual else "none",
         "music_context_confidence": confidence, **extra}
    r["context_evidence_count"] = pool.evidence_count(r)
    r["context_tier"] = pool.classify_tier(r)
    return r


class TestMusicEvidence(unittest.TestCase):
    def test_pico_comments_including_real_regression(self):
        self.assertEqual(pool.analyze_lua("function scene_1_init() cls(0)\n //music(1,50,0)\nend"), [])
        self.assertEqual(pool.analyze_lua("-- music(3)\n// music(3)\n--[=[music(3)]=]"), [])

    def test_strings_methods_and_quoted_comment_markers(self):
        self.assertEqual(pool.analyze_lua('x="//music(3)" y=[=[music(3)]=] obj:music(3) obj.music(3)'), [])
        self.assertEqual(evidence('x="//"; function _init() music(3) end')["music_context_confidence"], "strong")

    def test_multiple_shadowing(self):
        for prefix in ["local unused, music, another = 0, function(...) end, nil",
                       "music, other = replacement, 0", "other, music = 0, replacement",
                       "local unused,music", "local music=other"]:
            with self.subTest(prefix=prefix):
                self.assertEqual(evidence(prefix + "\nfunction _init() music(3) end")["music_context_confidence"], "weak")

    def test_redefinition_parameters_and_environment(self):
        for code in ["function _init(music) music(3) end", "local _ENV=other; function _init() music(3) end",
                     "function _init() music(3) end function _init() end", "function _init() music(3) end _init=other"]:
            self.assertEqual(evidence(code)["music_context_confidence"], "weak")

    def test_exact_start_and_named_state(self):
        self.assertEqual(evidence("function _init() music(0x3,0,7) end")["music_context_confidence"], "strong")
        self.assertEqual(evidence("function init_game() music(3) end")["music_context_confidence"], "moderate")

    def test_interior_stop_and_unrelated_patterns(self):
        self.assertEqual(evidence("function _init() music(4) end")["music_context_confidence"], "weak")
        self.assertFalse(evidence("music(4)")["music_start_pattern_exact_match"])
        self.assertEqual(evidence("music(-1) music(9)")["music_context_confidence"], "none")

    def test_dynamic_does_not_promote_segment(self):
        e = evidence("music(state.track) music(fn(2,3),500)")
        self.assertEqual(e["music_context_confidence"], "none")
        self.assertEqual(e["music_dynamic_call_count"], 2)
        self.assertEqual(evidence("music(3+0)")["music_context_confidence"], "none")

    def test_conditions_nested_functions_and_offsets(self):
        for code in ["function _init() if ok then music(3) end end",
                     "function _init() if(ok) music(3) end",
                     "function _init() x=ok and music(3) end",
                     "function _init() function nested() music(3) end end"]:
            self.assertEqual(evidence(code)["music_context_confidence"], "weak")
        self.assertEqual(pool.analyze_lua("\nmusic(3)", line_offset=4)[0]["line"], 6)


class TestDescriptionAndVisual(unittest.TestCase):
    def test_short_informative_descriptions_without_punctuation(self):
        for text in ["jogo de pesca de um botão", "This is a fishing one button game I made Enjoy",
                     "you are a raincloud", "Help ninja boy go to school before time runs out"]:
            with self.subTest(text=text):
                self.assertTrue(pool.description_quality(text)["description_informative"])

    def test_boilerplate_and_technical_discussion(self):
        for text in ["", "Updated version added new sprites fixed collisions " * 50,
                     "I learned Lua and enjoyed the built-in text editor I used all four tracks " * 10,
                     "Credits: music by Ada code by Bob", "Controls: arrows move Z jump X shoot"]:
            self.assertEqual(pool.description_quality(text)["description_quality"], 0)

    def test_gameplay_mentioning_controls_is_not_a_section(self):
        d = pool.description_quality("You control a robot and this game controls the weather Collect keys to escape")
        self.assertTrue(d["description_informative"])
        self.assertIn("weather", d["description_core"])

    def test_cart_alphanumeric_and_mixed_content(self):
        d = pool.description_quality("Cart #abc-123 | License: CC4-BY-NC-SA You are a knight Collect gems and rescue friends")
        self.assertTrue(d["description_informative"])
        self.assertNotIn("abc-123", d["description_core"])
        d = pool.description_quality("Updated sprites fixed bugs " * 30 + ". You are a knight. Collect gems.")
        self.assertLess(d["description_quality"], 3)

    def test_meta_request_is_not_player_role(self):
        text = "Teacher if you are looking at this please give me a 100 please please please"
        self.assertFalse(pool.description_quality(text)["description_informative"])
        self.assertFalse(pool.description_quality("You are looking at this submission")["description_informative"])
        self.assertFalse(pool.description_quality("Please give me a grade for my school assignment")["description_informative"])

    def test_meta_blocks_preserve_real_game_context(self):
        for text in [
            "Teacher please give me a 100. You are a robot. Traps kill you.",
            "Teacher please grade my submission You are a fisherman Catch fish using one button",
            "You are a robot Teacher please give me a 100",
            "You are a teacher. Rescue children from the maze.",
            "Please catch fish using one button please",
        ]:
            with self.subTest(text=text):
                self.assertTrue(pool.description_quality(text)["description_informative"])
        core = pool.description_quality("Teacher please give me a 100. You are a robot.")["description_core"]
        self.assertNotIn("give me", core)

    def test_local_role_support_without_formal_punctuation(self):
        for text in [
            "You are a robot. Traps kill you. Your corpse stays behind and reveals more of the map.",
            "You are a fisherman. Catch fish using one button.",
            "You are a robot collect keys to escape the maze",
            "You are inside a dungeon", "jogo de pesca de um botão",
            "You are a domesticated honey bee collecting nectar for your hive",
            "You are Ponyboy Francis working on your farm",
            "You are Beach Sign Cop",
            "Player character You are the spinning blue circle",
            "After shooting a ball you can boost or brake unless you are in no-brake mode",
        ]:
            with self.subTest(text=text):
                self.assertTrue(pool.description_quality(text)["description_informative"])

    def test_incidental_game_name_in_technical_planning(self):
        for text in [
            "testing cartridge loading and tetris map loading",
            "You are testing cartridge loading and tetris map loading",
            "Separate the heavy rooms(tetris) into carts with cartdata and map memory",
        ]:
            self.assertFalse(pool.description_quality(text)["description_informative"])
        self.assertTrue(pool.description_quality("A tetris game where you stack blocks")["description_informative"])

    def test_no_blank_label_signal(self):
        for p8 in ["", "__label__\n" + ("d" * 128 + "\n") * 128, "__label__\nINVALID"]:
            self.assertFalse(pool.pixel_section_evidence(p8, "label")["informative"])
        self.assertFalse(pool.pixel_section_evidence("__label__\n1", "label")["informative"])
        self.assertTrue(pool.pixel_section_evidence("__label__\n" + ("01" * 64 + "\n") * 128, "label")["informative"])

    def test_title_is_auxiliary_and_generic_credits_are_not_titles(self):
        for text in ["-- made by hiko", "-- music by Ada"]:
            self.assertEqual(pool.header_identity(text, "123.p8.png")["game_title"], "")
        d = pool.header_identity("-- My Game\n-- by Ada - music by Bob", "123.p8.png")
        self.assertEqual(d["author"], "")
        self.assertEqual(d["game_title_confidence"], "header_only")


class TestTiersAndServing(unittest.TestCase):
    def test_all_eight_signal_combinations(self):
        for text in [False, True]:
            for visual in [False, True]:
                for music in [False, True]:
                    r = row(text=text, visual=visual, confidence="moderate" if music else "weak")
                    count = sum([text, visual, music])
                    self.assertEqual(r["context_tier"], "DCBA"[count])
                    self.assertEqual(r["context_evidence_count"], count)

    def test_ranking_excludes_metadata_and_tier_dominates(self):
        r = row(); original = pool.priority_key(r)
        r.update(total_stars=100000, game_title="Any", author="Any", comments_present=True, tags='["x"]')
        self.assertEqual(original, pool.priority_key(r))
        rows = [row(text=True, visual=True, confidence="strong"), row(),
                row(text=False), row(text=False, visual=False)]
        self.assertEqual([r["context_tier"] for r in sorted(reversed(rows), key=pool.priority_key)], list("ABCD"))
        self.assertLess(pool.priority_key(row(tid="2")), pool.priority_key(row(tid="10")))

    def test_serve_order_rounds_and_tiers(self):
        rows = [row(tid="1", song=i) for i in range(3)] + [row(tid="2"), row(tid="3")]
        rows += [row(tid="4", text=False)]
        for i,r in enumerate(rows,1):r["context_priority"] = i
        served = pool.assign_serve_order(list(reversed(rows)))
        self.assertEqual([r["tid"] for r in served], ["1","2","3","1","1","4"])
        self.assertEqual([r["context_priority"] for r in served], [1,4,5,2,3,6])
        self.assertEqual([r["serve_order"] for r in served], list(range(1,7)))
        self.assertEqual(served, pool.assign_serve_order(rows))

    def test_identity_validation(self):
        pool.check_identities([row()], [("1",0)])
        for records, expected in [([row(),row()],[("1",0)]),([row()],[("2",0)]),([row(split="train")],[("1",0)])]:
            with self.assertRaises(ValueError):pool.check_identities(records,expected)


class TestIntegration(unittest.TestCase):
    def test_full_fixture_portability_determinism_and_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/"one"; dataset=root/"dataset"; splits=root/"splits"
            dataset.mkdir(parents=True);splits.mkdir()
            def put(relative, raw):
                p=dataset/relative;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(raw if isinstance(raw,bytes) else raw.encode())
            song={"song_id":0,"start_pattern":3,"end_pattern":5}
            put("dataset.jsonl",'{"tid":1}\n')
            put("metadata/1/game_info.json",json.dumps({"tid":1,"description":"a fishing one button game","total_stars":0}))
            put("audio/1/songs.json",json.dumps({"tid":1,"songs":[song]}))
            put("p8/1/1.p8","__lua__\nfunction _init() music(3) end\n__label__\n"+("d"*128+"\n")*128+"__gfx__\n"+"01"*64)
            def png(w,h):return b"\x89PNG\r\n\x1a\n"+struct.pack(">I",13)+b"IHDR"+struct.pack(">II",w,h)+bytes(9)
            put("cartridges/1/cart.p8.png",png(160,205));put("sprites/1/spritesheet.png",png(128,128))
            for name in ["test","train","val"]:
                (splits/(name+".jsonl")).write_text(json.dumps({**song,"tid":"1"})+"\n" if name=="test" else "")
            (splits/"split_hashes.sha256").write_text("".join(hashlib.sha256((splits/(n+".jsonl")).read_bytes()).hexdigest()+"  "+n+".jsonl\n" for n in ["train","val","test"]))
            (splits/"split_manifest.csv").write_text("split,tid,song_id\ntest,1,0\n")
            rows,audit=pool.prepare_pool(dataset,splits,1,1)
            self.assertFalse(rows[0]["cartridge_visual_informative"])
            self.assertEqual(rows[0]["preferred_visual_type"],"spritesheet")
            self.assertTrue(rows[0]["visual_informative"])
            first=pool.serialize_outputs(rows,audit)
            shutil.copytree(root,Path(folder)/"two")
            second=pool.serialize_outputs(*pool.prepare_pool(Path(folder)/"two/dataset",Path(folder)/"two/splits",1,1))
            self.assertEqual(first,second)
            put("p8/1/1.p8","__lua__\nx=1\n__label__\n"+"d"*128)
            rows,_=pool.prepare_pool(dataset,splits,1,1)
            self.assertEqual(rows[0]["preferred_visual_type"],"none")
            (splits/"test.jsonl").write_text("tampered")
            with self.assertRaisesRegex(ValueError,"hash mismatch"):pool.prepare_pool(dataset,splits,1,1)


if __name__ == "__main__":
    unittest.main()
