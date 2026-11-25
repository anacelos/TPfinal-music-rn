"""Small temporary fixtures test packaging, blinding and audit failures."""

import copy
import io
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import uuid

import numpy as np
from PIL import Image, PngImagePlugin

from evaluation.subjective import build_subjective_handoff as handoff


FRAMES = 256


def fixture(root):
    permutations = list(itertools.permutations(handoff.ORIGINS)) * 2
    rows, scenarios = [], []
    image = io.BytesIO()
    Image.new("RGB", (8, 8), (4, 8, 12)).save(image, format="PNG")
    png = image.getvalue()
    signals = {o: np.rint(9000 * np.sin(np.arange(FRAMES) * (0.1 + i * 0.1))).astype("<i2")
               for i, o in enumerate(handoff.audio.ORIGINS)}
    for i, permutation in enumerate(permutations):
        identifier = str(uuid.uuid5(uuid.NAMESPACE_URL, f"fixture-{i}"))
        hashes = {o: handoff.sha(f"{i}:{o}".encode()) for o in handoff.audio.ORIGINS}
        row = dict(selection_index=i, external_id=identifier, real_record_id=f"real:{i}:0", real_tid=str(i), real_song_id=0,
                   real_record_sha256=hashes["real"], markov_generated_id="sha256:" + hashes["markov"],
                   neural_generated_id="sha256:" + hashes["neural"], markov_record_sha256=hashes["markov"],
                   neural_record_sha256=hashes["neural"], markov_song_id=i, neural_song_id=i,
                   neural_system=handoff.subset.NEURAL_SYSTEM, markov_matching_cost=0.2, neural_matching_cost=0.4,
                   serve_order=i + 1, context_tier="A", public_description="Contexto do jogo.", public_context_heading="Contexto",
                   public_visual_sha256=handoff.sha(png), presentation=dict(permutation_protocol="abcd-balanced-rotation-v1",
                                                                        assignment=dict(zip(handoff.SLOTS, permutation))))
        rows.append(row)
        folder = root / "private/audio" / identifier
        folder.mkdir(parents=True)
        exposed = root / "public/scenarios" / identifier
        exposed.mkdir(parents=True)
        audits = {}
        for origin in handoff.audio.ORIGINS:
            raw = handoff.audio.wav_bytes(signals[origin])
            path = folder / f"{origin}.wav"
            path.write_bytes(raw)
            audits[origin] = {"private_file": path.relative_to(root).as_posix(), "wav_sha256": handoff.sha(raw),
                              "source_record_sha256": hashes[origin], "protocol": handoff.audio.PROTOCOL,
                              "sfx_filters_applied": origin == "real",
                              "sfx_filters_sha256": handoff.audio.PINNED["src/sfx_filters.json"] if origin == "real" else None,
                              "validation": {"conversion_clipped_samples": 0, "native_mix_saturation_rail_samples": 0}}
        parent_hash = audits["real"]["wav_sha256"]
        seed = handoff.attention.derive_seed(831, str(i), "0", parent_hash)
        signal = signals["real"].astype(np.float64)
        pcm, rms, noise_rms, gain = handoff.attention.mix_noise(signal, 0, seed)
        snr, error = handoff.attention.measure_effective_snr(signal, pcm, gain, 0)
        raw = handoff.audio.wav_bytes(pcm)
        path = folder / "attention.wav"
        path.write_bytes(raw)
        audits["attention"] = {"private_file": path.relative_to(root).as_posix(), "wav_sha256": handoff.sha(raw),
                               "protocol": handoff.attention.PROTOCOL, "parent_origin": "real", "parent_wav_sha256": parent_hash,
                               "global_seed": 831, "target_snr_db": 0, "source_tid": str(i), "source_song_id": "0",
                               "derived_seed": seed, "post_mix_gain": gain, "signal_rms": rms, "noise_rms": noise_rms,
                               "effective_snr_db": snr, "absolute_snr_error_db": error}
        for slot, origin in row["presentation"]["assignment"].items():
            shutil.copyfile(folder / f"{origin}.wav", exposed / f"{slot}.wav")
        (exposed / "context.png").write_bytes(png)
        scenarios.append(handoff.private_scenario(row, audits))
    manifest = handoff.json_bytes([handoff.public_projection(r) for r in rows])
    (root / "public/scenarios.json").write_bytes(manifest)
    (root / "private/scenario_key.json").write_bytes(handoff.json_bytes({"protocol": handoff.PROTOCOL,
        "public_manifest_sha256": handoff.sha(manifest), "scenarios": scenarios}))
    shutil.copyfile(handoff.DOC, root / "HANDOFF_GIOVANNI.md")
    return rows


class TestSubjectiveHandoff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = tempfile.TemporaryDirectory()
        cls.rows = fixture(Path(cls.template.name))

    @classmethod
    def tearDownClass(cls):
        cls.template.cleanup()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / "package"
        shutil.copytree(self.template.name, self.root)

    def validate(self):
        return handoff.validate_bundle(self.root, self.rows, target_frames=FRAMES)

    def key(self):
        return json.loads((self.root / "private/scenario_key.json").read_bytes())

    def save_key(self, key):
        (self.root / "private/scenario_key.json").write_bytes(handoff.json_bytes(key))

    def save_manifest(self, manifest):
        raw = handoff.json_bytes(manifest)
        (self.root / "public/scenarios.json").write_bytes(raw)
        key = self.key()
        key["public_manifest_sha256"] = handoff.sha(raw)
        self.save_key(key)

    def test_complete_blinded_fixture_validates_and_balances_all_slots(self):
        summary = self.validate()
        self.assertEqual((summary["scenarios"], summary["pngs"], summary["public_wavs"]), (48, 48, 192))
        self.assertEqual(summary["attention_recomputed_and_bound_to_real"], 48)
        self.assertTrue(all(c == {o: 12 for o in handoff.ORIGINS} for c in summary["origin_slot_counts"].values()))

    def test_projection_has_only_the_public_allowlist(self):
        public = handoff.public_projection(self.rows[0])
        self.assertEqual(set(public), handoff.PUBLIC_FIELDS)
        self.assertEqual(set(public["audios"]), set("ABCD"))
        self.assertNotIn("matching", json.dumps(public))
        self.assertNotIn("sha256", json.dumps(public))

    def test_private_mapping_cannot_be_added_to_public_manifest(self):
        manifest = json.loads((self.root / "public/scenarios.json").read_bytes())
        manifest[0]["mapping"] = self.rows[0]["presentation"]["assignment"]
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "Public schema"):
            self.validate()

    def test_traversal_or_origin_named_audio_paths_are_rejected(self):
        original = json.loads((self.root / "public/scenarios.json").read_bytes())
        for path in ("../private/audio/real.wav", "scenarios/real.wav", "/absolute/A.wav"):
            manifest = copy.deepcopy(original)
            manifest[0]["audios"]["A"] = path
            self.save_manifest(manifest)
            with self.assertRaisesRegex(ValueError, "Public projection"):
                self.validate()

    def test_changing_frozen_assignment_is_rejected(self):
        key = self.key()
        m = key["scenarios"][0]["mapping"]
        m["A"], m["B"] = m["B"], m["A"]
        self.save_key(key)
        with self.assertRaisesRegex(ValueError, "Frozen mapping"):
            self.validate()

    def test_swapping_actual_audio_is_rejected(self):
        folder = self.root / "public/scenarios" / self.rows[0]["external_id"]
        shutil.copyfile(folder / "B.wav", folder / "A.wav")
        with self.assertRaisesRegex(ValueError, "byte-identical"):
            self.validate()

    def test_private_file_inside_public_tree_is_rejected(self):
        shutil.copyfile(self.root / "private/scenario_key.json", self.root / "public/scenario_key.json")
        with self.assertRaisesRegex(ValueError, "Unexpected/missing public"):
            self.validate()

    def test_public_symlink_is_rejected(self):
        (self.root / "public/leak").symlink_to(self.root / "private", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.validate()

    def test_wav_tags_and_wrong_duration_are_rejected(self):
        raw = handoff.audio.wav_bytes(np.ones(FRAMES, dtype="<i2"))
        for modified in (raw + b"LISTreal", raw[:-2]):
            with self.assertRaisesRegex(ValueError, "metadata/size"):
                handoff.validate_public_wav(modified, FRAMES)

    def test_png_text_metadata_and_low_bits_are_rejected(self):
        for color, text in (((4, 8, 12), True), ((5, 8, 12), False)):
            raw = io.BytesIO()
            info = PngImagePlugin.PngInfo()
            if text:
                info.add_text("origin", "real")
            Image.new("RGB", (8, 8), color).save(raw, format="PNG", pnginfo=info)
            with self.assertRaises(ValueError):
                handoff.validate_png(raw.getvalue())

    def test_attention_wrong_parent_or_seed_is_rejected(self):
        original = self.key()
        for field, value in (("parent_wav_sha256", "0" * 64), ("derived_seed", 0)):
            key = copy.deepcopy(original)
            key["scenarios"][0]["audios"]["attention"][field] = value
            self.save_key(key)
            with self.assertRaisesRegex(ValueError, "Attention"):
                self.validate()

    def test_attention_uses_exact_final_real_bytes_and_is_deterministic(self):
        row = self.rows[0]
        folder = self.root / "private/audio" / row["external_id"]
        original_validator = handoff.audio.validate_wav
        with patch.object(handoff.audio, "TARGET_FRAMES", FRAMES), patch.object(handoff.audio, "validate_wav", side_effect=lambda raw: original_validator(raw, FRAMES)):
            first, audit = handoff.make_attention(folder / "real.wav", row)
            second, repeated = handoff.make_attention(folder / "real.wav", row)
        self.assertEqual(first, second)
        self.assertEqual(audit, repeated)
        self.assertEqual(first, (folder / "attention.wav").read_bytes())
        self.assertEqual(audit["parent_wav_sha256"], handoff.file_sha(folder / "real.wav"))

    @unittest.skipUnless(shutil.which("zstd"), "zstd not installed")
    def test_archive_excludes_private_and_is_byte_reproducible(self):
        first, second = self.directory / "first.tar.zst", self.directory / "second.tar.zst"
        handoff.write_archive(self.root, first)
        handoff.write_archive(self.root, second)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        summary = handoff.validate_archive(self.root, first)
        self.assertEqual((summary["regular_files"], summary["private_files"]), (242, 0))
        with self.assertRaisesRegex(ValueError, "already exists"):
            handoff.write_archive(self.root, first)

    @unittest.skipUnless(shutil.which("zstd"), "zstd not installed")
    def test_archive_with_private_member_is_rejected(self):
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as packed:
            member = tarfile.TarInfo("private/scenario_key.json")
            member.size = 2
            packed.addfile(member, io.BytesIO(b"{}"))
        archive = self.directory / "leak.tar.zst"
        with archive.open("wb") as handle:
            subprocess.run(["zstd", "-q", "-c"], input=raw.getvalue(), stdout=handle, check=True)
        with self.assertRaisesRegex(ValueError, "Nonpublic"):
            handoff.validate_archive(self.root, archive)

    def test_failed_build_leaves_no_published_output(self):
        output = self.directory / "failed"
        with patch.object(handoff, "load_inputs", return_value=([], {}, {}, {}, {})), patch.object(handoff, "render_bundle", side_effect=ValueError("injected failure")):
            with self.assertRaisesRegex(ValueError, "injected"):
                handoff.build_handoff(output)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".tar.zst").exists())
        self.assertEqual(list(self.directory.glob(".subjective-handoff-*")), [])

    def test_existing_package_is_not_modified(self):
        before = (self.root / "public/scenarios.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            handoff.build_handoff(self.root)
        self.assertEqual((self.root / "public/scenarios.json").read_bytes(), before)

    def test_cli_rejects_ssd_before_loading_inputs(self):
        with patch.object(handoff, "build_handoff", side_effect=AssertionError("must not build")):
            with self.assertRaisesRegex(ValueError, "HDD"):
                handoff.main(["--output-dir", "/tmp/not-final-audio"])


if __name__ == "__main__":
    unittest.main()
