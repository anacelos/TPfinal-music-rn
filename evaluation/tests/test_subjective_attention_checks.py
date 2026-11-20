import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

import subjective.create_attention_checks as attention_checks
from subjective.create_attention_checks import create_attention_checks


# Actual schema written by evaluation/synthesize_jsonl.py (no extra columns).
SYNTH_FIELDS = ("index", "song_id", "tid", "wav_file", "duration_sec", "num_patterns")
SCRIPT = Path(__file__).resolve().parents[1] / "subjective" / "create_attention_checks.py"


def read_wav(path):
    with wave.open(str(path), "rb") as wav:
        params = wav.getparams()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float64)
    return params, samples


class TestSubjectiveAttentionChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "audio" / "real.wav"
        self.source.parent.mkdir()
        self.sample_rate = 22050  # Ensure it is preserved, not hardcoded to 44100.
        self.signal = np.rint(
            28000 * np.sin(2 * np.pi * 440 * np.arange(8192) / self.sample_rate)
        ).astype("<i2")
        self.write_wav(self.source, self.signal.tobytes())
        self.row = {
            "index": "0", "song_id": "7", "tid": "1818",
            "wav_file": "audio/real.wav", "duration_sec": "0.371519",
            "num_patterns": "1",
        }
        self.manifest = self.root / "manifest.csv"
        self.write_manifest([self.row])
        self.output = self.root / "checks"

    def write_wav(self, path, pcm, channels=1, width=2):
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(width)
            wav.setframerate(self.sample_rate)
            wav.writeframes(pcm)

    def write_manifest(self, rows):
        with self.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SYNTH_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def generate(self, output=None, **kwargs):
        path = create_attention_checks(self.manifest, output or self.output, **kwargs)
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return path, rows

    def test_creates_different_wav_without_modifying_inputs(self):
        original = self.source.read_bytes()
        original_manifest = self.manifest.read_bytes()
        _, rows = self.generate()
        target = self.output / rows[0]["attention_wav"]
        self.assertTrue(target.is_file())
        self.assertNotEqual(target.read_bytes(), original)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(self.manifest.read_bytes(), original_manifest)

    def test_manifest_has_provenance_hashes_and_reproduction_parameters(self):
        _, rows = self.generate(seed=123)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["source_index"], "0")
        self.assertEqual(row["source_song_id"], "7")
        self.assertEqual(row["source_tid"], "1818")
        self.assertEqual(row["source_wav_file"], self.row["wav_file"])
        self.assertEqual(Path(row["source_wav"]), self.source)
        self.assertEqual(Path(row["source_manifest"]), self.manifest)
        self.assertEqual(row["source_sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        target = self.output / row["attention_wav"]
        self.assertEqual(row["attention_sha256"], hashlib.sha256(target.read_bytes()).hexdigest())
        self.assertEqual(row["source_manifest_sha256"], hashlib.sha256(self.manifest.read_bytes()).hexdigest())
        self.assertEqual(float(row["snr_db"]), 0.0)
        self.assertEqual(int(row["global_seed"]), 123)
        self.assertGreaterEqual(int(row["derived_seed"]), 0)
        self.assertLess(int(row["derived_seed"]), 2 ** 128)
        self.assertEqual(json.loads(row["source_identity"]), {
            key: self.row[key] for key in ("index", "tid", "song_id", "wav_file")
        })
        self.assertEqual(row["rng"], "numpy.random.PCG64.standard_normal")
        self.assertEqual(
            row["seed_derivation"],
            "sha256-first128-big-endian-json-utf8-[global_seed,tid,song_id,source_sha256]-v1",
        )
        self.assertEqual(row["numpy_version"], np.__version__)
        self.assertEqual(row["python_version"], platform.python_version())
        self.assertEqual(row["protocol"], "attention-check-v1")

    def test_same_seed_and_input_produce_identical_bytes(self):
        _, first = self.generate(seed=831)
        other = self.root / "other"
        _, second = self.generate(output=other, seed=831)
        self.assertEqual(first, second)
        self.assertEqual(
            (self.output / first[0]["attention_wav"]).read_bytes(),
            (other / second[0]["attention_wav"]).read_bytes(),
        )

    def test_different_seed_changes_noise(self):
        _, first = self.generate(seed=831)
        _, second = self.generate(output=self.root / "other", seed=832)
        self.assertNotEqual(first[0]["derived_seed"], second[0]["derived_seed"])
        self.assertNotEqual(first[0]["attention_sha256"], second[0]["attention_sha256"])

    def test_item_seeds_are_independent_of_manifest_order(self):
        second = {**self.row, "index": "1", "song_id": "8", "wav_file": "audio/second.wav"}
        shutil.copyfile(self.source, self.root / second["wav_file"])
        self.write_manifest([self.row, second])
        _, first_rows = self.generate()
        self.write_manifest([second, self.row])
        _, second_rows = self.generate(output=self.root / "reordered")
        self.assertNotEqual(first_rows[0]["derived_seed"], first_rows[1]["derived_seed"])
        for first, other in zip(first_rows, reversed(second_rows)):
            self.assertEqual(first["derived_seed"], other["derived_seed"])
            self.assertEqual(first["attention_sha256"], other["attention_sha256"])

    def test_relative_input_can_be_relocated_without_changing_noise(self):
        _, first = self.generate()
        moved = self.root / "relocated"
        (moved / "audio").mkdir(parents=True)
        shutil.copyfile(self.source, moved / "audio" / "real.wav")
        shutil.copyfile(self.manifest, moved / "manifest.csv")
        manifest = create_attention_checks(moved / "manifest.csv", moved / "checks")
        with manifest.open(newline="") as handle:
            second = next(csv.DictReader(handle))
        self.assertEqual(first[0]["derived_seed"], second["derived_seed"])
        self.assertEqual(first[0]["attention_sha256"], second["attention_sha256"])

    def test_preserves_sample_rate_frames_and_pcm_format(self):
        _, rows = self.generate()
        params, samples = read_wav(self.output / rows[0]["attention_wav"])
        self.assertEqual(params.framerate, self.sample_rate)
        self.assertEqual(params.nchannels, 1)
        self.assertEqual(params.sampwidth, 2)
        self.assertEqual(params.comptype, "NONE")
        self.assertEqual(params.nframes, self.signal.size)
        self.assertEqual(samples.size, self.signal.size)
        for key, expected in (("sample_rate", self.sample_rate), ("num_frames", self.signal.size),
                              ("num_channels", 1), ("sample_width_bytes", 2)):
            self.assertEqual(int(rows[0][key]), expected)

    def test_global_gain_avoids_clipping_and_preserves_requested_snr(self):
        signal = self.signal.astype(np.float64)
        for snr in (-6.0, 0.0, 6.0):
            with self.subTest(snr=snr):
                output = self.root / f"snr_{snr}"
                _, rows = self.generate(output=output, snr_db=snr)
                row = rows[0]
                _, samples = read_wav(output / row["attention_wav"])
                gain = float(row["post_mix_gain"])
                self.assertGreater(gain, 0)
                self.assertLess(gain, 1)  # Loud fixture requires attenuation.
                self.assertLess(np.max(samples), 32767)
                self.assertGreater(np.min(samples), -32768)
                # Subtract the attenuated original; include final PCM rounding.
                recovered_noise = samples - gain * signal
                actual_snr = 20 * math.log10(
                    np.sqrt(np.mean((gain * signal) ** 2))
                    / np.sqrt(np.mean(recovered_noise ** 2))
                )
                self.assertAlmostEqual(actual_snr, snr, delta=0.01)
                self.assertAlmostEqual(float(row["effective_snr_db"]), actual_snr, places=10)
                self.assertAlmostEqual(float(row["absolute_snr_error_db"]), abs(actual_snr - snr), places=10)
                self.assertEqual(float(row["max_absolute_snr_error_db"]), 0.1)
                self.assertLessEqual(float(row["absolute_snr_error_db"]), 0.1)
                self.assertAlmostEqual(
                    float(row["noise_rms"]),
                    float(row["signal_rms"]) / 10 ** (snr / 20),
                    places=8,
                )

    def test_quiet_input_is_not_unnecessarily_normalized(self):
        self.write_wav(self.source, (self.signal // 100).astype("<i2").tobytes())
        _, rows = self.generate()
        self.assertEqual(float(rows[0]["post_mix_gain"]), 1.0)

    def test_silent_input_fails_explicitly_without_output_wav(self):
        self.write_wav(self.source, np.zeros(512, dtype="<i2").tobytes())
        original = self.source.read_bytes()
        with self.assertRaisesRegex(ValueError, r"real.wav:.*silencioso"):
            self.generate()
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(list(self.output.glob("*.wav")), [])
        self.assertFalse((self.output / "attention_checks_manifest.csv").exists())

    def test_incompatible_channels_and_width_are_rejected(self):
        for channels, width in ((2, 2), (1, 1), (1, 3)):
            with self.subTest(channels=channels, width=width):
                self.write_wav(self.source, b"\x01" * (512 * channels * width), channels, width)
                with self.assertRaisesRegex(ValueError, r"real.wav:.*PCM mono 16-bit"):
                    self.generate()

    def test_empty_invalid_and_truncated_wavs_are_rejected(self):
        self.write_wav(self.source, b"")
        with self.assertRaisesRegex(ValueError, "zero frames"):
            self.generate()
        self.source.write_bytes(b"not a WAV")
        with self.assertRaisesRegex(ValueError, "WAV inválido"):
            self.generate()
        self.write_wav(self.source, self.signal.tobytes())
        self.source.write_bytes(self.source.read_bytes()[:-10])
        with self.assertRaisesRegex(ValueError, "truncado"):
            self.generate()

    def test_existing_wav_and_manifest_are_not_overwritten_by_default(self):
        manifest, rows = self.generate()
        wav = self.output / rows[0]["attention_wav"]
        original_wav = wav.read_bytes()
        original_manifest = manifest.read_bytes()
        with self.assertRaisesRegex(FileExistsError, "--overwrite"):
            self.generate(seed=999)
        self.assertEqual(wav.read_bytes(), original_wav)
        self.assertEqual(manifest.read_bytes(), original_manifest)

    def test_existing_manifest_alone_prevents_writing_wavs(self):
        self.output.mkdir()
        manifest = self.output / "attention_checks_manifest.csv"
        manifest.write_bytes(b"preserve me")
        with self.assertRaisesRegex(FileExistsError, "--overwrite"):
            self.generate()
        self.assertEqual(manifest.read_bytes(), b"preserve me")
        self.assertEqual(list(self.output.glob("*.wav")), [])

    def test_overwrite_replaces_outputs_but_preserves_original(self):
        _, first = self.generate()
        original = self.source.read_bytes()
        _, repeated = self.generate(overwrite=True)
        self.assertEqual(first, repeated)
        _, changed = self.generate(seed=999, overwrite=True)
        self.assertNotEqual(first[0]["attention_sha256"], changed[0]["attention_sha256"])
        self.assertEqual(changed[0]["global_seed"], "999")
        self.assertEqual(self.source.read_bytes(), original)

    def test_overwrite_cannot_modify_source_via_output_alias(self):
        self.output.mkdir()
        target = self.output / "attention_0000.wav"
        original = self.source.read_bytes()
        for kind in ("symlink", "hardlink"):
            with self.subTest(kind=kind):
                if kind == "symlink":
                    target.symlink_to(self.source)
                else:
                    os.link(self.source, target)
                with self.assertRaisesRegex(ValueError, "origem"):
                    self.generate(overwrite=True)
                self.assertEqual(self.source.read_bytes(), original)
                target.unlink()

    def test_overwrite_cannot_replace_input_manifest(self):
        same_manifest = self.root / "attention_checks_manifest.csv"
        self.manifest.rename(same_manifest)
        original = same_manifest.read_bytes()
        with self.assertRaisesRegex(ValueError, "origem"):
            create_attention_checks(same_manifest, self.root, overwrite=True)
        self.assertEqual(same_manifest.read_bytes(), original)

    def test_bad_manifest_missing_file_and_nonfinite_snr_are_rejected(self):
        for snr in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(snr=snr), self.assertRaisesRegex(ValueError, "finito"):
                self.generate(snr_db=snr)
        self.write_manifest([{**self.row, "wav_file": "missing.wav"}])
        with self.assertRaisesRegex(ValueError, "não encontrado"):
            self.generate()
        self.write_manifest([self.row, self.row])
        with self.assertRaisesRegex(ValueError, "duplicado"):
            self.generate()
        self.write_manifest([])
        with self.assertRaisesRegex(ValueError, "sem músicas"):
            self.generate()
        self.manifest.write_text("index,wav_file\n0,a.wav\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "colunas ausentes"):
            self.generate()
        self.assertFalse(self.output.exists())

    def test_cli_is_independent_of_python_hash_seed_and_supports_overwrite(self):
        outputs = []
        for hash_seed in ("1", "999"):
            output = self.root / f"cli_{hash_seed}"
            command = [sys.executable, "-B", str(SCRIPT), "--manifest", str(self.manifest),
                       "--output-dir", str(output), "--seed", "831"]
            env = {**os.environ, "PYTHONHASHSEED": hash_seed}
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            outputs.append((output / "attention_0000.wav").read_bytes())
        self.assertEqual(outputs[0], outputs[1])
        rejected = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(rejected.returncode, 1)
        self.assertIn("--overwrite", rejected.stderr)
        overwritten = subprocess.run(command + ["--overwrite"], capture_output=True, text=True)
        self.assertEqual(overwritten.returncode, 0, overwritten.stderr)
        self.assertEqual((output / "attention_0000.wav").read_bytes(), outputs[1])

    def test_absolute_and_relative_paths_produce_same_seed_and_bytes(self):
        _, relative = self.generate()
        self.write_manifest([{**self.row, "wav_file": str(self.source)}])
        _, absolute = self.generate(output=self.root / "absolute")
        self.assertEqual(relative[0]["derived_seed"], absolute[0]["derived_seed"])
        self.assertEqual(relative[0]["attention_sha256"], absolute[0]["attention_sha256"])
        self.assertEqual(
            (self.output / relative[0]["attention_wav"]).read_bytes(),
            (self.root / "absolute" / absolute[0]["attention_wav"]).read_bytes(),
        )

    def test_changing_only_wav_content_changes_seed(self):
        _, first = self.generate()
        changed = self.signal.copy()
        changed[0] += 1
        self.write_wav(self.source, changed.tobytes())
        _, second = self.generate(output=self.root / "changed")
        self.assertNotEqual(first[0]["source_sha256"], second[0]["source_sha256"])
        self.assertNotEqual(first[0]["derived_seed"], second[0]["derived_seed"])

    def test_changing_tid_changes_seed(self):
        _, first = self.generate()
        self.write_manifest([{**self.row, "tid": "other_game"}])
        _, second = self.generate(output=self.root / "changed")
        self.assertNotEqual(first[0]["derived_seed"], second[0]["derived_seed"])

    def test_changing_song_id_changes_seed(self):
        _, first = self.generate()
        self.write_manifest([{**self.row, "song_id": "other_song"}])
        _, second = self.generate(output=self.root / "changed")
        self.assertNotEqual(first[0]["derived_seed"], second[0]["derived_seed"])

    def test_changing_index_does_not_change_seed_or_bytes(self):
        _, first = self.generate()
        self.write_manifest([{**self.row, "index": "99"}])
        _, second = self.generate(output=self.root / "changed")
        self.assertEqual(second[0]["source_index"], "99")
        self.assertNotEqual(first[0]["attention_wav"], second[0]["attention_wav"])
        self.assertEqual(first[0]["derived_seed"], second[0]["derived_seed"])
        self.assertEqual(first[0]["attention_sha256"], second[0]["attention_sha256"])

    def test_seed_matches_exact_portable_serialization(self):
        _, rows = self.generate(seed=831)
        digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        payload = json.dumps([831, "1818", "7", digest], ensure_ascii=False, separators=(",", ":"))
        expected = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:16], "big")
        self.assertEqual(int(rows[0]["derived_seed"]), expected)

    def test_empty_tid_is_rejected_before_outputs(self):
        for value in ("", " \t"):
            with self.subTest(value=value):
                self.write_manifest([self.row, {**self.row, "index": "1", "tid": value}])
                with self.assertRaisesRegex(ValueError, "tid vazio"):
                    self.generate()
                self.assertFalse(self.output.exists())

    def test_empty_song_id_is_rejected_before_outputs(self):
        for value in ("", " \t"):
            with self.subTest(value=value):
                self.write_manifest([self.row, {**self.row, "index": "1", "song_id": value}])
                with self.assertRaisesRegex(ValueError, "song_id vazio"):
                    self.generate()
                self.assertFalse(self.output.exists())

    def test_duplicate_logical_identity_is_rejected_before_outputs(self):
        self.write_manifest([self.row, {**self.row, "index": "1"}])
        with self.assertRaisesRegex(ValueError, r"identidade .*duplicada"):
            self.generate()
        self.assertFalse(self.output.exists())

    def test_non_silent_input_losing_noise_in_quantization_is_rejected(self):
        tiny = np.zeros(8192, dtype="<i2")
        tiny[0] = 1
        self.write_wav(self.source, tiny.tobytes())
        original = self.source.read_bytes()
        with self.assertRaisesRegex(ValueError, r"real.wav:.*idêntico.*quantização"):
            self.generate()
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_quantization_error_over_tolerance_is_rejected(self):
        self.write_wav(self.source, np.ones(8192, dtype="<i2").tobytes())
        with self.assertRaisesRegex(ValueError, r"erro absoluto de SNR .*excede 0.1 dB"):
            self.generate()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_zero_residual_and_nonfinite_effective_snr_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "ruído efetivo desapareceu"):
            attention_checks.measure_effective_snr(np.array([2.]), np.array([1]), 0.5, 0.0)
        with self.assertRaisesRegex(ValueError, "SNR efetiva não finita"):
            attention_checks.measure_effective_snr(np.array([2.]), np.array([float("inf")]), 1.0, 0.0)

    def test_out_of_range_snr_is_rejected_before_processing(self):
        for snr in (-6070.0, -60.001, 60.001, 200.0):
            with self.subTest(snr=snr):
                with self.assertRaisesRegex(ValueError, "entre -60 e \\+60"):
                    self.generate(snr_db=snr)
                self.assertFalse(self.output.exists())

    def test_snr_range_endpoints_are_supported(self):
        for snr in (-60.0, 60.0):
            with self.subTest(snr=snr):
                _, rows = self.generate(output=self.root / f"endpoint_{snr}", snr_db=snr)
                self.assertLessEqual(float(rows[0]["absolute_snr_error_db"]), 0.1)

    def test_later_item_failure_publishes_no_manifest_or_wavs(self):
        second = self.root / "audio" / "silent.wav"
        self.write_wav(second, np.zeros(512, dtype="<i2").tobytes())
        self.write_manifest([self.row, {**self.row, "index": "1", "song_id": "8", "wav_file": "audio/silent.wav"}])
        with self.assertRaisesRegex(ValueError, r"silent.wav:.*silencioso"):
            self.generate()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_validation_failure_preserves_previous_complete_batch(self):
        manifest, rows = self.generate()
        target = self.output / rows[0]["attention_wav"]
        old_manifest = manifest.read_bytes()
        old_wav = target.read_bytes()
        silent = self.root / "audio" / "silent.wav"
        self.write_wav(silent, np.zeros(512, dtype="<i2").tobytes())
        self.write_manifest([self.row, {**self.row, "index": "1", "song_id": "8", "wav_file": "audio/silent.wav"}])
        with self.assertRaisesRegex(ValueError, "silencioso"):
            self.generate(seed=999, overwrite=True)
        self.assertEqual(manifest.read_bytes(), old_manifest)
        self.assertEqual(target.read_bytes(), old_wav)
        self.assertEqual(set(self.output.iterdir()), {manifest, target})

    def test_publication_failure_invalidates_old_manifest_and_cleans_temps(self):
        self.write_manifest([self.row, {**self.row, "index": "1", "song_id": "8"}])
        manifest, _ = self.generate()
        replace = os.replace

        def fail_second_wav(source, target):
            if Path(target).name == "attention_0001.wav":
                raise OSError("simulated publication failure")
            replace(source, target)

        with patch.object(attention_checks.os, "replace", side_effect=fail_second_wav):
            with self.assertRaisesRegex(OSError, "publication failure"):
                self.generate(seed=999, overwrite=True)
        self.assertFalse(manifest.exists())
        self.assertEqual(set(p.name for p in self.output.iterdir()), {"attention_0000.wav", "attention_0001.wav"})

    def test_atomic_wav_replacement_preserves_old_symlink_or_hardlink_inode(self):
        for kind in ("symlink", "hardlink"):
            with self.subTest(kind=kind):
                output = self.root / kind
                output.mkdir()
                previous = self.root / f"previous_{kind}.wav"
                previous.write_bytes(b"old inode must remain unchanged")
                inode = previous.stat().st_ino
                target = output / "attention_0000.wav"
                if kind == "symlink":
                    target.symlink_to(previous)
                else:
                    os.link(previous, target)
                self.generate(output=output, overwrite=True)
                self.assertEqual(previous.read_bytes(), b"old inode must remain unchanged")
                self.assertEqual(previous.stat().st_ino, inode)
                self.assertFalse(target.is_symlink())
                self.assertNotEqual(target.stat().st_ino, inode)

    def test_atomic_manifest_replacement_preserves_old_alias_inode(self):
        for kind in ("symlink", "hardlink"):
            with self.subTest(kind=kind):
                output = self.root / kind
                output.mkdir()
                previous = self.root / f"previous_{kind}.csv"
                previous.write_bytes(b"previous inode")
                inode = previous.stat().st_ino
                manifest = output / "attention_checks_manifest.csv"
                if kind == "symlink":
                    manifest.symlink_to(previous)
                else:
                    os.link(previous, manifest)
                self.generate(output=output, overwrite=True)
                self.assertEqual(previous.read_bytes(), b"previous inode")
                self.assertEqual(previous.stat().st_ino, inode)
                self.assertNotEqual(manifest.stat().st_ino, inode)

    def test_hardlinks_between_output_wavs_do_not_corrupt_hashes(self):
        self.output.mkdir()
        previous = self.root / "previous.wav"
        previous.write_bytes(b"shared inode")
        os.link(previous, self.output / "attention_0000.wav")
        os.link(previous, self.output / "attention_0001.wav")
        self.write_manifest([self.row, {**self.row, "index": "1", "song_id": "8"}])
        _, rows = self.generate(overwrite=True)
        self.assertEqual(previous.read_bytes(), b"shared inode")
        self.assertNotEqual(rows[0]["attention_sha256"], rows[1]["attention_sha256"])
        for row in rows:
            self.assertEqual(row["attention_sha256"], hashlib.sha256((self.output / row["attention_wav"]).read_bytes()).hexdigest())

    def test_temporaries_are_cleaned_after_success(self):
        manifest, rows = self.generate()
        self.assertEqual(set(self.output.iterdir()), {manifest, self.output / rows[0]["attention_wav"]})

    def test_temporaries_are_cleaned_after_wav_write_failure(self):
        with patch.object(wave.Wave_write, "writeframes", side_effect=OSError("simulated disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.generate()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_manifest_is_published_last_and_files_are_staged_in_output_dir(self):
        replace = os.replace
        published = []

        def observe_replace(source, target):
            source, target = Path(source), Path(target)
            self.assertEqual(source.parent, self.output)
            self.assertEqual(target.parent, self.output)
            self.assertTrue(source.name.startswith(".attention-"))
            if target.suffix == ".wav":
                self.assertFalse((self.output / "attention_checks_manifest.csv").exists())
            else:
                self.assertTrue((self.output / "attention_0000.wav").is_file())
            published.append(target.name)
            replace(source, target)

        with patch.object(attention_checks.os, "replace", side_effect=observe_replace):
            self.generate()
        self.assertEqual(published, ["attention_0000.wav", "attention_checks_manifest.csv"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
