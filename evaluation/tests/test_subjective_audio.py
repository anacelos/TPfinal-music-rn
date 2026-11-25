"""Small in-memory protocol tests; no production audio, matching or Attention."""

import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from evaluation.subjective import render_subjective_audio as audio
from src import synth


def song(*, ticks=4, instrument=0, end_reason="stop", loop_start=None):
    note = dict(pitch=33, instrument=instrument, volume=5, effect=0)
    channel = dict(active=True, speed=1, notes_unrolled=[note] * ticks, original_sfx_id=3, block_id=0)
    return dict(tid="1", song_id=0, end_reason=end_reason, loop_start_pattern=loop_start,
                patterns=[dict(pattern_id=0, pattern_ticks=ticks, ref_speed=1, channels={"ch1": channel})],
                blocks_dictionary={"0": dict(original_sfx_id=3, notes=[note] * ticks, loop_start=0, loop_end=0)})


class TestSubjectiveAudio(unittest.TestCase):
    def test_stress_plan_is_113_full_plays_not_113_extra_repeats(self):
        frames, plays = audio.repeat_plan(song(ticks=32))
        self.assertEqual((frames, plays), (11712, 113))
        self.assertEqual(frames * plays, 1323456)
        self.assertEqual(audio.TARGET_FRAMES, 1323000)

    def test_exact_threshold_and_long_music_do_not_repeat(self):
        source = song(ticks=4)
        self.assertEqual(audio.repeat_plan(source, 1464), (1464, 1))
        self.assertEqual(audio.repeat_plan(source, 1000), (1464, 1))
        self.assertEqual(audio.repeat_plan(source, 1465), (1464, 2))

    def test_empty_duration_fails_instead_of_padding(self):
        for source in (song(ticks=0), {"patterns": []}):
            with self.assertRaisesRegex(ValueError, "zero-duration"):
                audio.repeat_plan(source)

    def test_native_loop_parameter_is_suffix_only_and_ignores_stop(self):
        source = song(ticks=2)
        second = copy.deepcopy(source["patterns"][0])
        second["pattern_id"] = 1
        source["patterns"].append(second)
        self.assertEqual(len(synth._song_patterns(source, 2)), 2)
        source.update(end_reason="loop_end", loop_start_pattern=1)
        self.assertEqual([p["pattern_id"] for p in synth._song_patterns(source, 2)], [0, 1, 1, 1])

    def test_full_sequence_repeats_for_stop_and_loop_suffix_songs(self):
        for reason in ("stop", "loop_end"):
            source = song(ticks=2, end_reason=reason, loop_start=1)
            second = copy.deepcopy(source["patterns"][0])
            second["pattern_id"] = 1
            second["channels"]["ch1"]["notes_unrolled"][0]["pitch"] = 45
            source["patterns"].append(second)
            expected = synth.synthesize_song(dict(source, patterns=source["patterns"] * 2))[:2000]
            raw, audit = audio.render_clip(source, "markov", {}, target_frames=2000)
            np.testing.assert_array_equal(audio.validate_wav(raw, 2000), np.round(expected * 32767).astype("<i2"))
            self.assertEqual(audit["full_song_play_count"], 2)
            self.assertEqual(audit["engine_loop_extra_repeats"], 0)

    def test_native_state_continues_instead_of_tiling_waveform(self):
        source = song(ticks=2, instrument=6)
        original = synth.synthesize_song(source)
        target = len(original) * 2
        raw, _ = audio.render_clip(source, "neural", {}, target_frames=target)
        expected = synth.synthesize_song(dict(source, patterns=source["patterns"] * 2))
        pcm = audio.validate_wav(raw, target)
        np.testing.assert_array_equal(pcm, np.round(expected * 32767).astype("<i2"))
        self.assertFalse(np.array_equal(pcm, np.tile(np.round(original * 32767).astype("<i2"), 2)))

    def test_long_clip_is_exact_native_writer_prefix(self):
        source = song(ticks=8)
        buffer = io.BytesIO()
        native_open = wave.open
        with patch("src.synth.os.makedirs"), patch("src.synth.wave.open", side_effect=lambda *a, **k: native_open(buffer, "wb")):
            synth.synthesize_song_to_wav(source, "unused.wav")
        with native_open(io.BytesIO(buffer.getvalue()), "rb") as handle:
            native_pcm = handle.readframes(1000)
        raw, audit = audio.render_clip(source, "markov", {}, target_frames=1000)
        self.assertEqual(audio.validate_wav(raw, 1000).tobytes(), native_pcm)
        self.assertEqual(audit["extra_repeat_count"], 0)
        self.assertEqual(audit["crop_start_frame"], 0)

    def test_real_gets_filters_and_inputs_are_immutable(self):
        source = song(ticks=8)
        before = copy.deepcopy(source)
        table = {"1": {"3": 24}}
        raw, audit = audio.render_clip(source, "real", table, target_frames=2500)
        self.assertTrue(audit["sfx_filters_applied"])
        self.assertEqual(audit["filtered_entries"], {"blocks": 1, "channel_occurrences": 1})
        self.assertEqual(audit["sfx_filters_sha256"], audio.PINNED["src/sfx_filters.json"])
        self.assertEqual(source, before)
        self.assertEqual(audit["source_record_sha256"], audio.sha(audio.canonical(before)))
        self.assertTrue(raw)

    def test_generated_never_gets_table_despite_shared_tid(self):
        source = song()
        with patch.object(audio, "attach", side_effect=AssertionError("must not attach")):
            for origin in ("markov", "neural"):
                _, audit = audio.render_clip(source, origin, {"1": {"3": 24}}, target_frames=1000)
                self.assertFalse(audit["sfx_filters_applied"])
                self.assertIsNone(audit["sfx_filters_sha256"])
                self.assertFalse(any(audit["filtered_entries"].values()))

    def test_missing_real_table_and_unexpected_generated_filters_fail(self):
        with self.assertRaisesRegex(ValueError, "Real requires"):
            audio.prepare_song(song(), "real", None)
        source = song()
        source["blocks_dictionary"]["0"]["filters"] = 24
        with self.assertRaisesRegex(ValueError, "unexpectedly"):
            audio.prepare_song(source, "neural", {})
        with self.assertRaisesRegex(ValueError, "Attention"):
            audio.prepare_song(song(), "attention", {})

    def test_nonfinite_and_invalid_native_samples_fail(self):
        for value in (float("nan"), float("inf"), 2.0):
            with patch.object(synth, "synthesize_song", return_value=np.full(1464, value, dtype=np.float32)):
                with self.assertRaisesRegex(ValueError, "Invalid native"):
                    audio.render_clip(song(), "markov", {}, target_frames=1000)

    def test_noise_is_byte_deterministic_and_format_exact(self):
        source = song(instrument=6)
        first, first_audit = audio.render_clip(source, "neural", {}, target_frames=3000)
        second, second_audit = audio.render_clip(source, "neural", {}, target_frames=3000)
        self.assertEqual(first, second)
        self.assertEqual(first_audit, second_audit)
        with wave.open(io.BytesIO(first), "rb") as handle:
            self.assertEqual((handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getnframes()), (1, 2, 44100, 3000))
        self.assertEqual(first_audit["validation"]["conversion_clipped_samples"], 0)

    def test_native_mix_saturation_is_preserved_without_added_clipping_or_gain(self):
        source = song(ticks=8, instrument=0)
        channel = source["patterns"][0]["channels"]["ch1"]
        source["patterns"][0]["channels"] = {f"ch{i}": copy.deepcopy(channel) for i in range(1, 5)}
        raw, audit = audio.render_clip(source, "markov", {}, target_frames=1000)
        expected = synth.synthesize_song(source)[:1000]
        np.testing.assert_array_equal(audio.validate_wav(raw, 1000), np.round(expected * 32767).astype("<i2"))
        self.assertGreater(audit["validation"]["native_mix_saturation_rail_samples"], 0)
        self.assertGreater(audit["validation"]["pcm_rail_samples"], 0)
        self.assertEqual(audit["validation"]["conversion_clipped_samples"], 0)

    def test_silence_and_wrong_wav_duration_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "silent"):
            audio.validate_wav(audio.wav_bytes(np.zeros(100, dtype="<i2")), 100)
        with self.assertRaisesRegex(ValueError, "duration"):
            audio.validate_wav(audio.wav_bytes(np.ones(100, dtype="<i2")), 101)

    def test_mixed_smoke_choice_is_lowest_index_and_fixed_required_cases(self):
        rows = [dict(selection_index=i, **{f"{o}_duration_sec": 1 for o in audio.ORIGINS}) for i in range(48)]
        rows[1].update({f"{o}_duration_sec": 31 for o in audio.ORIGINS})
        rows[3]["neural_duration_sec"] = 31
        rows[6]["markov_duration_sec"] = 31
        self.assertEqual(audio.smoke_indices(list(reversed(rows))), [42, 1, 3])

    def test_existing_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            marker = path / "keep.txt"
            marker.write_text("unchanged")
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                audio.render_smoke(path, [], {}, {})
            self.assertEqual(marker.read_text(), "unchanged")

    def test_failed_render_does_not_publish_or_leave_staging_files(self):
        jobs = [dict(selection_index=i, origin=o, song=song()) for i in (42, 1, 3) for o in audio.ORIGINS]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new"
            with patch.object(audio, "render_clip", side_effect=ValueError("injected failure")), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "injected failure"):
                    audio.render_smoke(output, jobs, {}, {"scenario_indices": [42, 1, 3]})
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_cli_rejects_ssd_before_loading_or_rendering(self):
        with patch.object(audio, "load_smoke", side_effect=AssertionError("must not load")):
            with self.assertRaisesRegex(ValueError, "HDD"):
                audio.main(["--smoke", "--output-dir", "/tmp/not-authorized-audio"])


if __name__ == "__main__":
    unittest.main()
