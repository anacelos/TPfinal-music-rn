#!/usr/bin/env python3
"""subjective-audio-30s-v1: full symbolic cycles, native synthesis, exact crop.

The CLI deliberately implements only the authorized three-scenario smoke. All
manifests are internal provenance, not participant-facing assets or DTOs.
"""

import argparse
import io
import json
import os
from pathlib import Path
import platform
import tempfile
import wave

import numpy as np
import scipy

from src import synth
from src.sfx_filters import attach, load_table
from . import select_subjective_subset as subset


REPO = Path(__file__).resolve().parents[2]
PROTOCOL = "subjective-audio-30s-v1"
TARGET_FRAMES = 30 * synth.SAMPLE_RATE
ORIGINS = ("real", "markov", "neural")
DEFAULT_OUTPUT = Path("/mnt/hd/Documentos/mestrado/IAGenerativa/TP2/subjective-audio-30s-v1-smoke")
PINNED = {
    "src/synth.py": "de530068b3d8f752b35ea5506f7125791a48a976211a5020e0ec62a84e980b05",
    "src/sfx_filters.py": "21bda8556eab41c79a597023051e60619f65adcb74f32b1f96b3c1ef594e4e8c",
    "src/sfx_filters.json": "069f9c63de5ad1959743abfd1b4698d171bee346db3a7d712f46623c4794dc07",
    "outputs/subjective/subjective-48-v1/scenarios.jsonl": "6c8c966b99c51d4eef8fd2a24a93b061ee6f7954d803f051f454aa238649fae5",
    "outputs/subjective/subjective-48-v1/selection_config.json": "487197c15fc074b89daeb9e448548232a17fd06bde9f055d1d1d8facd6d3a692",
}
require = subset.require
sha = subset.sha
canonical = subset.canonical
json_bytes = subset.json_bytes


def original_frames(song):
    """Exactly the native synth's pattern clock, before any song repetition."""
    return sum(max(0, int(p.get("pattern_ticks", 512))) for p in song.get("patterns", [])) * synth.SAMPLES_PER_TICK


def repeat_plan(song, target_frames=TARGET_FRAMES):
    require(isinstance(target_frames, int) and target_frames > 0, "Invalid target frame count")
    frames = original_frames(song)
    require(frames > 0, "Cannot repeat an empty/zero-duration song")
    plays = max(1, (target_frames + frames - 1) // frames)
    return frames, plays


def filtered_entries(song):
    blocks = song.get("blocks_dictionary") or {}
    channels = [ch for p in song.get("patterns", []) for ch in p.get("channels", {}).values()]
    return {
        "blocks": sum(bool(b.get("filters")) for b in blocks.values()),
        "channel_occurrences": sum(bool(ch.get("filters")) for ch in channels),
    }


def prepare_song(song, origin, table):
    require(origin in ORIGINS, "Unsupported origin (Attention is not rendered here)")
    if origin == "real":
        require(table is not None, "Real requires the frozen SFX filter table")
        return attach(song, table)
    require(not any(filtered_entries(song).values()), "Generated input unexpectedly contains SFX filters")
    return song  # Never attach the real table, even if generated metadata includes a real tid.


def wav_bytes(pcm):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(synth.SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def validate_wav(raw, expected_frames=TARGET_FRAMES):
    with wave.open(io.BytesIO(raw), "rb") as handle:
        require((handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getcomptype())
                == (1, 2, 44100, "NONE"), "WAV format differs from canonical mono PCM16/44100")
        require(handle.getnframes() == expected_frames, "Incorrect WAV duration")
        frames = handle.readframes(expected_frames + 1)
    require(len(frames) == expected_frames * 2, "Incomplete WAV data")
    pcm = np.frombuffer(frames, dtype="<i2")
    require(np.isfinite(pcm.astype(np.float32)).all() and np.any(pcm), "Empty/silent or nonfinite WAV")
    return pcm


def render_clip(song, origin, table, *, target_frames=TARGET_FRAMES):
    """One continuous native render; no waveform tiling, added fades or gain."""
    require(synth.SAMPLE_RATE == 44100 and synth.SAMPLES_PER_TICK == 366, "Unexpected synth clock")
    frames, plays = repeat_plan(song, target_frames)
    source_digest = sha(canonical(song))
    prepared = prepare_song(song, origin, table)
    # Full ordered pattern sequence, including its intro, for EVERY end_reason.
    # Keep native SFX loops, note/effect semantics and all channel/block content.
    expanded = dict(prepared, patterns=list(prepared.get("patterns", [])) * plays)
    audio = synth.synthesize_song(expanded, loop_extra_repeats=0, seed=0, rate=synth.SAMPLE_RATE)
    require(audio.ndim == 1 and len(audio) == frames * plays, "Native render length mismatch")
    require(np.isfinite(audio).all() and np.max(np.abs(audio)) <= 1, "Invalid native samples")
    clip = audio[:target_frames]
    # EXACT arithmetic/dtype of src.synth.synthesize_song_to_wav, after cropping.
    rounded = np.round(clip * 32767.0)
    added_clipping = int(np.count_nonzero((rounded < -32767) | (rounded > 32767)))
    require(added_clipping == 0, "Unexpected clipping introduced during PCM conversion")
    pcm = np.clip(rounded, -32767, 32767).astype("<i2")
    raw = wav_bytes(pcm)
    validate_wav(raw, target_frames)
    require(sha(canonical(song)) == source_digest, "Rendering mutated the input record")
    differences = np.abs(np.diff(pcm.astype(np.int32)))
    boundaries = list(range(frames, target_frames, frames)) if plays > 1 else []
    boundary_deltas = [int(differences[b - 1]) for b in boundaries]
    audit = {
        "protocol": PROTOCOL, "source_record_sha256": source_digest,
        "original_symbolic_duration_sec": subset.symbolic_duration(song),
        "original_rendered_frames": frames, "original_rendered_duration_sec": frames / synth.SAMPLE_RATE,
        "full_song_play_count": plays, "extra_repeat_count": plays - 1,
        "engine_loop_extra_repeats": 0, "repeat_domain": "full symbolic pattern sequence",
        "rendered_frames_before_crop": len(audio), "rendered_duration_before_crop_sec": len(audio) / synth.SAMPLE_RATE,
        "crop_start_frame": 0, "final_frames": target_frames, "final_duration_sec": target_frames / synth.SAMPLE_RATE,
        "sample_rate": synth.SAMPLE_RATE, "channels": 1, "sample_width_bytes": 2,
        "encoding": "PCM_S16LE", "seed": 0, "wav_sha256": sha(raw),
        "synth_sha256": PINNED["src/synth.py"],
        "sfx_filters_sha256": PINNED["src/sfx_filters.json"] if origin == "real" else None,
        "sfx_filters_applied": origin == "real", "filtered_entries": filtered_entries(prepared),
        "original_end_reason": song.get("end_reason"), "original_loop_start_pattern": song.get("loop_start_pattern"),
        "validation": {
            "finite_native_samples": True, "nonzero_pcm_samples": int(np.count_nonzero(pcm)),
            "peak_absolute_pcm": int(np.max(np.abs(pcm.astype(np.int32)))),
            "pcm_rail_samples": int(np.count_nonzero(np.abs(pcm.astype(np.int32)) == 32767)),
            "native_mix_saturation_rail_samples": int(np.count_nonzero(np.abs(clip) >= np.float32(32767.9 / 32768.0))),
            "conversion_clipped_samples": added_clipping,
            "cycle_boundaries_in_clip": len(boundaries), "cycle_boundary_max_adjacent_delta_pcm": max(boundary_deltas, default=0),
            "global_max_adjacent_delta_pcm": int(differences.max()) if differences.size else 0,
            "final_sample_pcm": int(pcm[-1]),
        },
    }
    return raw, audit


def smoke_indices(rows):
    by_index = {r["selection_index"]: r for r in rows}
    require(len(by_index) == len(rows) == 48 and set(by_index) == set(range(48)), "Expected frozen 48 scenarios")
    require(all(by_index[42][f"{o}_duration_sec"] < 30 for o in ORIGINS), "Stress scenario changed")
    require(all(by_index[1][f"{o}_duration_sec"] >= 30 for o in ORIGINS), "Long scenario changed")
    mixed = [i for i, r in sorted(by_index.items())
             if any(r[f"{o}_duration_sec"] < 30 for o in ORIGINS)
             and any(r[f"{o}_duration_sec"] >= 30 for o in ORIGINS)]
    require(bool(mixed), "No mixed-duration smoke scenario")
    return [42, 1, mixed[0]]


def load_smoke(repo=REPO):
    hashes = {ref: sha((repo / ref).read_bytes()) for ref in PINNED}
    require(hashes == PINNED, "Frozen synth/filter/subset bytes changed; review protocol before rendering")
    config_path = "outputs/subjective/subjective-48-v1/selection_config.json"
    selection = json.loads((repo / config_path).read_bytes())
    rows = [json.loads(line) for line in (repo / "outputs/subjective/subjective-48-v1/scenarios.jsonl").read_bytes().splitlines()]
    require(selection["protocol"] == "subjective-48-v1", "Wrong subset protocol")
    require(selection["outputs_sha256"]["scenarios.jsonl"] == hashes["outputs/subjective/subjective-48-v1/scenarios.jsonl"],
            "Subset output checksum mismatch")
    indices = smoke_indices(rows)
    chosen = [next(r for r in rows if r["selection_index"] == i) for i in indices]
    jobs, source_files = [], {}
    for origin in ORIGINS:
        id_field = "real_record_id" if origin == "real" else f"{origin}_generated_id"
        frozen = selection["inputs"][origin]
        sources, file_hash, content_hash = subset.selected_sources(
            repo / frozen["artifact_ref"], origin, {r[id_field] for r in chosen}, expected_count=1854)
        require(file_hash == frozen["sha256"], f"Changed {origin} source file")
        source_files[origin] = dict(frozen, uncompressed_sha256=content_hash)
        for row in chosen:
            identity = row[id_field]
            song, digest, line_index = sources[identity]
            require(digest == row[f"{origin}_record_sha256"], f"Changed {origin} record")
            require(song["song_id"] == row[f"{origin}_song_id"], f"Incorrect {origin} song_id")
            if origin == "real":
                require(str(song["tid"]) == str(row["real_tid"]), "Incorrect Real tid")
            if origin == "neural":
                require(row["neural_system"] == subset.NEURAL_SYSTEM and song["song_id"] != 1416, "Incorrect Neural identity")
            require(abs(subset.symbolic_duration(song) - row[f"{origin}_duration_sec"]) < 1e-12, "Changed symbolic duration")
            require(abs(original_frames(song) / synth.SAMPLE_RATE - row[f"{origin}_duration_sec"]) < 1e-12,
                    "Symbolic/native duration discrepancy requires review")
            jobs.append({"selection_index": row["selection_index"], "external_id": row["external_id"],
                         "real_record_id": row["real_record_id"], "origin": origin, "source_identity": identity,
                         "source_record_sha256": digest, "source_song_id": song["song_id"], "source_tid": song.get("tid"),
                         "source_jsonl_line_1based": line_index + 1,
                         "neural_system": subset.NEURAL_SYSTEM if origin == "neural" else None, "song": song})
    jobs.sort(key=lambda j: (indices.index(j["selection_index"]), ORIGINS.index(j["origin"])))
    table = load_table(repo / "src/sfx_filters.json")
    require(table is not None, "Missing SFX filter table")
    provenance = {
        "protocol": PROTOCOL, "mode": "three-scenario-smoke-only", "scenario_indices": indices,
        "neural_system": subset.NEURAL_SYSTEM,
        "scenario_choice": "requested 42, requested 1, lowest frozen index with mixed durations",
        "input_sha256": hashes, "source_files": source_files,
        "implementation_sha256": sha(Path(__file__).read_bytes()),
        "shared_source_reader_sha256": sha(Path(subset.__file__).read_bytes()),
        "format": {"sample_rate": 44100, "channels": 1, "sample_width_bytes": 2, "target_frames": TARGET_FRAMES},
        "render": {"seed": 0, "start_frame": 0, "repeat": "entire ordered symbolic patterns; ceil(target_frames/original_frames)",
                   "loop_extra_repeats": 0, "new_crossfade": False, "normalization": False, "waveform_concatenation": False},
        "filters": {"real": "attach frozen SFX table", "markov": "no table", "neural": "no table"},
        "attention": {"rendered": False, "future_input": "exact final Real WAV/PCM bytes, pinned by manifest WAV SHA256",
                      "future_protocol": "attention-check-v1", "future_snr_db": 0},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "determinism_scope": "same input/code/dependency versions; no timestamps or output directory in serialized artifacts",
        "artifact_visibility": "internal provenance; contains blinded source identities",
    }
    return jobs, table, provenance


def render_smoke(output_dir, jobs, table, provenance):
    """Publish fresh outputs atomically; failed renders leave no partial output."""
    output_dir = Path(output_dir)
    require(not output_dir.exists(), "Output directory already exists; refusing overwrite")
    require(len(jobs) == 9 and len({(j["selection_index"], j["origin"]) for j in jobs}) == 9,
            "Smoke must contain exactly nine unique scenario/origin jobs")
    require({j["origin"] for j in jobs} == set(ORIGINS)
            and all(sum(j["selection_index"] == i for j in jobs) == 3 for i in provenance["scenario_indices"])
            and len(set(provenance["scenario_indices"])) == 3
            and set(provenance["scenario_indices"]) == {j["selection_index"] for j in jobs}, "Smoke coverage mismatch")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".subjective-audio-stage-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        records = []
        for job in jobs:
            print(f"render {job['selection_index']} {job['origin']}", flush=True)
            raw, audit = render_clip(job["song"], job["origin"], table)
            require(audit["source_record_sha256"] == job["source_record_sha256"], "Source provenance mismatch")
            name = f"scenario_{job['selection_index']:02d}_{job['origin']}.wav"
            (staging / name).write_bytes(raw)
            records.append(dict({k: v for k, v in job.items() if k != "song"}, **{k: v for k, v in audit.items() if k != "source_record_sha256"}, wav_file=name))
        (staging / "manifest.jsonl").write_bytes(b"".join(canonical(r) + b"\n" for r in records))
        config = dict(provenance, output_sha256={p.name: sha(p.read_bytes()) for p in sorted(staging.iterdir())})
        (staging / "render_config.json").write_bytes(json_bytes(config))
        require(not output_dir.exists(), "Output appeared during rendering; refusing overwrite")
        os.rename(staging, output_dir)
    return records


def verify_repeat(output_dir, jobs, table, provenance):
    """Independent second synthesis, also on HDD; keep only the first nine WAVs."""
    output_dir = Path(output_dir)
    with tempfile.TemporaryDirectory(prefix=".subjective-audio-repeat-", dir=output_dir.parent) as temporary:
        second = Path(temporary) / "artifact"
        render_smoke(second, jobs, table, provenance)
        first_files = {p.name for p in output_dir.iterdir()}
        require(first_files == {p.name for p in second.iterdir()}, "Repeat output coverage differs")
        checksums = {}
        for name in sorted(first_files):
            first = (output_dir / name).read_bytes()
            require(first == (second / name).read_bytes(), f"Nondeterministic output: {name}")
            checksums[name] = sha(first)
    proof = {"protocol": PROTOCOL, "independent_synthesis_passes": 2, "all_output_bytes_identical": True,
             "wav_count": len(list(output_dir.glob("*.wav"))), "sha256": checksums}
    (output_dir / "verification.json").write_bytes(json_bytes(proof))
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", required=True, help="Only indices 42, 1 and the first mixed scenario")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    require(args.output_dir.resolve().is_relative_to(Path("/mnt/hd")), "Smoke audio must be written on HDD /mnt/hd")
    require(Path("/mnt/hd").is_mount(), "HDD /mnt/hd is not mounted")
    require(not args.output_dir.exists(), "Output directory already exists; refusing overwrite")
    jobs, table, provenance = load_smoke()
    records = render_smoke(args.output_dir, jobs, table, provenance)
    proof = verify_repeat(args.output_dir, jobs, table, provenance)
    print(json.dumps({"protocol": PROTOCOL, "scenario_indices": provenance["scenario_indices"],
                      "wav_count": len(records), "byte_identical": proof["all_output_bytes_identical"]}, sort_keys=True))


if __name__ == "__main__":
    main()
