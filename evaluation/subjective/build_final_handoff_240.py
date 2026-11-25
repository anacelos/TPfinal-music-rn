"""Definitive 240-scenario handoff; approved speech and middle-slot balance."""
import argparse
from collections import Counter
import copy
import ctypes
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

import numpy as np
import scipy
from scipy.signal import resample_poly

from . import build_subjective_handoff as shared
from . import finalize_context_descriptions as context

REPO = shared.audio.REPO
PROTOCOL = "subjective-240-handoff-v1"
PERMUTATION_PROTOCOL = "attention-middle-balanced-v1"
OUTPUT = shared.DEFAULT_OUTPUT.parent / PROTOCOL
CACHE = shared.DEFAULT_OUTPUT.parent / ".subjective-240-musical-cache-v1"
DOC = Path(__file__).with_name("HANDOFF_GIOVANNI_240.md")
require, sha, json_bytes = shared.require, shared.sha, shared.json_bytes


def check_public_metadata(p):
    require(set(p) == shared.PUBLIC_FIELDS | {"collection_order_v2"}, "Public schema leak")
    # Portuguese 'real' can mean royal (e.g. múmia real); exact frozen text is
    # separately enforced. Reject source identifiers, rather than that adjective.
    require(not shared.re.search(r"\b(markov|neural|attention|lstm)\b|grid_block|sha256:|real:\d",
                                 shared.canonical(p).decode(), shared.re.I), "Public origin leak")


def permutation_cycle():
    origins = shared.audio.ORIGINS
    result = []
    for direction in (1, -1):
        for start in range(3):
            music = [origins[(start + i * direction) % 3] for i in range(3)]
            result.append(dict(zip("ABCD", [music[0], "attention", music[1], music[2]])))
            music = [origins[(start + 1 + i * direction) % 3] for i in range(3)]
            result.append(dict(zip("ABCD", [music[0], music[1], "attention", music[2]])))
    require(len({tuple(m[s] for s in "ABCD") for m in result}) == 12, "Invalid cycle")
    return result


def assignments(rows):
    ordered = sorted(rows, key=lambda r: r["collection_order_v2"])
    require([r["collection_order_v2"] for r in ordered] == list(range(240)), "Collection order changed")
    cycle = permutation_cycle()
    return {r["external_id"]: cycle[i % 12].copy() for i, r in enumerate(ordered)}


def decode_speech(path):
    """Native libsndfile MP3 decoding; no TTS, gain adjustment or repetition."""
    class Info(ctypes.Structure):
        _fields_ = [("frames", ctypes.c_int64), ("samplerate", ctypes.c_int), ("channels", ctypes.c_int),
                    ("format", ctypes.c_int), ("sections", ctypes.c_int), ("seekable", ctypes.c_int)]
    lib = ctypes.CDLL("libsndfile.so.1")
    lib.sf_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(Info)]
    lib.sf_open.restype = ctypes.c_void_p
    lib.sf_readf_float.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
    lib.sf_readf_float.restype = ctypes.c_int64
    lib.sf_close.argtypes = [ctypes.c_void_p]
    lib.sf_version_string.restype = ctypes.c_char_p
    info = Info()
    handle = lib.sf_open(os.fsencode(path), 0x10, ctypes.byref(info))
    require(bool(handle) and info.frames > 0 and info.channels > 0, "Cannot decode approved MP3")
    try:
        decoded = np.empty((info.frames, info.channels), dtype=np.float32)
        n = lib.sf_readf_float(handle, decoded.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), info.frames)
        require(n == info.frames and np.isfinite(decoded).all(), "Incomplete/nonfinite speech")
    finally:
        lib.sf_close(handle)
    mono = decoded.mean(axis=1) if info.channels > 1 else decoded[:, 0]
    divisor = math.gcd(44100, info.samplerate)
    speech = resample_poly(mono, 44100 // divisor, info.samplerate // divisor,
                           window=("kaiser", 5.0), padtype="constant")
    require(0 < len(speech) < shared.audio.TARGET_FRAMES and np.isfinite(speech).all(), "Speech must fit inside 30s")
    rounded = np.round(speech * 32767.0)
    require(np.max(np.abs(rounded)) <= 32767, "Unexpected speech clipping")
    pcm = np.zeros(shared.audio.TARGET_FRAMES, dtype="<i2")
    pcm[:len(speech)] = rounded.astype("<i2")
    raw = shared.audio.wav_bytes(pcm)
    shared.validate_public_wav(raw)
    return raw, {"protocol": "attention-check-spoken-v2", "source_mp3_sha256": shared.file_sha(path),
                 "source_filename": "attention-preview.mp3", "source_tool_user_reported": "edge-tts",
                 "voice_user_reported": "pt-BR-FranciscaNeural", "auditory_approval": "provided by user",
                 "text": context.SPOKEN_TEXT.replace("‘", "").replace("’", ""),
                 "decoder": lib.sf_version_string().decode(), "source_rate": info.samplerate,
                 "source_channels": info.channels, "decoded_frames": info.frames,
                 "resampler": "scipy.signal.resample_poly; kaiser(5.0); constant pad", "scipy_version": scipy.__version__,
                 "speech_frames": len(speech), "speech_duration_sec": len(speech) / 44100,
                 "silence_frames": len(pcm) - len(speech), "final_frames": len(pcm), "sample_rate": 44100,
                 "channels": 1, "sample_width_bytes": 2, "wav_sha256": sha(raw), "repeated": False,
                 "gain_normalization": False, "requested_rating": 1}


def load_inputs():
    root = context.OUTPUT
    cfg = json.loads((root / "selection_config.json").read_bytes())
    require(all(shared.file_sha(root / name) == digest for name, digest in cfg["outputs_sha256"].items()), "Final context bytes changed")
    rows = [json.loads(line) for line in (root / "scenarios.jsonl").read_bytes().splitlines()]
    require(len(rows) == 240 and len({r["external_id"] for r in rows}) == 240, "Expected 240 unique scenarios")
    for field in ("real_record_id", "markov_generated_id", "neural_generated_id"):
        require(len({r[field] for r in rows}) == 240, "Repeated source identity")
    for row in rows: check_public_metadata(public_record(row))
    for ref, digest in shared.audio.PINNED.items():
        require(shared.file_sha(REPO / ref) == digest, "Frozen engine/filter bytes changed")
    require(shared.file_sha(REPO / "evaluation/subjective/render_subjective_audio.py") == shared.AUDIO_SHA256, "Audio implementation changed")
    original_config = json.loads((REPO / "outputs/subjective/subjective-48-v1/selection_config.json").read_bytes())
    cfg["musical_source_inputs"] = {o: original_config["inputs"][o] for o in shared.audio.ORIGINS}
    sources = {}
    for origin in shared.audio.ORIGINS:
        idkey = "real_record_id" if origin == "real" else origin + "_generated_id"
        source = cfg["musical_source_inputs"][origin]
        sources[origin], digest, _ = shared.subset.selected_sources(REPO / source["artifact_ref"], origin,
            {r[idkey] for r in rows}, expected_count=1854)
        require(digest == source["sha256"], "Source file changed")
        for row in rows:
            song, record_hash, _ = sources[origin][row[idkey]]
            require(record_hash == row[origin + "_record_sha256"] and song["song_id"] == row[origin + "_song_id"], "Source binding changed")
            if origin == "real": require(str(song["tid"]) == row["real_tid"], "Real tid changed")
            if origin == "neural": require(song["system"] == "lstm-grid_block-k10" and song["song_id"] != 1416, "Wrong Neural")
    return rows, sources, cfg


def public_record(row):
    return dict(shared.public_projection(row), collection_order_v2=row["collection_order_v2"])


def build_files(root, rows, sources, cfg, speech_path):
    (root / "public").mkdir()
    (root / "private").mkdir(mode=0o700)
    shutil.copyfile(speech_path, root / "private/attention-preview.mp3")
    speech, provenance = decode_speech(root / "private/attention-preview.mp3")
    (root / "private/approved-spoken.wav").write_bytes(speech)
    oldroot = shared.DEFAULT_OUTPUT.parent / "subjective-48-handoff-v1"
    oldkey = json.loads((oldroot / "private/scenario_key.json").read_bytes())
    oldrows = {r["external_id"]: r for r in oldkey["scenarios"]}
    mapping = assignments(rows)
    table = shared.audio.load_table(REPO / "src/sfx_filters.json")
    key = {"protocol": PROTOCOL, "permutation_protocol": PERMUTATION_PROTOCOL,
           "context_config_sha256": shared.file_sha(context.OUTPUT / "selection_config.json"),
           "implementation_sha256": shared.file_sha(Path(__file__)), "sources": cfg["inputs"],
           "musical_source_inputs": cfg["musical_source_inputs"],
           "runtime": {"python": shared.platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
           "audio_protocol": shared.audio.PROTOCOL, "attention": provenance, "scenarios": []}
    reused = 0
    for row in sorted(rows, key=lambda r: r["collection_order_v2"]):
        identifier = row["external_id"]
        private = root / "private/audio" / identifier
        public = root / "public/scenarios" / identifier
        private.mkdir(parents=True)
        public.mkdir(parents=True)
        audits = {}
        for origin in shared.audio.ORIGINS:
            identity = row["real_record_id"] if origin == "real" else row[origin + "_generated_id"]
            song, digest, line = sources[origin][identity]
            path = private / (origin + ".wav")
            if identifier in oldrows:
                audit = copy.deepcopy(oldrows[identifier]["audios"][origin])
                require(audit["source_record_sha256"] == digest and audit["synth_sha256"] == shared.audio.PINNED["src/synth.py"]
                        and audit["protocol"] == shared.audio.PROTOCOL and audit["seed"] == 0
                        and audit["sfx_filters_applied"] == (origin == "real")
                        and audit["sfx_filters_sha256"] == (shared.audio.PINNED["src/sfx_filters.json"] if origin == "real" else None), "Incompatible reuse")
                original = oldroot / audit["private_file"]
                require(shared.file_sha(original) == audit["wav_sha256"], "Reuse hash mismatch")
                shutil.copyfile(original, path)
                reused += 1
            else:
                cached = CACHE / origin / (digest + ".wav")
                metadata = cached.with_suffix(".json")
                if cached.exists() and metadata.exists():
                    audit = json.loads(metadata.read_bytes())
                    require(audit["source_record_sha256"] == digest and audit["synth_sha256"] == shared.audio.PINNED["src/synth.py"]
                            and audit["protocol"] == shared.audio.PROTOCOL and audit["seed"] == 0
                            and audit["sfx_filters_applied"] == (origin == "real")
                            and audit["sfx_filters_sha256"] == (shared.audio.PINNED["src/sfx_filters.json"] if origin == "real" else None), "Incompatible audio cache")
                    raw = cached.read_bytes()
                    require(sha(raw) == audit["wav_sha256"], "Cache hash mismatch")
                    shared.validate_public_wav(raw)
                else:
                    raw, audit = shared.audio.render_clip(song, origin, table)
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    cached.write_bytes(raw)
                    metadata.write_bytes(json_bytes(audit))
                path.write_bytes(raw)
            audit.update(private_file=path.relative_to(root).as_posix(), source_jsonl_line_1based=line + 1)
            audits[origin] = audit
        audits["attention"] = dict(provenance, private_file="private/approved-spoken.wav")
        for slot, origin in mapping[identifier].items():
            shutil.copyfile(root / audits[origin]["private_file"], public / (slot + ".wav"))
        image = (context.OUTPUT / row["public_visual_source"]).read_bytes()
        require(sha(image) == row["public_visual_sha256"], "Frozen image changed")
        (public / "context.png").write_bytes(image)
        hidden = shared.private_scenario(row, audits)
        hidden.update(mapping=mapping[identifier], historical_presentation=row["presentation"],
                      collection_order_v2=row["collection_order_v2"])
        key["scenarios"].append(hidden)
        print(f"{row['collection_order_v2'] + 1}/240 ready", flush=True)
    manifest = json_bytes([public_record(r) for r in sorted(rows, key=lambda r: r["collection_order_v2"])])
    (root / "public/scenarios.json").write_bytes(manifest)
    study = {"heading": context.HEADING, "question": context.QUESTION,
             "scale": [{"value": i, "label": s} for i, s in enumerate(context.SCALE, 1)],
             "scenarios_per_participant": 2, "demographics": {"age_range": "faixa etária",
                "musical_experience": "experiência musical", "gaming_frequency": "frequência com jogos",
                "retro_familiarity": "familiaridade com jogos retrô/8-bit", "region": "região do Brasil ou exterior",
                "headphones": "uso de fones de ouvido"}}
    (root / "public/study.json").write_bytes(json_bytes(study))
    shutil.copyfile(DOC, root / "HANDOFF_GIOVANNI.md")
    key.update(public_manifest_sha256=sha(manifest), musical_clips_reused=reused, musical_clips_new=720 - reused)
    (root / "private/scenario_key.json").write_bytes(json_bytes(key))


def validate(root, rows):
    key = json.loads((root / "private/scenario_key.json").read_bytes())
    public = json.loads((root / "public/scenarios.json").read_bytes())
    ordered = sorted(rows, key=lambda r: r["collection_order_v2"])
    require(len(public) == len(key["scenarios"]) == 240, "Coverage mismatch")
    wanted = assignments(rows)
    expected = {"scenarios.json", "study.json"}
    expected_dirs = {"scenarios"}
    counts = Counter()
    positions = {s: Counter() for s in "ABCD"}
    attention_bytes, provenance = decode_speech(root / "private/attention-preview.mp3")
    require((root / "private/approved-spoken.wav").read_bytes() == attention_bytes and key["attention"] == provenance, "Speech provenance mismatch")
    for i, (p, h, r) in enumerate(zip(public, key["scenarios"], ordered)):
        identifier = r["external_id"]
        require(p == public_record(r) and h["external_id"] == identifier and h["collection_order_v2"] == i, "Public/frozen order binding changed")
        require(h["mapping"] == wanted[identifier] and h["historical_presentation"] == r["presentation"], "Assignment changed")
        check = shared.private_scenario(r, h["audios"])
        for field in ("identities", "matching_costs", "frozen_scenario_sha256", "context_tier", "serve_order"):
            require(h[field] == check[field], "Private identity/matching/context changed")
        check_public_metadata(p)
        counts[tuple(h["mapping"][s] for s in "ABCD")] += 1
        expected_dirs.add(f"scenarios/{identifier}")
        for slot, origin in h["mapping"].items():
            positions[slot][origin] += 1
            raw = (root / "public" / p["audios"][slot]).read_bytes()
            shared.validate_public_wav(raw)
            audit = h["audios"][origin]
            require(raw == (root / audit["private_file"]).read_bytes() and sha(raw) == audit["wav_sha256"], "Audio copy/identity mismatch")
            if origin == "attention": require(raw == attention_bytes and slot in "BC", "Wrong speech/slot")
            else:
                require(audit["source_record_sha256"] == r[origin + "_record_sha256"]
                        and audit["validation"]["conversion_clipped_samples"] == 0, "Music source/clipping mismatch")
            expected.add(p["audios"][slot])
        image = (root / "public" / p["context_image"]).read_bytes()
        shared.validate_png(image)
        require(sha(image) == r["public_visual_sha256"], "Image mismatch")
        expected.add(p["context_image"])
    actual, directories = set(), set()
    for path in (root / "public").rglob("*"):
        require(not path.is_symlink(), "Public symlink")
        (directories if path.is_dir() else actual).add(path.relative_to(root / "public").as_posix())
    require(actual == expected and directories == expected_dirs, "Unexpected public file")
    require(len(counts) == 12 and set(counts.values()) == {20}, "Permutation balance failed")
    prefix = {}
    for n in (2, 4, 6, 12, 24, 48, 120, 240):
        c = {s: dict(Counter(h["mapping"][s] for h in key["scenarios"][:n])) for s in "ABCD"}
        require(c["A"].get("attention", 0) == c["D"].get("attention", 0) == 0
                and c["B"].get("attention", 0) == c["C"].get("attention", 0) == n // 2, "Attention prefix imbalance")
        prefix[str(n)] = c
    return {"scenarios": 240, "pngs": 240, "public_wavs": 960, "permutation_counts": {"|".join(k): v for k, v in counts.items()},
            "position_counts": {k: dict(v) for k, v in positions.items()}, "prefix_counts": prefix,
            "musical_clips_reused": key["musical_clips_reused"], "musical_clips_new": key["musical_clips_new"],
            "speech": provenance, "all_final_wavs": "mono PCM16 44100 Hz; 1323000 frames", "public_origin_metadata_leaks": 0}


def validate_extract_archive(root, archive):
    """Check explicit member allowlist, extract on HDD, hash every extracted file."""
    entries = {p.relative_to(root).as_posix(): p for p in shared.archive_entries(root)}
    require(sum(p.is_file() for p in entries.values()) == 1203, "Archive file coverage")
    with tempfile.TemporaryDirectory(prefix=".handoff-extraction-", dir=archive.parent) as temporary:
        process = subprocess.Popen(["zstd", "-q", "-dc", str(archive)], stdout=subprocess.PIPE)
        names = set()
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") as packed:
                for member in packed:
                    require(member.name in entries and member.name not in names, "Nonpublic or duplicate archive member")
                    names.add(member.name)
                    original = entries[member.name]
                    require(member.isdir() if original.is_dir() else member.isfile(), "Unsafe archive member type")
                    require(not member.pax_headers and member.uid == member.gid == member.mtime == 0, "Archive metadata leak")
                    target = Path(temporary) / member.name
                    if member.isdir(): target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with packed.extractfile(member) as src, target.open("xb") as dst: shutil.copyfileobj(src, dst)
                        require(shared.file_sha(target) == shared.file_sha(original), "Extracted hash mismatch")
            require(process.wait() == 0 and names == set(entries), "Archive extraction incomplete")
        finally:
            process.stdout.close()
            if process.poll() is None: process.terminate(); process.wait()
    return {"sha256": shared.file_sha(archive), "size_bytes": archive.stat().st_size,
            "regular_files": 1203, "members": len(names), "extraction_hashes_verified": True, "private_files": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--speech", type=Path, default=Path("/tmp/attention-preview.mp3"))
    args = parser.parse_args()
    output = args.output_dir
    archive = output.with_name(output.name + ".tar.zst")
    require(output.resolve().is_relative_to(Path("/mnt/hd")) and Path("/mnt/hd").is_mount(), "Use mounted HDD")
    require(not output.exists() and not archive.exists(), "Fresh destinations required")
    rows, sources, cfg = load_inputs()
    with tempfile.TemporaryDirectory(prefix=".final-handoff-240-", dir=output.parent) as temporary:
        stage = Path(temporary) / "package"
        stage.mkdir()
        build_files(stage, rows, sources, cfg, args.speech)
        summary = validate(stage, rows)
        compressed = Path(temporary) / "final.tar.zst"
        shared.write_archive(stage, compressed)
        summary["archive"] = validate_extract_archive(stage, compressed)
        (stage / "private/validation_summary.json").write_bytes(json_bytes(summary))
        require(not output.exists() and not archive.exists(), "Destination appeared")
        os.rename(stage, output)
        os.rename(compressed, archive)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
