#!/usr/bin/env python3
"""Build and validate the local, blinded subjective-48-handoff-v1 package."""

import argparse
from collections import Counter
import io
import itertools
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import tarfile
import tempfile
import uuid

import numpy as np
from PIL import Image

from . import create_attention_checks as attention
from . import render_subjective_audio as audio
from . import select_subjective_subset as subset


PROTOCOL = "subjective-48-handoff-v1"
DEFAULT_OUTPUT = audio.DEFAULT_OUTPUT.parent / PROTOCOL
SLOTS = tuple("ABCD")
ORIGINS = (*audio.ORIGINS, "attention")
PUBLIC_FIELDS = {"external_id", "description", "context_heading", "context_image", "audios"}
GLOBAL_SEED = 831
ATTENTION_SHA256 = "553cce6b200d455e51a04086fd50f7bef9a77089efe4c107e02de91da3494cc2"
AUDIO_SHA256 = "354b81e612dbd17fe843a149baac93ef49a7f9c3518d15b520f009546b2c4c35"
DOC = Path(__file__).with_name("HANDOFF_GIOVANNI.md")
require = subset.require
sha = subset.sha
canonical = subset.canonical
json_bytes = subset.json_bytes


def file_sha(path):
    import hashlib
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_rows(rows):
    require(len(rows) == 48 and [r["selection_index"] for r in rows] == list(range(48)), "Expected frozen 48 in original order")
    require(len({r["external_id"] for r in rows}) == 48, "Duplicate external ID")
    for field in ("real_record_id", "markov_generated_id", "neural_generated_id"):
        require(len({r[field] for r in rows}) == 48, f"Repeated source: {field}")
    require(all(r["real_record_id"] != subset.UNMATCHED_REAL and r["neural_song_id"] != 1416 for r in rows), "Ineligible source")
    permutations = Counter()
    for row in rows:
        identifier = row["external_id"]
        require(str(uuid.UUID(identifier)) == identifier, "External ID must be canonical UUID")
        require(row["neural_system"] == subset.NEURAL_SYSTEM, "Incorrect Neural system")
        require(row["presentation"]["permutation_protocol"] == "abcd-balanced-rotation-v1", "Permutation protocol changed")
        mapping = row["presentation"]["assignment"]
        require(set(mapping) == set(SLOTS) and set(mapping.values()) == set(ORIGINS), "Invalid frozen permutation")
        permutations[tuple(mapping[s] for s in SLOTS)] += 1
        require(all(math.isfinite(row[f"{o}_matching_cost"]) for o in ("markov", "neural")), "Nonfinite provenance cost")
    require(set(permutations) == set(itertools.permutations(ORIGINS)) and set(permutations.values()) == {2}, "Frozen permutation quota changed")


def public_projection(row):
    base = f"scenarios/{row['external_id']}"
    return {"external_id": row["external_id"], "description": row["public_description"],
            "context_heading": row["public_context_heading"], "context_image": f"{base}/context.png",
            "audios": {s: f"{base}/{s}.wav" for s in SLOTS}}


def load_inputs(repo=audio.REPO):
    pins = {**audio.PINNED, "evaluation/subjective/render_subjective_audio.py": AUDIO_SHA256,
            "evaluation/subjective/create_attention_checks.py": ATTENTION_SHA256}
    hashes = {ref: file_sha(repo / ref) for ref in pins}
    require(hashes == pins, "Frozen input/implementation hash changed")
    subset_dir = repo / "outputs/subjective/subjective-48-v1"
    config = json.loads((subset_dir / "selection_config.json").read_bytes())
    rows = [json.loads(line) for line in (subset_dir / "scenarios.jsonl").read_bytes().splitlines()]
    require(config["protocol"] == subset.PROTOCOL and config["outputs_sha256"]["scenarios.jsonl"]
            == hashes["outputs/subjective/subjective-48-v1/scenarios.jsonl"], "Subset checksum mismatch")
    validate_rows(rows)
    sources, source_files = {}, {}
    for origin in audio.ORIGINS:
        id_field = "real_record_id" if origin == "real" else f"{origin}_generated_id"
        frozen = config["inputs"][origin]
        sources[origin], digest, content_hash = subset.selected_sources(
            repo / frozen["artifact_ref"], origin, {r[id_field] for r in rows}, expected_count=1854)
        require(digest == frozen["sha256"], f"Changed {origin} source file")
        source_files[origin] = dict(frozen, uncompressed_sha256=content_hash)
        for row in rows:
            song, record_hash, _ = sources[origin][row[id_field]]
            require(record_hash == row[f"{origin}_record_sha256"] and song["song_id"] == row[f"{origin}_song_id"], "Source identity/hash mismatch")
            if origin == "real":
                require(str(song["tid"]) == str(row["real_tid"]), "Real tid mismatch")
            if origin == "neural":
                require(song["system"] == subset.NEURAL_SYSTEM, "Neural source system mismatch")
            require(abs(audio.original_frames(song) / 44100 - row[f"{origin}_duration_sec"]) < 1e-12
                    and abs(subset.symbolic_duration(song) - row[f"{origin}_duration_sec"]) < 1e-12, "Source duration mismatch")
    images = {}
    for row in rows:
        ref = f"visuals/{row['external_id']}.png"
        require(row["visual_usable"] and row["public_visual_source"] == ref, "Frozen visual changed")
        raw = (subset_dir / ref).read_bytes()
        require(sha(raw) == row["public_visual_sha256"] == config["outputs_sha256"][ref], "Visual hash mismatch")
        validate_png(raw)
        images[row["external_id"]] = raw
    table = audio.load_table(repo / "src/sfx_filters.json")
    require(table is not None, "Missing Real filter table")
    provenance = {"protocol": PROTOCOL, "input_sha256": hashes, "source_files": source_files,
                  "implementation_sha256": file_sha(Path(__file__)), "instructions_sha256": file_sha(DOC),
                  "shared_source_reader_sha256": file_sha(Path(subset.__file__)),
                  "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": audio.scipy.__version__},
                  "protocols": {"subset": subset.PROTOCOL, "audio": audio.PROTOCOL, "attention": attention.PROTOCOL,
                                "real_markov_matching": "symbolic-balanced-v1", "real_neural_matching": subset.NEURAL_PROTOCOL,
                                "permutation": "abcd-balanced-rotation-v1", "visual": subset.VISUAL_PROTOCOL},
                  "neural_system": subset.NEURAL_SYSTEM, "audio_format": {"frames": audio.TARGET_FRAMES,
                      "duration_sec": 30, "sample_rate": 44100, "channels": 1, "sample_width_bytes": 2}}
    return rows, sources, images, table, provenance


def make_attention(real_path, row):
    # Read the exact FINAL Real WAV bytes, never rerender or loop after noise.
    signal, rate, source_hash = attention.read_pcm16(real_path)
    require(rate == 44100 and len(signal) == audio.TARGET_FRAMES, "Attention parent is not final 30-second Real")
    seed = attention.derive_seed(GLOBAL_SEED, str(row["real_tid"]), str(row["real_song_id"]), source_hash)
    pcm, signal_rms, noise_rms, gain = attention.mix_noise(signal, 0, seed)
    raw = audio.wav_bytes(pcm)
    decoded = audio.validate_wav(raw).astype(np.float64)
    require(np.array_equal(decoded, pcm), "Attention writer changed samples")
    snr, error = attention.measure_effective_snr(signal, decoded, gain, 0)
    return raw, {"protocol": attention.PROTOCOL, "parent_origin": "real", "parent_wav_sha256": source_hash,
                 "source_tid": str(row["real_tid"]), "source_song_id": str(row["real_song_id"]),
                 "global_seed": GLOBAL_SEED, "derived_seed": seed, "seed_derivation": attention.SEED_DERIVATION,
                 "target_snr_db": 0, "effective_snr_db": snr, "absolute_snr_error_db": error,
                 "max_absolute_snr_error_db": attention.MAX_ABSOLUTE_SNR_ERROR_DB,
                 "signal_rms": signal_rms, "noise_rms": noise_rms, "post_mix_gain": gain,
                 "rng": "numpy.random.PCG64.standard_normal", "wav_sha256": sha(raw)}


def private_scenario(row, audits):
    identities = {"real": {"record_id": row["real_record_id"], "tid": row["real_tid"], "song_id": row["real_song_id"],
                           "record_sha256": row["real_record_sha256"]}}
    for origin in ("markov", "neural"):
        identities[origin] = {"generated_id": row[f"{origin}_generated_id"], "song_id": row[f"{origin}_song_id"],
                              "record_sha256": row[f"{origin}_record_sha256"]}
    return {"external_id": row["external_id"], "selection_index": row["selection_index"], "serve_order": row["serve_order"],
            "context_tier": row["context_tier"], "mapping": dict(row["presentation"]["assignment"]),
            "frozen_scenario_sha256": sha(canonical(row)), "identities": identities,
            "matching_costs": {o: row[f"{o}_matching_cost"] for o in ("markov", "neural")}, "audios": audits,
            "context_png_sha256": row["public_visual_sha256"]}


def render_bundle(root, rows, sources, images, table, provenance):
    public, private = root / "public", root / "private"
    public.mkdir()
    private.mkdir(mode=0o700)
    key = dict(provenance, scenarios=[])
    for row in rows:
        identifier = row["external_id"]
        destination = public / "scenarios" / identifier
        originals = private / "audio" / identifier
        destination.mkdir(parents=True)
        originals.mkdir(parents=True)
        audits = {}
        for origin in audio.ORIGINS:
            identity = row["real_record_id"] if origin == "real" else row[f"{origin}_generated_id"]
            song, digest, line_index = sources[origin][identity]
            raw, audit = audio.render_clip(song, origin, table)
            require(audit["source_record_sha256"] == digest, "Render source hash mismatch")
            path = originals / f"{origin}.wav"
            path.write_bytes(raw)
            audits[origin] = dict(audit, private_file=path.relative_to(root).as_posix(), source_jsonl_line_1based=line_index + 1)
        raw, audit = make_attention(originals / "real.wav", row)
        path = originals / "attention.wav"
        path.write_bytes(raw)
        audits["attention"] = dict(audit, private_file=path.relative_to(root).as_posix())
        for slot, origin in row["presentation"]["assignment"].items():
            shutil.copyfile(originals / f"{origin}.wav", destination / f"{slot}.wav")
        (destination / "context.png").write_bytes(images[identifier])
        key["scenarios"].append(private_scenario(row, audits))
        print(f"scenario {row['selection_index']:02d}: 4 WAVs ready", flush=True)
    manifest = json_bytes([public_projection(r) for r in rows])
    (public / "scenarios.json").write_bytes(manifest)
    key["public_manifest_sha256"] = sha(manifest)
    (private / "scenario_key.json").write_bytes(json_bytes(key))
    shutil.copyfile(DOC, root / "HANDOFF_GIOVANNI.md")


def validate_png(raw):
    require(raw.startswith(b"\x89PNG\r\n\x1a\n"), "Invalid PNG")
    offset, chunks = 8, []
    while offset < len(raw):
        require(offset + 12 <= len(raw), "Truncated PNG")
        size = int.from_bytes(raw[offset:offset + 4], "big")
        chunks.append(raw[offset + 4:offset + 8])
        offset += 12 + size
    require(chunks and offset == len(raw) and chunks[0] == b"IHDR" and chunks[-1] == b"IEND"
            and set(chunks) == {b"IHDR", b"IDAT", b"IEND"}, "Unexpected PNG metadata/chunks")
    with Image.open(io.BytesIO(raw)) as image:
        require(image.mode == "RGB", "Expected sanitized RGB PNG")
        image.load()
        require(not np.any(np.asarray(image) & 3), "PNG contains unsanitized low bits")


def validate_public_wav(raw, target_frames=audio.TARGET_FRAMES):
    require(len(raw) == 44 + target_frames * 2 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE", "Unexpected WAV metadata/size")
    require(int.from_bytes(raw[4:8], "little") == len(raw) - 8 and raw[12:16] == b"fmt "
            and raw[16:20] == struct.pack("<I", 16) and raw[36:40] == b"data"
            and raw[40:44] == struct.pack("<I", target_frames * 2), "Unexpected WAV chunks")
    require(struct.unpack("<HHIIHH", raw[20:36]) == (1, 1, 44100, 88200, 2, 16), "Invalid WAV PCM format")
    return audio.validate_wav(raw, target_frames)


def validate_bundle(root, rows, *, target_frames=audio.TARGET_FRAMES):
    """Validate public allowlists, frozen bindings, byte copies and noise itself."""
    validate_rows(rows)
    public = root / "public"
    raw_manifest = (public / "scenarios.json").read_bytes()
    manifest = json.loads(raw_manifest)
    key = json.loads((root / "private/scenario_key.json").read_bytes())
    require(key["protocol"] == PROTOCOL and key["public_manifest_sha256"] == sha(raw_manifest), "Private manifest binding changed")
    require(len(manifest) == len(key["scenarios"]) == 48, "Incorrect scenario count")
    require([p["external_id"] for p in manifest] == [r["external_id"] for r in rows], "Public scenario order/identity changed")
    require([p["external_id"] for p in key["scenarios"]] == [r["external_id"] for r in rows], "Private scenario order/identity changed")
    expected_files = {"scenarios.json"}
    expected_dirs = {"scenarios"} | {f"scenarios/{r['external_id']}" for r in rows}
    counts = {s: Counter() for s in SLOTS}
    errors, native_rails = [], 0
    for row, exposed, hidden in zip(rows, manifest, key["scenarios"]):
        identifier = row["external_id"]
        require(set(exposed) == PUBLIC_FIELDS and isinstance(exposed["audios"], dict)
                and set(exposed["audios"]) == set(SLOTS), "Public schema leaks extra data")
        require(exposed == public_projection(row), "Public projection changed (unsafe path/content)")
        require(not re.search(r"\b(real|markov|neural|attention|lstm)\b|grid_block|sha256:|real:\d", canonical(exposed).decode(), re.I), "Origin/provenance token in public metadata")
        require(hidden["mapping"] == row["presentation"]["assignment"], "Frozen mapping changed")
        expected_hidden = private_scenario(row, hidden["audios"])
        require(hidden == expected_hidden, "Private identity/cost/context provenance changed")
        image_path = public / exposed["context_image"]
        image = image_path.read_bytes()
        validate_png(image)
        require(sha(image) == row["public_visual_sha256"], "Public image differs from frozen image")
        expected_files.add(exposed["context_image"])
        pcm_by_origin = {}
        for slot in SLOTS:
            origin = hidden["mapping"][slot]
            counts[slot][origin] += 1
            relative = exposed["audios"][slot]
            expected_files.add(relative)
            raw = (public / relative).read_bytes()
            pcm_by_origin[origin] = validate_public_wav(raw, target_frames)
            audit = hidden["audios"][origin]
            expected_private = f"private/audio/{identifier}/{origin}.wav"
            require(audit["private_file"] == expected_private, "Unsafe private audio path")
            original = (root / expected_private).read_bytes()
            require(raw == original and sha(raw) == audit["wav_sha256"], "Public WAV is not a byte-identical origin copy")
            if origin != "attention":
                require(audit["source_record_sha256"] == row[f"{origin}_record_sha256"]
                        and audit["protocol"] == audio.PROTOCOL
                        and audit["sfx_filters_applied"] == (origin == "real")
                        and audit["sfx_filters_sha256"] == (audio.PINNED["src/sfx_filters.json"] if origin == "real" else None), "Audio origin/filter provenance changed")
                require(audit["validation"]["conversion_clipped_samples"] == 0, "Added clipping")
                native_rails += audit["validation"]["native_mix_saturation_rail_samples"]
        parent = hidden["audios"]["real"]
        check = hidden["audios"]["attention"]
        require(check["protocol"] == attention.PROTOCOL and check["parent_origin"] == "real"
                and check["parent_wav_sha256"] == parent["wav_sha256"] and check["global_seed"] == GLOBAL_SEED
                and check["target_snr_db"] == 0 and check["source_tid"] == str(row["real_tid"])
                and check["source_song_id"] == str(row["real_song_id"]), "Attention parent/protocol mismatch")
        seed = attention.derive_seed(GLOBAL_SEED, str(row["real_tid"]), str(row["real_song_id"]), parent["wav_sha256"])
        signal = pcm_by_origin["real"].astype(np.float64)
        regenerated, signal_rms, noise_rms, gain = attention.mix_noise(signal, 0, seed)
        require(seed == check["derived_seed"] and np.array_equal(regenerated, pcm_by_origin["attention"])
                and gain == check["post_mix_gain"] and signal_rms == check["signal_rms"] and noise_rms == check["noise_rms"], "Attention noise/seed differs from frozen generation")
        snr, error = attention.measure_effective_snr(signal, pcm_by_origin["attention"], gain, 0)
        require(snr == check["effective_snr_db"] and error == check["absolute_snr_error_db"], "Attention SNR provenance differs")
        errors.append(error)
    require(all(dict(c) == {o: 12 for o in ORIGINS} for c in counts.values()), "Origin/slot balance differs from 12 each")
    actual_files, actual_dirs = set(), set()
    require(not public.is_symlink(), "Public root must not be a symlink")
    for path in public.rglob("*"):
        require(not path.is_symlink(), "Symlink in public package")
        relative = path.relative_to(public).as_posix()
        (actual_dirs if path.is_dir() else actual_files).add(relative)
    require(actual_files == expected_files and actual_dirs == expected_dirs, "Unexpected/missing public files or directories")
    return {"scenarios": 48, "pngs": 48, "public_wavs": 192, "slot_wavs": {s: 48 for s in SLOTS},
            "origin_slot_counts": {s: dict(c) for s, c in counts.items()}, "all_byte_copies": True,
            "public_allowlist_valid": True, "all_references_exist": True, "attention_recomputed_and_bound_to_real": 48,
            "max_attention_snr_error_db": max(errors), "native_mix_saturation_rail_samples_total": native_rails,
            "public_files": len(actual_files)}


def archive_entries(root):
    entries = [root / "HANDOFF_GIOVANNI.md", root / "public", *sorted((root / "public").rglob("*"))]
    require(all(not p.is_symlink() and (p.is_dir() or p.is_file()) for p in entries), "Unsafe archive input")
    return entries


def write_archive(root, archive):
    """Explicit public-only allowlist and normalized tar headers; deterministic zstd."""
    require(not archive.exists() and not archive.is_symlink(), "Archive already exists; refusing overwrite")
    with archive.open("xb") as destination:
        process = subprocess.Popen(["zstd", "-q", "-T1", "-9", "-c"], stdin=subprocess.PIPE, stdout=destination, stderr=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=process.stdin, mode="w|", format=tarfile.USTAR_FORMAT) as packed:
                for path in archive_entries(root):
                    member = tarfile.TarInfo(path.relative_to(root).as_posix())
                    member.uid = member.gid = member.mtime = 0
                    member.uname = member.gname = ""
                    member.mode = 0o755 if path.is_dir() else 0o644
                    member.type = tarfile.DIRTYPE if path.is_dir() else tarfile.REGTYPE
                    member.size = 0 if path.is_dir() else path.stat().st_size
                    if path.is_file():
                        with path.open("rb") as handle:
                            packed.addfile(member, handle)
                    else:
                        packed.addfile(member)
            process.stdin.close()
            error = process.stderr.read().decode()
            require(process.wait() == 0, f"zstd failed: {error}")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            process.stderr.close()


def validate_archive(root, archive):
    expected = {p.relative_to(root).as_posix(): p for p in archive_entries(root)}
    files, names = 0, set()
    process = subprocess.Popen(["zstd", "-q", "-d", "-c", str(archive)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as packed:
            for member in packed:
                require(member.name in expected and member.name not in names, "Nonpublic/duplicate archive member")
                names.add(member.name)
                path = expected[member.name]
                require(not member.pax_headers and member.uid == member.gid == member.mtime == 0
                        and not member.uname and not member.gname, "Unexpected archive metadata")
                require(member.isdir() if path.is_dir() else member.isfile(), "Unsafe archive member type")
                if member.isfile():
                    import hashlib
                    handle = packed.extractfile(member)
                    require(member.size == path.stat().st_size and hashlib.file_digest(handle, "sha256").hexdigest() == file_sha(path), "Archive bytes differ from public file")
                    files += 1
        error = process.stderr.read().decode()
        require(process.wait() == 0, f"Invalid zstd stream: {error}")
        require(names == set(expected) and files == 242, "Archive coverage mismatch")
    finally:
        process.stdout.close()
        process.stderr.close()
        if process.poll() is None:
            process.terminate()
            process.wait()
    return {"regular_files": files, "members": len(names), "private_files": 0,
            "sha256": file_sha(archive), "size_bytes": archive.stat().st_size}


def build_handoff(output_dir, repo=audio.REPO):
    output_dir = Path(output_dir)
    archive = output_dir.with_name(output_dir.name + ".tar.zst")
    require(not output_dir.exists() and not output_dir.is_symlink(), "Output already exists; refusing overwrite")
    require(not archive.exists() and not archive.is_symlink(), "Archive already exists; refusing overwrite")
    rows, sources, images, table, provenance = load_inputs(repo)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".subjective-handoff-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        render_bundle(staging, rows, sources, images, table, provenance)
        summary = validate_bundle(staging, rows)
        staged_archive = Path(temporary) / "package.tar.zst"
        write_archive(staging, staged_archive)
        summary["archive"] = validate_archive(staging, staged_archive)
        (staging / "private/validation_summary.json").write_bytes(json_bytes(summary))
        require(not output_dir.exists() and not archive.exists(), "Destination appeared during build")
        os.rename(staging, output_dir)
        os.rename(staged_archive, archive)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    require(args.output_dir.resolve().is_relative_to(Path("/mnt/hd")) and Path("/mnt/hd").is_mount(), "Final handoff must be on mounted HDD /mnt/hd")
    print(json.dumps(build_handoff(args.output_dir), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
