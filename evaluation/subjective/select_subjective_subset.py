#!/usr/bin/env python3
"""Freeze 48 scenarios by existing serve_order and two per frozen permutation.

Internal artifact, not a browser DTO. Costs and symbolic durations are accessed
only AFTER selection. No matching, permutation assignment, audio or publishing.
"""

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import gzip
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

import PIL
from PIL import Image

from . import build_scenario_manifest as manifest
from . import export_scenario_contexts as contexts
from ..pico8_metrics import SYNTH_SAMPLE_RATE, SYNTH_SAMPLES_PER_TICK


REPO = Path(__file__).resolve().parents[2]
PROTOCOL = "subjective-48-v1"
NEURAL_SYSTEM = "lstm-grid_block-k10"
NEURAL_PROTOCOL = "symbolic-balanced-rectangular-v1"
UNMATCHED_REAL = "real:31396:0"
SILENT_NEURAL = 1416
QUOTA = 2
PERMUTATIONS = tuple(itertools.permutations(manifest.ORIGINS))
VISUAL_PROTOCOL = "rgb6-png-v1"
REFS = {
    "pool": "outputs/subjective/context-pool-v1/context_pool.csv",
    "pool_config": "outputs/subjective/context-pool-v1/context_pool_config.json",
    "contexts": "outputs/subjective/scenario-context-v1/scenario_contexts_internal.json",
    "context_config": "outputs/subjective/scenario-context-v1/scenario_context_config.json",
    "visual_inventory": "outputs/subjective/scenario-context-v1/visual_assets_manifest.csv",
    "manifest": "outputs/subjective/final-scenario-manifest-v1/scenario_manifest_draft.jsonl",
    "manifest_config": "outputs/subjective/final-scenario-manifest-v1/scenario_manifest_config.json",
    "markov_matching": "outputs/subjective/symbolic-balanced-v1/markov/matching.csv",
    "markov_config": "outputs/subjective/symbolic-balanced-v1/markov/matching_config.json",
    "neural_matching": "outputs/subjective/symbolic-balanced-rectangular-v1/neural-grid_block-k10/matching.csv",
    "neural_config": "outputs/subjective/symbolic-balanced-rectangular-v1/neural-grid_block-k10/matching_config.json",
    "neural_excluded": "outputs/subjective/symbolic-balanced-rectangular-v1/neural-grid_block-k10/excluded_generated.csv",
    "neural_unmatched": "outputs/subjective/symbolic-balanced-rectangular-v1/neural-grid_block-k10/unmatched_real.csv",
    "real": "data/splits/test.jsonl",
    "markov": "results/markov_test_final.jsonl",
    "neural": "artefatos/predicoes-teste/grid_block-k10.jsonl.gz",
}
SCENARIO_FIELDS = (
    "selection_index", "external_id", "real_record_id", "real_tid", "real_song_id",
    "real_record_sha256", "serve_order", "context_tier", "public_title", "public_description",
    "description_fallback", "public_context_heading", "public_visual_type", "public_visual_label",
    "public_visual_source", "public_visual_sha256", "visual_usable", "visual_source_relative_path",
    "source_visual_sha256", "permutation_protocol", "origin_A", "origin_B", "origin_C", "origin_D",
    "markov_generated_id", "markov_record_sha256", "markov_song_id", "markov_matching_cost",
    "neural_generated_id", "neural_record_sha256", "neural_song_id", "neural_matching_cost",
    "neural_system", "real_duration_sec", "markov_duration_sec", "neural_duration_sec",
    "canonical_scenario_sha256",
)
SCAN_FIELDS = ("external_id", "serve_order", "permutation", "decision", "reason")
DURATION_FIELDS = ("selection_index", "external_id", "real_record_id", "serve_order",
                   "real_duration_sec", "markov_duration_sec", "neural_duration_sec", "origins_under_30s")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return manifest.canonical(value)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def csv_rows(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))


def csv_bytes(fields, rows):
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return text.getvalue().encode("utf-8")


def real_key(row):
    return str(row["real_tid"]), contexts.positive_integer(row["real_song_id"], "real_song_id")


def matching_table(rows, system):
    """Validate identities only: cost columns are not consulted here."""
    result, generated = {}, set()
    for row in rows:
        key, gid = real_key(row), row["generated_id"]
        require(key not in result and gid not in generated, "Matching reuses an identity")
        require(row["system_name"] == system, "Unexpected matching system")
        require(gid == "sha256:" + row["generated_record_sha256"], "Invalid generated identity/hash")
        if system == NEURAL_SYSTEM:
            require(int(row["generated_song_id"]) != SILENT_NEURAL, "Silent Neural must not be matched")
        result[key] = row
        generated.add(gid)
    return result


@dataclass(frozen=True)
class Candidate:
    external_id: str
    serve_order: int
    permutation: tuple
    reasons: tuple
    visual_type: str
    source_relative_path: str


def clean_png(raw, kind):
    """Remove the PICO-8 two-bit carrier, alpha and all PNG metadata."""
    expected_size = {"cartridge": (160, 205), "spritesheet": (128, 128)}[kind]
    with Image.open(io.BytesIO(raw)) as source:
        require(source.format == "PNG" and source.size == expected_size, "Invalid frozen PNG format/size")
        source.verify()
    with Image.open(io.BytesIO(raw)) as source:
        source.load()
        rgb = source.convert("RGB").tobytes()
    # PICO-8 stores a byte in the low two bits of B,G,R,A. Retain only visible
    # upper six RGB bits; RGB output has constant implicit alpha, no ROM bytes.
    pixels = rgb.translate(bytes(i & 0xFC for i in range(256)))
    clean = Image.frombytes("RGB", expected_size, pixels)
    output = io.BytesIO()
    clean.save(output, format="PNG", compress_level=9, optimize=False)
    packed = output.getvalue()
    with Image.open(io.BytesIO(packed)) as check:
        check.load()
        require(check.mode == "RGB" and not check.info, "Public PNG must be RGB without metadata")
        require(all((v & 3) == 0 for v in check.tobytes()), "PICO-8 carrier was not cleared")
    return packed


def visual_loader(dataset_root):
    root, cache = Path(dataset_root).resolve(), {}

    def load(candidate):
        key = candidate.visual_type, candidate.source_relative_path
        if key not in cache:
            relative = contexts.relative_asset_path(candidate.source_relative_path)
            path = (root / relative).resolve()
            require(path.is_relative_to(root), "Visual source escapes dataset root")
            try:
                raw = path.read_bytes()
                packed = clean_png(raw, candidate.visual_type)
                cache[key] = (packed, sha(raw), "")
            except (OSError, ValueError, SyntaxError) as exc:
                cache[key] = (None, None, "missing_or_unusable_image:" + type(exc).__name__)
        return cache[key]

    return load


def scan_candidates(candidates, load_visual):
    """The decision boundary has no cost, tier, duration, metrics or music fields."""
    ordered = sorted(candidates, key=lambda item: item.serve_order)
    require(len({c.serve_order for c in ordered}) == len(ordered), "Duplicate serve_order")
    require(len({c.external_id for c in ordered}) == len(ordered), "Duplicate external_id")
    counts = Counter({p: 0 for p in PERMUTATIONS})
    selected, audit, assets = [], [], {}
    for c in ordered:
        require(c.permutation in counts, "Invalid frozen permutation")
        reasons = list(c.reasons)
        visual = None
        if not reasons:
            visual = load_visual(c)
            if visual[0] is None:
                reasons.append(visual[2])
        if reasons:
            decision, reason = "ineligible", ";".join(reasons)
        elif counts[c.permutation] >= QUOTA:
            decision, reason = "quota_full", "frozen_permutation_already_has_two"
        else:
            decision, reason = "selected", "first_eligible_in_serve_order_with_quota"
            selected.append(c)
            assets[c.external_id] = visual
            counts[c.permutation] += 1
        audit.append(dict(external_id=c.external_id, serve_order=c.serve_order,
                          permutation="|".join(c.permutation), decision=decision, reason=reason))
        if all(counts[p] == QUOTA for p in PERMUTATIONS):
            break
    require(len(selected) == 48 and all(counts[p] == 2 for p in PERMUTATIONS),
            "Insufficient eligible scenarios to fill all 24 quotas")
    return selected, audit, assets


def load_contracts(paths, expected_count):
    raw = {role: Path(path).read_bytes() for role, path in paths.items()
           if role not in ("real", "markov", "neural")}
    hashes = {role: sha(value) for role, value in raw.items()}
    configs = {role: json.loads(raw[role]) for role in (
        "pool_config", "context_config", "manifest_config", "markov_config", "neural_config")}
    pc, cc, mc, kc, nc = (configs[k] for k in configs)
    require(pc["protocol"] == contexts.POOL_PROTOCOL and cc["protocol"] == contexts.PROTOCOL,
            "Unexpected frozen context protocol")
    require(mc["manifest_protocol"] == manifest.PROTOCOL and kc["protocol"] == manifest.MATCHING_PROTOCOL
            and nc["protocol"] == NEURAL_PROTOCOL and nc["system_name"] == NEURAL_SYSTEM,
            "Unexpected manifest/matching protocol")
    checks = {
        "pool": pc["context_pool_csv_sha256"], "pool_config": cc["source_pool_config_sha256"],
        "contexts": cc["outputs_sha256"]["scenario_contexts_internal.json"],
        "visual_inventory": cc["outputs_sha256"]["visual_assets_manifest.csv"],
        "manifest": mc["output_sha256"]["scenario_manifest_draft.jsonl"],
        "markov_matching": mc["inputs"]["markov_matching"]["sha256"],
        "markov_config": mc["inputs"]["markov_matching_config"]["sha256"],
        "context_config": mc["inputs"]["scenario_context_config"]["sha256"],
    }
    require(cc["source_pool_csv_sha256"] == hashes["pool"], "Context/pool hash mismatch")
    for role, digest in checks.items():
        require(hashes[role] == digest, f"Frozen input hash mismatch: {role}")
    require(hashes["contexts"] == mc["inputs"]["scenario_context"]["sha256"], "Manifest/context hash mismatch")
    for role, name in (("neural_matching", "matching.csv"), ("neural_excluded", "excluded_generated.csv"),
                       ("neural_unmatched", "unmatched_real.csv")):
        require(hashes[role] == nc["outputs_sha256"][name], f"Neural output checksum mismatch: {role}")
    pool = csv_rows(raw["pool"])
    context_rows = json.loads(raw["contexts"])
    drafts = [json.loads(line) for line in raw["manifest"].splitlines() if line.strip()]
    require(len(pool) == len(context_rows) == len(drafts) == expected_count, "Frozen input count mismatch")
    markov = matching_table(csv_rows(raw["markov_matching"]), "markov")
    neural = matching_table(csv_rows(raw["neural_matching"]), NEURAL_SYSTEM)
    require(len(markov) == expected_count and len(neural) == expected_count - 1, "Matching count mismatch")
    unmatched, excluded = csv_rows(raw["neural_unmatched"]), csv_rows(raw["neural_excluded"])
    require(len(unmatched) == len(excluded) == 1, "Expected one unmatched Real and one excluded Neural")
    require(unmatched[0]["real_record_id"] == UNMATCHED_REAL, "Unexpected unmatched Real")
    require(int(excluded[0]["generated_song_id"]) == SILENT_NEURAL
            and excluded[0]["generated_record_sha256"] == nc["eligibility"]["approved_exclusion"]["record_sha256"],
            "Unexpected Neural exclusion")
    require(set(markov) == set(neural) | {real_key(unmatched[0])}, "Matching coverage differs")
    inventory = {r["external_id"]: r for r in csv_rows(raw["visual_inventory"])}
    candidates, by_id, known_ineligible = [], {}, []
    for order, (p, c, d) in enumerate(zip(pool, context_rows, drafts), 1):
        key, identifier = manifest.identity(c), c["external_id"]
        require(identifier not in by_id and identifier == contexts.external_id(*key), "Invalid/duplicate context identity")
        require(manifest.identity(p) == key == manifest.identity(d["identity"]), "Context/manifest/pool join mismatch")
        require(int(p["serve_order"]) == c["serve_order"] == d["identity"]["serve_order"] == order,
                "Frozen serve_order changed")
        require(p["context_tier"] == c["context_tier"] == d["identity"]["context_tier"]
                and c["context_tier"] in ("A", "B", "C", "D"), "Context tier mismatch")
        require(d["identity"]["external_id"] == identifier and d["protocol"]["manifest_protocol"] == manifest.PROTOCOL,
                "Canonical scenario identity/protocol mismatch")
        manifest.validate_scenario(d)
        require(d["protocol"]["ready_for_import"] is False, "Canonical source must remain draft")
        require(d["presentation"]["permutation_protocol"] == manifest.PERMUTATION_PROTOCOL,
                "Unexpected frozen permutation protocol")
        assignment = d["presentation"]["assignment"]
        require(set(assignment) == set(manifest.POSITIONS), "Frozen assignment positions differ")
        permutation = tuple(assignment[position] for position in manifest.POSITIONS)
        require(permutation in PERMUTATIONS, "Invalid frozen assignment")
        kind, source = c["public_visual_type"], c["source_relative_path"]
        contexts.public_projection(c, published_visual_source=None if kind == "none" else f"visuals/{identifier}.png")
        require(all(d["context"][field] == c[field] for field in contexts.PUBLIC_FIELDS
                    if field not in ("external_id", "public_visual_source")), "Frozen public context differs")
        require(d["context"]["source_relative_path"] == source, "Frozen visual source differs")
        reasons = []
        rid = manifest.source_id(key)
        if key not in markov:
            reasons.append("missing_markov_matching")
        if key not in neural or rid == UNMATCHED_REAL:
            reasons.append("unmatched_real_no_neural")
        if kind == "none":
            reasons.append("interface_requires_image_but_frozen_context_has_none")
        else:
            contexts.relative_asset_path(source)
            require(identifier in inventory and inventory[identifier]["source_relative_path"] == source
                    and inventory[identifier]["visual_type"] == kind, "Frozen visual inventory mismatch")
        for table, origin in ((markov, "markov"), (neural, "neural")):
            if key in table:
                require(table[key]["real_record_sha256"] == d["audios"]["real"]["source_record_sha256"],
                        f"{origin}/Real record hash mismatch")
        if key in markov:
            require(d["audios"]["markov"]["generated_id"] == markov[key]["generated_id"]
                    and d["audios"]["markov"]["source_record_sha256"] == markov[key]["generated_record_sha256"],
                    "Markov association differs from canonical manifest")
        candidate = Candidate(identifier, order, permutation, tuple(reasons), kind, source)
        candidates.append(candidate)
        by_id[identifier] = (c, d, key)
        if reasons:
            known_ineligible.append(dict(external_id=identifier, serve_order=order, real_record_id=rid,
                                         reason=";".join(reasons)))
    require(set(markov) == {item[2] for item in by_id.values()}, "Context/matching identity coverage differs")
    return configs, hashes, candidates, by_id, markov, neural, known_ineligible


def selected_sources(path, role, wanted, expected_count):
    """Read and hash source records only after the quota scan has finished."""
    path = Path(path)
    blob = path.read_bytes()
    stream = gzip.GzipFile(fileobj=io.BytesIO(blob)) if path.suffix == ".gz" else io.BytesIO(blob)
    content, selected, seen = hashlib.sha256(), {}, set()
    with stream as handle:
        for index, line in enumerate(handle):
            content.update(line)
            if not line.strip():
                continue
            song = json.loads(line)
            key = manifest.source_id(manifest.identity(song)) if role == "real" else "sha256:" + sha(canonical(song))
            require(key not in seen, f"Duplicate source identity: {role}")
            seen.add(key)
            if key in wanted:
                selected[key] = (song, sha(canonical(song)), index)
    require(len(seen) == expected_count and set(selected) == set(wanted), f"Source coverage mismatch: {role}")
    return selected, sha(blob), content.hexdigest()


def symbolic_duration(song):
    # Exactly the evaluator's symbolic clock, with no song-loop repetitions.
    ticks = sum(max(0, int(p.get("pattern_ticks") or 0)) for p in song.get("patterns", []))
    return ticks * SYNTH_SAMPLES_PER_TICK / SYNTH_SAMPLE_RATE


def duration_statistics(records):
    summary = {}
    for origin in ("real", "markov", "neural"):
        values = [r[f"{origin}_duration_sec"] for r in records]
        summary[origin] = dict(min=min(values), median=statistics.median(values), mean=statistics.mean(values),
                               max=max(values), lt_10s=sum(v < 10 for v in values),
                               lt_20s=sum(v < 20 for v in values), lt_30s=sum(v < 30 for v in values),
                               ge_30s=sum(v >= 30 for v in values))
    return summary


def prepare_subset(paths, dataset_root, *, expected_count=1854):
    require(set(paths) == set(REFS), "Expected all frozen input roles")
    configs, hashes, candidates, by_id, markov, neural, known_ineligible = load_contracts(paths, expected_count)
    selected, scan, assets = scan_candidates(candidates, visual_loader(dataset_root))
    # Selection is now frozen. Neither costs nor music were available to scan_candidates.
    wanted = {role: set() for role in ("real", "markov", "neural")}
    for candidate in selected:
        _, _, key = by_id[candidate.external_id]
        wanted["real"].add(manifest.source_id(key))
        wanted["markov"].add(markov[key]["generated_id"])
        wanted["neural"].add(neural[key]["generated_id"])
    require(all(len(values) == 48 for values in wanted.values()), "Selected music identities are not unique")
    sources = {}
    for role in wanted:
        sources[role], hashes[role], content_hash = selected_sources(paths[role], role, wanted[role], expected_count)
        if role == "real":
            require(hashes[role] == configs["manifest_config"]["inputs"]["real_test"]["sha256"]
                    == configs["neural_config"]["inputs"]["real"]["sha256"], "Real input hash differs")
        elif role == "markov":
            require(hashes[role] == configs["markov_config"]["inputs"]["generated"]["sha256"]
                    == configs["manifest_config"]["inputs"]["markov_generated"]["sha256"], "Markov input hash differs")
        else:
            reference = configs["neural_config"]["inputs"]["generated"]
            require(hashes[role] == reference["sha256"] and content_hash == reference["content_sha256"],
                    "Official Neural input hash differs")
    records, visual_rows, image_files = [], [], {}
    for index, candidate in enumerate(selected):
        identifier = candidate.external_id
        c, d, key = by_id[identifier]
        km, kn = markov[key], neural[key]
        rid = manifest.source_id(key)
        require(rid != UNMATCHED_REAL and int(kn["generated_song_id"]) != SILENT_NEURAL, "Excluded identity selected")
        require(sources["real"][rid][1] == d["audios"]["real"]["source_record_sha256"], "Selected Real hash differs")
        for role, match in (("markov", km), ("neural", kn)):
            song, digest, line_index = sources[role][match["generated_id"]]
            require(digest == match["generated_record_sha256"] and str(song["song_id"]) == match["generated_song_id"]
                    and line_index == int(match["generated_index"]), "Selected generated provenance differs")
        require(sources["neural"][kn["generated_id"]][0]["system"] == NEURAL_SYSTEM, "Selected Neural system differs")
        packed, source_hash, _ = assets[identifier]
        public_source = f"visuals/{identifier}.png"
        image_files[public_source] = packed
        # This public projection now refers to a real sanitized local file.
        public = contexts.public_projection(c, published_visual_source=public_source)
        row = dict(selection_index=index, external_id=identifier, real_record_id=rid,
                   real_tid=key[0], real_song_id=key[1], real_record_sha256=sources["real"][rid][1],
                   serve_order=candidate.serve_order, context_tier=c["context_tier"],
                   **{k: public[k] for k in contexts.PUBLIC_FIELDS if k != "external_id"},
                   description_fallback=c["description_fallback"], visual_usable=True,
                   public_visual_sha256=sha(packed), visual_source_relative_path=c["source_relative_path"],
                   source_visual_sha256=source_hash, presentation=d["presentation"],
                   markov_generated_id=km["generated_id"], markov_record_sha256=km["generated_record_sha256"],
                   markov_song_id=int(km["generated_song_id"]), markov_matching_cost=float(km["total_cost"]),
                   neural_generated_id=kn["generated_id"], neural_record_sha256=kn["generated_record_sha256"],
                   neural_song_id=int(kn["generated_song_id"]), neural_matching_cost=float(kn["total_cost"]),
                   neural_system=NEURAL_SYSTEM, canonical_scenario_sha256=sha(canonical(d)))
        require(math.isfinite(row["markov_matching_cost"]) and math.isfinite(row["neural_matching_cost"]),
                "Nonfinite provenance cost")
        for role, source_id in (("real", rid), ("markov", km["generated_id"]), ("neural", kn["generated_id"])):
            row[f"{role}_duration_sec"] = symbolic_duration(sources[role][source_id][0])
        records.append(row)
        visual_rows.append(dict(external_id=identifier, public_visual_source=public_source,
                                public_visual_sha256=sha(packed), source_relative_path=c["source_relative_path"],
                                source_sha256=source_hash, protocol=VISUAL_PROTOCOL, local_ready=True,
                                web_published=False))
    flat = [{**{k: v for k, v in r.items() if k != "presentation"},
             "permutation_protocol": r["presentation"]["permutation_protocol"],
             **{f"origin_{p}": r["presentation"]["assignment"][p] for p in manifest.POSITIONS}} for r in records]
    duration_rows = [{k: r[k] for k in DURATION_FIELDS if k != "origins_under_30s"} | {
        "origins_under_30s": ";".join(origin for origin in ("real", "markov", "neural")
                                    if r[f"{origin}_duration_sec"] < 30)} for r in records]
    short = [r for r in duration_rows if r["origins_under_30s"]]
    for audit in scan:
        if audit["decision"] == "ineligible" and not any(r["external_id"] == audit["external_id"] for r in known_ineligible):
            c, _, key = by_id[audit["external_id"]]
            known_ineligible.append(dict(external_id=audit["external_id"], serve_order=c["serve_order"],
                                         real_record_id=manifest.source_id(key), reason=audit["reason"]))
    permutation_counts = Counter("|".join(r["presentation"]["assignment"][p] for p in manifest.POSITIONS) for r in records)
    require(len(permutation_counts) == 24 and set(permutation_counts.values()) == {2}, "Quota audit failed")
    summary = {
        "protocol": PROTOCOL, "total_candidates": expected_count, "candidates_examined": len(scan),
        "last_serve_order_examined": scan[-1]["serve_order"], "selected": len(records),
        "unique_real": len(wanted["real"]), "unique_markov": len(wanted["markov"]),
        "unique_neural": len(wanted["neural"]), "scan_decisions": dict(Counter(r["decision"] for r in scan)),
        "context_tiers": {tier: sum(r["context_tier"] == tier for r in records) for tier in "ABCD"},
        "description_fallbacks": sum(r["description_fallback"] for r in records),
        "permutation_counts": dict(sorted(permutation_counts.items())),
        "position_counts": {p: dict(Counter(r["presentation"]["assignment"][p] for r in records)) for p in manifest.POSITIONS},
        "usable_public_visuals": len(image_files), "web_published": False,
        "known_ineligible": sorted(known_ineligible, key=lambda r: r["serve_order"]),
        "duration_statistics": duration_statistics(records), "scenarios_with_tracks_under_30s": short,
        "checks": {"serve_order_preserved": True, "frozen_permutations_preserved": True,
                   "costs_and_durations_used_only_after_selection": True, "no_reassociation": True,
                   "canonical_manifest_unchanged": True},
    }
    files = {
        "scenarios.csv": csv_bytes(SCENARIO_FIELDS, flat),
        "scenarios.jsonl": b"".join(canonical(r) + b"\n" for r in records),
        "selection_summary.json": json_bytes(summary),
        "selection_scan.csv": csv_bytes(SCAN_FIELDS, scan),
        "ineligible_scenarios.csv": csv_bytes(("external_id", "serve_order", "real_record_id", "reason"),
                                              summary["known_ineligible"]),
        "duration_diagnostics.csv": csv_bytes(DURATION_FIELDS, duration_rows),
        "visual_assets_manifest.csv": csv_bytes(tuple(visual_rows[0]), visual_rows),
        **image_files,
    }
    config = {
        "protocol": PROTOCOL, "artifact_visibility": "internal; never send this artifact directly to participants",
        "inputs": {role: {"artifact_ref": REFS[role], "sha256": digest} for role, digest in hashes.items()},
        "selection": {"order": "frozen serve_order ascending", "permutation_source": "canonical manifest only",
                      "quota_per_permutation": 2, "permutations": 24, "target": 48,
                      "stop": "all 24 quotas filled", "ranking_inputs": ["serve_order"],
                      "eligibility": ["Markov matching", "Neural matching", "valid frozen context", "usable image"],
                      "text_fallback_eligible": True, "excluded_real": UNMATCHED_REAL,
                      "forbidden_decision_inputs": ["matching cost", "PE/SC/GC", "duration", "musical preference",
                                                    "Neural results", "perceived quality"]},
        "visuals": {"protocol": VISUAL_PROTOCOL, "source": "exact frozen visual choice, no replacement",
                    "processing": "decode PNG; RGB & 0xFC; discard alpha and metadata; fresh RGB PNG",
                    "scope": "local assets ready for interface; no web publication",
                    "checked_scope": "metadata for all contexts; image bytes for eligible candidates examined before stop",
                    "original_cart_copy_forbidden": True, "pillow_version": PIL.__version__},
        "duration_diagnostic": {"computed_after_selection": True, "clock": [SYNTH_SAMPLES_PER_TICK, SYNTH_SAMPLE_RATE],
                                "loop_extra_repeats": 0, "looping_padding_truncation_synthesis": False},
        "implementation_sha256": sha(Path(__file__).read_bytes()),
        "reused_context_contract_sha256": sha(Path(contexts.__file__).read_bytes()),
        "canonical_manifest_contract_sha256": sha(Path(manifest.__file__).read_bytes()),
        "outputs_sha256": {name: sha(value) for name, value in files.items()},
        "determinism": "same frozen input bytes, source images, code and Pillow version; no timestamps/output path",
    }
    files["selection_config.json"] = json_bytes(config)
    return files, summary


def write_subset(output_dir, files):
    output_dir = Path(output_dir).absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"{output_dir}: output exists; use a new directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temporary:
        staged = Path(temporary)
        for name, raw in files.items():
            relative = contexts.relative_asset_path(name)
            path = staged / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(raw)
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"{output_dir}: output appeared during execution")
        os.replace(staged, output_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=REPO.parent / "dataset_pico8_v2/dataset_pico8_v2")
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs/subjective/subjective-48-v1")
    args = parser.parse_args(argv)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.exit(1, "Output already exists; use a new directory\n")
    files, summary = prepare_subset({role: REPO / ref for role, ref in REFS.items()}, args.dataset_root)
    write_subset(args.output_dir, files)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
