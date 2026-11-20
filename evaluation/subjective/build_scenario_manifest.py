#!/usr/bin/env python3
"""Build the INTERNAL DRAFT final-scenario-manifest-v1; no audio or publishing.

Run: python3 -B -m evaluation.subjective.build_scenario_manifest
Only the six explicitly supplied symbolic/context inputs are opened.
"""

import argparse
from collections import Counter
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import re

from . import export_scenario_contexts as contexts


REPO = Path(__file__).resolve().parents[2]
PROTOCOL = "final-scenario-manifest-v1"
MATCHING_PROTOCOL = "symbolic-balanced-v1"
PERMUTATION_PROTOCOL = "abcd-balanced-rotation-v1"
# Frozen order, including lexical order of the INTEGER permutation indices.
ORIGINS = ("real", "markov", "neural", "attention")
POSITIONS = ("A", "B", "C", "D")
REFS = {
    "scenario_context": "outputs/subjective/scenario-context-v1/scenario_contexts_internal.json",
    "scenario_context_config": "outputs/subjective/scenario-context-v1/scenario_context_config.json",
    "real_test": "data/splits/test.jsonl",
    "markov_generated": "results/markov_test_final.jsonl",
    "markov_matching": "outputs/subjective/symbolic-balanced-v1/markov/matching.csv",
    "markov_matching_config": "outputs/subjective/symbolic-balanced-v1/markov/matching_config.json",
}
WAV_FIELDS = ("render_protocol", "wav_source_path", "duration_sec", "sample_rate",
              "channels", "sample_width_bytes", "wav_sha256", "wav_validation",
              "published_audio_id", "published_locator")
PUBLIC_AUDIO_FIELDS = ("position", "audio_id", "locator")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def is_sha256(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(row):
    tid, song = row["tid"], row["song_id"]
    require(type(tid) in (str, int) and type(song) in (str, int), "Invalid identity types")
    return (str(contexts.positive_integer(tid, "tid", 1)),
            contexts.positive_integer(song, "song_id"))


def source_id(key):
    return f"real:{key[0]}:{key[1]}"


def portable_ref(value):
    """Internal locators: canonical relative POSIX or subjective-audio:// only."""
    require(type(value) is str, "Expected relative/logical path")
    tail = value.removeprefix("subjective-audio://")
    return contexts.relative_asset_path(tail).as_posix()


def load_records(path, role, expected_count):
    """Keep only identity/hash/physical line, never WAV paths or symbolic duration."""
    records, digest = {}, hashlib.sha256()
    with Path(path).open("rb") as handle:
        for index, raw in enumerate(handle):
            digest.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            require(isinstance(row, dict), f"{role}: expected JSON object")
            record_hash = sha256(canonical(row))
            key = identity(row) if role == "real" else "sha256:" + record_hash
            require(key not in records, f"Duplicate {role} identity: {key}")
            records[key] = {"sha256": record_hash, "line_index": index,
                            "tid": str(row.get("tid")), "song_id": str(row.get("song_id"))}
    require(len(records) == expected_count, f"{role}: expected {expected_count} records")
    return records, digest.hexdigest()


def permutation_assignments(external_ids):
    """Six lexical representatives with index 0 first, each with 4 left rotations.

    Every block balances all positions; all 24 permutations occur in one cycle.
    Any prefix of the cycle has floor(n/4) or ceil(n/4) of each origin/position.
    The ONLY per-scenario input is external_id; input sequence is irrelevant.
    """
    ids = list(external_ids)
    require(len(ids) == len(set(ids)), "Duplicate external_id in permutation")
    cycle = []
    for tail in itertools.permutations(range(1, 4)):
        representative = (0,) + tail
        for shift in range(4):
            cycle.append(representative[shift:] + representative[:shift])
    ordered = sorted(ids, key=lambda value: (sha256(canonical([PERMUTATION_PROTOCOL, value])), value))
    return {identifier: dict(zip(POSITIONS, (ORIGINS[j] for j in cycle[i % 24])))
            for i, identifier in enumerate(ordered)}


def empty_slot(origin, external_id):
    return {
        "origin": origin, "internal_audio_id": f"{external_id}/{origin}",
        "source_record_id": None, "source_artifact_ref": None,
        "source_record_sha256": None, "source_line_index": None,
        "symbolic_ready": None if origin == "attention" else False,
        "matching_ready": False if origin in ("markov", "neural") else None,
        **dict.fromkeys(WAV_FIELDS), "wav_ready": False, "published_ready": False,
    }


def readiness_errors(row):
    """Metadata gate for future import; never reads WAVs or publishes anything.

    wav_validation must come from a future trusted byte-level WAV verifier.
    Flipping readiness booleans alone cannot satisfy this gate.
    """
    errors = []

    def check(ok, message):
        if not ok:
            errors.append(message)

    def ref_ok(value):
        try:
            portable_ref(value)
            return True
        except (ValueError, TypeError):
            return False

    def nonempty(value):
        return type(value) is str and bool(value.strip())

    def positive_number(value):
        return type(value) in (int, float) and math.isfinite(value) and value > 0

    protocol = row.get("protocol", {})
    check(protocol.get("manifest_protocol") == PROTOCOL, "manifest protocol")
    slots = row.get("audios", {})
    check(set(slots) == set(ORIGINS), "four audio slots required")
    assignment = row.get("presentation", {}).get("assignment", {})
    check(set(assignment) == set(POSITIONS) and Counter(assignment.values()) == Counter(ORIGINS),
          "complete permutation required")
    check(row.get("presentation", {}).get("permutation_protocol") == PERMUTATION_PROTOCOL,
          "permutation protocol")
    public_ids, public_locators = [], []
    external_id = row.get("identity", {}).get("external_id")
    for origin in ORIGINS:
        slot = slots.get(origin, {})
        prefix = origin + ": "
        check(slot.get("origin") == origin, prefix + "origin")
        check(slot.get("internal_audio_id") == f"{external_id}/{origin}", prefix + "internal relation")
        if origin != "attention":
            check(slot.get("symbolic_ready") is True, prefix + "symbolic pending")
            check(nonempty(slot.get("source_record_id")), prefix + "source record missing")
            check(ref_ok(slot.get("source_artifact_ref")), prefix + "source artifact missing/unsafe")
            check(is_sha256(slot.get("source_record_sha256")), prefix + "record checksum missing")
        if origin in ("markov", "neural"):
            check(slot.get("matching_ready") is True, prefix + "matching pending")
            match = slot.get("matching") or {}
            check(nonempty(match.get("protocol")) and ref_ok(match.get("artifact_ref"))
                  and is_sha256(match.get("artifact_sha256"))
                  and match.get("real_record_sha256") == slots.get("real", {}).get("source_record_sha256")
                  and match.get("generated_record_sha256") == slot.get("source_record_sha256")
                  and match.get("generated_id") == slot.get("source_record_id"),
                  prefix + "matching evidence missing/inconsistent")
        if origin == "markov":
            check(slot.get("generated_record_sha256") == slot.get("source_record_sha256")
                  and slot.get("generated_id") == "sha256:" + str(slot.get("source_record_sha256"))
                  and slot.get("source_record_id") == slot.get("generated_id"), prefix + "generated identity/hash")
        check(slot.get("wav_ready") is True, prefix + "WAV pending")
        check(nonempty(slot.get("render_protocol")), prefix + "render protocol pending")
        check(ref_ok(slot.get("wav_source_path")), prefix + "WAV path missing/unsafe")
        check(is_sha256(slot.get("wav_sha256")), prefix + "WAV checksum missing")
        check(positive_number(slot.get("duration_sec")), prefix + "WAV duration missing")
        check(all(type(slot.get(field)) is int and slot[field] > 0
                  for field in ("sample_rate", "channels", "sample_width_bytes")), prefix + "WAV format missing")
        validation = slot.get("wav_validation") or {}
        check(validation.get("protocol") == "wav-validation-v1" and validation.get("valid") is True
              and is_sha256(validation.get("wav_sha256"))
              and validation.get("wav_sha256") == slot.get("wav_sha256"), prefix + "WAV validation missing")
        check(slot.get("published_ready") is True, prefix + "publication pending")
        check(nonempty(slot.get("published_audio_id")), prefix + "published audio ID missing")
        check(ref_ok(slot.get("published_locator")), prefix + "published locator missing/unsafe")
        public_ids.append(slot.get("published_audio_id"))
        public_locators.append(slot.get("published_locator"))
    check(len(set(public_ids)) == 4, "four unique published audio IDs required")
    check(len(set(public_locators)) == 4, "four unique published locators required")
    attention, real = slots.get("attention", {}), slots.get("real", {})
    check(attention.get("generation_ready") is True, "attention: generation pending")
    check(attention.get("parent_origin") == "real"
          and attention.get("parent_audio_id") == real.get("internal_audio_id")
          and is_sha256(attention.get("parent_wav_sha256"))
          and attention.get("parent_wav_sha256") == real.get("wav_sha256"), "attention: parent WAV binding")
    check(attention.get("protocol") == "attention-check-v1"
          and type(attention.get("global_seed")) is int and attention["global_seed"] == 831
          and attention.get("target_snr_db") == 0, "attention: protocol/seed/target SNR")
    check(type(attention.get("derived_seed")) is int and 0 <= attention["derived_seed"] < 2**128,
          "attention: derived seed pending")
    snr, gain = attention.get("effective_snr_db"), attention.get("gain")
    check(type(snr) in (int, float) and math.isfinite(snr) and abs(snr) <= 0.1,
          "attention: effective SNR pending/outside tolerance")
    check(positive_number(gain) and gain <= 1, "attention: gain pending")
    context = row.get("context", {})
    kind, visual = context.get("public_visual_type"), context.get("visual", {})
    if kind == "none":
        check(visual.get("state") == "none" and visual.get("published_locator") is None,
              "visual: explicit none required")
    else:
        check(kind in ("cartridge", "spritesheet") and visual.get("state") == "published"
              and visual.get("published_ready") is True and ref_ok(visual.get("published_locator"))
              and is_sha256(visual.get("published_sha256")), "visual: publication pending")
    return errors


def validate_for_import(row):
    errors = readiness_errors(row)
    require(not errors, "Scenario incomplete: " + "; ".join(errors))


def validate_scenario(row):
    """Reject unsupported status or a false declaration of import readiness."""
    require(row["protocol"]["status"] == "draft", "Only draft status supported")
    require(type(row["protocol"]["ready_for_import"]) is bool, "Readiness must be boolean")
    if row["protocol"]["ready_for_import"]:
        validate_for_import(row)


def public_projection(row):
    """Future DTO allowlist only; not a backend importer. Drafts always fail."""
    validate_for_import(row)
    require(row["protocol"]["ready_for_import"] is True, "Import readiness not declared")
    public_context = {key: row["context"][key] for key in contexts.PUBLIC_FIELDS
                      if key not in ("external_id", "public_visual_source")}
    public_context["external_id"] = row["identity"]["external_id"]
    public_context["source_relative_path"] = row["context"]["source_relative_path"]
    payload = contexts.public_projection(
        public_context, published_visual_source=row["context"]["visual"]["published_locator"])
    payload["audios"] = [dict(zip(PUBLIC_AUDIO_FIELDS, (
        position, row["audios"][row["presentation"]["assignment"][position]]["published_audio_id"],
        row["audios"][row["presentation"]["assignment"][position]]["published_locator"])))
        for position in POSITIONS]
    return payload


def prepare_manifest(scenario_context, scenario_context_config, real_test, markov_generated,
                     markov_matching, markov_matching_config, *, expected_count=1854):
    paths = dict(zip(REFS, (scenario_context, scenario_context_config, real_test, markov_generated,
                           markov_matching, markov_matching_config)))
    # Never follow any paths from inside records/configs (including legacy WAVs).
    raw = {role: Path(path).read_bytes() for role, path in paths.items()
           if role not in ("real_test", "markov_generated")}
    hashes = {role: sha256(content) for role, content in raw.items()}
    real, hashes["real_test"] = load_records(real_test, "real", expected_count)
    generated, hashes["markov_generated"] = load_records(markov_generated, "markov", expected_count)
    context_config = json.loads(raw["scenario_context_config"])
    matching_config = json.loads(raw["markov_matching_config"])
    require(context_config["protocol"] == contexts.PROTOCOL, "Expected scenario-context-v1")
    require(context_config["outputs_sha256"]["scenario_contexts_internal.json"] == hashes["scenario_context"],
            "Scenario context hash mismatch")
    require(context_config["summary"]["total_contexts"] == expected_count, "Context count mismatch")
    require(matching_config["protocol"] == MATCHING_PROTOCOL and matching_config["system_name"] == "markov",
            "Expected symbolic-balanced-v1 markov")
    for role, input_name in (("real_test", "real"), ("markov_generated", "generated")):
        frozen = matching_config["inputs"][input_name]
        require(frozen["sha256"] == hashes[role], f"{role} file hash mismatch")
        require(frozen["count"] == expected_count, f"{role} frozen count mismatch")
    matches = list(csv.DictReader(io.StringIO(raw["markov_matching"].decode("utf-8"), newline="")))
    require(len(matches) == expected_count, "Matching count mismatch")
    by_real, used_generated = {}, set()
    for match in matches:
        key = identity({"tid": match["real_tid"], "song_id": match["real_song_id"]})
        gid = match["generated_id"]
        require(key not in by_real, "Real reused in matching")
        require(gid not in used_generated, "Markov reused in matching")
        require(key in real and gid in generated, "Matching identity missing from symbolic artifact")
        r, g = real[key], generated[gid]
        require(match["system_name"] == "markov", "Wrong matching system")
        require(match["real_record_sha256"] == r["sha256"], "Real record hash mismatch")
        require(match["generated_record_sha256"] == g["sha256"] and gid == "sha256:" + g["sha256"],
                "Markov generated_id/hash mismatch")
        require(match["generated_tid"] == g["tid"] and match["generated_song_id"] == g["song_id"],
                "Markov record provenance mismatch")
        require(int(match["real_index"]) == r["line_index"]
                and int(match["generated_index"]) == g["line_index"], "Matching line provenance mismatch")
        by_real[key] = match
        used_generated.add(gid)
    require(set(by_real) == set(real) and used_generated == set(generated), "Matching coverage mismatch")
    source_contexts = json.loads(raw["scenario_context"])
    require(isinstance(source_contexts, list) and len(source_contexts) == expected_count, "Context count mismatch")
    seen, ids = set(), set()
    for index, context in enumerate(source_contexts, 1):
        key, identifier = identity(context), context["external_id"]
        require(key not in seen and identifier not in ids, "Duplicate context identity/external_id")
        require(identifier == contexts.external_id(*key), "external_id does not bind context identity")
        require(context["serve_order"] == index, "Frozen context serve_order mismatch")
        require(type(context["context_priority"]) is int and context["context_priority"] > 0
                and context["context_tier"] in ("A", "B", "C", "D"), "Invalid frozen priority/tier")
        kind, source = context["public_visual_type"], context["source_relative_path"]
        contexts.public_projection(context, published_visual_source=(
            None if kind == "none" else f"visuals/{identifier}.png"))
        if kind != "none":
            portable_ref(source)
        seen.add(key)
        ids.add(identifier)
    require(seen == set(real), "Context/Real identity coverage mismatch")
    assignments = permutation_assignments(ids)
    rows = []
    for original in source_contexts:
        key, identifier = identity(original), original["external_id"]
        match, r = by_real[key], real[key]
        g = generated[match["generated_id"]]
        slots = {origin: empty_slot(origin, identifier) for origin in ORIGINS}
        slots["real"].update(source_record_id=source_id(key), source_artifact_ref=REFS["real_test"],
                             source_record_sha256=r["sha256"], source_line_index=r["line_index"], symbolic_ready=True)
        matching_ref = {
            "protocol": MATCHING_PROTOCOL, "artifact_ref": REFS["markov_matching"],
            "artifact_sha256": hashes["markov_matching"], "real_record_id": source_id(key),
            "real_record_sha256": r["sha256"], "generated_id": match["generated_id"],
            "generated_record_sha256": g["sha256"],
        }
        slots["markov"].update(
            source_record_id=match["generated_id"], source_artifact_ref=REFS["markov_generated"],
            source_record_sha256=g["sha256"], source_line_index=g["line_index"],
            generated_id=match["generated_id"], generated_record_sha256=g["sha256"],
            generated_index=g["line_index"], symbolic_ready=True, matching_ready=True, matching=matching_ref)
        slots["neural"]["matching"] = None
        slots["attention"].update(
            parent_origin="real", parent_audio_id=slots["real"]["internal_audio_id"],
            protocol="attention-check-v1", global_seed=831, target_snr_db=0,
            parent_wav_sha256=None, derived_seed=None, effective_snr_db=None, gain=None,
            generation_ready=False)
        row = {
            "identity": {field: original[field] for field in (
                "external_id", "tid", "song_id", "serve_order", "context_priority", "context_tier")},
            "protocol": {"manifest_protocol": PROTOCOL, "status": "draft", "ready_for_import": False},
            "context": {
                "artifact_ref": REFS["scenario_context"], "record_external_id": identifier,
                **{field: original[field] for field in contexts.PUBLIC_FIELDS
                   if field not in ("external_id", "public_visual_source")},
                "source_relative_path": original["source_relative_path"],
                "visual": {"state": "none" if original["public_visual_type"] == "none" else "pending",
                           "published_ready": False, "published_locator": None, "published_sha256": None},
            },
            "provenance": {
                "real_test_sha256": hashes["real_test"], "real_record_sha256": r["sha256"],
                "scenario_context_protocol": contexts.PROTOCOL,
                "scenario_context_sha256": hashes["scenario_context"],
                "scenario_context_config_sha256": hashes["scenario_context_config"],
                "markov_matching_protocol": MATCHING_PROTOCOL,
                "markov_matching_sha256": hashes["markov_matching"],
                "markov_matching_config_sha256": hashes["markov_matching_config"],
                "markov_generated_file_sha256": hashes["markov_generated"],
            },
            "presentation": {"permutation_protocol": PERMUTATION_PROTOCOL, "assignment": assignments[identifier]},
            "audios": slots,
        }
        validate_scenario(row)
        require(bool(readiness_errors(row)), "Draft unexpectedly importable")
        rows.append(row)
    content = "".join(canonical(row).decode("utf-8") + "\n" for row in rows)
    summary = {
        "total": len(rows), "unique_real": len(real), "unique_markov_generated_id": len(generated),
        "symbolic_ready": {o: sum(r["audios"][o]["symbolic_ready"] is True for r in rows) for o in ORIGINS},
        "matching_ready": {o: sum(r["audios"][o]["matching_ready"] is True for r in rows) for o in ORIGINS},
        "wav_ready": {o: sum(r["audios"][o]["wav_ready"] for r in rows) for o in ORIGINS},
        "published_ready": {o: sum(r["audios"][o]["published_ready"] for r in rows) for o in ORIGINS},
        "attention_generation_ready": sum(r["audios"]["attention"]["generation_ready"] for r in rows),
        "ready_for_import": sum(r["protocol"]["ready_for_import"] for r in rows),
        "position_counts": {o: {p: sum(r["presentation"]["assignment"][p] == o for r in rows)
                                for p in POSITIONS} for o in ORIGINS},
        "distinct_permutations": len({tuple(r["presentation"]["assignment"][p] for p in POSITIONS) for r in rows}),
        "neural_state": "pending", "attention_state": "pending",
    }
    require(all(n in (len(rows) // 4, (len(rows) + 3) // 4)
                for counts in summary["position_counts"].values() for n in counts.values()), "Unbalanced positions")
    audit = {
        "manifest_protocol": PROTOCOL, "status": "draft", "ready_for_import": False,
        "inputs": {role: {"artifact_ref": REFS[role], "sha256": hashes[role]} for role in REFS},
        "implementation_sha256": sha256(Path(__file__).read_bytes()),
        "context_contract_implementation_sha256": sha256(Path(contexts.__file__).read_bytes()),
        "output_sha256": {"scenario_manifest_draft.jsonl": sha256(content.encode("utf-8"))},
        "record_hash_policy": "SHA256 canonical complete JSON; sort_keys; ensure_ascii=false; compact separators; allow_nan=false; UTF-8",
        "line_index_policy": "zero-based physical JSONL line; provenance only; never a join key",
        "permutation": {"protocol": PERMUTATION_PROTOCOL, "canonical_origins": list(ORIGINS),
                        "scenario_sort": "SHA256 compact JSON [protocol,external_id], then external_id",
                        "cycle": "lexical permutations of indices 1,2,3 prefixed by 0; four left rotations each; repeat 24-cycle",
                        "scope": "frozen external_id set; assignment must be persisted; offline/internal only"},
        "public_audio_ids": "null until future persistent opaque allocation; no secret generated; never derive from external_id+origin",
        "future_browser_schema": {"context_fields": list(contexts.PUBLIC_FIELDS),
                                  "audio_fields": list(PUBLIC_AUDIO_FIELDS),
                                  "rule": "allowlist only; no origin, internal identity, permutation or provenance"},
        "storage": {"logical_root": "subjective-audio://", "future_layout":
                    "render-<id>/{real,markov,neural,attention,excerpts,published,manifests}/"},
        "render_protocol": None,
        "render_pending": ["approved synth commit/hash", "synth file SHA256", "loop_extra_repeats",
                           "excerpt/full-track policy", "gain policy", "final format"],
        "summary": summary,
        "checks": {"scenario_real_bijection": True, "real_markov_bijection": True,
                   "matching_record_hashes_verified": True, "input_file_hashes_verified": True,
                   "no_reuse_or_discard": True, "no_audio_inputs": True, "no_neural_input": True,
                   "all_records_draft_and_incomplete": True, "balanced_positions": True},
    }
    return {"scenario_manifest_draft.jsonl": content,
            "scenario_manifest_config.json": contexts.json_text(audit)}, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role, relative in REFS.items():
        parser.add_argument("--" + role.replace("_", "-"), type=Path, default=REPO / relative)
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs/subjective" / PROTOCOL)
    args = parser.parse_args()
    files, audit = prepare_manifest(**{role: getattr(args, role) for role in REFS})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (args.output_dir / name).write_text(content, encoding="utf-8", newline="")
    print(contexts.json_text(audit["summary"]), end="")


if __name__ == "__main__":
    main()
