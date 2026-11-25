#!/usr/bin/env python3
"""subjective-240-v1 plus static semantic context-description-ptbr-v1 curation.

Selection sees only frozen Candidate objects and technical image eligibility.
Descriptions, costs and historical subset joins are processed after selection.
No audio, Attention, new matching, app integration or canonical rewrites.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import statistics

import PIL

from . import select_subjective_subset as frozen


REPO = frozen.REPO
PROTOCOL = "subjective-240-v1"
DESCRIPTION_PROTOCOL = "context-description-ptbr-v1"
QUOTA = 10
TARGET = 240
CURATION = Path(__file__).with_name("context_descriptions_ptbr_v1.json")
OLD_SUBSET = "outputs/subjective/subjective-48-v1"
OLD_PINS = {
    "scenarios.jsonl": "6c8c966b99c51d4eef8fd2a24a93b061ee6f7954d803f051f454aa238649fae5",
    "selection_config.json": "487197c15fc074b89daeb9e448548232a17fd06bde9f055d1d1d8facd6d3a692",
}
WORDS = r"[^\W_]+(?:[-’'][^\W_]+)*"
CHECK_PATTERNS = {
    "url": r"https?://|www\.|\b\S+\.(?:com|org|net|io)(?:/|\b)",
    "music_audio": r"\b(?:m[uú]sic\w*|trilhas?|sonor\w*|[aá]udio\w*|ritmo\w*|melodi\w*|canç(?:ão|ões)|soundtracks?|sounds?)\b",
    "keys_controls": r"\b(?:teclas?|teclados?|bot(?:ão|ões)|controles?|arrow\s+keys?|pression\w*|aperte|cliqu\w*)\b|[⬅️➡️⬆️⬇️]",
    "operational": r"\b(?:menus?|downloads?|como jogar|para iniciar|selecione|score|highscore|pontuaç\w*|placar|recordes?)\b|\b(?:instal\w*|configur\w*)\s+(?:d[oa]\s+|[oa]\s+)?(?:jogo|software|programa|aplicativo|pacote)\b",
    "english_phrase": r"\b(?:how to play|game over|you (?:are|have|can)|use the|press the|left/right)\b",
}
ENGLISH_WORDS = {"the", "you", "your", "with", "this", "that", "these", "those", "have", "from", "and", "where", "when"}
AUDIT_FIELDS = ("selection_index", "external_id", "real_tid", "context_tier", "description_original", "description_ptbr",
                "word_count", "original_description_fallback", "manual_review_recommended", "review_reason",
                "short_description_rationale", "description_evidence_sha256", "evidence_fields", "evidence_source", "checks")
require = frozen.require
sha = frozen.sha
canonical = frozen.canonical
json_bytes = frozen.json_bytes


def scan_candidates(candidates, load_visual, *, quota=QUOTA):
    """Same frozen order/eligibility rule, increased quota only; never read music."""
    require(type(quota) is int and quota > 0, "Invalid quota")
    ordered = sorted(candidates, key=lambda c: c.serve_order)
    require(len({c.serve_order for c in ordered}) == len(ordered), "Duplicate serve_order")
    require(len({c.external_id for c in ordered}) == len(ordered), "Duplicate external_id")
    counts = Counter({p: 0 for p in frozen.PERMUTATIONS})
    selected, scan, assets = [], [], {}
    for c in ordered:
        require(c.permutation in counts, "Invalid frozen permutation")
        reasons, visual = list(c.reasons), None
        if not reasons:
            visual = load_visual(c)
            if visual[0] is None:
                reasons.append(visual[2])
        if reasons:
            decision, reason = "ineligible", ";".join(reasons)
        elif counts[c.permutation] >= quota:
            decision, reason = "quota_full", "frozen_permutation_quota_full"
        else:
            decision, reason = "selected", "first_eligible_in_serve_order_with_quota"
            selected.append(c)
            assets[c.external_id] = visual
            counts[c.permutation] += 1
        scan.append(dict(external_id=c.external_id, serve_order=c.serve_order,
                         permutation="|".join(c.permutation), decision=decision, reason=reason))
        if all(n == quota for n in counts.values()):
            break
    require(len(selected) == len(frozen.PERMUTATIONS) * quota and all(n == quota for n in counts.values()), "Insufficient eligible candidates")
    return selected, scan, assets


def evidence_bundle(pool, context, asset):
    """Only contextual evidence; no music-call metadata, outcomes or costs."""
    packed, source_hash, _ = asset
    return {"description_core": pool["description_core"], "description": pool["description"],
            "original_public_description": context["public_description"], "tags": json.loads(pool["tags"]),
            "visual": {"type": context["public_visual_type"], "source_relative_path": context["source_relative_path"],
                       "source_sha256": source_hash, "sanitized_png_sha256": sha(packed)}}


def prepare_selection(paths, dataset_root):
    configs, hashes, candidates, by_id, markov, neural, known = frozen.load_contracts(paths, 1854)
    selected, scan, assets = scan_candidates(candidates, frozen.visual_loader(dataset_root))
    require(len(selected) == TARGET, "Incorrect target")
    # Nothing below can change selected, scan or the frozen assignments.
    old_raw = {name: (REPO / OLD_SUBSET / name).read_bytes() for name in OLD_PINS}
    require({name: sha(raw) for name, raw in old_raw.items()} == OLD_PINS, "Historical 48 bytes changed")
    old_rows = [json.loads(line) for line in old_raw["scenarios.jsonl"].splitlines()]
    old = {r["external_id"]: r for r in old_rows}
    require(len(old) == 48 and set(old) <= {c.external_id for c in selected}, "Historical 48 not contained")
    pool_rows = frozen.csv_rows(Path(paths["pool"]).read_bytes())
    pool = {p["tid"] + ":" + p["song_id"]: p for p in pool_rows}
    records, evidence, images = [], {}, {}
    for index, candidate in enumerate(selected):
        identifier = candidate.external_id
        context, draft, key = by_id[identifier]
        km, kn = markov[key], neural[key]
        rid = frozen.manifest.source_id(key)
        require(rid != frozen.UNMATCHED_REAL and int(kn["generated_song_id"]) != frozen.SILENT_NEURAL, "Ineligible source selected")
        packed, source_hash, _ = assets[identifier]
        image_ref = f"visuals/{identifier}.png"
        images[image_ref] = packed
        public = frozen.contexts.public_projection(context, published_visual_source=image_ref)
        row = dict(selection_index=index, external_id=identifier, real_record_id=rid, real_tid=key[0], real_song_id=key[1],
                   real_record_sha256=km["real_record_sha256"], serve_order=candidate.serve_order,
                   context_tier=context["context_tier"], **{k: public[k] for k in frozen.contexts.PUBLIC_FIELDS if k != "external_id"},
                   public_description_original=public["public_description"], original_description_fallback=context["description_fallback"],
                   public_visual_sha256=sha(packed), source_visual_sha256=source_hash,
                   visual_source_relative_path=context["source_relative_path"], visual_usable=True,
                   presentation=draft["presentation"], canonical_scenario_sha256=sha(canonical(draft)),
                   previous_48_selection_index=old[identifier]["selection_index"] if identifier in old else None,
                   markov_generated_id=km["generated_id"], markov_record_sha256=km["generated_record_sha256"],
                   markov_song_id=int(km["generated_song_id"]), markov_matching_cost=float(km["total_cost"]),
                   neural_generated_id=kn["generated_id"], neural_record_sha256=kn["generated_record_sha256"],
                   neural_song_id=int(kn["generated_song_id"]), neural_matching_cost=float(kn["total_cost"]), neural_system=frozen.NEURAL_SYSTEM)
        require(all(math.isfinite(row[f"{o}_matching_cost"]) for o in ("markov", "neural")), "Nonfinite provenance cost")
        if identifier in old:
            fields = ("real_record_id", "real_record_sha256", "real_tid", "real_song_id", "serve_order", "context_tier", "presentation",
                      "markov_generated_id", "markov_record_sha256", "markov_song_id", "markov_matching_cost",
                      "neural_generated_id", "neural_record_sha256", "neural_song_id", "neural_matching_cost", "public_visual_sha256")
            require(all(row[k] == old[identifier][k] for k in fields)
                    and row["public_description_original"] == old[identifier]["public_description"], "Historical association/visual/context changed")
        evidence[identifier] = evidence_bundle(pool[key[0] + ":" + str(key[1])], context, assets[identifier])
        records.append(row)
    for field in ("real_record_id", "markov_generated_id", "neural_generated_id"):
        require(len({r[field] for r in records}) == TARGET, f"Repeated identity: {field}")
    return records, evidence, images, scan, hashes, configs, known


def check_description(text):
    issues = []
    if not isinstance(text, str) or not text.strip():
        return {"word_count": 0, "sentence_count": 0, "issues": ["empty"]}
    words = re.findall(WORDS, text)
    for name, pattern in CHECK_PATTERNS.items():
        if re.search(pattern, text, re.I):
            issues.append(name)
    english = sum(w.casefold() in ENGLISH_WORDS for w in words)
    if english >= 3 and english / max(1, len(words)) >= 0.12:
        issues.append("significant_english_words")
    sentences = len(re.findall(r"[.!?](?:[\"”']?)(?:\s|$)", text))
    if sentences not in (1, 2):
        issues.append("sentence_count")
    if len(words) > 60:
        issues.append("over_60_words")
    return {"word_count": len(words), "sentence_count": sentences, "issues": issues}


def apply_descriptions(records, evidence, curation):
    require(curation["protocol"] == DESCRIPTION_PROTOCOL, "Unexpected description protocol")
    entries = {}
    for item in curation["entries"]:
        require(bool(item["evidence_fields"]) and set(item["evidence_fields"]) <= {"description", "description_core", "tags", "visual"}, "Invalid evidence fields")
        require(type(item["manual_review_recommended"]) is bool, "Review flag must be boolean")
        require(bool(item["review_reason"].strip()) == item["manual_review_recommended"], "Review rationale missing/inconsistent")
        for identifier in item["external_ids"]:
            require(identifier not in entries, "Duplicate curation external ID")
            entries[identifier] = item
    require(set(entries) == {r["external_id"] for r in records}, "Curation coverage differs from selected scenarios")
    result, audits = [], []
    for original in records:
        row = dict(original)
        identifier = row["external_id"]
        item, sources = entries[identifier], evidence[identifier]
        require(item["tid"] == row["real_tid"] and item["evidence_sha256"] == sha(canonical(sources)), "Curated description source fingerprint changed")
        text = item["description_ptbr"]
        checks = check_description(text)
        require(not checks["issues"], f"Description needs semantic revision: {identifier}: {checks['issues']}")
        require(checks["word_count"] >= 25 or bool(item["short_description_rationale"].strip()), "Short description requires evidence rationale")
        row.update(public_description=text, description_ptbr=text, description_protocol=DESCRIPTION_PROTOCOL,
                   description_evidence_sha256=item["evidence_sha256"], description_word_count=checks["word_count"],
                   manual_review_recommended=item["manual_review_recommended"], description_review_reason=item["review_reason"])
        result.append(row)
        audits.append(dict(selection_index=row["selection_index"], external_id=identifier, real_tid=row["real_tid"],
                           context_tier=row["context_tier"], description_original=row["public_description_original"], description_ptbr=text,
                           word_count=checks["word_count"], original_description_fallback=row["original_description_fallback"],
                           manual_review_recommended=item["manual_review_recommended"], review_reason=item["review_reason"],
                           short_description_rationale=item["short_description_rationale"], description_evidence_sha256=item["evidence_sha256"],
                           evidence_fields=item["evidence_fields"], evidence_source={
                               "pool_ref": frozen.REFS["pool"], "pool_serve_order": row["serve_order"],
                               "context_ref": frozen.REFS["contexts"], **sources}, checks=checks))
    return result, audits


def description_summary(audits):
    counts = [a["word_count"] for a in audits]
    return {"protocol": DESCRIPTION_PROTOCOL, "produced": len(audits),
            "word_count": {"min": min(counts), "median": statistics.median(counts), "mean": statistics.mean(counts), "max": max(counts)},
            "original_fallback_count": sum(a["original_description_fallback"] for a in audits),
            "manual_review_recommended_count": sum(a["manual_review_recommended"] for a in audits),
            "manual_review_cases": [{k: a[k] for k in ("selection_index", "external_id", "context_tier", "description_ptbr", "review_reason")}
                                    for a in audits if a["manual_review_recommended"]],
            "short_evidence_limited_count": sum(a["word_count"] < 25 for a in audits),
            "short_evidence_limited_cases": [{k: a[k] for k in ("external_id", "word_count", "short_description_rationale")}
                                            for a in audits if a["word_count"] < 25],
            "check_issue_counts": {name: sum(name in a["checks"]["issues"] for a in audits)
                                   for name in (*CHECK_PATTERNS, "empty", "significant_english_words", "sentence_count", "over_60_words")},
            "checks_limit": "lexical guards plus semantic curation; not an automatic proof of language or factual entailment"}


def prepare_subset(paths, dataset_root, curation_path=CURATION):
    records, evidence, images, scan, hashes, configs, known = prepare_selection(paths, dataset_root)
    curation_raw = Path(curation_path).read_bytes()
    curation = json.loads(curation_raw)
    records, audits = apply_descriptions(records, evidence, curation)
    permutations = Counter("|".join(r["presentation"]["assignment"][p] for p in frozen.manifest.POSITIONS) for r in records)
    require(len(permutations) == 24 and set(permutations.values()) == {10}, "Quota audit failed")
    positions = {p: dict(Counter(r["presentation"]["assignment"][p] for r in records)) for p in frozen.manifest.POSITIONS}
    require(all(set(v.values()) == {60} and set(v) == set(frozen.manifest.ORIGINS) for v in positions.values()), "Slot balance failed")
    description_audit = description_summary(audits)
    summary = {"protocol": PROTOCOL, "selected": TARGET, "candidates_examined": len(scan), "last_serve_order_examined": scan[-1]["serve_order"],
               "scan_decisions": dict(Counter(a["decision"] for a in scan)), "context_tiers": {t: sum(r["context_tier"] == t for r in records) for t in "ABCD"},
               "permutation_counts": dict(sorted(permutations.items())), "position_counts": positions,
               "unique_real": len({r["real_record_id"] for r in records}), "unique_markov": len({r["markov_generated_id"] for r in records}),
               "unique_neural": len({r["neural_generated_id"] for r in records}), "usable_visuals": len(images),
               "historical_48_contained": sum(r["previous_48_selection_index"] is not None for r in records),
               "historical_48_unchanged_associations": True, "selected_serve_order_increasing": True,
               "descriptions": description_audit, "scenarios_per_participant": 2, "audio_rendered": False, "attention_generated": False}
    flat_rows = [{**{k: v for k, v in r.items() if k != "presentation"},
                  "permutation_protocol": r["presentation"]["permutation_protocol"],
                  **{f"origin_{p}": r["presentation"]["assignment"][p] for p in frozen.manifest.POSITIONS}} for r in records]
    audit_csv = [{k: (canonical(a[k]).decode() if isinstance(a[k], (dict, list)) else a[k]) for k in AUDIT_FIELDS} for a in audits]
    files = {"scenarios.csv": frozen.csv_bytes(tuple(flat_rows[0]), flat_rows),
             "scenarios.jsonl": b"".join(canonical(r) + b"\n" for r in records),
             "descriptions_ptbr.csv": frozen.csv_bytes(AUDIT_FIELDS, audit_csv),
             "selection_summary.json": json_bytes(summary), "description_audit.json": json_bytes(dict(description_audit, records=audits)),
             "selection_scan.csv": frozen.csv_bytes(frozen.SCAN_FIELDS, scan),
             "ineligible_scenarios.csv": frozen.csv_bytes(("external_id", "serve_order", "real_record_id", "reason"), known), **images}
    config = {"protocol": PROTOCOL, "description_protocol": DESCRIPTION_PROTOCOL, "neural_system": frozen.NEURAL_SYSTEM,
              "inputs": {k: {"artifact_ref": frozen.REFS[k], "sha256": v} for k, v in hashes.items()},
              "historical_48_inputs": {k: {"artifact_ref": OLD_SUBSET + "/" + k, "sha256": v} for k, v in OLD_PINS.items()},
              "selection": {"target": TARGET, "quota_per_permutation": QUOTA, "permutations": 24,
                            "order": "frozen serve_order ascending", "eligibility": "same frozen matching/context/image eligibility as subjective-48-v1",
                            "description_fallback_eligible": True, "description_review_flags_do_not_affect_selection": True,
                            "decision_inputs": ["serve_order", "technical eligibility", "frozen permutation quota"],
                            "forbidden_decision_inputs": ["matching cost", "duration", "PE/SC/GC", "Neural results", "musical quality"],
                            "excluded_real": frozen.UNMATCHED_REAL, "excluded_neural_song_id": frozen.SILENT_NEURAL},
              "collection_design": {"scenarios_per_participant": 2, "serve_selected_order": True, "app_integration": False},
              "descriptions": {"curation_ref": "evaluation/subjective/" + CURATION.name, "curation_sha256": sha(curation_raw),
                               "method": curation["method"], "distinct_source_contexts": len(curation["entries"]),
                               "word_count_rule": WORDS, "target_words": [25, 60], "limited_evidence_may_be_shorter": True,
                               "check_patterns": CHECK_PATTERNS, "english_function_words": sorted(ENGLISH_WORDS),
                               "no_runtime_translation_or_text_generation": True, "original_descriptions_preserved": True},
              "visuals": {"protocol": frozen.VISUAL_PROTOCOL, "pillow_version": PIL.__version__, "source_choice_unchanged": True},
              "implementation_sha256": sha(Path(__file__).read_bytes()), "frozen_helpers_sha256": sha(Path(frozen.__file__).read_bytes()),
              "artifact_visibility": "INTERNAL: contains origins/costs/source evidence; never serve directly to browser",
              "audio_attention_or_canonical_manifest_changes": False,
              "determinism": "same input/curation/code/image bytes and Pillow version; no output path or timestamp",
              "outputs_sha256": {k: sha(v) for k, v in files.items()}}
    files["selection_config.json"] = json_bytes(config)
    return files, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=REPO.parent / "dataset_pico8_v2/dataset_pico8_v2")
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs/subjective/subjective-240-v1")
    args = parser.parse_args(argv)
    require(not args.output_dir.exists() and not args.output_dir.is_symlink(), "Output exists; use a new directory")
    files, summary = prepare_subset({k: REPO / v for k, v in frozen.REFS.items()}, args.dataset_root)
    frozen.write_subset(args.output_dir, files)
    print(json.dumps({k: v for k, v in summary.items() if k != "descriptions"}, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary["descriptions"].items() if not k.endswith("cases")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
