"""Finalize documentary context review in a separate artifact; never render audio."""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

from . import select_subjective_240 as source

REPO = source.REPO
REVIEW = Path(__file__).with_name("context_description_final_review_v1.json")
BASE = REPO / "outputs/subjective/subjective-240-v1"
OUTPUT = REPO / "outputs/subjective/subjective-240-v1-context-final"
HEADING = "Contexto do jogo"
QUESTION = "Considerando a imagem e a descrição do jogo, quanto esta música combina com esse contexto?"
SCALE = ["Não combina nada", "Combina pouco", "Mais ou menos", "Combina bem", "Combina muito bem"]
SPOKEN_TEXT = "Este áudio é uma checagem de atenção. Para confirmar que você está prestando atenção à avaliação, selecione a opção ‘Não combina nada’ para esta faixa."
REVIEWED = {31, 36, 41, 61, 62, 99, 171, 183, 188, 194, 198, 211, 220, 236, 239}
GROUPS = {"strong": 0, "moderate": 1, "normal": 2, "weak": 3}
CHANGED_FIELDS = {"public_description", "description_ptbr", "description_word_count", "manual_review_recommended",
                  "description_review_reason", "public_context_heading"}
require = source.require


def collection_orders(records):
    """Only evidence category and original serve_order; do not inspect songs."""
    require(len({r["external_id"] for r in records}) == len(records), "Duplicate collection ID")
    require(len({r["serve_order"] for r in records}) == len(records), "Duplicate serve_order")
    def group(r):
        strength = r.get("evidence_strength", "not_assessed")
        require(strength in ("strong", "moderate", "weak", "not_assessed"), "Invalid evidence strength")
        return "normal" if strength == "not_assessed" else strength
    ordered = sorted(records, key=lambda r: (GROUPS[group(r)], r["serve_order"]))
    return {r["external_id"]: (index, group(r)) for index, r in enumerate(ordered)}


def finalize_rows(originals, original_audits, review):
    entries = {e["selection_index"]: e for e in review["entries"]}
    require(len(entries) == len(review["entries"]) == 15 and set(entries) == REVIEWED, "Expected exact 15 reviewed cases")
    records, audits = copy.deepcopy(originals), copy.deepcopy(original_audits)
    for row, audit in zip(records, audits):
        require(row["external_id"] == audit["external_id"], "Audit identity mismatch")
        row.update(public_context_heading=HEADING, evidence_strength="not_assessed")
        if row["selection_index"] in entries:
            entry = entries[row["selection_index"]]
            require((row["external_id"], row["real_tid"], row["real_song_id"], row["description_ptbr"])
                    == (entry["external_id"], entry["real_tid"], entry["real_song_id"], entry["before"]), "Review source identity/text changed")
            checks = source.check_description(entry["after"])
            require(not checks["issues"], f"Invalid revised description: {checks}")
            row.update(public_description=entry["after"], description_ptbr=entry["after"],
                       description_word_count=checks["word_count"], evidence_strength=entry["evidence_strength"],
                       manual_review_recommended=entry["manual_review_recommended"], description_review_reason=entry["review_reason"])
            audit.update(description_ptbr=entry["after"], word_count=checks["word_count"], checks=checks,
                         manual_review_recommended=entry["manual_review_recommended"], review_reason=entry["review_reason"],
                         short_description_rationale=entry["short_description_rationale"], documentary_final_review=entry)
        audit["evidence_strength"] = row["evidence_strength"]
    require(records[31]["description_ptbr"] == records[198]["description_ptbr"], "PewCo must share factual context")
    require(records[188]["description_ptbr"] == originals[188]["description_ptbr"] and records[188]["evidence_strength"] == "weak", "Preserve generic collection context")
    orders = collection_orders(records)
    for before, row, audit in zip(originals, records, audits):
        row["collection_order_v2"], row["collection_priority_group"] = orders[row["external_id"]]
        audit["collection_order_v2"] = row["collection_order_v2"]
        require(all(row[k] == before[k] for k in before if k not in CHANGED_FIELDS), "Frozen selection/matching/permutation field changed")
        if row["selection_index"] not in REVIEWED:
            require(row["description_ptbr"] == before["description_ptbr"], "Unreviewed description changed")
    require(records[188]["collection_order_v2"] == 239, "Index 188 must be last")
    return records, audits


def prepare(base=BASE, review_path=REVIEW, dataset=None):
    dataset = dataset or REPO.parent / "dataset_pico8_v2/dataset_pico8_v2"
    config = json.loads((base / "selection_config.json").read_bytes())
    files = {name: (base / name).read_bytes() for name in config["outputs_sha256"]}
    require(all(source.sha(raw) == config["outputs_sha256"][name] for name, raw in files.items()), "Frozen base artifact changed")
    review_raw = review_path.read_bytes()
    review = json.loads(review_raw)
    for entry in review["entries"]:
        for evidence in entry["sources"]:
            path = dataset / evidence["dataset_ref"]
            require(path.resolve().is_relative_to(dataset.resolve()) and source.sha(path.read_bytes()) == evidence["sha256"], "Documentary evidence changed")
    originals = [json.loads(line) for line in files["scenarios.jsonl"].splitlines()]
    original_audit = json.loads(files["description_audit.json"])
    require(len(originals) == 240 and [r["selection_index"] for r in originals] == list(range(240)), "Invalid frozen base selection")
    records, audits = finalize_rows(originals, original_audit["records"], review)
    flat = [{**{k: v for k, v in r.items() if k != "presentation"},
             "permutation_protocol": r["presentation"]["permutation_protocol"],
             **{f"origin_{p}": r["presentation"]["assignment"][p] for p in "ABCD"}} for r in records]
    audit_fields = tuple(audits[0]) + ("documentary_final_review",)
    csv_audits = [{k: source.canonical(a[k]).decode() if isinstance(a.get(k), (dict, list)) else a.get(k)
                   for k in audit_fields} for a in audits]
    desc_summary = source.description_summary(audits)
    summary = json.loads(files["selection_summary.json"])
    summary.update(descriptions=desc_summary, final_review_count=15,
                   evidence_strength_counts=dict(Counter(r["evidence_strength"] for r in records)),
                   collection_order_v2={"index_188": 239, "zero_based": True, "last_position_1based": 240,
                       "priority": ["strong", "moderate", "normal", "weak"], "stable_within_group": "original serve_order"})
    study = {"heading": HEADING, "question": QUESTION,
             "scale": [{"value": i, "label": label} for i, label in enumerate(SCALE, 1)],
             "scenarios_per_participant": 2,
             "future_attention": {"protocol": "attention-check-spoken-v2", "text": SPOKEN_TEXT,
                 "speech_start_sec": 0, "remaining_segment": "silence until exactly 30 seconds", "target_duration_sec": 30,
                 "requested_rating": 1, "state": "documented_only_not_generated",
                 "replaces_gaussian_for_future_collection_only": True}}
    queue = [{k: r[k] for k in ("collection_order_v2", "selection_index", "external_id", "serve_order", "collection_priority_group", "evidence_strength")}
             for r in sorted(records, key=lambda r: r["collection_order_v2"])]
    files.update({"scenarios.jsonl": b"".join(source.canonical(r) + b"\n" for r in records),
                  "scenarios.csv": source.frozen.csv_bytes(tuple(flat[0]), flat),
                  "descriptions_ptbr.csv": source.frozen.csv_bytes(audit_fields, csv_audits),
                  "description_audit.json": source.json_bytes(dict(desc_summary, records=audits)),
                  "selection_summary.json": source.json_bytes(summary), "study_protocol.json": source.json_bytes(study),
                  "collection_queue_v2.csv": source.frozen.csv_bytes(tuple(queue[0]), queue),
                  "final_review.json": review_raw, "source_selection_config.json": (base / "selection_config.json").read_bytes()})
    config["base_selection_config_sha256"] = source.sha(files["source_selection_config.json"])
    config["source_implementation_sha256"] = config["implementation_sha256"]
    config["implementation_sha256"] = source.sha(Path(__file__).read_bytes())
    config["finalization"] = {"description_protocol": source.DESCRIPTION_PROTOCOL, "review_sha256": source.sha(review_raw),
                              "implementation_sha256": source.sha(Path(__file__).read_bytes()),
                              "preserve_base_serve_order": True, "collection_order_v2_is_separate": True,
                              "unassessed_contexts": "normal queue group, preserving previous review flags; no new strength assessment",
                              "attention_audio_changes": False}
    config["collection_design"].update(collection_order_field="collection_order_v2", serve_selected_order=False,
                                        original_selection_order_preserved=True)
    config["outputs_sha256"] = {k: source.sha(v) for k, v in files.items()}
    files["selection_config.json"] = source.json_bytes(config)
    return files, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "Use a fresh output directory")
    files, summary = prepare()
    source.frozen.write_subset(args.output_dir, files)
    print(json.dumps({"selected": summary["selected"], "strengths": summary["evidence_strength_counts"],
                      "manual_review": summary["descriptions"]["manual_review_recommended_count"],
                      "index188_collection_order_v2": 239, "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
