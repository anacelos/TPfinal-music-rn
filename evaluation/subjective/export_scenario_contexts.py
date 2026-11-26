#!/usr/bin/env python3
"""Export frozen context-pool-v1 into an INTERNAL scenario integration artifact.

Run: python3 -B -m evaluation.subjective.export_scenario_contexts
See SCENARIO_CONTEXTS.md for the browser boundary and publication requirements.
No audio, model outputs, pixel analysis, ranking or pool generation is involved.
"""

import argparse
from collections import Counter
import csv
import hashlib
import html
import io
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import uuid

from .context_pool import DESCRIPTION_CUES, ROLE_CONTEXT


REPO = Path(__file__).resolve().parents[2]
PROTOCOL = "scenario-context-v1"
POOL_PROTOCOL = "context-pool-v1"
ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "urn:tpfinal:subjective:scenario-context-v1")
TITLE_FALLBACK = ""
DESCRIPTION_FALLBACK = "Descrição contextual não disponível para esta avaliação."
CONTEXT_HEADING = "Cartucho de origem e contexto visual"
# The frozen pool defines only header_only / unknown: neither certifies a title.
# A future approved metadata source requires an explicit policy/version change.
TRUSTED_TITLE_CONFIDENCES = frozenset()
VISUAL_LABELS = {
    "cartridge": "Imagem do cartucho de origem",
    "spritesheet": "Elementos gráficos do cartucho de origem",
    "none": "Sem imagem",
}
PUBLIC_FIELDS = (
    "external_id", "public_title", "public_description", "public_visual_label",
    "public_visual_type", "public_visual_source", "public_context_heading",
)
INTERNAL_ONLY_FIELDS = (
    "tid", "song_id", "serve_order", "context_priority", "context_tier",
    "start_pattern", "end_pattern", "title_fallback", "description_fallback", "source_relative_path",
)
FIELDS = ("external_id",) + INTERNAL_ONLY_FIELDS + tuple(k for k in PUBLIC_FIELDS[1:] if k != "public_visual_source")
ASSET_POLICY = "INTERNAL SOURCE ONLY; publishing required; direct public copy forbidden"
MANIFEST_FIELDS = ("external_id", "source_relative_path", "visual_type", "publication_policy")
REQUIRED_POOL_FIELDS = {
    "tid", "song_id", "split", "serve_order", "context_priority", "context_tier",
    "start_pattern", "end_pattern", "game_title", "game_title_confidence",
    "description_core", "description_informative", "preferred_visual_type",
    "preferred_visual_path",
}

# Headers require section punctuation/line boundaries, except conventional
# multiword headings. A lexical 'player controls' or 'Carbon Credits' is not one.
HEADERS = re.compile(
    r"\b(?:to-do|look forward to|how to play|behind the scenes|technical details|development status|"
    r"main influences|original post)\b|"
    r"\b(?:instructions?|controls|controles|credits|cr[eé]ditos|changelog|"
    r"manual|buttons|commentary|roadmap|to-do)\s*:|"
    r"^\s*(?:instructions?|controls|controles|credits|cr[eé]ditos)\s*$", re.I | re.M)
BAD_HEADER = re.compile(r"credits|cr[eé]ditos|changelog|behind the scenes|technical|development|influences|commentary|roadmap|to-do|look forward to", re.I)
META_RULES = {
    "author_report": r"\b(?:I(?:['’](?:m|ve|d|ll))?|we|my|myself)\b|\b(?:wanted to challenge|finished up|work on making|made (?:in|over|by)|created by)\b",
    "reader_request": r"\b(?:hopefully you|in case anyone|anyone can solve|comment/reach out|leave me a comment|please (?:post|report|let)|report bugs|if you find (?:any|anything)|hopefully|let me know|feedback welcome|thanks? to|thank you|find (?:all my|more of my))\b",
    "planning": r"\b(?:(?:gotta|need to|trying to) find a (?:better|more consistent) way|will (?:probably )?(?:add|implement)|plan(?:ning)? to|to be implemented|not yet implemented)\b",
    "change_report": r"\b(?:tweaked|bugfix|fixed|updated|changelog|previously|removed|replaced|changes? \d|now they correctly|new (?:animated )?(?:levels|animations|screens)|add (?:lava|levels|sfx)|shorter jump in normal mode)\b|^\s*(?:added|changed)\b|\d{2}/\d{2}/\d{4}",
    "development": r"\b(?:work in progress|WIP|mode is implemented|proof of concept|source code|particle effect refers|physics of dash|code by|go through the code|most of the code|debug menu|code cleaned|further limit|has been increased|a bug where|fix .*bug|minor tweak|learn(?:ed|ing)? lua|tutorials?|collision detection|optimization|implementation|token (?:limit|use)|generic sprites|save on space|sprite editor|data mode|developers?|youtube|github|discord|blog|forums?|photoshop|rendered out|built entirely in PICO|custom isometric rendering)\b",
    "source_residue": r"__external_url__|\bBBS\b|\bCart\s*#|\blicense\s*:|\|\s*\|",
    # Public musical associations are excluded independently of internal evidence.
    "musical_reference": r"\b(?:theme|soundtrack|song|music)\b",
}
META_PATTERNS = tuple(re.compile(pattern, re.I) for pattern in META_RULES.values())
AUTHOR_OR_MARKUP = re.compile(r"\bby\s+(?:@|[A-Z][\w-]*)|\|\s*\||@\w+|<[^>]+>")
CLAUSE_BREAK = re.compile(
    r"(?=\b(?:I (?:made|tried|wanted|learned|was|need|will)|in case anyone|"
    r"hopefully you|If you find (?:any|anything)|Thanks to|please report|"
    r"Leave me a comment|enjoy yourself|Things to do|made in|Still WIP|You have to find)\b)", re.I)


def contextual_segment(text):
    """Small lexical presentation guard; never scores/rebuilds the frozen pool."""
    cues = {k for k, pattern in DESCRIPTION_CUES.items() if re.search(pattern, text, re.I)}
    if "role" in cues and not ROLE_CONTEXT.search(text) and not (cues - {"role"}):
        cues.remove("role")
    extra = re.search(r"\b(?:upgrade|add unique effects|earn|restor(?:e|ing)|plant|harvest|glid(?:e|ing)|bosses)\b", text, re.I)
    words = re.findall(r"[^\W\d_]+", text)
    return bool(cues or extra) and len(words) >= 2 and sum(c.isalpha() for c in text) >= 10


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def positive_integer(value, name, minimum=0):
    text = str(value)
    if not re.fullmatch(r"0|[1-9][0-9]*", text) or int(text) < minimum:
        raise ValueError(f"Invalid {name}: {value!r}")
    return int(text)


def external_id(tid, song_id):
    """Opaque UUIDv5 of canonical identity; no tier/order/system/audio inputs."""
    identity = [str(positive_integer(tid, "tid", 1)),
                positive_integer(song_id, "song_id")]
    return str(uuid.uuid5(ID_NAMESPACE, json.dumps(identity, separators=(",", ":"))))


def public_description(row):
    if row["description_informative"] not in (True, "True"):
        return DESCRIPTION_FALLBACK
    core = html.unescape(row["description_core"])
    # Mask complete URLs BEFORE splitting on punctuation inside domains. Reject
    # the containing fragment; the marker can never become participant text.
    core = re.sub(r"(?:https?://|www\.)\S+", " __external_url__ ", core, flags=re.I)
    fragments, cursor, blocked = [], 0, False
    sections = []
    for match in HEADERS.finditer(core):
        sections.append((core[cursor:match.start()], blocked))
        blocked = bool(BAD_HEADER.search(match[0]))
        cursor = match.end()
    sections.append((core[cursor:], blocked))
    for section, blocked in sections:
        if blocked:
            continue
        for fragment in re.split(r"[.!?\n;]+", section):
            for clause in CLAUSE_BREAK.split(fragment):
                clause = re.sub(r"\s+", " ", clause).strip(' \t|:-\"')
                clause = re.sub(r"\s+in which$", "", clause, flags=re.I)
                if clause.count('(') > clause.count(')') and clause.rfind('(') > 0:
                    clause = clause[:clause.rfind('(')].rstrip()
                if (any(p.search(clause) for p in META_PATTERNS)
                        or AUTHOR_OR_MARKUP.search(clause)):
                    continue
                if contextual_segment(clause) and clause.casefold() not in {s.casefold() for s in fragments}:
                    fragments.append(clause)
    result = ". ".join(fragments)
    return result if len(re.findall(r"[^\W\d_]+", result)) >= 3 else DESCRIPTION_FALLBACK


def relative_asset_path(value):
    """Reject absolute, traversal, URI, Windows and noncanonical POSIX paths."""
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or PureWindowsPath(value).drive
            or "\\" in value or ":" in value or ".." in path.parts
            or path.as_posix() != value or value == "."
            or any(ord(c) < 32 for c in value)):
        raise ValueError(f"Asset path must be relative and portable: {value!r}")
    return path


def validate_visual(row, dataset_root):
    kind, value = row["preferred_visual_type"], row["preferred_visual_path"]
    if kind not in VISUAL_LABELS:
        raise ValueError(f"Unsupported frozen visual type: {kind!r}")
    if kind == "none":
        if value:
            raise ValueError("Visual type none must have no asset")
        return ""
    relative = relative_asset_path(value)
    root = Path(dataset_root).resolve()
    asset = (root / relative).resolve()
    if not asset.is_relative_to(root):
        raise ValueError(f"Asset escapes dataset root: {value!r}")
    if not asset.is_file():
        raise ValueError(f"Missing visual asset: {value!r}")
    return relative.as_posix()


def public_projection(row, *, published_visual_source=None):
    """Allowlist for a FUTURE browser payload, after assets are published.

    Only an explicitly supplied opaque publication path can enter the payload.
    Source locators never enter this allowlist. This does not publish assets.
    """
    identifier = row.get("external_id")
    try:
        parsed = uuid.UUID(identifier) if type(identifier) is str else None
    except ValueError:
        parsed = None
    if parsed is None or parsed.version != 5 or str(parsed) != identifier:
        raise ValueError("Expected canonical UUIDv5 external_id")
    kind = row.get("public_visual_type")
    if type(kind) is not str or kind not in VISUAL_LABELS:
        raise ValueError("Invalid public_visual_type")
    for key in PUBLIC_FIELDS:
        if key != "public_visual_source" and type(row.get(key)) is not str:
            raise ValueError(f"Expected string: {key}")
    if row["public_visual_label"] != VISUAL_LABELS[kind] or row["public_context_heading"] != CONTEXT_HEADING:
        raise ValueError("Invalid public label/heading")
    if kind == "none":
        if published_visual_source is not None or row.get("source_relative_path") not in (None, ""):
            raise ValueError("Visual type none must have no asset")
        source = None
    else:
        expected = f"visuals/{identifier}.png"
        if type(published_visual_source) is not str or published_visual_source != expected:
            raise ValueError(f"Published opaque visual path required: {expected}")
        source = published_visual_source
    return {key: source if key == "public_visual_source" else row[key]
            for key in PUBLIC_FIELDS}


def project_rows(pool_rows, dataset_root, *, expected_count=1854, expected_tids=873):
    """Validate 1:1 coverage and physical serving order; never sort/re-rank."""
    if len(pool_rows) != expected_count:
        raise ValueError(f"Expected {expected_count} contexts, got {len(pool_rows)}")
    identities, rows = [], []
    for index, original in enumerate(pool_rows, 1):
        missing = REQUIRED_POOL_FIELDS - original.keys()
        if missing:
            raise ValueError(f"Missing pool fields: {sorted(missing)}")
        tid = str(positive_integer(original["tid"], "tid", 1))
        song = positive_integer(original["song_id"], "song_id")
        identities.append((tid, song))
        if original["split"] != "test":
            raise ValueError("Only frozen TEST contexts are allowed")
        order = positive_integer(original["serve_order"], "serve_order", 1)
        if order != index:
            raise ValueError("Pool physical sequence must exactly follow frozen serve_order")
        if original["context_tier"] not in "ABCD" or len(original["context_tier"]) != 1:
            raise ValueError("Invalid context tier")
        title = original["game_title"].strip()
        if original["game_title_confidence"] not in TRUSTED_TITLE_CONFIDENCES or not title:
            title = TITLE_FALLBACK
        description = public_description(original)
        kind = original["preferred_visual_type"]
        source = validate_visual(original, dataset_root)
        rows.append({
            "external_id": external_id(tid, song), "tid": tid, "song_id": song,
            "serve_order": order,
            "context_priority": positive_integer(original["context_priority"], "context_priority", 1),
            "context_tier": original["context_tier"],
            "start_pattern": positive_integer(original["start_pattern"], "start_pattern"),
            "end_pattern": positive_integer(original["end_pattern"], "end_pattern"),
            "title_fallback": title == TITLE_FALLBACK,
            "description_fallback": description == DESCRIPTION_FALLBACK,
            "public_title": title, "public_description": description,
            "public_visual_label": VISUAL_LABELS[kind], "public_visual_type": kind,
            "source_relative_path": source, "public_context_heading": CONTEXT_HEADING,
        })
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate pool identity")
    if len({tid for tid, _ in identities}) != expected_tids:
        raise ValueError(f"Expected {expected_tids} distinct TIDs")
    if len({row["external_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate external_id")
    if {row["context_priority"] for row in rows} != set(range(1, expected_count + 1)):
        raise ValueError("Frozen context_priority must be a permutation of 1..N")
    if [(row["tid"], row["song_id"]) for row in rows] != identities:
        raise ValueError("Export identities differ from pool sequence")
    if [row["serve_order"] for row in rows] != [int(row["serve_order"]) for row in pool_rows]:
        raise ValueError("Export serve_order differs from pool")
    return rows


def csv_text(rows, fields):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def prepare_export(pool_csv, pool_config, dataset_root, *, expected_count=1854, expected_tids=873):
    raw, config_raw = Path(pool_csv).read_bytes(), Path(pool_config).read_bytes()
    pool_audit = json.loads(config_raw)
    if pool_audit["protocol"] != POOL_PROTOCOL:
        raise ValueError("Expected context-pool-v1")
    if sha256(raw) != pool_audit["context_pool_csv_sha256"]:
        raise ValueError("Frozen context pool CSV hash mismatch")
    originals = list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))
    rows = project_rows(originals, dataset_root, expected_count=expected_count, expected_tids=expected_tids)
    tiers = dict(sorted(Counter(row["context_tier"] for row in rows).items()))
    if (pool_audit["summary"]["total"] != len(rows)
            or pool_audit["summary"]["tids"] != expected_tids
            or pool_audit["summary"]["tiers"] != tiers):
        raise ValueError("Frozen pool summary does not match CSV")
    manifest = [{"external_id": row["external_id"],
                 "source_relative_path": row["source_relative_path"],
                 "visual_type": row["public_visual_type"], "publication_policy": ASSET_POLICY}
                for row in rows if row["public_visual_type"] != "none"]
    files = {
        "scenario_contexts_internal.csv": csv_text(rows, FIELDS),
        "scenario_contexts_internal.json": json_text(rows),
        "visual_assets_manifest.csv": csv_text(manifest, MANIFEST_FIELDS),
    }
    summary = {
        "total_contexts": len(rows), "total_tids": expected_tids,
        "visual_types": {kind: sum(r["public_visual_type"] == kind for r in rows) for kind in VISUAL_LABELS},
        "public_titles": sum(not r["title_fallback"] for r in rows),
        "title_fallbacks": sum(r["title_fallback"] for r in rows),
        "informative_public_descriptions": sum(not r["description_fallback"] for r in rows),
        "text_fallbacks": sum(r["description_fallback"] for r in rows),
        "frozen_informative_descriptions": sum(r["description_informative"] == "True" for r in originals),
        "informative_cores_using_public_fallback": sum(r["description_informative"] == "True" and out["description_fallback"]
                                                      for r, out in zip(originals, rows)),
        "without_image": sum(r["public_visual_type"] == "none" for r in rows),
        "tiers_internal_only": tiers, "asset_references": len(manifest),
        "unique_source_assets": len({r["source_relative_path"] for r in manifest}),
    }
    audit = {
        "protocol": PROTOCOL, "source_protocol": POOL_PROTOCOL,
        "source_pool_csv_sha256": sha256(raw), "source_pool_config_sha256": sha256(config_raw),
        "implementation_sha256": sha256(Path(__file__).read_bytes()),
        "text_guard_sha256": sha256(Path(__file__).with_name("context_pool.py").read_bytes()),
        "outputs_sha256": {name: sha256(value.encode("utf-8")) for name, value in files.items()},
        "physical_order": "unchanged source CSV sequence, exactly serve_order",
        "id_policy": {"algorithm": "UUIDv5", "namespace": str(ID_NAMESPACE),
                      "name": 'compact JSON [canonical decimal tid string, integer song_id]',
                      "inputs_exclude": ["tier", "priority", "order", "system", "audio origin"]},
        "rules": {"trusted_title_confidences": sorted(TRUSTED_TITLE_CONFIDENCES),
                  "title_fallback": TITLE_FALLBACK, "description_fallback": DESCRIPTION_FALLBACK,
                  "revision": "public-segments-2",
                  "description": "frozen core; section-aware clause extraction; contextual lexical guard; no LLM or pool scoring",
                  "header_regex": HEADERS.pattern,
                  "meta_rules": META_RULES,
                  "author_or_markup_regex": AUTHOR_OR_MARKUP.pattern,
                  "visual": "unchanged pool selection; existence and portable path only; no pixel analysis",
                  "visual_labels": VISUAL_LABELS},
        "schema": {"integration_fields": list(FIELDS), "browser_allowlist": list(PUBLIC_FIELDS),
                   "internal_only_fields": list(INTERNAL_ONLY_FIELDS),
                   "source_relative_path": "INTERNAL dataset-relative source only; never a public locator",
                   "public_visual_source": "public_projection only; explicit published visuals/<external_id>.png or null",
                   "manifest": "INTERNAL; one entry per scenario with an image; repeated source assets allowed",
                   "forbidden_browser_data": ["tid", "song_id", "tier", "context_priority", "serve_order", "stars",
                                              "music evidence", "Lua evidence", "description scoring", "hashes", "matching", "audio origin"]},
        "roots": {"visual_source": "dataset (CLI --dataset-root); no physical root serialized"},
        "asset_publication": {"policy": ASSET_POLICY, "direct_copy_authorized": False,
                              "cartridge": "MUST render clean visual appearance without recoverable Lua/ROM/music; renaming/copying p8.png is forbidden",
                              "spritesheet": "MUST pass through asset publishing layer too"},
        "summary": summary,
        "checks": {"external_ids_unique": True, "exact_pool_identity_set": True,
                   "exact_pool_identity_sequence": True, "serve_order_exactly_preserved": True,
                   "all_visual_assets_exist": True, "all_source_paths_relative": True,
                   "tier_d_preserved": sum(r["context_tier"] == "D" for r in rows) == sum(r["context_tier"] == "D" for r in originals)},
        "limitations": ["Integration artifacts and manifest are not browser payloads.",
                        "UUIDs are opaque identifiers, not secrets or authorization tokens.",
                        "No title confidence in frozen v1 is approved for public use.",
                        "Text extraction is deterministic EN/PT heuristics, not semantic validation or translation.",
                        "Visuals identify source cartridge/assets only; no runtime scene or music-scene claim.",
                        "Assets have not been copied or published; no audio or backend import."],
    }
    files["scenario_context_config.json"] = json_text(audit)
    return files, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    pool_dir = REPO / "outputs" / "subjective" / POOL_PROTOCOL
    parser.add_argument("--pool-csv", type=Path, default=pool_dir / "context_pool.csv")
    parser.add_argument("--pool-config", type=Path, default=pool_dir / "context_pool_config.json")
    parser.add_argument("--dataset-root", type=Path, default=REPO.parent / "dataset_pico8_v2" / "dataset_pico8_v2")
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs" / "subjective" / PROTOCOL)
    args = parser.parse_args()
    files, audit = prepare_export(args.pool_csv, args.pool_config, args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (args.output_dir / name).write_text(content, encoding="utf-8", newline="")
    # Remove only obsolete names owned by this export, after successful writes.
    for obsolete in ("scenario_contexts.csv", "scenario_contexts.json"):
        (args.output_dir / obsolete).unlink(missing_ok=True)
    print(json_text(audit["summary"]), end="")


if __name__ == "__main__":
    main()
