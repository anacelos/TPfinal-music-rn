#!/usr/bin/env python3
"""Audited rectangular Neural matching; the frozen Markov matcher is unchanged.

Only the approved silent record is ineligible for the subjective study. Every
other generated record must pass the frozen extractor. The objective input is
never rewritten. Output requires a new directory and contains no timestamps or
runtime measurements, so repeated runs can be compared byte for byte.
"""

import argparse
import csv
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import scipy

try:
    from . import match_hungarian as frozen
except ImportError:  # Direct execution from evaluation/subjective/.
    import match_hungarian as frozen


PROTOCOL = "symbolic-balanced-rectangular-v1"
SYSTEM_NAME = "lstm-grid_block-k10"
EXCLUSION_REASON = "absence_of_audible_onsets_and_undefined_pitch_features"
UNMATCHED_REASON = "unassigned_by_global_rectangular_hungarian"
EXPECTED_COUNTS = dict(frozen.EXPECTED_COUNTS)
EXCLUDED_FIELDS = (
    "generated_index", "generated_id", "generated_tid", "generated_song_id",
    "generated_record_sha256", "system_name", "reason", "note_onsets",
    "audible_note_steps", "pitch_entropy", "pitch_range", "pitch_center",
)
UNMATCHED_FIELDS = (
    "real_index", "real_record_id", "real_tid", "real_song_id",
    "real_record_sha256", "reason",
)


@dataclass(frozen=True)
class ExclusionSpec:
    song_id: int
    record_sha256: str


APPROVED_EXCLUSION = ExclusionSpec(
    1416, "5bcc9482c42f5d53bb0ee7a62d99ebb145073a0ba6f2c73716408c53a2964225",
)


@dataclass
class GeneratedDataset:
    path: Path
    sha256: str
    content_sha256: str
    compression: str
    original_count: int
    items: list[frozen.SongFeatures]
    excluded: list[dict]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_record_sha256(song):
    raw = json.dumps(song, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_generated(path, expected_count, exclusion=APPROVED_EXCLUSION):
    """Read the exact original input; reject every unapproved invalid record."""
    path = Path(path).resolve()
    source = path.read_bytes()
    compressed = path.suffix == ".gz"
    stream = gzip.GzipFile(fileobj=io.BytesIO(source)) if compressed else io.BytesIO(source)
    digest = hashlib.sha256()
    items, excluded, hashes, song_ids = [], [], set(), set()
    original_count = 0
    with stream as handle:
        for index, raw in enumerate(handle):
            digest.update(raw)
            if not raw.strip():
                continue
            original_count += 1
            try:
                song = json.loads(raw)
                require(isinstance(song, dict), "expected a JSON song object")
                require(song.get("system") == SYSTEM_NAME, "unexpected Neural system")
                sid = song.get("song_id")
                require(type(sid) is int, "generated song_id must be an integer")
                record_hash = canonical_record_sha256(song)
                require(record_hash not in hashes, "duplicate canonical generated record")
                require(sid not in song_ids, "duplicate generated song_id")
                hashes.add(record_hash)
                song_ids.add(sid)
                if sid == exclusion.song_id or record_hash == exclusion.record_sha256:
                    require(sid == exclusion.song_id and record_hash == exclusion.record_sha256,
                            "approved exclusion identity/hash mismatch")
                    metrics = frozen.evaluate_song_data(song)
                    require(metrics["note_onsets"] == 0 and metrics["audible_note_steps"] == 0
                            and metrics["audible_ticks"] == 0
                            and all(metrics[k] is None for k in (
                                "pitch_min", "pitch_max", "pitch_range",
                                "pitch_entropy", "scale_consistency")),
                            "approved exclusion must have no audible onsets and undefined pitch")
                    duration = metrics["duration_seconds"]
                    require(duration is not None and math.isfinite(duration) and duration > 0,
                            "approved exclusion has an additional duration problem")
                    excluded.append({
                        "generated_index": index, "generated_id": "sha256:" + record_hash,
                        "generated_tid": frozen.identity_value(song.get("tid")),
                        "generated_song_id": sid, "generated_record_sha256": record_hash,
                        "system_name": SYSTEM_NAME, "reason": EXCLUSION_REASON,
                        "note_onsets": 0, "audible_note_steps": 0,
                        "pitch_entropy": None, "pitch_range": None, "pitch_center": None,
                    })
                else:
                    item = frozen.extract_features(song, index)
                    require(item.record_sha256 == record_hash, "canonical hash differs from frozen extractor")
                    items.append(item)
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                raise ValueError(f"generated {path}, line {index + 1}: {exc}") from exc
    require(original_count == expected_count,
            f"generated: expected {expected_count} original records, found {original_count}")
    require(len(excluded) == 1, "exactly the one approved exclusion must be present")
    require(len(items) == original_count - 1, "unexpected eligible generated count")
    return GeneratedDataset(path, hashlib.sha256(source).hexdigest(), digest.hexdigest(),
                            "gzip" if compressed else "none", original_count, items, excluded)


def solve_rectangular_assignment(matrix):
    """Cover every column exactly once; rows may remain unmatched."""
    matrix = np.asarray(matrix, dtype=np.float64)
    require(matrix.ndim == 2 and 0 < matrix.shape[1] <= matrix.shape[0],
            "expected a nonempty matrix with at least as many Real rows as generated columns")
    require(np.all(np.isfinite(matrix)), "cost matrix contains nonfinite values")
    rows, columns = frozen.linear_sum_assignment(matrix)
    n = matrix.shape[1]
    require(len(rows) == n and len(set(rows)) == n and set(columns) == set(range(n)),
            "solver did not produce complete one-to-one generated coverage")
    return rows, columns


def match_items(train, real, generated):
    require(len(real) == len(generated) + 1 and len(generated) > 0,
            "this protocol requires exactly one more Real than eligible generated record")
    scaler = frozen.fit_scaler(train)
    real = sorted(real, key=lambda item: (item.tid, item.song_id))
    generated = sorted(generated, key=lambda item: item.generated_id)
    matrix = frozen.build_cost_matrix(real, generated, scaler)
    rows, columns = solve_rectangular_assignment(matrix)
    chosen_real = [real[i] for i in rows]
    chosen_generated = [generated[j] for j in columns]
    components = frozen.cost_components(
        frozen.normalize(chosen_real, scaler), frozen.normalize(chosen_generated, scaler),
        np.stack([item.groove for item in chosen_real]),
        np.stack([item.groove for item in chosen_generated]),
    )
    require(all(np.all(np.isfinite(components[k])) for k in frozen.COST_FIELDS),
            "selected cost components are nonfinite")
    require(np.allclose(components["total_cost"], matrix[rows, columns], rtol=0, atol=1e-15),
            "selected cost decomposition differs from the frozen matrix")
    matches = [{
        "real_index": r.index, "real_tid": r.tid, "real_song_id": r.song_id,
        "real_record_sha256": r.record_sha256,
        "generated_index": g.index, "generated_id": g.generated_id,
        "generated_tid": g.tid, "generated_song_id": g.song_id,
        "generated_record_sha256": g.record_sha256, "system_name": SYSTEM_NAME,
        **{key: float(components[key][i]) for key in frozen.COST_FIELDS},
    } for i, (r, g) in enumerate(zip(chosen_real, chosen_generated))]
    selected = set(rows)
    unmatched = [{
        "real_index": item.index, "real_record_id": f"real:{item.tid}:{item.song_id}",
        "real_tid": item.tid, "real_song_id": item.song_id,
        "real_record_sha256": item.record_sha256, "reason": UNMATCHED_REASON,
    } for i, item in enumerate(real) if i not in selected]
    return (sorted(matches, key=lambda row: row["real_index"]),
            sorted(unmatched, key=lambda row: row["real_index"]), scaler,
            {"shape": list(matrix.shape), "dtype": str(matrix.dtype),
             "bytes": matrix.nbytes, "row_block_size": 64})


def csv_bytes(fields, rows):
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return text.getvalue().encode("utf-8")


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def run_matching(train_path, real_path, generated_path, output_dir, *,
                 expected_counts=None, exclusion=APPROVED_EXCLUSION):
    """Overrides are for explicit small fixtures; the CLI fixes the official policy."""
    counts = dict(EXPECTED_COUNTS if expected_counts is None else expected_counts)
    require(set(counts) == set(EXPECTED_COUNTS)
            and all(type(n) is int and n > 0 for n in counts.values()), "invalid expected counts")
    require(counts["real"] == counts["generated"] and counts["generated"] > 1,
            "Real and original generated counts must be equal and greater than one")
    require(type(exclusion.song_id) is int and len(exclusion.record_sha256) == 64
            and all(c in "0123456789abcdef" for c in exclusion.record_sha256), "invalid exclusion spec")
    output_dir = Path(output_dir).absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"{output_dir}: output already exists; use a new directory")
    train = frozen.load_dataset(train_path, "train", counts["train"])
    real = frozen.load_dataset(real_path, "real", counts["real"])
    generated = load_generated(generated_path, counts["generated"], exclusion)
    matches, unmatched, scaler, matrix = match_items(train.items, real.items, generated.items)
    real_ids = {(row["real_tid"], row["real_song_id"]) for row in matches}
    generated_ids = {row["generated_id"] for row in matches}
    unmatched_ids = {(row["real_tid"], row["real_song_id"]) for row in unmatched}
    require(len(matches) == len(real_ids) == len(generated_ids) == len(generated.items),
            "pairs are incomplete or reuse an identity")
    require(generated_ids == {item.generated_id for item in generated.items},
            "eligible generated coverage differs")
    require(len(unmatched) == 1 and not (real_ids & unmatched_ids)
            and real_ids | unmatched_ids == {(item.tid, item.song_id) for item in real.items},
            "Real coverage does not partition into matched and one unmatched")
    costs = np.asarray([row["total_cost"] for row in matches])
    actual_counts = {
        "train": len(train.items), "real": len(real.items),
        "generated_original": generated.original_count, "generated_eligible": len(generated.items),
        "excluded_generated": len(generated.excluded), "unmatched_real": len(unmatched),
    }
    summary = {
        "protocol": PROTOCOL, "system_name": SYSTEM_NAME, "counts": actual_counts,
        "pairs": len(matches), "unique_real": len(real_ids), "unique_generated": len(generated_ids),
        "reused_real": 0, "reused_generated": 0, "eligible_generated_unmatched": 0,
        "total_cost": {
            "min": float(costs.min()), "mean": float(costs.mean()), "median": float(np.median(costs)),
            "p90": float(np.quantile(costs, 0.9, method="linear")),
            "p95": float(np.quantile(costs, 0.95, method="linear")), "max": float(costs.max()),
        },
        "cost_matrix": matrix,
    }
    config = {
        "protocol": PROTOCOL, "system_name": SYSTEM_NAME, "based_on_protocol": frozen.PROTOCOL,
        "inputs": {
            "train": {"path": str(train.path), "sha256": train.sha256, "count": len(train.items)},
            "real": {"path": str(real.path), "sha256": real.sha256, "count": len(real.items)},
            "generated": {"path": str(generated.path), "sha256": generated.sha256,
                          "content_sha256": generated.content_sha256, "compression": generated.compression,
                          "count": generated.original_count, "eligible_count": len(generated.items)},
        },
        "expected_counts": counts, "counts": actual_counts,
        "eligibility": {"scope": "subjective matching only; objective input and evaluation unchanged",
                        "approved_exclusion": {"song_id": exclusion.song_id,
                                               "record_sha256": exclusion.record_sha256,
                                               "reason": EXCLUSION_REASON},
                        "other_invalid_records": "error", "imputation": False,
                        "regeneration": False, "cost_based_pair_filtering": False},
        "family_weights": {"pitch": 1 / 3, "rhythm": 1 / 3, "texture": 1 / 3},
        "within_family_weights": {
            "pitch": {f: 1 / 3 for f in frozen.SCALAR_FEATURES[:3]},
            "rhythm": {"onset_rate": 1 / 2, "mean_groove_profile": 1 / 2},
            "texture": {"audible_channel_utilization": 1.0},
        },
        "features": {
            "scalars_in_order": list(frozen.SCALAR_FEATURES), "transforms": frozen.TRANSFORMS,
            "pitch_center": "(pitch_min + pitch_max) / 2",
            "onset_rate": "note_onsets / duration_seconds",
            "mean_groove_profile": {"bins": frozen.GROOVE_BINS,
                                    "aggregation": "unweighted mean over positive-duration patterns",
                                    "distance": "mean(abs(profile_real - profile_generated))",
                                    "extractor": "pico8_metrics._extract_onset_features"},
        },
        "scalar_distance": "u / (1 + u), u = abs(z_real - z_generated)",
        "normalization": {"fit_split": "train", "quantile_method": "linear", "std_ddof": 0,
                          "fallback": "IQR > 0, else std > 0, else 1 (constant)", "parameters": scaler},
        "identities": {"real": "unique (str(tid), str(song_id))",
                       "generated": "sha256 of canonical complete JSON record, including metadata",
                       "canonical_json": {"sort_keys": True, "ensure_ascii": False,
                                          "separators": [",", ":"], "allow_nan": False, "encoding": "utf-8"},
                       "index": "zero-based physical JSONL line, including blanks; provenance only"},
        "solver": {"function": "scipy.optimize.linear_sum_assignment", "matrix_dtype": "float64",
                   "order": "real sorted by (tid, song_id); generated sorted by generated_id",
                   "tie_policy": "canonical input order; no cost perturbation",
                   "coverage": "every eligible generated exactly once; one Real unmatched by global optimum"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "implementation_sha256": frozen.sha256_file(__file__),
        "frozen_implementation_sha256": frozen.sha256_file(frozen.__file__),
        "metrics_source_sha256": frozen.sha256_file(frozen.EVALUATION_DIR / "pico8_metrics.py"),
        "reused_frozen_functions": ["extract_features", "load_dataset", "fit_scaler", "normalize",
                                    "scalar_distance", "cost_components", "build_cost_matrix"],
        "determinism": "same input bytes, code and versions; no timestamps, timings or output path",
    }
    files = {
        "matching.csv": csv_bytes(frozen.CSV_FIELDS, matches),
        "excluded_generated.csv": csv_bytes(EXCLUDED_FIELDS, generated.excluded),
        "unmatched_real.csv": csv_bytes(UNMATCHED_FIELDS, unmatched),
        "matching_summary.json": json_bytes(summary),
    }
    config["outputs_sha256"] = {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
    files["matching_config.json"] = json_bytes(config)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temporary:
        staged = Path(temporary)
        for name, raw in files.items():
            with (staged / name).open("xb") as handle:
                handle.write(raw)
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"{output_dir}: output appeared during execution")
        os.replace(staged, output_dir)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = run_matching(args.train, args.real, args.generated, args.output_dir)
    except (ValueError, OSError, EOFError) as exc:
        parser.exit(1, f"Erro: {exc}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
