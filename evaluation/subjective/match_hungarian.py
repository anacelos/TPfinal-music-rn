#!/usr/bin/env python3
"""Matching global 1:1, protocolo congelado symbolic-balanced-v1.

Exemplo na raiz do repositório:
    python evaluation/subjective/match_hungarian.py \
        --train data/splits/train.jsonl --real data/splits/test.jsonl \
        --generated results/markov_test_final.jsonl \
        --output-dir outputs/subjective/symbolic-balanced-v1/markov \
        --system-name markov

O CLI exige 8669 TRAIN, 1854 Real e 1854 Generated. Não há tuning de pesos,
seleção de subconjuntos nem custo de duração independente: a duração entra
somente no denominador de onset_rate. GC escalar e blocos não entram no custo.

Real: identidade lógica (str(tid), str(song_id)), não vazia e única.
Generated: sha256 do registro JSON canônico UTF-8 (sort_keys=True,
ensure_ascii=False, separators=(',', ':'), allow_nan=False), incluindo os
metadados que existirem. Não inventa song_id. Registros canônicos duplicados
são rejeitados: sem IDs adicionais, não há como distingui-los de forma estável
independentemente da ordem. O índice físico da linha (base zero) é proveniência.

As linhas/colunas do solver são ordenadas por identidade estável, inclusive
para desempates sem adicionar epsilon aos custos. A ordem física do Generated
não altera o assignment por identidade no mesmo ambiente/versão do solver.

Reutiliza evaluate_song_data e a função privada _extract_onset_features do
evaluator congelado. Essa dependência é deliberada para não duplicar/redefinir
onsets e groove; o hash do evaluator é registrado. Inputs sem descritores
finitos (por exemplo, silêncio sem pitches ou duração zero) causam erro; não
há descarte nem imputação fora da especificação aprovada.

Exige diretório de execução novo. Publica o diretório completo por rename no
mesmo filesystem, depois de validar o assignment. Um writer por destino.
"""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

EVALUATION_DIR = Path(__file__).resolve().parents[1]
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))
from pico8_metrics import _extract_onset_features, evaluate_song_data


PROTOCOL = "symbolic-balanced-v1"
GROOVE_BINS = 32
SCALAR_FEATURES = (
    "pitch_entropy", "pitch_range", "pitch_center", "onset_rate",
    "audible_channel_utilization",
)
TRANSFORMS = {name: "log1p" if name == "onset_rate" else "identity" for name in SCALAR_FEATURES}
EXPECTED_COUNTS = {"train": 8669, "real": 1854, "generated": 1854}
COST_FIELDS = (
    "total_cost", "pitch_cost", "rhythm_cost", "texture_cost",
    "pitch_entropy_cost", "pitch_range_cost", "pitch_center_cost",
    "onset_rate_cost", "groove_profile_cost", "audible_channel_utilization_cost",
)
CSV_FIELDS = (
    "real_index", "real_tid", "real_song_id", "real_record_sha256",
    "generated_index", "generated_id", "generated_tid", "generated_song_id",
    "generated_record_sha256", "system_name",
) + COST_FIELDS


@dataclass
class SongFeatures:
    index: int
    tid: str | None
    song_id: str | None
    record_sha256: str
    scalars: np.ndarray
    groove: np.ndarray

    @property
    def generated_id(self):
        return "sha256:" + self.record_sha256


@dataclass
class Dataset:
    path: Path
    sha256: str
    items: list[SongFeatures]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_value(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    text = str(value)
    return text if text.strip() else None


def extract_features(song, index=0):
    metrics = evaluate_song_data(song)
    _, vectors = _extract_onset_features(song.get("patterns", []), resolution=GROOVE_BINS)
    duration = metrics["duration_seconds"]
    if duration is None or not math.isfinite(duration) or duration <= 0:
        raise ValueError("duração deve ser positiva e finita")
    if not vectors:
        raise ValueError("nenhum pattern de duração positiva para o groove")
    low, high = metrics["pitch_min"], metrics["pitch_max"]
    if low is None or high is None:
        raise ValueError("pitch ausente: música sem onsets audíveis")
    scalars = np.array([
        metrics["pitch_entropy"], metrics["pitch_range"], (low + high) / 2.0,
        metrics["note_onsets"] / duration, metrics["audible_channel_utilization"],
    ], dtype=np.float64)
    groove = np.mean(np.asarray(vectors, dtype=np.float64), axis=0)
    if not np.all(np.isfinite(scalars)) or not np.all(np.isfinite(groove)):
        raise ValueError("features não finitas; não há imputação neste protocolo")
    canonical = json.dumps(
        song, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return SongFeatures(
        index, identity_value(song.get("tid")), identity_value(song.get("song_id")),
        hashlib.sha256(canonical).hexdigest(), scalars, groove,
    )


def load_dataset(path, role, expected_count):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    items = []
    identities = set()
    with path.open("rb") as handle:
        for index, raw in enumerate(handle):
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                song = json.loads(raw)
                if not isinstance(song, dict):
                    raise ValueError("cada linha deve conter uma música JSON")
                item = extract_features(song, index)
                if role == "real":
                    if item.tid is None or item.song_id is None:
                        raise ValueError("Real exige tid e song_id não vazios")
                    identity = (item.tid, item.song_id)
                else:
                    identity = item.generated_id
                if role in ("real", "generated") and identity in identities:
                    raise ValueError(f"identidade {role} duplicada: {identity}")
                identities.add(identity)
                items.append(item)
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                raise ValueError(f"{role} {path}, linha {index + 1}: {exc}") from exc
    if len(items) != expected_count:
        raise ValueError(f"{role}: esperadas {expected_count} músicas, encontradas {len(items)}")
    return Dataset(path, digest.hexdigest(), items)


def transformed_scalars(items):
    values = np.stack([item.scalars for item in items]).copy()
    values[:, SCALAR_FEATURES.index("onset_rate")] = np.log1p(values[:, SCALAR_FEATURES.index("onset_rate")])
    return values


def fit_scaler(train_items):
    """Somente TRAIN; quantis lineares e desvio populacional (ddof=0)."""
    values = transformed_scalars(train_items)
    result = {}
    for i, name in enumerate(SCALAR_FEATURES):
        column = values[:, i]
        q25, median, q75 = np.quantile(column, [0.25, 0.5, 0.75], method="linear")
        iqr = float(q75 - q25)
        std = float(np.std(column, ddof=0))
        scale = iqr if iqr > 0 else std if std > 0 else 1.0
        result[name] = {
            "transform": TRANSFORMS[name], "median_train": float(median),
            "q25_train": float(q25), "q75_train": float(q75), "iqr_train": iqr,
            "std_train": std, "scale": scale,
            "scale_source": "iqr" if iqr > 0 else "std" if std > 0 else "one",
            "constant": iqr == 0 and std == 0, "n_train": len(train_items),
        }
    return result


def normalize(items, scaler):
    values = transformed_scalars(items)
    medians = np.array([scaler[f]["median_train"] for f in SCALAR_FEATURES])
    scales = np.array([scaler[f]["scale"] for f in SCALAR_FEATURES])
    return (values - medians) / scales


def scalar_distance(z_real, z_generated):
    difference = np.abs(np.asarray(z_real) - np.asarray(z_generated))
    return difference / (1.0 + difference)


def cost_components(z_real, z_generated, groove_real, groove_generated):
    """Broadcast no último eixo: 5 escalares, 32 bins de groove."""
    distances = scalar_distance(z_real, z_generated)
    groove = np.mean(np.abs(np.asarray(groove_real) - np.asarray(groove_generated)), axis=-1)
    pitch = np.mean(distances[..., :3], axis=-1)
    rhythm = (distances[..., 3] + groove) / 2.0
    texture = distances[..., 4]
    return {
        "total_cost": (pitch + rhythm + texture) / 3.0,
        "pitch_cost": pitch, "rhythm_cost": rhythm, "texture_cost": texture,
        "pitch_entropy_cost": distances[..., 0], "pitch_range_cost": distances[..., 1],
        "pitch_center_cost": distances[..., 2], "onset_rate_cost": distances[..., 3],
        "groove_profile_cost": groove, "audible_channel_utilization_cost": texture,
    }


def build_cost_matrix(real, generated, scaler, block_size=64):
    z_real, z_generated = normalize(real, scaler), normalize(generated, scaler)
    groove_real = np.stack([item.groove for item in real])
    groove_generated = np.stack([item.groove for item in generated])
    matrix = np.empty((len(real), len(generated)), dtype=np.float64)
    for start in range(0, len(real), block_size):
        stop = min(len(real), start + block_size)
        components = cost_components(
            z_real[start:stop, None, :], z_generated[None, :, :],
            groove_real[start:stop, None, :], groove_generated[None, :, :],
        )
        matrix[start:stop] = components["total_cost"]
    return matrix


def solve_assignment(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("assignment completo exige matriz quadrada não vazia")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matriz de custos contém valores não finitos")
    rows, columns = linear_sum_assignment(matrix)
    n = matrix.shape[0]
    if len(rows) != n or len(set(rows)) != n or len(set(columns)) != n:
        raise RuntimeError("solver não produziu assignment completo 1:1")
    return rows, columns


def match_items(train, real, generated):
    if len(real) != len(generated):
        raise ValueError("Real e Generated devem ter a mesma contagem")
    scaler = fit_scaler(train)
    # Canonical order for deterministic ties; never perturb the frozen cost.
    real = sorted(real, key=lambda item: (item.tid, item.song_id))
    generated = sorted(generated, key=lambda item: item.generated_id)
    start = time.perf_counter()
    matrix = build_cost_matrix(real, generated, scaler)
    matrix_seconds = time.perf_counter() - start
    start = time.perf_counter()
    rows, columns = solve_assignment(matrix)
    solver_seconds = time.perf_counter() - start
    selected_real = [real[i] for i in rows]
    selected_generated = [generated[j] for j in columns]
    components = cost_components(
        normalize(selected_real, scaler), normalize(selected_generated, scaler),
        np.stack([item.groove for item in selected_real]),
        np.stack([item.groove for item in selected_generated]),
    )
    if not np.allclose(components["total_cost"], matrix[rows, columns], rtol=0, atol=1e-15):
        raise RuntimeError("decomposição dos pares diverge da matriz")
    matches = []
    for i, (r, g) in enumerate(zip(selected_real, selected_generated)):
        matches.append({
            "real_index": r.index, "real_tid": r.tid, "real_song_id": r.song_id,
            "real_record_sha256": r.record_sha256,
            "generated_index": g.index, "generated_id": g.generated_id,
            "generated_tid": g.tid, "generated_song_id": g.song_id,
            "generated_record_sha256": g.record_sha256,
            **{key: float(components[key][i]) for key in COST_FIELDS},
        })
    return sorted(matches, key=lambda row: row["real_index"]), scaler, {
        "cost_matrix_seconds": matrix_seconds, "assignment_seconds": solver_seconds,
        "cost_matrix_shape": list(matrix.shape), "cost_matrix_bytes": matrix.nbytes,
        "cost_matrix_dtype": str(matrix.dtype), "row_block_size": 64,
    }


def run_matching(train_path, real_path, generated_path, output_dir, system_name, *, expected_counts=None):
    counts = EXPECTED_COUNTS if expected_counts is None else expected_counts
    if set(counts) != set(EXPECTED_COUNTS) or any(not isinstance(n, int) or n <= 0 for n in counts.values()):
        raise ValueError("contagens esperadas devem ser inteiros positivos para train, real e generated")
    if counts["real"] != counts["generated"]:
        raise ValueError("contagens esperadas de Real e Generated devem ser iguais")
    if not system_name.strip():
        raise ValueError("system_name não pode ser vazio")
    output_dir = Path(output_dir).absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"{output_dir}: diretório de execução já existe; use um novo destino")
    start = time.perf_counter()
    datasets = {}
    load_seconds = {}
    for role, path in (("train", train_path), ("real", real_path), ("generated", generated_path)):
        phase = time.perf_counter()
        datasets[role] = load_dataset(path, role, counts[role])
        load_seconds[role] = time.perf_counter() - phase
    matches, scaler, performance = match_items(*(datasets[role].items for role in ("train", "real", "generated")))
    for row in matches:
        row["system_name"] = system_name
    costs = np.array([row["total_cost"] for row in matches])
    real_ids = {(row["real_tid"], row["real_song_id"]) for row in matches}
    generated_ids = {row["generated_id"] for row in matches}
    if len(matches) != counts["real"] or len(real_ids) != len(matches) or len(generated_ids) != len(matches):
        raise RuntimeError("assignment não cobre todas as identidades exatamente uma vez")
    summary = {
        "protocol": PROTOCOL, "system_name": system_name,
        "pairs": len(matches), "unique_real": len(real_ids), "unique_generated": len(generated_ids),
        "discarded_real": 0, "discarded_generated": 0,
        "total_cost": {
            "min": float(costs.min()), "mean": float(costs.mean()), "median": float(np.median(costs)),
            "p90": float(np.quantile(costs, 0.9, method="linear")),
            "p95": float(np.quantile(costs, 0.95, method="linear")), "max": float(costs.max()),
        },
        "performance": {
            **performance, "load_and_features_seconds": load_seconds,
            "elapsed_before_publication_seconds": time.perf_counter() - start,
            "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024),
        },
    }
    config = {
        "protocol": PROTOCOL, "system_name": system_name,
        "inputs": {role: {"path": str(data.path), "sha256": data.sha256, "count": len(data.items)} for role, data in datasets.items()},
        "expected_counts": counts,
        "family_weights": {"pitch": 1 / 3, "rhythm": 1 / 3, "texture": 1 / 3},
        "within_family_weights": {
            "pitch": {f: 1 / 3 for f in SCALAR_FEATURES[:3]},
            "rhythm": {"onset_rate": 1 / 2, "mean_groove_profile": 1 / 2},
            "texture": {"audible_channel_utilization": 1.0},
        },
        "features": {
            "scalars_in_order": list(SCALAR_FEATURES), "transforms": TRANSFORMS,
            "pitch_center": "(pitch_min + pitch_max) / 2",
            "onset_rate": "note_onsets / duration_seconds",
            "mean_groove_profile": {"bins": GROOVE_BINS, "aggregation": "unweighted mean over positive-duration patterns", "distance": "mean(abs(profile_real - profile_generated))", "extractor": "pico8_metrics._extract_onset_features"},
        },
        "scalar_distance": "u / (1 + u), u = abs(z_real - z_generated)",
        "normalization": {"fit_split": "train", "quantile_method": "linear", "std_ddof": 0, "fallback": "IQR > 0, else std > 0, else 1 (constant)", "parameters": scaler},
        "missing_policy": "error without discarding any song or imputing features",
        "identities": {
            "real": "unique (str(tid), str(song_id)); both nonempty",
            "generated": "sha256:<SHA-256 of canonical complete JSON record>; duplicate records rejected",
            "canonical_json": {"sort_keys": True, "ensure_ascii": False, "separators": [",", ":"], "allow_nan": False, "encoding": "utf-8"},
            "index": "zero-based physical JSONL line; provenance only",
        },
        "solver": {"function": "scipy.optimize.linear_sum_assignment", "matrix_dtype": "float64", "order": "real sorted by (tid, song_id); generated sorted by generated_id", "tie_policy": "canonical input order; no cost perturbation"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "implementation_sha256": sha256_file(__file__),
        "metrics_source_sha256": sha256_file(EVALUATION_DIR / "pico8_metrics.py"),
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temporary:
        staged = Path(temporary)
        with (staged / "matching.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(matches)
        for name, document in (("matching_config.json", config), ("matching_summary.json", summary)):
            with (staged / name).open("x", encoding="utf-8") as handle:
                json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"{output_dir}: destino apareceu durante a execução")
        os.replace(staged, output_dir)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--real", required=True, type=Path)
    parser.add_argument("--generated", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--system-name", required=True)
    args = parser.parse_args(argv)
    try:
        summary = run_matching(args.train, args.real, args.generated, args.output_dir, args.system_name)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Erro: {exc}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Diretório publicado: {args.output_dir}")


if __name__ == "__main__":
    main()
