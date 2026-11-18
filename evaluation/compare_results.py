import argparse
import csv
import math
import statistics
from pathlib import Path


DEFAULT_METRICS = [
    "pitch_entropy",
    "scale_consistency",
    "groove_consistency",
]


def read_metric_values(csv_path, metric):
    values = []

    with Path(csv_path).open(
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        if metric not in reader.fieldnames:
            raise ValueError(
                f"Métrica '{metric}' ausente em {csv_path}"
            )

        for row in reader:
            raw = row.get(metric, "")

            if raw in ("", "NA", "None", None):
                continue

            value = float(raw)

            if math.isfinite(value):
                values.append(value)

    return values


def percentile(values, q):
    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    pos = (len(values) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)

    if lower == upper:
        return values[lower]

    fraction = pos - lower

    return (
        values[lower] * (1 - fraction)
        + values[upper] * fraction
    )


def describe(values):
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "q1": None,
            "q3": None,
            "min": None,
            "max": None,
        }

    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": (
            statistics.stdev(values)
            if len(values) > 1
            else 0.0
        ),
        "q1": percentile(values, 0.25),
        "q3": percentile(values, 0.75),
        "min": min(values),
        "max": max(values),
    }


def ks_statistic(a, b):
    """
    Estatística de Kolmogorov-Smirnov de duas amostras:
    máximo |F_a(x) - F_b(x)|.

    Retorna valor entre 0 e 1.
    """

    if not a or not b:
        return None

    a = sorted(a)
    b = sorted(b)

    points = sorted(set(a + b))

    ia = 0
    ib = 0
    max_diff = 0.0

    for x in points:
        while ia < len(a) and a[ia] <= x:
            ia += 1

        while ib < len(b) and b[ib] <= x:
            ib += 1

        fa = ia / len(a)
        fb = ib / len(b)

        max_diff = max(
            max_diff,
            abs(fa - fb),
        )

    return max_diff


def wasserstein_1d(a, b):
    """
    Distância Wasserstein-1 exata entre duas distribuições
    empíricas unidimensionais.

    A unidade do resultado é a mesma da métrica avaliada.
    """

    if not a or not b:
        return None

    a = sorted(a)
    b = sorted(b)

    points = sorted(set(a + b))

    if len(points) <= 1:
        return 0.0

    ia = 0
    ib = 0
    distance = 0.0

    for left, right in zip(
        points[:-1],
        points[1:],
    ):
        while ia < len(a) and a[ia] <= left:
            ia += 1

        while ib < len(b) and b[ib] <= left:
            ib += 1

        fa = ia / len(a)
        fb = ib / len(b)

        distance += (
            abs(fa - fb)
            * (right - left)
        )

    return distance


def compare_systems(
    systems,
    metrics=None,
):
    metrics = metrics or DEFAULT_METRICS

    cache = {}

    for system_name, path in systems.items():
        cache[system_name] = {}

        for metric in metrics:
            cache[system_name][metric] = (
                read_metric_values(path, metric)
            )

    real = cache["real"]

    rows = []

    for system_name in systems:
        for metric in metrics:
            values = cache[system_name][metric]
            stats = describe(values)

            if system_name == "real":
                ks = 0.0
                wasserstein = 0.0
            else:
                ks = ks_statistic(
                    real[metric],
                    values,
                )

                wasserstein = wasserstein_1d(
                    real[metric],
                    values,
                )

            rows.append({
                "system": system_name,
                "metric": metric,
                **stats,
                "ks_vs_real": ks,
                "wasserstein_vs_real": wasserstein,
            })

    return rows


def write_summary_csv(rows, output_path):
    fieldnames = [
        "system",
        "metric",
        "n",
        "mean",
        "median",
        "std",
        "q1",
        "q3",
        "min",
        "max",
        "ks_vs_real",
        "wasserstein_vs_real",
    ]

    with Path(output_path).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    print(
        f"{'system':<10}"
        f"{'metric':<30}"
        f"{'n':>6}"
        f"{'mean':>12}"
        f"{'median':>12}"
        f"{'KS':>10}"
        f"{'W1':>12}"
    )

    print("-" * 92)

    def fmt(value):
        if value is None:
            return "NA"

        if isinstance(value, float):
            return f"{value:.4f}"

        return str(value)

    for row in rows:
        print(
            f"{row['system']:<10}"
            f"{row['metric']:<30}"
            f"{row['n']:>6}"
            f"{fmt(row['mean']):>12}"
            f"{fmt(row['median']):>12}"
            f"{fmt(row['ks_vs_real']):>10}"
            f"{fmt(row['wasserstein_vs_real']):>12}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compara distribuições das métricas entre "
            "músicas reais, baseline e modelo neural."
        )
    )

    parser.add_argument(
        "--real",
        required=True,
    )

    parser.add_argument(
        "--baseline",
        required=True,
    )

    parser.add_argument(
        "--neural",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="comparison_summary.csv",
    )

    args = parser.parse_args()

    systems = {
        "real": args.real,
        "baseline": args.baseline,
        "neural": args.neural,
    }

    rows = compare_systems(systems)

    write_summary_csv(
        rows,
        args.output,
    )

    print_summary(rows)

    print()
    print(
        f"Resumo salvo em: {args.output}"
    )


if __name__ == "__main__":
    main()
