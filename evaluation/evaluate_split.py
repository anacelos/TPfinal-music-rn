import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median

from pico8_metrics import evaluate_song_data


SCALAR_FIELDS = [
    "tid",
    "song_id",
    "pattern_count",
    "total_ticks",
    "duration_seconds",
    "pitch_min",
    "pitch_max",
    "pitch_range",
    "pitch_class_entropy",
    "audible_step_rate",
    "audible_channel_utilization",
    "active_channel_reference_utilization",
    "audible_note_steps",
    "audible_ticks",
    "active_refs",
    "structural_note_density",
    "block_repetition_ratio",
    "structural_active_slots",
    "structural_available_slots",
]

JSON_FIELDS = [
    "instruments",
    "effects",
    "speeds",
]

PRIMARY_METRICS = [
    "pitch_range",
    "pitch_class_entropy",
    "audible_step_rate",
    "audible_channel_utilization",
]

AUXILIARY_METRICS = [
    "structural_note_density",
    "block_repetition_ratio",
]


def load_jsonl(path):
    path = Path(path)

    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON inválido em {path}, linha {line_number}"
                ) from exc


def evaluate_split(input_path):
    return [
        evaluate_song_data(song)
        for song in load_jsonl(input_path)
    ]


def write_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = SCALAR_FIELDS + JSON_FIELDS

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for result in rows:
            row = {}

            for field in SCALAR_FIELDS:
                row[field] = result.get(field)

            for field in JSON_FIELDS:
                row[field] = json.dumps(
                    result.get(field),
                    ensure_ascii=False,
                    sort_keys=True,
                )

            writer.writerow(row)


def print_metric_summary(rows, metrics):
    for metric in metrics:
        values = [
            row[metric]
            for row in rows
            if row.get(metric) is not None
        ]

        missing = len(rows) - len(values)

        if not values:
            print(
                f"{metric:32s}"
                f" n=0"
                f" missing={missing}"
            )
            continue

        print(
            f"{metric:32s}"
            f" n={len(values):4d}"
            f" min={min(values):.6f}"
            f" median={median(values):.6f}"
            f" mean={mean(values):.6f}"
            f" max={max(values):.6f}"
            f" missing={missing}"
        )


def print_summary(rows):
    print(f"Músicas avaliadas: {len(rows)}")

    print()
    print("Métricas primárias:")
    print_metric_summary(rows, PRIMARY_METRICS)

    print()
    print("Métricas estruturais auxiliares:")
    print_metric_summary(rows, AUXILIARY_METRICS)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Avalia músicas no formato canônico songs.json/JSONL."
        )
    )

    parser.add_argument(
        "--input",
        default="data/splits/test.jsonl",
        help="JSONL com uma música canônica por linha.",
    )

    parser.add_argument(
        "--output",
        default="results/real_test_metrics.csv",
        help="CSV de saída.",
    )

    args = parser.parse_args()

    rows = evaluate_split(args.input)
    write_csv(rows, args.output)
    print_summary(rows)

    print()
    print(f"CSV salvo em: {args.output}")


if __name__ == "__main__":
    main()
