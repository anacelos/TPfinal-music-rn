import argparse
import csv
import json
from pathlib import Path

from pico8_metrics import evaluate_piece_file


SCALAR_FIELDS = [
    "piece_id",
    "tid",
    "pattern_count",
    "pitch_min",
    "pitch_max",
    "pitch_range",
    "pitch_class_entropy",
    "structural_note_density",
    "audible_channel_utilization",
    "enabled_channel_utilization",
    "active_note_slots",
    "available_note_slots",
    "enabled_refs",
    "audible_refs",
    "implicit_empty_refs",
    "explicit_empty_refs",
    "sfx_repetition_ratio",
]

JSON_FIELDS = [
    "pattern_ids",
    "missing_pattern_ids",
    "instruments",
    "effects",
    "speeds",
    "semantic_flags",
]


def load_manifest(path):
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)

    pieces = data.get("pieces")

    if not isinstance(pieces, list):
        raise ValueError(
            "O manifesto deve conter uma lista em 'pieces'."
        )

    return pieces


def evaluate_manifest(dataset_root, manifest_path):
    dataset_root = Path(dataset_root)

    rows = []

    for piece in load_manifest(manifest_path):
        piece_id = piece["piece_id"]
        tid = str(piece["tid"])
        pattern_ids = piece["patterns"]

        audio_json = (
            dataset_root
            / "audio"
            / tid
            / "audio.json"
        )

        if not audio_json.is_file():
            raise FileNotFoundError(
                f"{piece_id}: audio.json não encontrado: {audio_json}"
            )

        result = evaluate_piece_file(
            audio_json=audio_json,
            pattern_ids=pattern_ids,
            tid=tid,
        )

        result["piece_id"] = piece_id

        rows.append(result)

    return rows


def write_csv(rows, output_path):
    output_path = Path(output_path)

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


def print_summary(rows):
    print(f"Peças avaliadas: {len(rows)}")

    print()
    print(
        f"{'piece_id':<15}"
        f"{'patterns':>10}"
        f"{'range':>10}"
        f"{'entropy':>12}"
        f"{'density':>12}"
        f"{'channels':>12}"
        f"{'sfx_rep':>12}"
    )

    print("-" * 83)

    for r in rows:

        def fmt(value):
            if value is None:
                return "NA"
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value)

        print(
            f"{r['piece_id']:<15}"
            f"{r['pattern_count']:>10}"
            f"{fmt(r['pitch_range']):>10}"
            f"{fmt(r['pitch_class_entropy']):>12}"
            f"{fmt(r['structural_note_density']):>12}"
            f"{fmt(r['audible_channel_utilization']):>12}"
            f"{fmt(r['sfx_repetition_ratio']):>12}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Avaliação estrutural de peças PICO-8."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Raiz de dataset_pico8/",
    )

    parser.add_argument(
        "--manifest",
        required=True,
        help="JSON contendo as peças e pattern IDs.",
    )

    parser.add_argument(
        "--output",
        default="results.csv",
        help="CSV de saída.",
    )

    args = parser.parse_args()

    rows = evaluate_manifest(
        dataset_root=args.dataset,
        manifest_path=args.manifest,
    )

    write_csv(rows, args.output)
    print_summary(rows)

    print()
    print(f"CSV salvo em: {args.output}")


if __name__ == "__main__":
    main()
