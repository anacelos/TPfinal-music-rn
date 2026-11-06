import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = {
    "pitch_range": "Pitch range",
    "pitch_class_entropy": "Pitch-class entropy",
    "structural_note_density": "Structural note density",
    "audible_channel_utilization": "Audible channel utilization",
    "sfx_repetition_ratio": "SFX repetition ratio",
}


def read_values(path, metric):
    values = []

    with Path(path).open(
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        if metric not in reader.fieldnames:
            raise ValueError(
                f"Métrica '{metric}' ausente em {path}"
            )

        for row in reader:
            raw = row.get(metric)

            if raw in ("", None, "NA", "None"):
                continue

            values.append(float(raw))

    return values


def plot_metric(
    metric,
    systems,
    output_dir,
):
    data = []
    labels = []

    for label, path in systems.items():
        values = read_values(path, metric)

        if values:
            data.append(values)
            labels.append(label)

    if not data:
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.boxplot(
        data,
        tick_labels=labels,
        showmeans=True,
    )

    ax.set_ylabel(METRICS[metric])
    ax.set_title(
        f"Distribution of {METRICS[metric]}"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    output_path = (
        Path(output_dir)
        / f"{metric}.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Gera boxplots das métricas para "
            "real, baseline e modelo neural."
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
        "--output-dir",
        default="plots",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    systems = {
        "Real": args.real,
        "Baseline": args.baseline,
        "Neural": args.neural,
    }

    for metric in METRICS:
        plot_metric(
            metric,
            systems,
            output_dir,
        )

        print(
            f"Gerado: "
            f"{output_dir / (metric + '.png')}"
        )


if __name__ == "__main__":
    main()
