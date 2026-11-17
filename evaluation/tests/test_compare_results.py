import csv
import tempfile
import unittest
from pathlib import Path

from compare_results import (
    compare_systems,
    describe,
    ks_statistic,
    percentile,
    wasserstein_1d,
)


METRICS = [
    "pitch_range",
    "pitch_class_entropy",
    "audible_step_rate",
    "audible_channel_utilization",
]


def write_fake_csv(path, rows):
    with Path(path).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["piece_id"] + METRICS,
        )

        writer.writeheader()

        for i, row in enumerate(rows):
            writer.writerow({
                "piece_id": f"p{i}",
                **row,
            })


class TestDistributionFunctions(unittest.TestCase):

    def test_percentile(self):
        values = [0, 10, 20, 30, 40]

        self.assertEqual(
            percentile(values, 0.25),
            10,
        )

        self.assertEqual(
            percentile(values, 0.75),
            30,
        )

    def test_describe(self):
        r = describe([1, 2, 3])

        self.assertEqual(r["n"], 3)
        self.assertEqual(r["mean"], 2)
        self.assertEqual(r["median"], 2)
        self.assertEqual(r["min"], 1)
        self.assertEqual(r["max"], 3)

    def test_ks_identical(self):
        self.assertEqual(
            ks_statistic(
                [1, 2, 3],
                [1, 2, 3],
            ),
            0.0,
        )

    def test_ks_separated(self):
        self.assertEqual(
            ks_statistic(
                [0, 0],
                [1, 1],
            ),
            1.0,
        )

    def test_wasserstein_identical(self):
        self.assertEqual(
            wasserstein_1d(
                [1, 2, 3],
                [1, 2, 3],
            ),
            0.0,
        )

    def test_wasserstein_shift(self):
        self.assertAlmostEqual(
            wasserstein_1d(
                [0, 1],
                [1, 2],
            ),
            1.0,
        )


class TestComparisonPipeline(unittest.TestCase):

    def test_compare_systems(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            real = tmp / "real.csv"
            baseline = tmp / "baseline.csv"
            neural = tmp / "neural.csv"

            real_rows = [
                {
                    "pitch_range": 40,
                    "pitch_class_entropy": 0.7,
                    "audible_step_rate": 0.5,
                    "audible_channel_utilization": 0.5,
                },
                {
                    "pitch_range": 50,
                    "pitch_class_entropy": 0.8,
                    "audible_step_rate": 0.6,
                    "audible_channel_utilization": 0.75,
                },
            ]

            baseline_rows = [
                {
                    "pitch_range": 10,
                    "pitch_class_entropy": 0.2,
                    "audible_step_rate": 0.2,
                    "audible_channel_utilization": 0.25,
                },
                {
                    "pitch_range": 20,
                    "pitch_class_entropy": 0.3,
                    "audible_step_rate": 0.3,
                    "audible_channel_utilization": 0.25,
                },
            ]

            neural_rows = [
                {
                    "pitch_range": 38,
                    "pitch_class_entropy": 0.68,
                    "audible_step_rate": 0.48,
                    "audible_channel_utilization": 0.5,
                },
                {
                    "pitch_range": 48,
                    "pitch_class_entropy": 0.78,
                    "audible_step_rate": 0.58,
                    "audible_channel_utilization": 0.7,
                },
            ]

            write_fake_csv(
                real,
                real_rows,
            )

            write_fake_csv(
                baseline,
                baseline_rows,
            )

            write_fake_csv(
                neural,
                neural_rows,
            )

            rows = compare_systems(
                {
                    "real": real,
                    "baseline": baseline,
                    "neural": neural,
                }
            )

            # 3 sistemas x 4 métricas.
            self.assertEqual(
                len(rows),
                12,
            )

            real_pitch = next(
                row
                for row in rows
                if row["system"] == "real"
                and row["metric"] == "pitch_range"
            )

            baseline_pitch = next(
                row
                for row in rows
                if row["system"] == "baseline"
                and row["metric"] == "pitch_range"
            )

            neural_pitch = next(
                row
                for row in rows
                if row["system"] == "neural"
                and row["metric"] == "pitch_range"
            )

            self.assertEqual(
                real_pitch["mean"],
                45,
            )

            self.assertEqual(
                real_pitch["ks_vs_real"],
                0.0,
            )

            self.assertGreater(
                baseline_pitch["wasserstein_vs_real"],
                neural_pitch["wasserstein_vs_real"],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
