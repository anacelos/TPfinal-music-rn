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
    "pitch_entropy",
    "scale_consistency",
    "groove_consistency",
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
                    "pitch_entropy": 2.8,
                    "scale_consistency": 0.90,
                    "groove_consistency": 0.85,
                },
                {
                    "pitch_entropy": 3.0,
                    "scale_consistency": 0.86,
                    "groove_consistency": 0.80,
                },
            ]

            baseline_rows = [
                {
                    "pitch_entropy": 1.5,
                    "scale_consistency": 0.65,
                    "groove_consistency": 0.55,
                },
                {
                    "pitch_entropy": 1.7,
                    "scale_consistency": 0.68,
                    "groove_consistency": 0.60,
                },
            ]

            neural_rows = [
                {
                    "pitch_entropy": 2.7,
                    "scale_consistency": 0.88,
                    "groove_consistency": 0.82,
                },
                {
                    "pitch_entropy": 2.9,
                    "scale_consistency": 0.84,
                    "groove_consistency": 0.78,
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

            # 3 sistemas x 3 métricas.
            self.assertEqual(
                len(rows),
                9,
            )

            real_entropy = next(
                row
                for row in rows
                if row["system"] == "real"
                and row["metric"] == "pitch_entropy"
            )

            baseline_entropy = next(
                row
                for row in rows
                if row["system"] == "baseline"
                and row["metric"] == "pitch_entropy"
            )

            neural_entropy = next(
                row
                for row in rows
                if row["system"] == "neural"
                and row["metric"] == "pitch_entropy"
            )

            self.assertAlmostEqual(
                real_entropy["mean"],
                2.9,
            )

            self.assertEqual(
                real_entropy["ks_vs_real"],
                0.0,
            )

            self.assertGreater(
                baseline_entropy["wasserstein_vs_real"],
                neural_entropy["wasserstein_vs_real"],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
