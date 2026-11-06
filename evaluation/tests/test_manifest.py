import csv
import json
import tempfile
import unittest
from pathlib import Path

from evaluate_manifest import evaluate_manifest, write_csv


def empty_note():
    return {
        "pitch": 0,
        "instrument": 0,
        "volume": 0,
        "effect": 0,
    }


class TestManifestPipeline(unittest.TestCase):

    def test_manifest_to_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            dataset = root / "dataset_pico8"
            audio_dir = dataset / "audio" / "123"
            audio_dir.mkdir(parents=True)

            notes = [empty_note() for _ in range(32)]

            notes[0] = {
                "pitch": 60,
                "instrument": 1,
                "volume": 7,
                "effect": 0,
            }

            notes[1] = {
                "pitch": 64,
                "instrument": 2,
                "volume": 6,
                "effect": 3,
            }

            audio = {
                "sfx": [
                    {
                        "id": 0,
                        "speed": 16,
                        "loop_start": 0,
                        "loop_end": 0,
                        "notes": notes,
                    }
                ],
                "music": [
                    {
                        "id": 0,
                        "flags": 0,
                        "ch1": 0,
                        "ch2": 66,
                        "ch3": 67,
                        "ch4": 68,
                    }
                ],
            }

            with (audio_dir / "audio.json").open(
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(audio, f)

            manifest = {
                "pieces": [
                    {
                        "piece_id": "piece_test",
                        "tid": "123",
                        "patterns": [0],
                    }
                ]
            }

            manifest_path = root / "manifest.json"

            with manifest_path.open(
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(manifest, f)

            rows = evaluate_manifest(
                dataset_root=dataset,
                manifest_path=manifest_path,
            )

            self.assertEqual(len(rows), 1)

            r = rows[0]

            self.assertEqual(r["piece_id"], "piece_test")
            self.assertEqual(r["tid"], "123")
            self.assertEqual(r["pattern_count"], 1)

            self.assertEqual(r["pitch_min"], 60)
            self.assertEqual(r["pitch_max"], 64)
            self.assertEqual(r["pitch_range"], 4)

            self.assertEqual(r["active_note_slots"], 2)
            self.assertEqual(r["available_note_slots"], 32)

            self.assertAlmostEqual(
                r["structural_note_density"],
                2 / 32,
            )

            self.assertAlmostEqual(
                r["audible_channel_utilization"],
                1 / 4,
            )

            output_csv = root / "results.csv"

            write_csv(rows, output_csv)

            self.assertTrue(output_csv.is_file())

            with output_csv.open(
                newline="",
                encoding="utf-8",
            ) as f:
                csv_rows = list(csv.DictReader(f))

            self.assertEqual(len(csv_rows), 1)

            row = csv_rows[0]

            self.assertEqual(
                row["piece_id"],
                "piece_test",
            )

            self.assertEqual(
                row["tid"],
                "123",
            )

            self.assertEqual(
                int(row["pitch_range"]),
                4,
            )

            instruments = json.loads(
                row["instruments"]
            )

            self.assertEqual(
                instruments,
                {
                    "1": 1,
                    "2": 1,
                },
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
