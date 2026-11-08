from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent
DATASET = REPO / "dataset_pico8"
INTERIM = REPO / "data" / "interim"
PROCESSED = REPO / "data" / "processed"
FIGURES = PACKAGE / "figures"
CONFIG = PACKAGE / "config.json"
