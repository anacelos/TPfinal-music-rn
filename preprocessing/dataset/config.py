import json
from pathlib import Path
from typing import Any

Config = dict[str, Any]

REQUIRED_KEYS = frozenset({
    "drop_all_rest", "drop_incompatible_speeds", "drop_loop_external", "drop_no_loop",
    "min_patterns", "min_notes", "max_steps", "dedupe", "group_by_shared_sfx",
    "shared_sfx_min_notes", "split_ratios", "force_train_group_frac", "seed",
})


def load_config(path: Path) -> Config:
    with open(path) as f:
        config = json.load(f)
    missing = REQUIRED_KEYS - config.keys()
    unknown = config.keys() - REQUIRED_KEYS
    if missing or unknown:
        raise ValueError(f"{path}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
    return config
