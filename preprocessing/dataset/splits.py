import random
from collections import Counter

from .config import Config

TRAIN = "train"


def assign_splits(group_sizes: Counter, config: Config) -> dict[str, str]:
    ratios = config["split_ratios"]
    total = sum(group_sizes.values())
    rng = random.Random(config["seed"])
    groups = sorted(group_sizes)
    rng.shuffle(groups)
    filled = {split: 0 for split in ratios}
    split_of_group = {}
    huge = config["force_train_group_frac"] * total
    for group in sorted(groups, key=lambda g: group_sizes[g] <= huge):
        if group_sizes[group] > huge:
            split = TRAIN
        else:
            split = max(ratios, key=lambda k: ratios[k] * total - filled[k])
        split_of_group[group] = split
        filled[split] += group_sizes[group]
    return split_of_group
