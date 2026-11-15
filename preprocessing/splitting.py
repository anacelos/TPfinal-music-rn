import random
from collections import Counter


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def group_games(kept_tids: list[str], block_keys: list[set[str]]) -> dict[str, int]:
    uf = UnionFind()
    owner: dict[str, str] = {}
    for tid, keys in zip(kept_tids, block_keys):
        uf.find(tid)
        for key in sorted(keys):
            if key in owner:
                uf.union(owner[key], tid)
            else:
                owner[key] = tid
    members: dict[str, list[int]] = {}
    for tid in uf.parent:
        members.setdefault(uf.find(tid), []).append(int(tid))
    return {str(tid): min(group) for group in members.values() for tid in group}


def assign_splits(group_sizes: Counter, ratios: dict[str, float], force_train_frac: float,
                  seed: int) -> dict[int, str]:
    names = list(ratios)
    total = sum(group_sizes.values())
    order = sorted(group_sizes)
    random.Random(seed).shuffle(order)

    filled = dict.fromkeys(names, 0)
    assignment: dict[int, str] = {}
    for group in order:
        if group_sizes[group] > force_train_frac * total:
            assignment[group] = "train"
            filled["train"] += group_sizes[group]
    rank = {name: i for i, name in enumerate(names)}
    for group in order:
        if group in assignment:
            continue
        name = max(names, key=lambda n: (ratios[n] * total - filled[n], -rank[n]))
        assignment[group] = name
        filled[name] += group_sizes[group]
    return assignment
