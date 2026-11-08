import json

from .config import Config
from .records import count_audible_notes


class DisjointSets:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[max(root_a, root_b)] = min(root_a, root_b)


def _union_by_key(sets: DisjointSets, first_tid: dict, key, tid: str) -> None:
    if key in first_tid:
        sets.union(first_tid[key], tid)
    else:
        first_tid[key] = tid


def group_tids(songs: list[dict], records: list[dict], config: Config) -> dict[str, str]:
    sets = DisjointSets()
    first_tid_by_song: dict = {}
    for song in songs:
        sets.find(song["tid"])
        _union_by_key(sets, first_tid_by_song, song["content_hash"], song["tid"])
    if config["group_by_shared_sfx"]:
        min_audible = config["shared_sfx_min_notes"]
        first_tid_by_sfx: dict = {}
        for record in records:
            for sfx in record["sfx"].values():
                if count_audible_notes(sfx["notes"]) >= min_audible:
                    _union_by_key(sets, first_tid_by_sfx, json.dumps(sfx["notes"]), record["tid"])
    return {tid: sets.find(tid) for tid in sets.parent}
