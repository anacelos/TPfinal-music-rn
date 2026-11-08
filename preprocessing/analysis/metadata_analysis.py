import hashlib
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..paths import DATASET, FIGURES, INTERIM
from ..pico8.audio_json import pattern_from_dict

OUT_CSV = INTERIM / "tid_metadata.csv"
BAR_COLOR = "#3b6ea5"
HASH_HEX_CHARS = 16
TOP_TAGS_FOR_COOCCURRENCE = 40
TOP_TAGS_PLOTTED = 30
MIN_COOCCURRENCE_FOR_LIFT = 30
STARS_CLIP_QUANTILE = 0.99
TAG_KEYWORDS = {
    "music": (r"music|song|tracker|sfx|chiptune|soundtrack|synth|jukebox|beat|composer|instrument"
              r"|midi|audio|sound"),
    "jam": r"jam|ludum|ld\d+|gamejam|compo|contest|entry",
    "demo_wip": r"demo|wip|prototype|unfinished|alpha|beta|test",
    "tweet": r"tweet|140|256|1k|2k|minimal|tiny",
}

MUSIC_CALL = re.compile(r"(?<![\w.:])music\s*\(\s*(-?\d+)?")
LUA_LINE_COMMENT = re.compile(r"--(?!\[\[).*")


def tids_in(subdir: str) -> set[int]:
    return {int(p.name) for p in (DATASET / subdir).iterdir() if p.is_dir() and p.name.isdigit()}


def short_hash(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(payload).hexdigest()[:HASH_HEX_CHARS]


def count_active_patterns(music: list[dict]) -> int:
    return sum(not pattern_from_dict(m).is_empty for m in music)


def p8_sections(path: Path) -> dict[str, list[str]]:
    sections = {}
    current = None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("__") and line.endswith("__") and len(line) > 4:
            current = line.strip("_")
            sections[current] = []
        elif current:
            sections[current].append(line)
    return sections


def classify_cart_without_audio(tid: int) -> str:
    sections = p8_sections(DATASET / "p8" / str(tid) / f"{tid}.p8")
    sfx, music = sections.get("sfx"), sections.get("music")
    if sfx is None and music is None:
        return "no __sfx__/__music__ section"
    sfx_non_empty = any(line.strip() and set(line.strip()) - {"0"} for line in (sfx or []))
    music_non_empty = any(line.strip() for line in (music or []))
    if not sfx_non_empty and not music_non_empty:
        return "sections present but empty or all zeros"
    return "non-empty sections but no audio.json"


def lua_music_fields(tid: int) -> dict:
    path = DATASET / "lua" / str(tid) / f"{tid}.lua"
    source = path.read_text(errors="replace") if path.exists() else ""
    calls = MUSIC_CALL.findall(LUA_LINE_COMMENT.sub("", source))
    entry_points = sorted({int(c) for c in calls if c != ""})
    return {"n_music_calls": len(calls), "music_entry_points": "|".join(map(str, entry_points))}


def build_row(tid: int, audio: set[int], p8: set[int], lua: set[int],
              no_audio_reasons: Counter, description_lengths: dict) -> dict:
    row = dict(tid=tid, has_audio=tid in audio, has_p8=tid in p8, has_lua=tid in lua,
               stars=None, license=None, n_tags=0, tags="", audio_hash="", music_hash="",
               n_music_calls=0, music_entry_points="")
    info_path = DATASET / "metadata" / str(tid) / "game_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        tags = info.get("tags") or []
        row["stars"] = info.get("total_stars")
        row["license"] = info.get("license")
        row["n_tags"] = len(tags)
        row["tags"] = "|".join(tags)
        description_lengths[tid] = len(info.get("description") or "")
    n_active, n_sfx = 0, 0
    if tid in audio:
        data = json.loads((DATASET / "audio" / str(tid) / "audio.json").read_text())
        n_active = count_active_patterns(data.get("music") or [])
        n_sfx = len(data.get("sfx") or [])
        row["audio_hash"] = short_hash({"sfx": data.get("sfx"), "music": data.get("music")})
        row["music_hash"] = short_hash(data.get("music")) if n_active else ""
    row["_n_music_active"] = n_active
    row["_n_sfx"] = n_sfx
    if tid in p8 and tid not in audio:
        no_audio_reasons[classify_cart_without_audio(tid)] += 1
    if tid in lua:
        row.update(lua_music_fields(tid))
    return row


def print_coverage(sets: dict[str, set], universe: set, ouro: dict) -> None:
    print("== coverage ==")
    for name, tids in sets.items():
        print(name, len(tids))
    print("universe", len(universe))
    audio, p8, meta, lua = sets["audio"], sets["p8"], sets["metadata"], sets["lua"]
    print("audio&p8", len(audio & p8), "audio-p8", len(audio - p8), "p8-audio", len(p8 - audio))
    print("meta-p8", len(meta - p8), "p8-meta", len(p8 - meta), "ouro-p8", len(set(ouro) - p8),
          "p8-ouro", len(p8 - set(ouro)), "lua^p8", len(lua ^ p8), "meta-ouro", len(meta - set(ouro)))


def print_audio_availability(df: pd.DataFrame, no_audio_reasons: Counter) -> None:
    print("\n== no-audio reasons (p8 present, no audio.json) ==")
    print(dict(no_audio_reasons))
    print("audio with 0 sfx and 0 active music:",
          int(((df.has_audio) & (df._n_sfx == 0) & (df._n_music_active == 0)).sum()))
    print("audio with active music:", int((df._n_music_active > 0).sum()),
          "audio with sfx but no active music:",
          int(((df.has_audio) & (df._n_sfx > 0) & (df._n_music_active == 0)).sum()))


def tag_counts(with_metadata: pd.DataFrame) -> Counter:
    return Counter(t for s in with_metadata.tags for t in s.split("|") if t)


def print_metadata_stats(md: pd.DataFrame, description_lengths: dict, tc: Counter) -> None:
    print("\n== metadata ==")
    print("stars describe", md.stars.describe(percentiles=[.5, .75, .9, .99]).round(1).to_dict())
    print("stars==0", int((md.stars == 0).sum()))
    print("license", md.license.fillna("NONE").value_counts().to_dict())
    lengths = pd.Series(description_lengths)
    print("desc len describe", lengths.describe(percentiles=[.5, .75, .9]).round(1).to_dict(),
          "empty", int((lengths == 0).sum()))
    print("with tags", int((md.n_tags > 0).sum()), "of", len(md), "; n_tags dist",
          md.n_tags.value_counts().sort_index().to_dict())
    print("distinct tags", len(tc))
    print("top60", tc.most_common(60))
    top = [t for t, _ in tc.most_common(TOP_TAGS_FOR_COOCCURRENCE)]
    cooccurrence = Counter()
    for s in md.tags:
        cooccurrence.update(combinations(sorted(set(t for t in s.split("|") if t in top)), 2))
    print("top cooc", [(a, b, n) for (a, b), n in cooccurrence.most_common(30)])
    n_tagged = (md.n_tags > 0).sum()
    lift = [(round(c * n_tagged / (tc[a] * tc[b]), 2), a, b, c)
            for (a, b), c in cooccurrence.items() if c >= MIN_COOCCURRENCE_FOR_LIFT]
    print("high lift", sorted(lift, reverse=True)[:20])
    for name, pattern in TAG_KEYWORDS.items():
        rx = re.compile(pattern, re.I)
        hits = {t: c for t, c in tc.items() if rx.search(t)}
        carts = md[md.tags.apply(lambda s: any(rx.search(t) for t in s.split("|") if t))]
        print(f"kw[{name}] tags", sorted(hits.items(), key=lambda x: -x[1])[:25], "carts", len(carts))


def print_duplicates(df: pd.DataFrame) -> None:
    print("\n== duplicates ==")
    for col in ("audio_hash", "music_hash"):
        sub = df[df[col] != ""]
        groups = sub.groupby(col).tid.apply(list)
        dup = groups[groups.apply(len) > 1]
        print(col, "tids", len(sub), "unique", sub[col].nunique(), "dup groups", len(dup),
              "tids in dup groups", int(dup.apply(len).sum()), "largest",
              sorted(dup.apply(len), reverse=True)[:10])
        for digest, ids in dup.sort_values(key=lambda s: s.apply(len), ascending=False).head(6).items():
            members = df[df.tid.isin(ids)]
            print("  group", digest, len(ids), "ids", ids[:8], "tid span", min(ids), max(ids),
                  "n_sfx", int(df.loc[df.tid == ids[0], "_n_sfx"].iloc[0]),
                  "active music", int(df.loc[df.tid == ids[0], "_n_music_active"].iloc[0]),
                  "tags", tag_counts(members).most_common(3))
        if col == "music_hash":
            sizes = dup.apply(len)
            print("  groups size==2:", int((sizes == 2).sum()), "size 3-5:",
                  int(((sizes >= 3) & (sizes <= 5)).sum()), "size>5:", int((sizes > 5).sum()))
            spans = [max(ids) - min(ids) for ids in dup]
            print("  groups with tid span <=50:", sum(1 for x in spans if x <= 50), "of", len(spans))
    non_empty = df[(df.audio_hash != "") & ((df._n_sfx > 0) | (df._n_music_active > 0))]
    groups = non_empty.groupby("audio_hash").tid.apply(list)
    dup = groups[groups.apply(len) > 1]
    print("non-empty audio: tids", len(non_empty), "unique", non_empty.audio_hash.nunique(),
          "dup groups", len(dup), "tids in dup", int(dup.apply(len).sum()))
    empty = df[(df.audio_hash != "") & (df._n_sfx == 0) & (df._n_music_active == 0)]
    print("empty-audio tids", len(empty), "distinct hashes", empty.audio_hash.nunique())


def print_lua_music(df: pd.DataFrame) -> int:
    """Prints the Lua music() usage report and returns how many Lua carts call music()."""
    print("\n== lua music ==")
    lua = df[df.has_lua]
    calling = lua[lua.n_music_calls > 0]
    print("lua tids", len(lua), "call music()", len(calling), "total calls", int(lua.n_music_calls.sum()))
    entries = Counter(e for s in lua.music_entry_points for e in s.split("|") if e)
    print("distinct entries", len(entries), "top", entries.most_common(15))
    print("tids with any literal entry", int((lua.music_entry_points != "").sum()),
          "calls without literal", int(((lua.n_music_calls > 0) & (lua.music_entry_points == "")).sum()))
    print("tids calling music(-1)", sum(1 for s in lua.music_entry_points if "-1" in s.split("|")))
    active = df[df._n_music_active > 0]
    print("tids with active music patterns", len(active), "of which have lua", int(active.has_lua.sum()),
          "call music()", int((active.n_music_calls > 0).sum()),
          "never call music()", int(((active.n_music_calls == 0) & active.has_lua).sum()))
    print("calls music() but no active patterns",
          int(((df.n_music_calls > 0) & (df._n_music_active == 0)).sum()))
    print("entry==0 share", round(entries.get("0", 0) / max(1, sum(entries.values())), 3))
    return len(calling)


def save_figure(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name)
    plt.close(fig)


def plot_stars_histogram(md: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(md.stars.clip(upper=md.stars.quantile(STARS_CLIP_QUANTILE)), bins=50, color=BAR_COLOR)
    ax.set_yscale("log")
    ax.set_xlabel("total_stars (clipped at 99th percentile)")
    ax.set_ylabel("number of TIDs (log)")
    ax.set_title("Distribution of BBS stars per cart")
    save_figure(fig, "meta_stars_hist.png")


def plot_top_tags(tc: Counter) -> None:
    top = tc.most_common(TOP_TAGS_PLOTTED)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh([t for t, _ in top][::-1], [c for _, c in top][::-1], color=BAR_COLOR)
    ax.set_xlabel("number of TIDs")
    ax.set_title(f"Top {TOP_TAGS_PLOTTED} tags")
    save_figure(fig, "meta_top_tags.png")


def plot_licenses(md: pd.DataFrame) -> None:
    counts = md.license.fillna("none").value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(counts.index, counts.values, color=BAR_COLOR)
    ax.set_yscale("log")
    ax.set_ylabel("number of TIDs (log)")
    ax.set_title("License values")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save_figure(fig, "meta_license.png")


def plot_coverage(coverage: dict[str, int]) -> None:
    labels, values = list(coverage)[::-1], list(coverage.values())[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(labels, values, color=BAR_COLOR)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v}", va="center")
    ax.set_xlabel("number of TIDs")
    ax.set_title("Coverage per source")
    ax.set_xlim(0, max(coverage.values()) * 1.12)
    save_figure(fig, "meta_coverage.png")


def main() -> None:
    sets = {name: tids_in(name) for name in ("metadata", "audio", "p8", "lua")}
    sets["cartridges"] = tids_in("cartridges")
    ouro = {}
    with open(DATASET / "dataset_ouro.jsonl") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                ouro[record["tid"]] = record
    sets["ouro_jsonl"] = set(ouro)
    sets["visited"] = {int(x) for x in (DATASET / "visited_tids.txt").read_text().split() if x.isdigit()}
    universe = sets["metadata"] | sets["audio"] | sets["p8"] | sets["lua"] | set(ouro)
    print_coverage(sets, universe, ouro)

    no_audio_reasons, description_lengths = Counter(), {}
    rows = [build_row(tid, sets["audio"], sets["p8"], sets["lua"], no_audio_reasons, description_lengths)
            for tid in sorted(universe)]
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=[c for c in df if c.startswith("_")]).to_csv(OUT_CSV, index=False)

    print_audio_availability(df, no_audio_reasons)
    md = df[df.stars.notna()].copy()
    tc = tag_counts(md)
    print_metadata_stats(md, description_lengths, tc)
    print_duplicates(df)
    n_calling_music = print_lua_music(df)

    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "figure.dpi": 130})
    plot_stars_histogram(md)
    plot_top_tags(tc)
    plot_licenses(md)
    plot_coverage({"metadata": len(sets["metadata"]), "p8": len(sets["p8"]), "lua": len(sets["lua"]),
                   "audio.json": len(sets["audio"]), "audio with sfx": int((df._n_sfx > 0).sum()),
                   "audio with active music": int((df._n_music_active > 0).sum()),
                   "lua calls music()": n_calling_music})


if __name__ == "__main__":
    main()
