"""Figures and summary numbers for the extracted music corpus: music_*.png and corpus_stats.json."""
import json
from collections import Counter
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..lua_calls import (CALL_EMPTY_PATTERN, CALL_MID_SONG, CALL_OUT_OF_RANGE, CALL_SONG_START,
                         CALL_STOP)
from ..paths import FIGURES, INTERIM
from ..pico8.model import (N_EFFECTS, N_INSTRUMENTS, N_PITCHES, N_VOLUMES, NOTES_PER_SFX,
                           SPEED_KINDS, TERMINATION_LOOP, TERMINATION_LOOP_EXTERNAL,
                           TERMINATION_STOP)

COLOR = "#2a6f97"
COLOR2 = "#d1495b"
MAX_SPEED_SHOWN = 64
SLOW_SPEED = 32
SONGS_PER_TID_CLIP = 20
TOP_SPEEDS = 15
WIDE_HIST_BINS = 16
LONG_SONG_SECONDS = 300
SHORT_SONG_SECONDS = 5
SHORT_SONG_STEPS = (64, 32)
HIGH_REST_RATIOS = (0.9, 0.75)
CUSTOM_NOTE_SHARE = 0.1
MAJORITY = 0.5
DESCRIBE_COLUMNS = ["n_patterns", "n_steps_full", "n_steps_loop", "duration_s_full", "duration_s_loop",
                    "rest_ratio", "active_ch_mean"]
DESCRIBE_PERCENTILES = [.05, .25, .5, .75, .95, .99]
EFFECT_AXIS_LABEL = ("effect (0 none, 1 slide, 2 vibrato, 3 drop, 4 fade in, 5 fade out, "
                     "6 arp fast, 7 arp slow)")


@dataclass
class Corpus:
    rows: list[dict]
    songs: pd.DataFrame
    tids: pd.DataFrame
    calls: pd.DataFrame


def configure_style() -> None:
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / f"music_{name}.png")
    plt.close(fig)


def log_hist(ax, data, bins=60, label=None, color=COLOR, alpha=0.8) -> None:
    data = np.asarray(data, dtype=float)
    data = data[data > 0]
    edges = np.logspace(np.log10(max(data.min(), 1)), np.log10(data.max() + 1), bins)
    ax.hist(data, bins=edges, color=color, alpha=alpha, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")


def bar(ax, labels, values, color=COLOR, logy=False, annotate=True) -> None:
    x = np.arange(len(labels))
    ax.bar(x, values, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if logy:
        ax.set_yscale("log")
    if annotate:
        for xi, v in zip(x, values):
            ax.text(xi, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)


def pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 2) if whole else 0.0


def load_corpus() -> Corpus:
    with open(INTERIM / "songs.jsonl") as f:
        rows = [json.loads(line) for line in f]
    songs = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in rows])
    songs["tid"] = songs["tid"].astype(str)
    tids = pd.read_csv(INTERIM / "tids_summary.csv", dtype={"tid": str})
    calls = pd.read_csv(INTERIM / "music_calls.csv", dtype={"tid": str})
    return Corpus(rows, songs, tids, calls)


def add_derived_columns(corpus: Corpus) -> None:
    S = corpus.songs
    S["has_loop"] = S.termination.isin([TERMINATION_LOOP, TERMINATION_LOOP_EXTERNAL])


def describe(S: pd.DataFrame, col: str) -> dict:
    return {k: float(v) for k, v in S[col].describe(percentiles=DESCRIBE_PERCENTILES).items()}


def summarize_sizes(out: dict, S: pd.DataFrame) -> None:
    for col in DESCRIBE_COLUMNS:
        out["desc_" + col] = describe(S, col)


def summarize_songs_per_tid(out: dict, T: pd.DataFrame) -> pd.Series:
    songs_per_tid = T.set_index("tid").n_songs
    with_songs = songs_per_tid[songs_per_tid > 0]
    out["songs_per_tid"] = {"mean_all": float(songs_per_tid.mean()),
                            "mean_with_songs": float(with_songs.mean()),
                            "median_with_songs": float(with_songs.median()),
                            "max": int(songs_per_tid.max()),
                            "tids_zero": int((songs_per_tid == 0).sum()),
                            "tids_one": int((songs_per_tid == 1).sum()),
                            "tids_ge2": int((songs_per_tid >= 2).sum()),
                            "tids_ge5": int((songs_per_tid >= 5).sum())}
    out["tids_table_nonempty"] = int((T.n_patterns_table - T.n_empty_patterns > 0).sum())
    out["tids_no_music_table"] = int((T.n_patterns_table == 0).sum())
    out["tids_all_empty_patterns"] = int(((T.n_patterns_table > 0) &
                                          (T.n_patterns_table == T.n_empty_patterns)).sum())
    out["tids_with_sfx_no_music"] = int(((T.n_patterns_table == 0) & (T.n_sfx > 0)).sum())
    return songs_per_tid


def plot_songs_per_tid(songs_per_tid: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = songs_per_tid.clip(upper=SONGS_PER_TID_CLIP).value_counts().sort_index()
    labels = [str(i) if i < SONGS_PER_TID_CLIP else f"{SONGS_PER_TID_CLIP}+" for i in counts.index]
    bar(ax, labels, counts.values, logy=True, annotate=False)
    ax.set_title(f"Songs per TID (all TIDs, count clipped at {SONGS_PER_TID_CLIP}+)")
    ax.set_xlabel("songs in TID")
    ax.set_ylabel("number of TIDs (log)")
    save(fig, "songs_per_tid")


def plot_patterns_per_song(S: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = S.n_patterns.value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=COLOR)
    ax.set_yscale("log")
    ax.set_title("Patterns per song")
    ax.set_xlabel("patterns in song")
    ax.set_ylabel("number of songs (log)")
    save(fig, "patterns_per_song")


def plot_steps_per_song(S: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    log_hist(ax, S.n_steps_full, label="full (start to end)")
    log_hist(ax, S.n_steps_loop, label="loop only (loop start to loop end)", color=COLOR2, alpha=0.55)
    ax.set_title("Steps per song (steps = notes of the pattern-defining channel)")
    ax.set_xlabel("steps (log)")
    ax.set_ylabel("number of songs (log)")
    ax.legend()
    save(fig, "steps_per_song")


def plot_duration(S: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    log_hist(ax, S.duration_s_full, label="full")
    log_hist(ax, S.duration_s_loop, label="loop only", color=COLOR2, alpha=0.55)
    ax.set_title("Song duration (one pass, master-channel speed)")
    ax.set_xlabel("seconds (log)")
    ax.set_ylabel("number of songs (log)")
    ax.legend()
    save(fig, "duration_seconds")


def count_pattern_features(rows: list[dict]) -> tuple[Counter, Counter, Counter]:
    speeds, kinds, active_channels = Counter(), Counter(), Counter()
    for row in rows:
        for pattern in row["patterns"]:
            kinds[pattern["kind"]] += 1
            active_channels[sum(1 for s in pattern["sfx"] if s >= 0)] += 1
            speeds.update(s for s in pattern["speed"] if s > 0)
    return speeds, kinds, active_channels


def summarize_patterns(out: dict, speeds: Counter, kinds: Counter, active_channels: Counter) -> None:
    out["n_patterns_in_songs"] = sum(kinds.values())
    out["pattern_kinds"] = dict(kinds)
    out["active_channels_per_pattern"] = {str(k): v for k, v in sorted(active_channels.items())}
    out["speed_counts_top"] = {str(k): v for k, v in speeds.most_common(TOP_SPEEDS)}
    out["speed_total_channels"] = sum(speeds.values())
    out["speed_gt_32"] = sum(v for k, v in speeds.items() if k > SLOW_SPEED)


def plot_speed_distribution(speeds: Counter) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    xs = np.arange(1, MAX_SPEED_SHOWN + 1)
    ax.bar(xs, [speeds.get(i, 0) for i in xs], color=COLOR)
    ax.set_yscale("log")
    n_above = sum(v for k, v in speeds.items() if k > MAX_SPEED_SHOWN)
    ax.set_title(f"SFX speed of enabled channels (speeds above {MAX_SPEED_SHOWN}: {n_above:,})")
    ax.set_xlabel("speed (effective, 0 treated as 1; ticks per note)")
    ax.set_ylabel("channel uses in song patterns (log)")
    save(fig, "speed_distribution")


def plot_speed_kinds(kinds: Counter) -> None:
    n_patterns = sum(kinds.values())
    fig, ax = plt.subplots(figsize=(7, 4))
    values = [kinds.get(k, 0) for k in SPEED_KINDS]
    bar(ax, list(SPEED_KINDS), values, logy=True)
    ax.set_title("Pattern speed compatibility (percent of patterns annotated)")
    ax.set_xlabel("single: 1 usable channel; uniform: same speed; multiple: integer multiples of fastest")
    ax.set_ylabel("patterns (log)")
    for i, v in enumerate(values):
        ax.text(i, v / 3 if v else 1, f"{pct(v, n_patterns)}%", ha="center", color="white", fontsize=8)
    save(fig, "speed_compat_patterns")


def summarize_song_speeds(out: dict, S: pd.DataFrame, incompatible_fraction: pd.Series) -> None:
    n_mixed = int(((S.n_pat_multiple + S.n_pat_incompatible) > 0).sum())
    n_incompatible = int((S.n_pat_incompatible > 0).sum())
    fig, ax = plt.subplots(figsize=(7, 4))
    bar(ax, ["any non-uniform\nspeeds", "any incompatible\n(non-multiple)"], [n_mixed, n_incompatible])
    ax.set_title(f"Songs with mixed channel speeds (of {len(S):,} songs)")
    ax.set_ylabel("songs")
    save(fig, "speed_compat_songs")
    out["songs_mixed_speeds"] = n_mixed
    out["songs_incompatible_speeds"] = n_incompatible
    out["tids_incompatible_speeds"] = int(S[S.n_pat_incompatible > 0].tid.nunique())
    out["songs_incompat_frac_gt_half"] = int((incompatible_fraction > MAJORITY).sum())
    out["songs_incompat_frac_all"] = int((incompatible_fraction == 1).sum())


def plot_active_channels(active_channels: Counter) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    keys = sorted(active_channels)
    bar(ax, [str(k) for k in keys], [active_channels[k] for k in keys], logy=False)
    ax.set_title("Enabled channels per pattern (patterns inside songs)")
    ax.set_xlabel("number of enabled channels")
    ax.set_ylabel("patterns")
    save(fig, "active_channels")


def plot_note_histogram(rows: list[dict], key: str, n_bins: int, title: str, xlabel: str,
                        name: str) -> list[int]:
    totals = np.sum([r[key] for r in rows], axis=0)
    fig, ax = plt.subplots(figsize=(9 if n_bins > WIDE_HIST_BINS else 6, 4))
    ax.bar(np.arange(n_bins), totals, color=COLOR)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("non-rest notes (full unroll)")
    save(fig, name)
    return [int(x) for x in totals]


def summarize_note_histograms(out: dict, rows: list[dict]) -> None:
    out["pitch_hist"] = plot_note_histogram(
        rows, "pitch_hist", N_PITCHES, "Pitch histogram (non-rest notes, 0 = C-0)", "pitch (0-63)",
        "pitch_hist")
    out["instr_hist"] = plot_note_histogram(
        rows, "instr_hist", N_INSTRUMENTS, "Instrument histogram (8-15 = custom SFX instruments)",
        "instrument", "instrument_hist")
    out["eff_hist"] = plot_note_histogram(
        rows, "eff_hist", N_EFFECTS, "Effect histogram (non-rest notes)", EFFECT_AXIS_LABEL,
        "effect_hist")
    out["vol_hist"] = plot_note_histogram(
        rows, "vol_hist", N_VOLUMES, "Volume histogram (non-rest notes; rests, volume 0, excluded)",
        "volume (1-7)", "volume_hist")


def plot_rest_ratio(S: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(S.rest_ratio.dropna(), bins=50, color=COLOR)
    ax.set_yscale("log")
    ax.set_title("Rest ratio per song (rests / enabled channel-steps)")
    ax.set_xlabel("rest ratio")
    ax.set_ylabel("songs (log)")
    save(fig, "rest_ratio")


def summarize_terminations(out: dict, corpus: Corpus) -> None:
    S = corpus.songs
    counts = S.termination.value_counts()
    external = S.termination == TERMINATION_LOOP_EXTERNAL
    out["termination"] = {k: int(v) for k, v in counts.items()}
    out["loop_external"] = int(external.sum())
    out["loop_external_tids"] = int(S[external].tid.nunique())
    out["extra_flag_patterns"] = int(S.n_extra_flag_patterns.sum())
    out["extra_flag_songs"] = int((S.n_extra_flag_patterns > 0).sum())
    out["len_mode_patterns"] = int(S.n_len_mode_patterns.sum())
    out["len_mode_songs"] = int((S.n_len_mode_patterns > 0).sum())
    out["patterns_len_lt32"] = int(sum(1 for r in corpus.rows for p in r["patterns"]
                                       if p["len_notes"] < NOTES_PER_SFX))
    out["songs_steps_ne_32n"] = int((S.n_steps_full != NOTES_PER_SFX * S.n_patterns).sum())
    out["blank_sfx_ref_songs"] = int((S.n_blank_sfx_refs > 0).sum())
    fig, ax = plt.subplots(figsize=(7, 4))
    bar(ax, list(counts.index), counts.values)
    ax.set_title("Song termination type")
    ax.set_ylabel("songs")
    save(fig, "termination")


def summarize_duplicates(out: dict, S: pd.DataFrame) -> None:
    n_songs = len(S)
    grouped = S.groupby("content_hash")
    sizes = grouped.size()
    n_unique = len(sizes)
    duplicates_within_tid = int(n_songs - S.groupby(["tid", "content_hash"]).ngroups)
    duplicates_global = int(n_songs - n_unique)
    out["duplicates"] = {
        "unique_hashes": n_unique, "redundant_songs_global": duplicates_global,
        "redundant_within_tid": duplicates_within_tid,
        "redundant_across_tids": duplicates_global - duplicates_within_tid,
        "hash_groups_spanning_tids": int((grouped.tid.nunique() > 1).sum()),
        "max_group_size": int(sizes.max()),
        "unique_loop_hashes": int(S.content_hash_loop.nunique()),
        "songs_in_groups_ge2": int(sizes[sizes >= 2].sum()),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bar(axes[0], ["songs", "unique\n(global)", "redundant\nwithin TID", "redundant\nacross TIDs"],
        [n_songs, n_unique, duplicates_within_tid, duplicates_global - duplicates_within_tid])
    axes[0].set_title("Exact duplicates of the unrolled content")
    axes[0].set_ylabel("songs")
    group_sizes = sizes.value_counts().sort_index()
    axes[1].bar(group_sizes.index, group_sizes.values, color=COLOR)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Size of duplicate groups")
    axes[1].set_xlabel("songs sharing the same content hash (log)")
    axes[1].set_ylabel("number of groups (log)")
    save(fig, "duplicates")


def summarize_lua_calls(out: dict, corpus: Corpus) -> None:
    S, T, C = corpus.songs, corpus.tids, corpus.calls
    by_class = C["class"].value_counts()
    literal = C[C["class"] != CALL_STOP]
    out["lua_calls"] = {k: int(v) for k, v in by_class.items()}
    out["lua_literal_targets"] = int(len(literal))
    out["lua_tids_with_literal_target"] = int(literal.tid.nunique())
    out["lua_tids_with_mid_song"] = int(C[C["class"] == CALL_MID_SONG].tid.nunique())
    out["lua_nonliteral_calls"] = int(T.n_lua_nonliteral_calls.sum())
    out["lua_tids_nonliteral"] = int((T.n_lua_nonliteral_calls > 0).sum())
    out["tids_with_songs_and_no_literal_target"] = int(len(set(S.tid) - set(literal.tid)))
    fig, ax = plt.subplots(figsize=(7, 4))
    order = [k for k in [CALL_SONG_START, CALL_MID_SONG, CALL_EMPTY_PATTERN, CALL_OUT_OF_RANGE]
             if k in by_class.index]
    bar(ax, order, [by_class[k] for k in order])
    ax.set_title("Literal music(N) targets in lua vs song segmentation")
    ax.set_ylabel("distinct (TID, N) targets")
    save(fig, "lua_entry_points")


def filter_candidates(S: pd.DataFrame, incompatible_fraction: pd.Series) -> dict:
    first_in_tid = ~S.duplicated(["tid", "content_hash"], keep="first")
    first_global = ~S.duplicated("content_hash", keep="first")
    high_rest_a, high_rest_b = HIGH_REST_RATIOS
    short_a, short_b = SHORT_SONG_STEPS
    return {
        "all_rest (no audible note)": S.all_rest,
        "single pattern": S.n_patterns == 1,
        "at most 2 patterns": S.n_patterns <= 2,
        "no loop (stop / end_of_table / fallthrough)": ~S.has_loop,
        "loop target outside the song (loop_external)": S.termination == TERMINATION_LOOP_EXTERNAL,
        "no loop, stop termination only": S.termination == TERMINATION_STOP,
        "any incompatible-speed pattern": S.n_pat_incompatible > 0,
        f"incompatible patterns > {MAJORITY:.0%} of song": incompatible_fraction > MAJORITY,
        "any non-uniform speeds (multiple or incompatible)": (S.n_pat_multiple + S.n_pat_incompatible) > 0,
        "exact duplicate within TID (keep first)": ~first_in_tid,
        "exact duplicate across corpus (keep first)": ~first_global,
        f"very long: duration > {LONG_SONG_SECONDS} s": S.duration_s_full > LONG_SONG_SECONDS,
        f"very short: duration < {SHORT_SONG_SECONDS} s": S.duration_s_full < SHORT_SONG_SECONDS,
        f"very short: n_steps_full < {short_a}": S.n_steps_full < short_a,
        f"very short: n_steps_full < {short_b}": S.n_steps_full < short_b,
        "uses custom instruments (8-15)": S.n_custom_instr_notes > 0,
        f"custom instruments > {CUSTOM_NOTE_SHARE:.0%} of notes":
            S.n_custom_instr_notes > CUSTOM_NOTE_SHARE * S.n_notes.clip(lower=1),
        "references blank (absent) sfx": S.n_blank_sfx_refs > 0,
        f"rest ratio > {high_rest_a}": S.rest_ratio > high_rest_a,
        f"rest ratio > {high_rest_b}": S.rest_ratio > high_rest_b,
        "single active channel (max 1)": S.active_ch_max == 1,
        f"slow: speed_max > {SLOW_SPEED}": S.speed_max > SLOW_SPEED,
        "baseline combo: all_rest | single pattern | incompatible | global duplicate":
            S.all_rest | (S.n_patterns == 1) | (S.n_pat_incompatible > 0) | ~first_global,
    }


def filter_table(S: pd.DataFrame, incompatible_fraction: pd.Series) -> list[dict]:
    n_songs = len(S)
    all_tids = set(S.tid)
    table = []
    for name, mask in filter_candidates(S, incompatible_fraction).items():
        mask = mask.fillna(False).astype(bool)
        kept_tids = set(S[~mask].tid)
        table.append({"filter": name, "songs_removed": int(mask.sum()),
                      "songs_pct": pct(int(mask.sum()), n_songs),
                      "tids_with_removed_songs": int(S[mask].tid.nunique()),
                      "tids_fully_removed": int(len(all_tids - kept_tids)),
                      "songs_kept": int((~mask).sum())})
    return table


def main() -> None:
    configure_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    add_derived_columns(corpus)
    S, rows = corpus.songs, corpus.rows
    out = {"n_tids": len(corpus.tids), "n_songs": len(S), "n_tids_with_songs": int(S.tid.nunique())}

    summarize_sizes(out, S)
    songs_per_tid = summarize_songs_per_tid(out, corpus.tids)
    plot_songs_per_tid(songs_per_tid)
    plot_patterns_per_song(S)
    plot_steps_per_song(S)
    plot_duration(S)

    speeds, kinds, active_channels = count_pattern_features(rows)
    summarize_patterns(out, speeds, kinds, active_channels)
    plot_speed_distribution(speeds)
    plot_speed_kinds(kinds)
    incompatible_fraction = S.n_pat_incompatible / S.n_patterns
    summarize_song_speeds(out, S, incompatible_fraction)
    plot_active_channels(active_channels)
    summarize_note_histograms(out, rows)
    plot_rest_ratio(S)
    summarize_terminations(out, corpus)
    summarize_duplicates(out, S)
    summarize_lua_calls(out, corpus)
    out["filters"] = filter_table(S, incompatible_fraction)

    with open(INTERIM / "corpus_stats.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"figures in {FIGURES}; stats in {INTERIM}/corpus_stats.json")


if __name__ == "__main__":
    main()
