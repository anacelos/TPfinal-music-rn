import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import StrMethodFormatter  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = REPO_ROOT / "data" / "splits"
FIG_DIR = Path(__file__).resolve().parent / "figures"
SPLITS = ("train", "val", "test")
CHANNELS = ("ch1", "ch2", "ch3", "ch4")

THOUSANDS = StrMethodFormatter("{x:,.0f}")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
BAR = "#2a78d6"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.size": 10,
})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)


def bars(name: str, title: str, labels: list, values: list, xlabel: str, ylabel: str, annotate: bool,
         tick_every: int = 1) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rects = ax.bar(range(len(values)), values, color=BAR, width=0.8, edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(range(0, len(values), tick_every), [str(x) for x in labels][::tick_every])
    if annotate:
        ax.bar_label(rects, labels=[f"{v:,}" for v in values], padding=3, color=MUTED, fontsize=9)
    ax.yaxis.set_major_formatter(THOUSANDS)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    save(fig, name)


def funnel(name: str, title: str, labels: list, values: list) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    rows = range(len(values))[::-1]
    rects = ax.barh(rows, values, color=BAR, height=0.7, edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(rows, labels)
    ax.bar_label(rects, labels=[f"{v:,}" for v in values], padding=4, color=MUTED, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(0, max(values) * 1.12)
    ax.xaxis.set_major_formatter(THOUSANDS)
    ax.set(title=title, xlabel="songs left")
    save(fig, name)


def hist(name: str, title: str, values: list, bins, xlabel: str, log: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(values, bins=bins, color=BAR, edgecolor=SURFACE, linewidth=0.8)
    if log:
        ax.set_yscale("log")
    ax.set(title=title, xlabel=xlabel, ylabel="songs" + (" (log scale)" if log else ""))
    save(fig, name)


def kept_songs():
    for name in SPLITS:
        with open(SPLIT_DIR / f"{name}.jsonl") as f:
            for line in f:
                yield json.loads(line)


def main() -> None:
    with open(SPLIT_DIR / "build_stats.json") as f:
        stats = json.load(f)
    with open(SPLIT_DIR / "filter_report.csv", newline="") as f:
        report = list(csv.DictReader(f))

    patterns, durations = [], []
    end_reasons, pitches, instruments = Counter(), Counter(), Counter()
    for song in kept_songs():
        patterns.append(song["num_patterns"])
        durations.append(song["duration_sec"])
        end_reasons[song["end_reason"]] += 1
        for pattern in song["patterns"]:
            for ch in CHANNELS:
                channel = pattern["channels"][ch]
                if not channel["active"]:
                    continue
                for note in channel["notes_unrolled"]:
                    if note["volume"] > 0:
                        pitches[note["pitch"]] += 1
                        instruments[note["instrument"]] += 1

    FIG_DIR.mkdir(exist_ok=True)
    for old in FIG_DIR.glob("*.png"):
        old.unlink()

    remaining = stats["songs_in"]
    funnel_labels, funnel_values = ["all songs.json songs"], [remaining]
    for reason, count in stats["dropped"].items():
        remaining -= count
        funnel_labels.append(f"minus {reason}")
        funnel_values.append(remaining)
    funnel("filter_funnel.png", "Songs left after each filter, in the order they run", funnel_labels, funnel_values)
    hist("patterns_per_song.png", "Patterns per kept song", patterns,
         [n - 0.5 for n in range(1, max(patterns) + 2)], "patterns", log=True)
    hist("duration_seconds.png", "Duration of kept songs (one pass, no loop repeat)", durations, 60, "seconds",
         log=True)
    order = [r for r in ("loop_end", "stop", "eof", "empty") if end_reasons[r]]
    bars("end_reason.png", "How kept songs end", order, [end_reasons[r] for r in order],
         "end_reason in songs.json", "songs", annotate=True)
    bars("pitch_hist.png", "Pitch of audible note steps in kept songs", list(range(64)),
         [pitches[p] for p in range(64)], "pitch (0 = C0, 63 = D#5)", "note steps", annotate=False,
         tick_every=12)
    bars("instrument_hist.png", "Instrument of audible note steps in kept songs", list(range(8)),
         [instruments[i] for i in range(8)], "instrument", "note steps", annotate=True)

    kept_tid_of_hash = {r["content_hash"]: r["tid"] for r in report if r["decision"] == "keep"}
    duplicates = [r for r in report if r["decision"] == "duplicate"]
    cross_game = sum(kept_tid_of_hash[r["content_hash"]] != r["tid"] for r in duplicates)
    passed_quality = stats["kept"] + stats["dropped"]["duplicate"]
    print(f"duplicates: {len(duplicates):,} of {passed_quality:,} songs that pass the quality filters "
          f"({len(duplicates) / passed_quality:.1%}); {cross_game:,} of them copy a song from another game "
          f"({cross_game / len(duplicates):.1%} of the duplicates)")
    for name, split in stats["splits"].items():
        print(f"{name}: {split['songs']:,} songs ({split['songs'] / stats['kept']:.1%}), {split['games']:,} games")
    print(f"groups: {stats['groups']:,}; largest: "
          + ", ".join(f"{g['songs']} songs in {g['games']} games ({g['split']})" for g in stats["largest_groups"][:2]))
    print(f"wrote {len(list(FIG_DIR.glob('*.png')))} figures to {FIG_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
