import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..paths import DATASET
from ..pico8.model import (CUSTOM_INSTRUMENT_MIN, N_CHANNELS, N_SLOTS, NOTES_PER_SFX,
                           channel_sfx_id, is_channel_disabled)
from ..pico8.p8_cart import P8Cart, editor_features, parse_p8

DEFAULT_STEP = 62
MAX_EXAMPLES = 5
MAX_ENCODABLE_EDITOR_BYTE = 0xD7
DEFAULT_DISABLED_IDS = (0, 1, 2, 3, 4)
NOTE_KEYS = {"pitch", "instrument", "volume", "effect"}
SFX_KEYS = {"id", "speed", "loop_start", "loop_end", "notes"}
MUSIC_KEYS = {"id", "flags", "ch1", "ch2", "ch3", "ch4"}
SFX_HEADER_FIELDS = ("speed", "loop_start", "loop_end")
INVENTORY_PREFIXES = ("p8_files", "p8_dir", "p8_with", "p8_audio", "p8_has", "p8_malformed",
                      "p8_read", "audio_dir")
ALWAYS_REPORTED = (
    "json_parse_error", "verify_skipped_no_p8", "mismatch_pitch", "mismatch_volume",
    "mismatch_effect", "mismatch_instrument_vs_wave4", "mismatch_sfx_speed",
    "mismatch_sfx_loop_start", "mismatch_sfx_loop_end", "mismatch_music_flags",
    "mismatch_music_ch", "json_sfx_ids_not_sequential", "json_sfx_count_ne_p8_lines",
    "json_music_count_ne_p8_lines", "p8_malformed_sfx", "p8_malformed_music", "p8_read_error",
    "p8_dir_without_file", "audio_dir_without_p8", "p8_audio_section_no_audio_dir",
    "audio_dir_p8_has_no_audio_section", "audio_dir_without_json", "p8_editor_byte_gt_0xd7")


@dataclass
class Findings:
    counts: Counter = field(default_factory=Counter)
    editor_values: Counter = field(default_factory=Counter)
    examples: dict = field(default_factory=dict)
    sfx_per_tid: list = field(default_factory=list)
    music_per_tid: list = field(default_factory=list)
    n_notes_compared: int = 0

    def add_example(self, key: str, item: tuple) -> None:
        self.examples.setdefault(key, []).append(item)


def p8_path(root: Path, tid: str) -> Path:
    return root / "p8" / tid / f"{tid}.p8"


def count_cart_inventory(cart: P8Cart, has_audio_dir: bool, counts: Counter) -> None:
    counts["p8_files"] += 1
    counts["p8_version_" + str(cart.version)] += 1
    if cart.read_error:
        counts["p8_read_error"] += 1
    if cart.has_sfx or cart.has_music:
        counts["p8_with_audio_section"] += 1
        if not has_audio_dir:
            counts["p8_audio_section_no_audio_dir"] += 1
    if cart.has_sfx:
        counts["p8_has_sfx_section"] += 1
    if cart.has_music:
        counts["p8_has_music_section"] += 1
    if cart.bad_sfx_lines:
        counts["p8_malformed_sfx"] += 1
    if cart.bad_music_lines:
        counts["p8_malformed_music"] += 1


def count_inventory(root: Path, audio_tids: list[str], p8_tids: list[str], counts: Counter) -> None:
    audio_set = set(audio_tids)
    has_audio_section = {}
    for tid in p8_tids:
        path = p8_path(root, tid)
        if not path.exists():
            counts["p8_dir_without_file"] += 1
            continue
        cart = parse_p8(path)
        has_audio_section[tid] = cart.has_sfx or cart.has_music
        count_cart_inventory(cart, tid in audio_set, counts)
    for tid in audio_tids:
        if tid not in has_audio_section:
            counts["audio_dir_without_p8"] += 1
        elif not has_audio_section[tid]:
            counts["audio_dir_p8_has_no_audio_section"] += 1
        if not (root / "audio" / tid / "audio.json").exists():
            counts["audio_dir_without_json"] += 1
    counts["audio_dirs"] = len(audio_tids)


def count_json_shape(audio: dict, cart: P8Cart, findings: Findings) -> None:
    c = findings.counts
    sfx_entries, music_entries = audio.get("sfx", []), audio.get("music", [])
    findings.sfx_per_tid.append(len(sfx_entries))
    findings.music_per_tid.append(len(music_entries))
    sfx_ids = [s["id"] for s in sfx_entries]
    if sfx_ids != list(range(len(sfx_entries))):
        c["json_sfx_ids_not_sequential"] += 1
    if len(sfx_entries) != cart.sfx_lines:
        c["json_sfx_count_ne_p8_lines"] += 1
    if len(music_entries) != cart.music_lines:
        c["json_music_count_ne_p8_lines"] += 1
    c["tids_json_sfx_lt_64"] += len(sfx_entries) < N_SLOTS
    c["tids_json_music_lt_64"] += len(music_entries) < N_SLOTS
    if len(set(sfx_ids)) != len(sfx_ids):
        c["json_dup_sfx_ids"] += 1
    music_ids = [m["id"] for m in music_entries]
    if len(set(music_ids)) != len(music_ids):
        c["json_dup_music_ids"] += 1
    c["json_sfx_entries"] += len(sfx_entries)
    c["json_music_entries"] += len(music_entries)
    for s in sfx_entries:
        if len(s["notes"]) != NOTES_PER_SFX:
            c["json_sfx_not_32_notes"] += 1


def count_p8_sfx(cart: P8Cart, json_ids: set[int], findings: Findings) -> None:
    c = findings.counts
    for i, p8_sfx in cart.sfx.items():
        sfx = p8_sfx.sfx
        blank = p8_sfx.is_blank
        sound = sfx.n_audible_notes > 0
        in_json = "in_json" if i in json_ids else "not_in_json"
        c["p8_sfx_lines"] += 1
        c["p8_sfx_blank" if blank else "p8_sfx_nonblank"] += 1
        c[("sfx_blank_" if blank else "sfx_nonblank_") + in_json] += 1
        if not blank:
            c[("sfx_silent_nonblank_" if not sound else "sfx_sound_") + in_json] += 1
        findings.editor_values[p8_sfx.editor] += 1
        if p8_sfx.editor > MAX_ENCODABLE_EDITOR_BYTE:
            c["p8_editor_byte_gt_0xd7"] += 1
        for feature in editor_features(p8_sfx.editor):
            c["sfx_" + feature] += 1
        if sfx.speed == 0:
            c["p8_sfx_speed0"] += 1
        if sfx.loop_start > sfx.loop_end and sfx.loop_end != 0:
            c["p8_sfx_loop_start_gt_end_nonzero"] += 1
        if sfx.len_mode:
            c["p8_sfx_len_mode"] += 1
        if sfx.loop_end > 0 and not sfx.loops:
            c["p8_sfx_loop_off_end_le_start"] += 1
        if p8_sfx.editor != 0:
            c["p8_sfx_editor_byte_nonzero"] += 1
        if any(n.is_custom_instrument for n in sfx.notes):
            c["p8_sfx_with_custom_instr_bit"] += 1
    missing = json_ids - set(cart.sfx)
    if missing:
        c["json_sfx_id_missing_in_p8"] += len(missing)


def compare_note(tid: str, sfx_id: int, note: dict, p8_note, findings: Findings) -> None:
    c = findings.counts
    if note["pitch"] != p8_note.pitch:
        c["mismatch_pitch"] += 1
        findings.add_example("mismatch_pitch", (tid, sfx_id, note["pitch"], p8_note.pitch))
    if note["volume"] != p8_note.volume:
        c["mismatch_volume"] += 1
    if note["effect"] != p8_note.effect:
        c["mismatch_effect"] += 1
    if note["instrument"] != p8_note.instrument:
        c["mismatch_instrument_vs_wave4"] += 1
        findings.add_example("mismatch_instrument",
                             (tid, sfx_id, note["instrument"], p8_note.instrument))
    if p8_note.is_custom_instrument:
        c["notes_with_custom_bit_p8"] += 1
        if note["instrument"] >= CUSTOM_INSTRUMENT_MIN:
            c["custom_bit_kept_as_instrument_gt7"] += 1
    elif note["instrument"] >= CUSTOM_INSTRUMENT_MIN:
        c["json_instrument_gt7_without_p8_custom_bit"] += 1
    if set(note) - NOTE_KEYS:
        c["json_note_extra_keys"] += 1


def compare_sfx(tid: str, entries: list[dict], cart: P8Cart, findings: Findings) -> None:
    c = findings.counts
    for entry in entries:
        p8_sfx = cart.sfx.get(entry["id"])
        if p8_sfx is None:
            continue
        c["sfx_compared"] += 1
        for name in SFX_HEADER_FIELDS:
            p8_value = getattr(p8_sfx.sfx, name)
            if entry[name] != p8_value:
                c["mismatch_sfx_" + name] += 1
                findings.add_example("mismatch_sfx_" + name, (tid, entry["id"], entry[name], p8_value))
        for note, p8_note in zip(entry["notes"], p8_sfx.sfx.notes):
            findings.n_notes_compared += 1
            compare_note(tid, entry["id"], note, p8_note, findings)
        extra = set(entry) - SFX_KEYS
        if extra:
            c["json_sfx_extra_keys"] += 1
            findings.examples.setdefault("json_sfx_extra_keys", set()).update(extra)


def count_p8_music(cart: P8Cart, json_ids: set[int], findings: Findings) -> None:
    c = findings.counts
    for i, pattern in cart.music.items():
        all_disabled = pattern.is_empty
        blank = pattern.flags == 0 and all_disabled
        in_json = "in_json" if i in json_ids else "not_in_json"
        c["p8_music_lines"] += 1
        c["p8_music_blank" if blank else "p8_music_nonblank"] += 1
        c[("music_blank_" if blank else "music_nonblank_") + in_json] += 1
        if all_disabled and not blank:
            c["music_alldisabled_flagged_" + in_json] += 1
        if pattern.loop_start:
            c["p8_music_loop_start"] += 1
        if pattern.loop_end:
            c["p8_music_loop_end"] += 1
        if pattern.stop:
            c["p8_music_stop"] += 1
        disabled = [b for b in pattern.channels if is_channel_disabled(b)]
        if not all_disabled and disabled:
            c["p8_music_partial_channels"] += 1
        if any(channel_sfx_id(b) not in DEFAULT_DISABLED_IDS for b in disabled):
            c["p8_music_disabled_byte_unusual"] += 1
        if pattern.has_extra_flags:
            c["p8_music_flags_gt7"] += 1
    missing = json_ids - set(cart.music)
    if missing:
        c["json_music_id_missing_in_p8"] += len(missing)


def compare_music(tid: str, entries: list[dict], cart: P8Cart, findings: Findings) -> None:
    c = findings.counts
    for entry in entries:
        pattern = cart.music.get(entry["id"])
        if pattern is None:
            continue
        c["music_compared"] += 1
        if entry["flags"] != pattern.flags:
            c["mismatch_music_flags"] += 1
            findings.add_example("mismatch_music_flags", (tid, entry["id"], entry["flags"], pattern.flags))
        for k in range(N_CHANNELS):
            if entry[f"ch{k + 1}"] != pattern.channels[k]:
                c["mismatch_music_ch"] += 1
                findings.add_example("mismatch_music_ch", (tid, entry["id"], k + 1,
                                                           entry[f"ch{k + 1}"], pattern.channels[k]))
        if set(entry) - MUSIC_KEYS:
            c["json_music_extra_keys"] += 1


def verify_tid(root: Path, tid: str, findings: Findings) -> None:
    c = findings.counts
    try:
        with open(root / "audio" / tid / "audio.json") as f:
            audio = json.load(f)
    except Exception as e:
        c["json_parse_error"] += 1
        findings.add_example("json_parse_error", (tid, str(e)[:80]))
        return
    path = p8_path(root, tid)
    if not path.exists():
        c["verify_skipped_no_p8"] += 1
        return
    cart = parse_p8(path)
    c["json_keys_" + ",".join(sorted(audio.keys()))] += 1
    sfx_entries, music_entries = audio.get("sfx", []), audio.get("music", [])
    count_json_shape(audio, cart, findings)
    count_p8_sfx(cart, {s["id"] for s in sfx_entries}, findings)
    compare_sfx(tid, sfx_entries, cart, findings)
    count_p8_music(cart, {m["id"] for m in music_entries}, findings)
    compare_music(tid, music_entries, cart, findings)


def print_report(findings: Findings, step_label: str) -> None:
    c = findings.counts
    print("== inventory (whole dataset) ==")
    inventory_keys = [k for k in sorted(c) if k.startswith(INVENTORY_PREFIXES)]
    for k in inventory_keys:
        print("  %-42s %d" % (k, c[k]))
    versions = Counter({k[len("p8_version_"):]: v for k, v in c.items() if k.startswith("p8_version_")})
    by_version = sorted(versions.items(), key=lambda kv: -1 if kv[0] == "None" else int(kv[0]))
    print("  p8 header versions: " + ", ".join("%s:%d" % kv for kv in by_version))
    print("== verification (%d TIDs, step %s) ==" % (c["verified_tids"], step_label))
    for k in ALWAYS_REPORTED:
        c[k] += 0
    for k in [k for k in sorted(c) if k not in inventory_keys]:
        print("  %-42s %d" % (k, c[k]))
    print("  note comparisons: %d" % findings.n_notes_compared)
    if findings.sfx_per_tid:
        sfx, music = np.array(findings.sfx_per_tid), np.array(findings.music_per_tid)
        print("  json sfx per tid: min %d med %d max %d" % (sfx.min(), np.median(sfx), sfx.max()))
        print("  json music per tid: min %d med %d max %d" % (music.min(), np.median(music), music.max()))
    print("  p8 editor-byte values (hex:count):",
          ", ".join("%02x:%d" % kv for kv in sorted(findings.editor_values.items())))
    for key, values in findings.examples.items():
        print("  examples %s: %s" % (key, list(values)[:MAX_EXAMPLES]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DATASET)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP,
                        help="verify every Nth sorted audio dir")
    parser.add_argument("--all", action="store_true", help="verify every audio dir")
    parser.add_argument("--out", default=None, help="write the counters as JSON")
    args = parser.parse_args()

    root = args.root.absolute()
    audio_tids = sorted(os.listdir(root / "audio"))
    p8_tids = sorted(os.listdir(root / "p8"))
    findings = Findings()
    count_inventory(root, audio_tids, p8_tids, findings.counts)
    sample = audio_tids if args.all else audio_tids[::args.step]
    findings.counts["verified_tids"] = len(sample)
    for tid in sample:
        verify_tid(root, tid, findings)
    print_report(findings, "all" if args.all else args.step)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"counts": dict(findings.counts),
                       "editor_values": {"%02x" % k: v for k, v in findings.editor_values.items()}},
                      f, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
