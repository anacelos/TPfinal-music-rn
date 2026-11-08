"""Self-contained song records: the patterns a song plays plus the sfx they reference."""
from ..pico8.model import Note, Pattern, Sfx

VOLUME_FIELD = Note._fields.index("volume")


def pack_sfx(sfx: Sfx) -> dict:
    return {"speed": sfx.speed, "loop_start": sfx.loop_start, "loop_end": sfx.loop_end,
            "notes": [list(note) for note in sfx.notes]}


def unpack_sfx(packed: dict) -> Sfx:
    return Sfx(packed["speed"], packed["loop_start"], packed["loop_end"],
               tuple(Note(*note) for note in packed["notes"]))


def count_audible_notes(packed_notes: list[list[int]]) -> int:
    return sum(note[VOLUME_FIELD] > 0 for note in packed_notes)


def build_record(song: dict, patterns: list[Pattern], sfx_slots: list[Sfx], meta: dict) -> dict:
    """Carries the sfx used as custom instruments too, so the record decodes back into an
    audio.json without the source cart."""
    song_patterns = [patterns[p["id"]] for p in song["patterns"]]
    used = {sid for p in song_patterns for sid in p.sfx_ids if sid is not None}
    instruments = {i for sid in used for i in sfx_slots[sid].custom_instrument_sfx_ids}
    return {
        "song_id": f"{song['tid']}_{song['song_idx']}",
        "tid": song["tid"],
        "song_idx": song["song_idx"],
        "termination": song["termination"],
        "loop_start": song["loop_start_idx"],
        "loop_end": song["loop_end_idx"],
        "n_steps": song["n_steps_full"],
        "n_steps_loop": song["n_steps_loop"],
        "uses_custom_instruments": bool(instruments),
        "patterns": [{"src_pattern": info["id"], "channels": list(p.sfx_ids),
                      "ticks": info["ticks"], "len_notes": info["len_notes"]}
                     for p, info in zip(song_patterns, song["patterns"])],
        "sfx": {str(i): pack_sfx(sfx_slots[i]) for i in sorted(used | instruments)},
        "tags": meta.get("tags", []),
        "stars": meta.get("stars"),
    }
