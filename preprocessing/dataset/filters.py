from typing import Optional

from ..pico8.model import TERMINATION_LOOP, TERMINATION_LOOP_EXTERNAL
from .config import Config

DEDUPE_OFF = "none"
DEDUPE_GLOBAL = "global"

SongKey = tuple[str, int]


def song_key(song: dict) -> SongKey:
    return song["tid"], song["song_idx"]


def drop_reason(song: dict, config: Config) -> Optional[str]:
    """First quality filter that rejects the song; the order only affects the reported reason."""
    if config["drop_all_rest"] and song["all_rest"]:
        return "all_rest"
    if song["n_patterns"] < config["min_patterns"]:
        return "too_few_patterns"
    if song["n_notes"] < config["min_notes"]:
        return "too_few_notes"
    if config["drop_incompatible_speeds"] and song["n_pat_incompatible"] > 0:
        return "incompatible_speeds"
    if config["drop_loop_external"] and song["termination"] == TERMINATION_LOOP_EXTERNAL:
        return "loop_external"
    if config["drop_no_loop"] and song["termination"] != TERMINATION_LOOP:
        return "no_loop"
    if config["max_steps"] and song["n_steps_full"] > config["max_steps"]:
        return "too_long"
    return None


def decide_songs(songs: list[dict], config: Config) -> dict[SongKey, Optional[str]]:
    """Maps each song to its drop reason, or None when kept. The first copy of a duplicate is kept."""
    reasons = {}
    seen = set()
    for song in songs:
        reason = drop_reason(song, config)
        if reason is None and config["dedupe"] != DEDUPE_OFF:
            digest = (song["content_hash"] if config["dedupe"] == DEDUPE_GLOBAL
                      else (song["tid"], song["content_hash"]))
            if digest in seen:
                reason = "duplicate"
            seen.add(digest)
        reasons[song_key(song)] = reason
    return reasons
