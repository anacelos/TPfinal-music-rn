"""Decoding of malformed grid sequences, as a sampled model can produce them."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tokenizer import Pico8Tokenizer  # noqa: E402


def test_speed_after_step_resets_the_step_pointer():
    # SPEED_32 after STEP_20 rebuilds ch1 with 512 / 32 = 16 steps; INST_3 has no step to land on.
    tokens = ["BOS", "PATTERN_START", "REF_SPEED_16", "CH1_START", "SPEED_16", "STEP_20", "PITCH_30",
              "SPEED_32", "INST_3", "STEP_2", "PITCH_40", "EOS"]
    song = Pico8Tokenizer(mode="grid_unrolled").decode_tokens(tokens)
    notes = song["patterns"][0]["channels"]["ch1"]["notes_unrolled"]
    assert len(notes) == 16
    assert notes[2]["pitch"] == 40
    assert all(n["instrument"] == 0 for n in notes)
