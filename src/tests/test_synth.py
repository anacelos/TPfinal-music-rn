import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sfx_filters import attach, describe, sfx_modes  # noqa: E402
from synth import PICO_RATE, SAMPLE_RATE, SAMPLES_PER_TICK, synthesize_song  # noqa: E402


def note(pitch=33, inst=0, vol=5, fx=0):
    return {"pitch": pitch, "instrument": inst, "volume": vol, "effect": fx}


def one_channel_song(notes, speed=16, ticks=None, filters=None):
    ch = {"active": True, "speed": speed, "notes_unrolled": notes, "original_sfx_id": 3, "block_id": 0}
    off = {"active": False, "speed": None, "notes_unrolled": []}
    block = {"original_sfx_id": 3, "speed": speed, "loop_start": 0, "loop_end": 0, "notes": notes[:32]}
    if filters is not None:
        block["filters"] = filters
    pattern = {"pattern_id": 0, "ref_speed": speed, "pattern_ticks": ticks or speed * len(notes),
               "channels": {"ch1": ch, "ch2": off, "ch3": off, "ch4": off}}
    return {"song_id": 0, "tid": "1", "patterns": [pattern], "blocks_dictionary": {"0": block}}


def peak_hz(x, rate):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / rate)[np.argmax(spec)]


def test_length_and_rate():
    song = one_channel_song([note()] * 32)
    audio = synthesize_song(song)
    assert len(audio) == 32 * 16 * SAMPLES_PER_TICK
    native = synthesize_song(song, rate=PICO_RATE)
    assert len(audio) == 2 * len(native)
    # the web player doubles the rate by linear interpolation
    np.testing.assert_allclose(audio[0::2], native, atol=1e-6)
    np.testing.assert_allclose(audio[1:-1:2], 0.5 * (native[:-1] + native[1:]), atol=1e-6)


def test_pitch_33_is_a440_for_every_tonal_instrument():
    for inst in (0, 1, 2, 3, 4, 7):
        audio = synthesize_song(one_channel_song([note(33, inst, 7)] * 8), rate=PICO_RATE)
        assert abs(peak_hz(audio[2000:-2000], PICO_RATE) - 440) < 5, inst
    # the organ is two triangles per period: its strongest partial is the octave, 880 Hz
    organ = synthesize_song(one_channel_song([note(33, 5, 7)] * 8), rate=PICO_RATE)
    assert abs(peak_hz(organ[2000:-2000], PICO_RATE) - 880) < 10


def test_organ_has_no_sub_octave():
    # the old synth mixed a sub-octave into the organ, putting energy at 220 Hz for an A440
    audio = synthesize_song(one_channel_song([note(33, 5, 7)] * 8), rate=PICO_RATE)[2000:-2000]
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    freqs = np.fft.rfftfreq(len(audio), 1 / PICO_RATE)
    at = lambda hz: spec[np.argmin(np.abs(freqs - hz))]
    assert at(220) < 0.05 * at(440)


def test_silence_and_volume():
    audio = synthesize_song(one_channel_song([note(vol=0)] * 4))
    assert np.all(audio == 0)
    quiet = synthesize_song(one_channel_song([note(vol=2)] * 8))
    loud = synthesize_song(one_channel_song([note(vol=6)] * 8))
    assert np.sqrt(np.mean(loud ** 2)) > 2.5 * np.sqrt(np.mean(quiet ** 2))


def test_noise_is_deterministic_per_seed():
    song = one_channel_song([note(40, 6, 5)] * 4)
    a, b, c = synthesize_song(song, seed=1), synthesize_song(song, seed=1), synthesize_song(song, seed=2)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_slide_and_drop_move_the_pitch():
    step = 16 * 183
    slide = synthesize_song(one_channel_song([note(24, 0, 7)] * 2 + [note(36, 0, 7, fx=1)] * 2), rate=PICO_RATE)
    drop = synthesize_song(one_channel_song([note(45, 0, 7, fx=3)] * 2), rate=PICO_RATE)
    # slide starts near the previous note (C, 262 Hz) and ends at the new one (C, 523 Hz)
    assert peak_hz(slide[2 * step:2 * step + 600], PICO_RATE) < 400
    assert abs(peak_hz(slide[3 * step + 1000:4 * step - 200], PICO_RATE) - 523) < 15
    # drop falls from 880 Hz toward 0 within the note
    assert peak_hz(drop[:800], PICO_RATE) > 700
    assert peak_hz(drop[step - 1200:step - 200], PICO_RATE) < 300


def test_arpeggio_cycles_over_the_group_of_four():
    notes = [note(24, 0, 7, fx=6), note(28, 0, 7, fx=6), note(31, 0, 7, fx=6), note(36, 0, 7, fx=6)]
    audio = synthesize_song(one_channel_song(notes * 2, speed=32), rate=PICO_RATE)
    win = PICO_RATE // 30  # fast arpeggio at speed > 8: a new note every 4 ticks (~33 ms)
    peaks = {round(peak_hz(audio[s + 100:s + win - 100], PICO_RATE) / 10) * 10
             for s in range(0, 8 * win, win)}
    assert len(peaks) >= 3


def test_filters_change_the_sound():
    notes = [note(33, 3, 7)] * 2 + [note(vol=0)] * 6
    dry = synthesize_song(one_channel_song(notes), rate=PICO_RATE)
    reverb = synthesize_song(one_channel_song(notes, filters=24 * 2), rate=PICO_RATE)  # reverb 2
    damp = synthesize_song(one_channel_song(notes, filters=72 * 2), rate=PICO_RATE)  # dampen 2
    tail = slice(2 * 16 * 183 + 400, 3 * 16 * 183)
    assert np.abs(dry[tail]).max() < 1e-3 < np.abs(reverb[tail]).max()  # reverb rings after the notes
    hf = lambda x: np.abs(np.fft.rfft(x[:5000]))[np.fft.rfftfreq(5000, 1 / PICO_RATE) > 3000].sum()
    assert hf(damp) < 0.5 * hf(dry)


def test_filters_table_round_trip():
    cart = "__sfx__\n" + "15120000" + "0" * 160 + "\n" + "00100000" + "0" * 160 + "\n__music__\n"
    assert sfx_modes(cart) == {0: 0x15, 1: 0}
    assert describe(0x15) == {"noiz": 0, "buzz": 1, "detune": 2, "reverb": 0, "dampen": 0}
    assert describe(0x79) == {"noiz": 0, "buzz": 0, "detune": 0, "reverb": 2, "dampen": 1}
    song = one_channel_song([note()] * 4)
    with_filters = attach(song, {"1": {"3": 0x79}})
    assert with_filters["blocks_dictionary"]["0"]["filters"] == 0x79
    assert with_filters["patterns"][0]["channels"]["ch1"]["filters"] == 0x79
    assert "filters" not in song["blocks_dictionary"]["0"]  # the input is not modified
    assert attach(song, {"2": {"3": 0x79}}) is song  # other games: untouched
    assert SAMPLE_RATE == 2 * PICO_RATE
