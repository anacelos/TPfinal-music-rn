"""
Sintetizador de áudio do PICO-8 para o projeto final (DCC831 - UFMG).

Converte músicas no formato `songs.json` (ou decodificadas dos tokens) em `.wav`
(44,1 kHz, 16 bits, mono), sem passar por MIDI.

O motor segue o som do PICO-8 como o zepto8 o reconstruiu (emulador aberto cujas
formas de onda foram medidas em exportações WAV do PICO-8; src/synth.cpp e
src/pico8/sfx.cpp em github.com/samhocevar/zepto8):

* o PICO-8 gera 22050 Hz, 183 amostras por tick; o player web entrega 44100 Hz por
  interpolação linear (conferido numa gravação do player oficial), e a saída daqui faz o mesmo;
* as 8 formas de onda, com as variações de buzz e noiz;
* os efeitos slide, vibrato, drop, fade in, fade out e arpejos (em grupos de 4 notas);
* a fase contínua entre notas e o crossfade de ~7,7 ms que o PICO-8 faz quando nota,
  volume ou instrumento mudam de repente;
* os filtros de SFX (detune, reverb, dampen), quando a música os traz: no bloco
  (`blocks_dictionary[id]["filters"]`) ou no canal (`"filters"`), com o byte de modo do SFX.
  As músicas geradas pelo modelo não têm filtros (o tokenizador não os representa).

Instrumentos customizados (SFX como instrumento) não são suportados: o dataset os filtra.
"""

import math
import os
import wave
from typing import Any, Dict, List, Optional

import numpy as np

try:  # scipy acelera os filtros recursivos; sem ele há um laço em Python, mais lento
    from scipy.signal import lfilter as _lfilter
except ImportError:  # pragma: no cover - depende do ambiente
    _lfilter = None

SAMPLE_RATE = 44100
SAMPLES_PER_TICK = 366  # 183 amostras @ 22050 Hz = 366 amostras @ 44100 Hz (~120,49 ticks/s)
PICO_RATE = 22050
PICO_SAMPLES_PER_TICK = 183

INST_TRIANGLE, INST_TILTED_SAW, INST_SAW, INST_SQUARE, INST_PULSE, INST_ORGAN, INST_NOISE, INST_PHASER = range(8)
FX_SLIDE, FX_VIBRATO, FX_DROP, FX_FADE_IN, FX_FADE_OUT, FX_ARP_FAST, FX_ARP_SLOW = range(1, 8)

FADE_STEP = 130.0 / PICO_RATE  # o crossfade entre estados dura 1/130 s
NOISE_TSCALE = 8.858923  # 22050 / key_to_freq(63), do zepto8
SEMITONE = 2.0 ** (1.0 / 12.0)


def pitch_to_freq(pitch: float) -> float:
    """Pitch do PICO-8 (0..63) em Hz: o pitch 33 é A = 440 Hz (24 = C = 261,63 Hz)."""
    return 440.0 * (2.0 ** ((pitch - 33.0) / 12.0))


# ----------------------------------------------------------------------------
# Formas de onda (vetorizadas sobre a fase acumulada `phi`, em ciclos)
# ----------------------------------------------------------------------------
def _tilted(t: np.ndarray, a: float) -> np.ndarray:
    return np.where(t < a, 2.0 * t / a - 1.0, 2.0 * (1.0 - t) / (1.0 - a) - 1.0)


def _waveform(inst: int, phi: np.ndarray, filters: int) -> np.ndarray:
    """Uma forma de onda tonal (todas menos o noise), como no zepto8."""
    t = np.mod(phi, 1.0)
    buzz = bool(filters & 0x4)
    if inst == INST_TRIANGLE:
        ret = 1.0 - np.abs(4.0 * t - 2.0)
        if buzz:  # média com um dente de serra inclinado
            ret = ret * 0.75 + _tilted(t, 0.875) * 0.25
        return ret * 0.5
    if inst == INST_TILTED_SAW:
        return _tilted(t, 0.975 if buzz else 0.875) * 0.5
    if inst == INST_SAW:
        ret = np.where(t < 0.5, t, t - 1.0)
        if buzz:
            ret = ret * 0.83 - np.where(np.abs(np.mod(phi, 2.0) - 1.0) < 0.5, 0.085, 0.0)
        return 0.653 * ret
    if inst == INST_SQUARE:
        return np.where(t < (0.4 if buzz else 0.5), 0.25, -0.25)
    if inst == INST_PULSE:
        return np.where(t < (0.255 if buzz else 0.316), 0.25, -0.25)
    if inst == INST_ORGAN:  # dois triângulos no mesmo período, o segundo menor
        ret = np.where(t < 0.5, 3.0 - np.abs(24.0 * t - 6.0), 1.0 - np.abs(16.0 * t - 12.0))
        if buzz:
            ret = np.where(t < 0.5, ret * 2.0 + 3.0, ret)
            ret = np.where((t < 0.5) & (ret > -1.875), ret * 0.2 - 1.0, ret + 0.5)
        return ret / 9.0
    if inst == INST_PHASER:  # dois triângulos com frequências levemente diferentes
        ret = 2.0 - np.abs(8.0 * t - 4.0)
        ret = ret + 1.0 - np.abs(4.0 * np.mod(phi * 109.0 / 110.0, 1.0) - 2.0)
        if buzz:
            ret = ret + 0.25 - np.abs(np.mod(phi * 2.0 + 0.5, 1.0) - 0.5)
            ret = ret + 0.125 - np.abs(0.5 * np.mod(phi * 4.0, 1.0) - 0.25)
        return ret / 6.0
    return np.zeros_like(phi)


def _detune_wave(inst: int, phi: np.ndarray, filters: int) -> np.ndarray:
    """Segunda onda do filtro detune (meia amplitude), como no zepto8."""
    detune = (filters // 8) % 3
    if detune == 0 or inst == INST_NOISE:
        return 0.0
    if inst == INST_TRIANGLE:
        factor = 3.0 / 4.0 if detune == 1 else 3.0 / 2.0
    elif inst == INST_ORGAN:
        factor = 200.0 / 199.0 if detune == 1 else 800.0 / 199.0
    elif inst == INST_PHASER:
        factor = 49.0 / 50.0 if detune == 1 else 400.0 / 199.0
    else:
        factor = 200.0 / 199.0 if detune == 1 else 400.0 / 199.0
    second = INST_TRIANGLE if (detune == 2 and inst == INST_ORGAN) else inst
    return _waveform(second, phi * factor, filters) * 0.5


def _first_order(x: np.ndarray, a: float, y0: float) -> np.ndarray:
    """y[n] = a * y[n-1] + x[n], com y[-1] = y0."""
    if _lfilter is not None:
        y, _ = _lfilter([1.0], [1.0, -a], x, zi=[a * y0])
        return y
    y = np.empty_like(x)
    prev = y0
    for i, v in enumerate(x):
        prev = a * prev + v
        y[i] = prev
    return y


def _noise(freq: np.ndarray, key: int, phi: np.ndarray, filters: int, last: float, rng) -> tuple:
    """Ruído do PICO-8: passeio aleatório filtrado cuja cor acompanha a frequência da nota."""
    out = np.empty_like(freq)
    chunk = 64  # a frequência pode mudar dentro da nota (slide, drop, vibrato): blocos curtos
    for s in range(0, len(freq), chunk):
        e = min(s + chunk, len(freq))
        scale = float(np.mean(freq[s:e])) / PICO_RATE * NOISE_TSCALE
        r = rng.uniform(-1.0, 1.0, e - s)
        y = _first_order(r * (scale / (1.0 + scale)), 1.0 / (1.0 + scale), last)
        out[s:e] = y
        last = float(y[-1])
    factor = 1.0 - key / 63.0
    ret = out * 1.5 * (1.0 + factor * factor)
    if filters & 0x2:  # noiz: modulado por um dente de serra
        t = np.mod(phi, 1.0)
        ret = ret * 2.0 * np.where(t < 0.5, t, t - 1.0)
    return ret, last


# ----------------------------------------------------------------------------
# Canal: notas desenroladas -> segmentos com frequência e volume por amostra
# ----------------------------------------------------------------------------
class _Synth:
    """Estado sonoro de um canal (o `synth_param` do zepto8)."""

    __slots__ = ("inst", "key", "freq", "vol", "filters", "phi", "noise")

    def __init__(self, inst=0, key=0, freq=0.0, vol=0.0, filters=0, phi=0.0, noise=0.0):
        self.inst, self.key, self.freq, self.vol = inst, key, freq, vol
        self.filters, self.phi, self.noise = filters, phi, noise

    def copy(self):
        return _Synth(self.inst, self.key, self.freq, self.vol, self.filters, self.phi, self.noise)


class _Channel:
    """Renderiza um canal ao longo da música inteira, guardando o estado entre patterns."""

    def __init__(self, n_samples: int, rng):
        self.out = np.zeros(n_samples, dtype=np.float64)
        self.reverb = np.zeros(n_samples, dtype=np.int8)
        self.dampen = np.zeros(n_samples, dtype=np.int8)
        self.last = _Synth()
        self.fade = 0.0
        self.fade_synth = _Synth()
        self.rng = rng

    # -- um trecho com instrumento, nota e filtros constantes -------------------------------
    def segment(self, pos: int, freq: np.ndarray, vol: np.ndarray, inst: int, key: int, filters: int):
        n = len(freq)
        if n == 0:
            return
        new = _Synth(inst, key, float(freq[0]), float(vol[0]), filters, self.last.phi, self.last.noise)
        if vol[0] <= 0 and np.all(vol <= 0):
            new = _Synth(phi=self.last.phi, noise=self.last.noise)  # silêncio: estado padrão
        last = self.last
        thr = min(new.freq, last.freq) * 0.01
        if abs(new.vol - last.vol) > 0.1 or abs(new.freq - last.freq) > thr or new.inst != last.inst:
            if self.fade <= 0.0:
                self.fade_synth = last.copy()
            self.fade = 1.0
            new.phi = math.fmod(new.phi, 1.0)

        value = np.zeros(n)
        if new.vol > 0 or np.any(vol > 0):
            phi = new.phi + np.cumsum(freq) / PICO_RATE
            if inst == INST_NOISE:
                wave_, new.noise = _noise(freq, key, phi, filters, new.noise, self.rng)
            else:
                wave_ = _waveform(inst, phi, filters) + _detune_wave(inst, phi, filters)
            value = np.clip(wave_ * vol, -1.0, 1.0)
            new.phi = float(phi[-1])
            new.freq, new.vol = float(freq[-1]), float(vol[-1])

        if self.fade > 0.0:  # crossfade a partir do estado anterior, que continua tocando
            m = min(n, int(math.ceil(self.fade / FADE_STEP)))
            w = self.fade - FADE_STEP * np.arange(m)
            fs = self.fade_synth
            if fs.vol > 0:
                fphi = fs.phi + np.arange(1, m + 1) * fs.freq / PICO_RATE
                if fs.inst == INST_NOISE:
                    fw, fs.noise = _noise(np.full(m, fs.freq), fs.key, fphi, fs.filters, fs.noise, self.rng)
                else:
                    fw = _waveform(fs.inst, fphi, fs.filters) + _detune_wave(fs.inst, fphi, fs.filters)
                fval = np.clip(fw * fs.vol, -1.0, 1.0)
                fs.phi = float(fphi[-1])
            else:
                fval = 0.0
            value[:m] = value[:m] * (1.0 - w) + fval * w
            self.fade -= FADE_STEP * m

        self.out[pos:pos + n] = value
        self.last = new

    def silence(self, pos: int, n: int):
        self.segment(pos, np.zeros(n), np.zeros(n), 0, 0, 0)

    # -- um pattern deste canal --------------------------------------------------------------
    def pattern(self, pos: int, n_samples: int, ch: Dict[str, Any], block: Optional[Dict[str, Any]],
                filters: int, default_speed: int):
        notes = ch.get("notes_unrolled") or []
        if not ch.get("active", False) or not notes:
            self.silence(pos, n_samples)
            return
        speed = max(1, int(ch.get("speed") or default_speed or 16))
        step = speed * PICO_SAMPLES_PER_TICK
        reverb, dampen = (filters // 24) % 3, (filters // 72) % 3
        self.reverb[pos:pos + n_samples] = reverb
        self.dampen[pos:pos + n_samples] = dampen

        src = (block or {}).get("notes") or notes  # notas do SFX (para os grupos do arpejo)
        index = _sfx_indices(len(notes), block) if block else list(range(len(notes)))
        prev_key, prev_vol = 24, 0.0  # valores iniciais do zepto8 ao disparar um SFX
        for j, n in enumerate(notes):
            start = j * step
            if start >= n_samples:
                break
            length = min(step, n_samples - start)
            key, vol, inst, fx = (int(n.get("pitch", 0)), int(n.get("volume", 0)) / 7.0,
                                  int(n.get("instrument", 0)) % 8, int(n.get("effect", 0)))
            if vol <= 0:
                self.silence(pos + start, length)
            else:
                k = np.arange(length, dtype=np.float64)
                u = k / step  # posição dentro da nota, 0..1
                secs = (start + k) / PICO_RATE  # tempo desde o início do SFX
                freq = np.full(length, pitch_to_freq(key))
                amp = np.full(length, vol)
                if fx == FX_SLIDE:
                    freq = pitch_to_freq(prev_key) + (freq - pitch_to_freq(prev_key)) * u
                    if prev_vol > 0:
                        amp = prev_vol + (amp - prev_vol) * u
                elif fx == FX_VIBRATO:
                    t = np.abs(np.mod(7.5 * secs, 1.0) - 0.5) - 0.25
                    freq = freq * (1.0 + (SEMITONE - 1.0) * t)
                elif fx == FX_DROP:
                    freq = freq * (1.0 - u)
                elif fx == FX_FADE_IN:
                    amp = amp * u
                elif fx == FX_FADE_OUT:
                    amp = amp * (1.0 - u)
                if fx in (FX_ARP_FAST, FX_ARP_SLOW):
                    m = (32 if speed <= 8 else 16) // (4 if fx == FX_ARP_FAST else 8)
                    step_id = np.floor(m * 7.5 * secs).astype(np.int64)
                    group = (index[j] if index[j] >= 0 else j) & ~3
                    cuts = np.flatnonzero(np.diff(step_id)) + 1
                    for a, b in zip(np.r_[0, cuts], np.r_[cuts, length]):
                        arp = group | int(step_id[a] & 3)
                        arp_key = int(src[arp].get("pitch", 0)) if arp < len(src) else key
                        self.segment(pos + start + a, np.full(b - a, pitch_to_freq(arp_key)), amp[a:b],
                                     inst, key, filters)
                else:
                    self.segment(pos + start, freq, amp, inst, key, filters)
            prev_key, prev_vol = key, vol
        played = min(len(notes) * step, n_samples)
        if played < n_samples:
            self.silence(pos + played, n_samples - played)

    # -- reverb e dampen, depois de todas as notas --------------------------------------------
    def finish(self) -> np.ndarray:
        y = self.out
        if self.reverb.any():  # eco com realimentação de 2 ou 4 ticks
            g1 = 0.5 * (self.reverb == 1)
            g2 = 0.5 * (self.reverb == 2)
            d1, d2 = 2 * PICO_SAMPLES_PER_TICK, 4 * PICO_SAMPLES_PER_TICK
            for s in range(0, len(y), d1):
                e = min(s + d1, len(y))
                if s >= d1:
                    y[s:e] += g1[s:e] * y[s - d1:e - d1]
                if s >= d2:
                    y[s:e] += g2[s:e] * y[s - d2:e - d2]
        if self.dampen.any():  # passa-baixa suave (high shelf) em 2400 Hz / -6 dB ou 1000 Hz / -12 dB
            z1 = _biquad(_HIGHSHELF_DAMP1, y)
            y = np.where(self.dampen == 1, z1, y)
            z2 = _biquad(_HIGHSHELF_DAMP2, y)
            y = np.where(self.dampen == 2, z2, y)
        return np.clip(y, -0.99, 0.99)


def _sfx_indices(n_steps: int, block: Dict[str, Any]) -> List[int]:
    """Índice da nota do SFX em cada passo desenrolado (-1 = silêncio), como `unroll_sfx_block`."""
    ls = min(32, max(0, int(block.get("loop_start", 0))))
    le = min(32, max(0, int(block.get("loop_end", 0))))
    has_loop = le > ls and le > 0 and ls < 32
    early_stop = le == 0 and 0 < ls < 32
    out, step = [], 0
    for _ in range(n_steps):
        if early_stop and step >= ls:
            out.append(-1)
            continue
        if step >= 32:
            if not has_loop:
                out.append(-1)
                continue
            step = ls
        if has_loop and step >= le:
            step = ls
        out.append(step)
        step += 1
    return out


def _highshelf(freq: float, gain_db: float, q: float = 1.0):
    """Coeficientes (b, a) de um high shelf do audio EQ cookbook, a 22050 Hz."""
    w0 = 2.0 * math.pi * freq / PICO_RATE
    cosw0, a = math.cos(w0), 10.0 ** (gain_db / 40.0)
    alpha = math.sin(w0) / 2.0 * math.sqrt((a + 1.0 / a) * (1.0 / q - 1.0) + 2.0)
    sqra = 2.0 * math.sqrt(a) * alpha
    a0 = (a + 1) - (a - 1) * cosw0 + sqra
    a1 = 2 * ((a - 1) - (a + 1) * cosw0)
    a2 = (a + 1) - (a - 1) * cosw0 - sqra
    b0 = a * ((a + 1) + (a - 1) * cosw0 + sqra)
    b1 = -2 * a * ((a - 1) + (a + 1) * cosw0)
    b2 = a * ((a + 1) + (a - 1) * cosw0 - sqra)
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


_HIGHSHELF_DAMP1 = _highshelf(2400.0, -6.0)
_HIGHSHELF_DAMP2 = _highshelf(1000.0, -12.0)


def _biquad(coefs, x: np.ndarray) -> np.ndarray:
    b, a = coefs
    if _lfilter is not None:
        return _lfilter(b, a, x)
    y = np.empty_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for i, v in enumerate(x):
        out = b[0] * v + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        x2, x1, y2, y1 = x1, v, y1, out
        y[i] = out
    return y


# ----------------------------------------------------------------------------
# Música inteira
# ----------------------------------------------------------------------------
def _song_patterns(song: Dict[str, Any], loop_extra_repeats: int) -> List[Dict[str, Any]]:
    patterns = list(song.get("patterns", []))
    if loop_extra_repeats > 0 and song.get("end_reason") == "loop_end" and song.get("loop_start_pattern") is not None:
        loop_slice = [p for p in patterns if p.get("pattern_id", 0) >= song["loop_start_pattern"]]
        for _ in range(loop_extra_repeats):
            patterns.extend(loop_slice)
    return patterns


def _channel_filters(ch: Dict[str, Any], block: Optional[Dict[str, Any]]) -> int:
    if ch.get("filters") is not None:
        return int(ch["filters"])
    if block and block.get("filters") is not None:
        return int(block["filters"])
    return 0


def synthesize_song(song: Dict[str, Any], loop_extra_repeats: int = 0, seed: int = 0,
                    rate: int = SAMPLE_RATE) -> np.ndarray:
    """Áudio da música em float32 (-1..1), a 44100 Hz (ou 22050 Hz, a taxa nativa do PICO-8)."""
    patterns = _song_patterns(song, loop_extra_repeats)
    lengths = [max(0, int(p.get("pattern_ticks", 512))) * PICO_SAMPLES_PER_TICK for p in patterns]
    total = int(sum(lengths))
    if total == 0:
        return np.zeros(0, dtype=np.float32)
    rng = np.random.default_rng(seed)
    blocks = song.get("blocks_dictionary") or {}
    channels = [_Channel(total, rng) for _ in range(4)]
    pos = 0
    for pat, n in zip(patterns, lengths):
        if n == 0:
            continue
        chans = pat.get("channels", {})
        for c, state in enumerate(channels):
            ch = chans.get(f"ch{c + 1}", {}) or {}
            block = blocks.get(str(ch.get("block_id"))) if ch.get("block_id") is not None else None
            state.pattern(pos, n, ch, block, _channel_filters(ch, block), int(pat.get("ref_speed", 16) or 16))
        pos += n

    # cada canal vira int16 e os 4 se somam com saturação, como no PICO-8
    mix = np.zeros(total, dtype=np.float64)
    for state in channels:
        mix += np.trunc(32767.99 * state.finish())
    mix = np.clip(mix, -32767.9, 32767.9) / 32768.0
    if rate == PICO_RATE:
        return mix.astype(np.float32)
    if rate != SAMPLE_RATE:
        raise ValueError(f"rate must be {PICO_RATE} or {SAMPLE_RATE}")
    up = np.empty(2 * total, dtype=np.float64)  # interpolação linear 22050 -> 44100, como o player web
    up[0::2] = mix
    up[1:-1:2] = 0.5 * (mix[:-1] + mix[1:])
    up[-1] = mix[-1]
    return up.astype(np.float32)


def waveform(inst: int, phase, filters: int = 0) -> np.ndarray:
    """Forma de onda de um instrumento tonal (todos menos o 6, ruído) na fase dada em ciclos; para gráficos."""
    if inst == INST_NOISE:
        raise ValueError("o ruído não tem forma fixa: use synthesize_notes")
    phase = np.asarray(phase, dtype=np.float64)
    return _waveform(inst, phase, filters) + _detune_wave(inst, phase, filters)


def synthesize_notes(notes: List[Dict[str, int]], speed: int, filters: int = 0, seed: int = 0,
                     rate: int = SAMPLE_RATE) -> np.ndarray:
    """Um canal isolado: `notes` tocadas uma após a outra, `speed` ticks cada (para exemplos e testes)."""
    speed = max(1, int(speed))
    channel = {"active": True, "speed": speed, "notes_unrolled": list(notes), "filters": filters}
    silent = {"active": False, "speed": None, "notes_unrolled": []}
    pattern = {"pattern_id": 0, "ref_speed": speed, "pattern_ticks": speed * len(notes),
               "channels": {"ch1": channel, "ch2": silent, "ch3": silent, "ch4": silent}}
    return synthesize_song({"patterns": [pattern]}, seed=seed, rate=rate)


def synthesize_song_to_wav(song: Dict[str, Any], out_wav_path: str, loop_extra_repeats: int = 0,
                           seed: int = 0) -> float:
    """
    Renderiza um dicionário `song` (formato `songs.json`) para um `.wav` de 44,1 kHz.
    Retorna a duração total em segundos.
    """
    audio = synthesize_song(song, loop_extra_repeats=loop_extra_repeats, seed=seed)
    if audio.size == 0:
        return 0.0
    pcm = np.clip(np.round(audio * 32767.0), -32767, 32767).astype("<i2")
    os.makedirs(os.path.dirname(os.path.abspath(out_wav_path)), exist_ok=True)
    with wave.open(out_wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return len(audio) / float(SAMPLE_RATE)
