"""
Sintetizador nativo de áudio do PICO-8 para o projeto final (DCC831 - UFMG).
Converte dicionários de músicas (no formato `songs.json` ou decodificados de tokens)
diretamente em arquivos de áudio `.wav` (44.1 kHz, 16-bit PCM mono), sem passar por MIDI.
"""

import array
import math
import os
import wave
from typing import Dict, Any, List

SAMPLE_RATE = 44100
SAMPLES_PER_TICK = 366  # 183 amostras @ 22050 Hz = 366 amostras @ 44100 Hz (~120.49 Hz)


def pitch_to_freq(pitch: float) -> float:
    """Converte pitch do PICO-8 (0..63, onde 24 = C2 = 130.81 Hz) para frequência em Hz."""
    return 440.0 * (2.0 ** ((pitch - 45.0) / 12.0))


def render_oscillator(phase: float, inst: int) -> float:
    """
    Gera amostra das 8 formas de onda nativas do PICO-8 (0..7) a partir da fase [0.0, 1.0).
    """
    inst_base = inst % 8
    p = phase % 1.0

    if inst_base == 0:  # Triangle
        return (1.0 - abs(4.0 * p - 2.0)) * 0.7
    elif inst_base == 1:  # Tilted Saw
        a = 0.875
        return (2.0 * (p / a) - 1.0 if p < a else 2.0 * (1.0 - (p - a) / (1.0 - a)) - 1.0) * 0.7
    elif inst_base == 2:  # Saw
        return (2.0 * p - 1.0) * 0.65
    elif inst_base == 3:  # Square (50% duty)
        return (1.0 if p < 0.5 else -1.0) * 0.5
    elif inst_base == 4:  # Pulse (~31.6% duty)
        return (1.0 if p < 0.316 else -1.0) * 0.5
    elif inst_base == 5:  # Organ (fundamental + sub-oitava)
        p2 = (phase * 0.5) % 1.0
        t1 = 1.0 - abs(4.0 * p - 2.0)
        t2 = 1.0 - abs(4.0 * p2 - 2.0)
        return (t1 * 0.6 + t2 * 0.4) * 0.7
    elif inst_base == 7:  # Phaser (dois triângulos desafinados na razão 109/110)
        p2 = (phase * (109.0 / 110.0)) % 1.0
        t1 = 1.0 - abs(4.0 * p - 2.0)
        t2 = 1.0 - abs(4.0 * p2 - 2.0)
        return (t1 + t2) * 0.45
    return 0.0


def synthesize_unrolled_channel(notes: List[Dict[str, int]], speed: int, target_samples: int) -> List[float]:
    """
    Sintetiza uma trilha de notas desenroladas (32 steps ou mais) mantendo fase contínua.
    """
    speed = max(1, int(speed))
    step_samples = speed * SAMPLES_PER_TICK
    out: List[float] = []

    phase = 0.0
    prev_pitch = -1
    prev_inst = -1
    prev_vol = 0

    noise_val = 0.0
    noise_lp = 0.0
    lfsr = 0x7FFF

    for idx, n in enumerate(notes):
        if len(out) >= target_samples:
            break

        pitch = int(n.get("pitch", 0))
        inst = int(n.get("instrument", 0))
        vol = int(n.get("volume", 0))
        effect = int(n.get("effect", 0))

        next_vol = int(notes[idx + 1].get("volume", 0)) if (idx + 1) < len(notes) else 0
        next_pitch = int(notes[idx + 1].get("pitch", 0)) if (idx + 1) < len(notes) else -1
        next_inst = int(notes[idx + 1].get("instrument", 0)) if (idx + 1) < len(notes) else -1

        n_samp = min(step_samples, target_samples - len(out))
        if vol <= 0:
            out.extend([0.0] * n_samp)
            prev_vol = 0
            continue

        base_freq = pitch_to_freq(pitch)
        prev_freq = pitch_to_freq(prev_pitch) if (prev_vol > 0 and prev_pitch >= 0) else base_freq
        amp_base = vol / 7.0

        same_prev = (prev_vol > 0 and prev_pitch == pitch and prev_inst == inst and effect not in (4, 5))
        same_next = (next_vol > 0 and next_pitch == pitch and next_inst == inst and effect not in (3, 4, 5))
        attack_len = min(int(0.002 * SAMPLE_RATE), max(1, step_samples // 10))
        release_len = min(int(0.002 * SAMPLE_RATE), max(1, step_samples // 10))
        inv_sr = 1.0 / SAMPLE_RATE
        inst_base = inst % 8

        for i in range(n_samp):
            u = i / float(step_samples)
            freq = base_freq
            amp = amp_base

            if effect == 1:  # Slide
                freq = prev_freq + (base_freq - prev_freq) * u
            elif effect == 2:  # Vibrato
                freq = base_freq * (1.0 + 0.015 * math.sin(2.0 * math.pi * 7.5 * (i * inv_sr)))
            elif effect == 3:  # Drop
                freq = base_freq * (1.0 - 0.85 * u)
            elif effect == 4:  # Fade In
                amp = amp_base * u
            elif effect == 5:  # Fade Out
                amp = amp_base * (1.0 - u)
            elif effect == 6:  # Arpeggio Fast
                arp_idx = (i // (SAMPLES_PER_TICK * 2)) % 2
                if arp_idx == 1 and next_vol > 0 and next_pitch >= 0:
                    freq = pitch_to_freq(next_pitch)
            elif effect == 7:  # Arpeggio Slow
                arp_idx = (i // (SAMPLES_PER_TICK * 4)) % 2
                if arp_idx == 1 and next_vol > 0 and next_pitch >= 0:
                    freq = pitch_to_freq(next_pitch)

            if not same_prev and i < attack_len and effect != 4:
                amp *= (i / float(attack_len))
            elif same_prev and prev_vol != vol and i < attack_len and effect != 4:
                prev_amp = prev_vol / 7.0
                amp = prev_amp + (amp - prev_amp) * (i / float(attack_len))

            if not same_next and (step_samples - i) < release_len and effect != 5:
                amp *= ((step_samples - i) / float(release_len))

            phase += freq * inv_sr
            if inst_base == 6:  # Noise com filtro passa-baixa rastreando a frequência da nota
                if phase >= 0.5:
                    phase -= 0.5
                    bit = ((lfsr >> 0) ^ (lfsr >> 1)) & 1
                    lfsr = (lfsr >> 1) | (bit << 14)
                    noise_val = 1.0 if (lfsr & 1) else -1.0
                alpha = min(0.85, max(0.02, (2.0 * math.pi * freq) * inv_sr))
                noise_lp += alpha * (noise_val - noise_lp)
                s = noise_lp * 0.8 * amp
            else:
                s = render_oscillator(phase, inst) * amp

            out.append(s)

        prev_pitch = pitch
        prev_inst = inst
        prev_vol = vol

    if len(out) < target_samples:
        out.extend([0.0] * (target_samples - len(out)))
    return out


def synthesize_song_to_wav(song: Dict[str, Any], out_wav_path: str, loop_extra_repeats: int = 0) -> float:
    """
    Renderiza um dicionário `song` (formato `songs.json`) para um arquivo `.wav`.
    Retorna a duração total em segundos.
    """
    patterns = list(song.get("patterns", []))
    if loop_extra_repeats > 0 and song.get("end_reason") == "loop_end" and song.get("loop_start_pattern") is not None:
        loop_start_pat = song["loop_start_pattern"]
        loop_slice = [p for p in patterns if p.get("pattern_id", 0) >= loop_start_pat]
        if loop_slice:
            for _ in range(loop_extra_repeats):
                patterns.extend(loop_slice)

    full_mix: List[float] = []

    for pat in patterns:
        pat_ticks = int(pat.get("pattern_ticks", 512))
        target_samples = pat_ticks * SAMPLES_PER_TICK
        pat_buf = [0.0] * target_samples
        channels = pat.get("channels", {})
        active_count = 0

        for ch_key in ("ch1", "ch2", "ch3", "ch4"):
            ch_info = channels.get(ch_key, {})
            if not ch_info.get("active", False):
                continue
            notes = ch_info.get("notes_unrolled", [])
            if not notes:
                continue
            ch_speed = int(ch_info.get("speed", pat.get("ref_speed", 16)))
            ch_audio = synthesize_unrolled_channel(notes, ch_speed, target_samples)
            active_count += 1
            for i in range(target_samples):
                pat_buf[i] += ch_audio[i]

        if active_count > 0:
            for i in range(target_samples):
                pat_buf[i] *= 0.28

        full_mix.extend(pat_buf)

    if not full_mix:
        return 0.0

    peak = max(abs(x) for x in full_mix)
    scale = (0.92 / peak) if peak > 0.92 else 1.0
    pcm_samples = array.array(
        "h",
        (max(-32767, min(32767, int(x * scale * 32767.0))) for x in full_mix)
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_wav_path)), exist_ok=True)
    with wave.open(out_wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_samples.tobytes())

    return len(full_mix) / float(SAMPLE_RATE)

