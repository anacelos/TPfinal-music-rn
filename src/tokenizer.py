"""
Tokenizadores simbólicos 100% reversíveis (lossless round-trip) para músicas do PICO-8 (`songs.json`).

Implementa três representações fundamentadas na literatura de geração musical simbólica:
1. `nlm` (NES Music Language / LakhNES - Donahue et al., ISMIR 2019):
   Representação orientada a eventos de mudança de estado por canal (`CHc_NOTEON_p`,
   `CHc_NOTEOFF`, `CHc_INST_i`, `CHc_VOL_v`, `CHc_FX_f`) intercalados por tokens de
   avanço temporal em ticks de hardware (`WT_t`).
2. `grid_unrolled` (REMI+ Tracker Grid - adaptado de Huang & Yang, ACM MM 2020):
   Representação de grade polifônica por pattern e canal, posicionando notas ativas
   via tokens explícitos de posição (`STEP_k`, `PITCH_p`, `INST_i`, `VOL_v`, `[FX_f]`),
   totalmente desenrolada (sem ponteiros de reutilização de blocos).
3. `grid_block` (Structured Tracker Grid com Ponteiros de Bloco):
   Mesma gramática local do `grid_unrolled`, porém preservando a estrutura macro de
   reutilização de blocos (`blocks_dictionary` / `is_repeat` do `songs.json`): na
   primeira ocorrência de um bloco emite `CHc_DEF_BLOCK_b ... END_BLOCK` com os 32
   steps canônicos do SFX e seus marcadores de loop interno (`BLK_LSTART_ls`, `BLK_LEND_le`),
   e nas repetições subsequentes emite um único token `CHc_USE_BLOCK_b`.
"""

import math
from typing import Dict, List, Any, Optional, Tuple

MAX_WAIT_TICKS = 64
MAX_STEPS = 512
MAX_BLOCKS = 64
MAX_SPEED = 255


def unroll_sfx_block(
    notes_32: List[Dict[str, int]],
    speed: int,
    loop_start: int,
    loop_end: int,
    target_ticks: int,
) -> List[Dict[str, int]]:
    """
    Mesma função exata (`unroll_channel_notes`) de `compose_music.py` para desenrolar um bloco
    canônico de 32 notas para `target_ticks` ticks do pattern.
    """
    if not notes_32 or target_ticks <= 0:
        return []
    speed = max(1, int(speed))
    target_steps = max(1, int(round(float(target_ticks) / speed)))
    ls = min(32, max(0, int(loop_start)))
    le = min(32, max(0, int(loop_end)))
    has_loop = (le > ls and le > 0 and ls < 32)
    early_stop = (le == 0 and 0 < ls < 32)

    unrolled: List[Dict[str, int]] = []
    step = 0
    silent_note = {"pitch": 0, "instrument": 0, "volume": 0, "effect": 0}

    for _ in range(target_steps):
        if early_stop and step >= ls:
            unrolled.append(dict(silent_note))
            continue
        if step >= 32:
            if has_loop:
                step = ls
            else:
                unrolled.append(dict(silent_note))
                continue
        if has_loop and step >= le:
            step = ls

        if step < len(notes_32):
            unrolled.append(dict(notes_32[step]))
        else:
            unrolled.append(dict(silent_note))
        step += 1

    return unrolled


class Pico8Tokenizer:
    """
    Tokenizador reversível unificado para `songs.json` do PICO-8.
    Suporta `mode` in {"nlm", "grid_unrolled", "grid_block"}.
    """

    def __init__(
        self,
        mode: str = "nlm",
        unroll_song_loop: bool = False,
        include_fx_zero: bool = False,
    ):
        if mode not in ("nlm", "grid_unrolled", "grid_block"):
            raise ValueError(f"Modo inválido: {mode}. Escolha entre 'nlm', 'grid_unrolled', 'grid_block'.")
        self.mode = mode
        self.unroll_song_loop = unroll_song_loop
        self.include_fx_zero = include_fx_zero

        self.id_to_token: List[str] = []
        self.token_to_id: Dict[str, int] = {}
        self._build_vocab()

    def _add_token(self, tok: str) -> None:
        if tok not in self.token_to_id:
            self.token_to_id[tok] = len(self.id_to_token)
            self.id_to_token.append(tok)

    def _build_vocab(self) -> None:
        # Tokens especiais estruturais comuns
        for special in ("PAD", "BOS", "EOS", "PATTERN_START", "LOOP_START", "LOOP_END"):
            self._add_token(special)

        for s in range(1, MAX_SPEED + 1):
            self._add_token(f"REF_SPEED_{s}")
        for st in range(1, 33):
            self._add_token(f"REF_STEPS_{st}")

        if self.mode == "nlm":
            for c in range(1, 5):
                for p in range(64):
                    self._add_token(f"CH{c}_NOTEON_{p}")
                self._add_token(f"CH{c}_NOTEOFF")
                for inst in range(8):
                    self._add_token(f"CH{c}_INST_{inst}")
                for vol in range(1, 8):
                    self._add_token(f"CH{c}_VOL_{vol}")
                for fx in range(8):
                    self._add_token(f"CH{c}_FX_{fx}")
            for wt in range(1, MAX_WAIT_TICKS + 1):
                self._add_token(f"WT_{wt}")
        else:
            # Vocabulário para grid_unrolled e grid_block
            for s in range(1, MAX_SPEED + 1):
                self._add_token(f"SPEED_{s}")
            for c in range(1, 5):
                self._add_token(f"CH{c}_START")
                if self.mode == "grid_block":
                    for b in range(MAX_BLOCKS):
                        self._add_token(f"CH{c}_DEF_BLOCK_{b}")
                        self._add_token(f"CH{c}_USE_BLOCK_{b}")
            if self.mode == "grid_block":
                for ls in range(33):
                    self._add_token(f"BLK_LSTART_{ls}")
                    self._add_token(f"BLK_LEND_{ls}")
                self._add_token("END_BLOCK")

            max_st = 32 if self.mode == "grid_block" else MAX_STEPS
            for step in range(max_st):
                self._add_token(f"STEP_{step}")
            for p in range(64):
                self._add_token(f"PITCH_{p}")
            for inst in range(8):
                self._add_token(f"INST_{inst}")
            for vol in range(1, 8):
                self._add_token(f"VOL_{vol}")
            for fx in range(8):
                self._add_token(f"FX_{fx}")

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["PAD"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["BOS"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["EOS"]

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id[t] for t in tokens if t in self.token_to_id]

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.id_to_token[i] for i in ids if 0 <= i < len(self.id_to_token)]

    def _prepare_patterns(self, song: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[int], str]:
        patterns = [p for p in song.get("patterns", []) if int(p.get("pattern_ticks", 0)) > 0]
        loop_start = song.get("loop_start_pattern")
        end_reason = song.get("end_reason", "stop")

        if self.unroll_song_loop and end_reason == "loop_end" and loop_start is not None:
            loop_slice = [p for p in patterns if p.get("pattern_id", 0) >= loop_start]
            if loop_slice:
                patterns = patterns + loop_slice
        return patterns, loop_start, end_reason

    # -------------------------------------------------------------------------
    # Codificação (song_dict -> List[str])
    # -------------------------------------------------------------------------
    def encode_song(self, song: Dict[str, Any]) -> List[str]:
        if self.mode == "nlm":
            return self._encode_nlm(song)
        return self._encode_grid(song)

    def encode_song_to_ids(self, song: Dict[str, Any]) -> List[int]:
        return self.tokens_to_ids(self.encode_song(song))

    def _emit_wait(self, tokens: List[str], dt: int) -> None:
        while dt > 0:
            step = min(dt, MAX_WAIT_TICKS)
            tokens.append(f"WT_{step}")
            dt -= step

    def _encode_nlm(self, song: Dict[str, Any]) -> List[str]:
        patterns, loop_start, end_reason = self._prepare_patterns(song)
        tokens: List[str] = ["BOS"]

        # Estado corrente por canal (1..4)
        ch_active = {c: False for c in range(1, 5)}
        ch_pitch = {c: -1 for c in range(1, 5)}
        ch_inst = {c: -1 for c in range(1, 5)}
        ch_vol = {c: -1 for c in range(1, 5)}
        ch_fx = {c: 0 for c in range(1, 5)}

        for pat_idx, pat in enumerate(patterns):
            if not self.unroll_song_loop and loop_start is not None and pat.get("pattern_id") == loop_start and pat_idx < len(song.get("patterns", [])):
                tokens.append("LOOP_START")

            tokens.append("PATTERN_START")
            ref_speed = max(1, min(MAX_SPEED, int(pat.get("ref_speed", 16))))
            tokens.append(f"REF_SPEED_{ref_speed}")

            pat_ticks = int(pat.get("pattern_ticks", 32 * ref_speed))
            ref_steps = max(1, min(32, pat_ticks // ref_speed))
            if ref_steps != 32:
                tokens.append(f"REF_STEPS_{ref_steps}")

            channels = pat.get("channels", {})

            # Coleta eventos por tick t em [0, pat_ticks)
            events_by_tick: Dict[int, List[str]] = {}

            for c in range(1, 5):
                ch_info = channels.get(f"ch{c}", {})
                if not ch_info.get("active", False) or not ch_info.get("notes_unrolled"):
                    if ch_active[c]:
                        events_by_tick.setdefault(0, []).append(f"CH{c}_NOTEOFF")
                        ch_active[c] = False
                        ch_pitch[c] = -1
                    continue

                ch_speed = max(1, int(ch_info.get("speed", ref_speed)))
                notes = ch_info.get("notes_unrolled", [])

                for step_idx, n in enumerate(notes):
                    tick = step_idx * ch_speed
                    if tick >= pat_ticks:
                        break

                    pitch = max(0, min(63, int(n.get("pitch", 0))))
                    inst = int(n.get("instrument", 0)) % 8
                    vol = max(0, min(7, int(n.get("volume", 0))))
                    fx = max(0, min(7, int(n.get("effect", 0))))

                    if vol == 0:
                        if ch_active[c]:
                            events_by_tick.setdefault(tick, []).append(f"CH{c}_NOTEOFF")
                            ch_active[c] = False
                            ch_pitch[c] = -1
                    else:
                        # Verifica se é sustentação contínua idêntica ao step anterior
                        is_sustained = (
                            ch_active[c]
                            and ch_pitch[c] == pitch
                            and ch_inst[c] == inst
                            and ch_vol[c] == vol
                            and ch_fx[c] == fx
                            and fx not in (3, 4, 5)
                        )
                        if not is_sustained:
                            ev_list = events_by_tick.setdefault(tick, [])
                            if ch_inst[c] != inst:
                                ev_list.append(f"CH{c}_INST_{inst}")
                                ch_inst[c] = inst
                            if ch_vol[c] != vol:
                                ev_list.append(f"CH{c}_VOL_{vol}")
                                ch_vol[c] = vol
                            if ch_fx[c] != fx:
                                ev_list.append(f"CH{c}_FX_{fx}")
                                ch_fx[c] = fx
                            ev_list.append(f"CH{c}_NOTEON_{pitch}")
                            ch_active[c] = True
                            ch_pitch[c] = pitch

                # Se a lista de notas encerrou antes de pat_ticks, silencia o canal em end_tick
                end_tick = len(notes) * ch_speed
                if end_tick < pat_ticks and ch_active[c]:
                    events_by_tick.setdefault(end_tick, []).append(f"CH{c}_NOTEOFF")
                    ch_active[c] = False
                    ch_pitch[c] = -1

            cur_tick = 0
            for tick in sorted(events_by_tick.keys()):
                if tick > cur_tick:
                    self._emit_wait(tokens, tick - cur_tick)
                    cur_tick = tick
                tokens.extend(events_by_tick[tick])

            if pat_ticks > cur_tick:
                self._emit_wait(tokens, pat_ticks - cur_tick)

        if not self.unroll_song_loop and end_reason == "loop_end":
            tokens.append("LOOP_END")

        tokens.append("EOS")
        return tokens

    def _encode_steps(self, notes: List[Dict[str, int]], tokens: List[str], max_st: int) -> None:
        for step_idx, n in enumerate(notes[:max_st]):
            vol = max(0, min(7, int(n.get("volume", 0))))
            if vol <= 0:
                continue
            pitch = max(0, min(63, int(n.get("pitch", 0))))
            inst = int(n.get("instrument", 0)) % 8
            fx = max(0, min(7, int(n.get("effect", 0))))

            tokens.append(f"STEP_{step_idx}")
            tokens.append(f"PITCH_{pitch}")
            tokens.append(f"INST_{inst}")
            tokens.append(f"VOL_{vol}")
            if self.include_fx_zero or fx > 0:
                tokens.append(f"FX_{fx}")

    def _encode_grid(self, song: Dict[str, Any]) -> List[str]:
        patterns, loop_start, end_reason = self._prepare_patterns(song)
        blocks_dict = song.get("blocks_dictionary", {})
        tokens: List[str] = ["BOS"]
        defined_blocks = set()

        for pat_idx, pat in enumerate(patterns):
            if not self.unroll_song_loop and loop_start is not None and pat.get("pattern_id") == loop_start and pat_idx < len(song.get("patterns", [])):
                tokens.append("LOOP_START")

            tokens.append("PATTERN_START")
            ref_speed = max(1, min(MAX_SPEED, int(pat.get("ref_speed", 16))))
            tokens.append(f"REF_SPEED_{ref_speed}")
            pat_ticks = int(pat.get("pattern_ticks", 32 * ref_speed))
            ref_steps = max(1, min(32, pat_ticks // ref_speed))
            if ref_steps != 32:
                tokens.append(f"REF_STEPS_{ref_steps}")

            channels = pat.get("channels", {})
            for c in range(1, 5):
                ch_info = channels.get(f"ch{c}", {})
                if not ch_info.get("active", False) or not ch_info.get("notes_unrolled"):
                    continue

                ch_speed = max(1, min(MAX_SPEED, int(ch_info.get("speed", ref_speed))))
                notes = ch_info.get("notes_unrolled", [])
                block_id_raw = ch_info.get("block_id")
                block_id = int(block_id_raw) if (block_id_raw is not None and str(block_id_raw).isdigit()) else -1

                if self.mode == "grid_block" and 0 <= block_id < MAX_BLOCKS and str(block_id) in blocks_dict:
                    if block_id in defined_blocks:
                        tokens.append(f"CH{c}_USE_BLOCK_{block_id}")
                    else:
                        defined_blocks.add(block_id)
                        b_meta = blocks_dict[str(block_id)]
                        b_speed = max(1, min(MAX_SPEED, int(b_meta.get("speed", ch_speed))))
                        b_ls = max(0, min(32, int(b_meta.get("loop_start", 0))))
                        b_le = max(0, min(32, int(b_meta.get("loop_end", 0))))
                        b_notes = b_meta.get("notes", notes[:32])

                        tokens.append(f"CH{c}_DEF_BLOCK_{block_id}")
                        tokens.append(f"SPEED_{b_speed}")
                        if b_ls > 0:
                            tokens.append(f"BLK_LSTART_{b_ls}")
                        if b_le > 0:
                            tokens.append(f"BLK_LEND_{b_le}")
                        self._encode_steps(b_notes, tokens, max_st=32)
                        tokens.append("END_BLOCK")
                else:
                    tokens.append(f"CH{c}_START")
                    tokens.append(f"SPEED_{ch_speed}")
                    self._encode_steps(notes, tokens, max_st=MAX_STEPS)

        if not self.unroll_song_loop and end_reason == "loop_end":
            tokens.append("LOOP_END")

        tokens.append("EOS")
        return tokens

    # -------------------------------------------------------------------------
    # Decodificação (List[str] -> song_dict compatível com `songs.json` e `synth.py`)
    # -------------------------------------------------------------------------
    def decode_tokens(self, tokens: List[str]) -> Dict[str, Any]:
        if self.mode == "nlm":
            return self._decode_nlm(tokens)
        return self._decode_grid(tokens)

    def decode_ids(self, ids: List[int]) -> Dict[str, Any]:
        return self.decode_tokens(self.ids_to_tokens(ids))

    def _decode_nlm(self, tokens: List[str]) -> Dict[str, Any]:
        patterns: List[Dict[str, Any]] = []
        loop_start_pattern: Optional[int] = None
        end_reason = "stop"
        next_pat_is_loop_start = False

        ch_active = {c: False for c in range(1, 5)}
        ch_pitch = {c: 24 for c in range(1, 5)}
        ch_inst = {c: 0 for c in range(1, 5)}
        ch_vol = {c: 5 for c in range(1, 5)}
        ch_fx = {c: 0 for c in range(1, 5)}

        cur_pat_ticks: List[Dict[int, Dict[str, int]]] = []
        cur_ref_speed = 16
        in_pattern = False

        def flush_pattern() -> None:
            nonlocal cur_pat_ticks, in_pattern
            if not in_pattern or not cur_pat_ticks:
                return
            pat_id = len(patterns)
            total_ticks = len(cur_pat_ticks)
            step_size = max(1, cur_ref_speed)
            channels_dict: Dict[str, Any] = {}
            for c in range(1, 5):
                change_ticks = [
                    t_idx for t_idx in range(1, total_ticks)
                    if cur_pat_ticks[t_idx][c] != cur_pat_ticks[t_idx - 1][c]
                ]
                ch_speed = step_size if total_ticks % step_size == 0 else math.gcd(total_ticks, step_size)
                if any(t % ch_speed != 0 for t in change_ticks):
                    g = total_ticks
                    for t in change_ticks:
                        g = math.gcd(g, t)
                    ch_speed = max(1, g)

                notes_unrolled: List[Dict[str, int]] = []
                has_sound = False
                for t_idx in range(0, total_ticks, ch_speed):
                    st = cur_pat_ticks[t_idx][c]
                    if st["volume"] > 0:
                        has_sound = True
                    notes_unrolled.append(dict(st))
                channels_dict[f"ch{c}"] = {
                    "active": has_sound,
                    "speed": ch_speed if has_sound else None,
                    "notes_unrolled": notes_unrolled if has_sound else [],
                }

            patterns.append({
                "pattern_id": pat_id,
                "ref_speed": cur_ref_speed,
                "pattern_ticks": total_ticks,
                "channels": channels_dict,
            })
            cur_pat_ticks = []
            in_pattern = False

        for tok in tokens:
            if tok in ("BOS", "PAD"):
                continue
            elif tok == "EOS":
                break
            elif tok == "LOOP_START":
                next_pat_is_loop_start = True
            elif tok == "LOOP_END":
                end_reason = "loop_end"
            elif tok == "PATTERN_START":
                flush_pattern()
                if next_pat_is_loop_start:
                    loop_start_pattern = len(patterns)
                    next_pat_is_loop_start = False
                in_pattern = True
                cur_ref_speed = 16
                cur_pat_ticks = []
            elif tok.startswith("REF_SPEED_"):
                cur_ref_speed = int(tok.split("_")[-1])
            elif tok.startswith("CH"):
                parts = tok.split("_")
                c = int(parts[0][2:])
                kind = parts[1]
                if kind == "INST":
                    ch_inst[c] = int(parts[2])
                elif kind == "VOL":
                    ch_vol[c] = int(parts[2])
                elif kind == "FX":
                    ch_fx[c] = int(parts[2])
                elif kind == "NOTEON":
                    ch_pitch[c] = int(parts[2])
                    ch_active[c] = True
                elif kind == "NOTEOFF":
                    ch_active[c] = False
            elif tok.startswith("WT_"):
                if not in_pattern:
                    in_pattern = True
                dt = int(tok.split("_")[1])
                for _ in range(dt):
                    snap: Dict[int, Dict[str, int]] = {}
                    for c in range(1, 5):
                        if ch_active[c]:
                            snap[c] = {
                                "pitch": ch_pitch[c],
                                "instrument": ch_inst[c],
                                "volume": ch_vol[c],
                                "effect": ch_fx[c],
                            }
                        else:
                            snap[c] = {"pitch": 0, "instrument": 0, "volume": 0, "effect": 0}
                    cur_pat_ticks.append(snap)

        flush_pattern()
        return {
            "start_pattern": 0,
            "end_pattern": max(0, len(patterns) - 1),
            "loop_start_pattern": loop_start_pattern,
            "end_reason": end_reason,
            "num_patterns": len(patterns),
            "patterns": patterns,
        }

    def _decode_grid(self, tokens: List[str]) -> Dict[str, Any]:
        patterns: List[Dict[str, Any]] = []
        blocks_memory: Dict[int, Tuple[int, int, int, List[Dict[str, int]]]] = {}
        loop_start_pattern: Optional[int] = None
        end_reason = "stop"
        next_pat_is_loop_start = False

        cur_pat: Optional[Dict[str, Any]] = None
        cur_ref_steps: int = 32
        cur_ch: Optional[int] = None
        cur_block_def: Optional[int] = None
        cur_block_ls: int = 0
        cur_block_le: int = 0
        cur_step: Optional[int] = None

        def default_empty_steps(num_steps: int = 32) -> List[Dict[str, int]]:
            return [{"pitch": 0, "instrument": 0, "volume": 0, "effect": 0} for _ in range(num_steps)]

        def finalize_channel_block() -> None:
            nonlocal cur_block_def, cur_block_ls, cur_block_le
            if cur_pat is not None and cur_ch is not None and cur_block_def is not None:
                ch_obj = cur_pat["channels"][f"ch{cur_ch}"]
                b_speed = int(ch_obj["speed"] or cur_pat["ref_speed"])
                canonical_32 = [dict(n) for n in ch_obj["notes_unrolled"][:32]]
                while len(canonical_32) < 32:
                    canonical_32.append({"pitch": 0, "instrument": 0, "volume": 0, "effect": 0})
                blocks_memory[cur_block_def] = (b_speed, cur_block_ls, cur_block_le, canonical_32)
                pat_ticks = cur_ref_steps * int(cur_pat["ref_speed"])
                ch_obj["notes_unrolled"] = unroll_sfx_block(
                    canonical_32, b_speed, cur_block_ls, cur_block_le, pat_ticks
                )
            cur_block_def = None
            cur_block_ls = 0
            cur_block_le = 0

        def flush_pattern() -> None:
            nonlocal cur_pat, cur_ch, cur_step
            finalize_channel_block()
            if cur_pat is not None:
                ref_s = int(cur_pat.get("ref_speed", 16))
                cur_pat["pattern_ticks"] = cur_ref_steps * ref_s
                patterns.append(cur_pat)
            cur_pat = None
            cur_ch = None
            cur_step = None

        for tok in tokens:
            if tok in ("BOS", "PAD"):
                continue
            elif tok == "EOS":
                break
            elif tok == "LOOP_START":
                next_pat_is_loop_start = True
            elif tok == "LOOP_END":
                end_reason = "loop_end"
            elif tok == "PATTERN_START":
                flush_pattern()
                pat_id = len(patterns)
                if next_pat_is_loop_start:
                    loop_start_pattern = pat_id
                    next_pat_is_loop_start = False
                cur_ref_steps = 32
                cur_pat = {
                    "pattern_id": pat_id,
                    "ref_speed": 16,
                    "pattern_ticks": 512,
                    "channels": {
                        f"ch{c}": {"active": False, "speed": None, "notes_unrolled": []}
                        for c in range(1, 5)
                    },
                }
            elif tok.startswith("REF_SPEED_") and cur_pat is not None:
                cur_pat["ref_speed"] = int(tok.split("_")[-1])
                cur_pat["pattern_ticks"] = cur_ref_steps * cur_pat["ref_speed"]
            elif tok.startswith("REF_STEPS_") and cur_pat is not None:
                cur_ref_steps = int(tok.split("_")[-1])
                cur_pat["pattern_ticks"] = cur_ref_steps * cur_pat["ref_speed"]
            elif tok.startswith("CH") and cur_pat is not None:
                finalize_channel_block()
                cur_step = None
                parts = tok.split("_")
                c = int(parts[0][2:])
                cur_ch = c
                if parts[1] == "START":
                    cur_pat["channels"][f"ch{c}"] = {
                        "active": True,
                        "speed": cur_pat["ref_speed"],
                        "notes_unrolled": default_empty_steps(cur_ref_steps),
                    }
                elif parts[1] == "DEF" and parts[2] == "BLOCK":
                    b_id = int(parts[3])
                    cur_block_def = b_id
                    cur_block_ls = 0
                    cur_block_le = 0
                    cur_pat["channels"][f"ch{c}"] = {
                        "active": True,
                        "block_id": str(b_id),
                        "is_repeat": False,
                        "speed": cur_pat["ref_speed"],
                        "notes_unrolled": default_empty_steps(32),
                    }
                elif parts[1] == "USE" and parts[2] == "BLOCK":
                    b_id = int(parts[3])
                    if b_id in blocks_memory:
                        b_speed, b_ls, b_le, b_notes32 = blocks_memory[b_id]
                        pat_ticks = cur_ref_steps * int(cur_pat["ref_speed"])
                        cur_pat["channels"][f"ch{c}"] = {
                            "active": True,
                            "block_id": str(b_id),
                            "is_repeat": True,
                            "speed": b_speed,
                            "notes_unrolled": unroll_sfx_block(b_notes32, b_speed, b_ls, b_le, pat_ticks),
                        }
                    cur_ch = None
            elif tok.startswith("BLK_LSTART_") and cur_block_def is not None:
                cur_block_ls = int(tok.split("_")[-1])
            elif tok.startswith("BLK_LEND_") and cur_block_def is not None:
                cur_block_le = int(tok.split("_")[-1])
            elif tok == "END_BLOCK":
                finalize_channel_block()
                cur_ch = None
                cur_step = None
            elif tok.startswith("SPEED_") and cur_pat is not None and cur_ch is not None:
                sp = int(tok.split("_")[1])
                cur_pat["channels"][f"ch{cur_ch}"]["speed"] = sp
                if cur_block_def is None:
                    pat_ticks = cur_ref_steps * int(cur_pat["ref_speed"])
                    expected_steps = max(1, int(round(float(pat_ticks) / sp)))
                    cur_pat["channels"][f"ch{cur_ch}"]["notes_unrolled"] = default_empty_steps(expected_steps)
            elif tok.startswith("STEP_") and cur_pat is not None and cur_ch is not None:
                st = int(tok.split("_")[1])
                notes_list = cur_pat["channels"][f"ch{cur_ch}"]["notes_unrolled"]
                while len(notes_list) <= st:
                    notes_list.append({"pitch": 0, "instrument": 0, "volume": 0, "effect": 0})
                cur_step = st
            elif cur_pat is not None and cur_ch is not None and cur_step is not None:
                notes_list = cur_pat["channels"][f"ch{cur_ch}"]["notes_unrolled"]
                if tok.startswith("PITCH_"):
                    notes_list[cur_step]["pitch"] = int(tok.split("_")[1])
                elif tok.startswith("INST_"):
                    notes_list[cur_step]["instrument"] = int(tok.split("_")[1])
                elif tok.startswith("VOL_"):
                    notes_list[cur_step]["volume"] = int(tok.split("_")[1])
                elif tok.startswith("FX_"):
                    notes_list[cur_step]["effect"] = int(tok.split("_")[1])

        flush_pattern()
        return {
            "start_pattern": 0,
            "end_pattern": max(0, len(patterns) - 1),
            "loop_start_pattern": loop_start_pattern,
            "end_reason": end_reason,
            "num_patterns": len(patterns),
            "patterns": patterns,
        }