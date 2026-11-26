#!/usr/bin/env python3
"""Deterministic, context-only inventory of every Real TEST example.

Run from the repository root: python -m evaluation.subjective.context_pool
Rules and limitations: evaluation/subjective/CONTEXT_POOL.md.
Only Python's standard library is required. No evaluation/model outputs are read.
"""

import argparse
import bisect
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import re
import struct
import sys


REPO = Path(__file__).resolve().parents[2]
PROTOCOL = "context-pool-v1"
RULES = {
    "revision": "observable-evidence-2-text-guard-1",
    "description": "guarded cleaned core; >=3 words, >=15 letters (10 for corroborated role); quality 1/2/3; see documentation",
    "visual": "nontrivial __label__ or __gfx__; PNG header validated; maps inventory only",
    "music": "segment-specific constants only; dynamic calls inventory only",
    "tiers": {"A": 3, "B": 2, "C": 1, "D": 0},
    "priority_ascending": ["tier_rank", "-description_quality", "-music_confidence",
                           "-visual_evidence_rank", "-min(description_core_words,100)", "numeric_tid", "song_id"],
    "serve_order": "tier-first rounds; one head per TID; order heads by context_priority",
}
CONFIDENCE = {"none": 0, "weak": 1, "moderate": 2, "strong": 3}
FIELDS = (
    "context_priority", "serve_order", "context_tier", "context_evidence_count", "priority_key",
    "tid", "song_id", "split", "start_pattern", "end_pattern", "game_title", "author",
    "game_title_confidence", "author_confidence", "identity_evidence", "total_stars",
    "description_present", "description_length", "description_informative", "description_quality",
    "description_core", "description_core_words", "description_core_length", "description_cue_families",
    "description_boilerplate_ratio", "description", "comments_present", "tags", "license",
    "original_pid", "original_cart_url", "cartridge_present", "cartridge_visual_informative",
    "spritesheet_present", "map_json_present", "rendered_map_present", "media_urls_count", "media_urls",
    "preferred_visual_type", "preferred_visual_path", "visual_informative", "visual_evidence_level",
    "visual_representativeness", "visual_evidence", "music_call_found", "music_call_kind",
    "music_call_count", "music_exact_call_count", "music_dynamic_call_count",
    "music_start_pattern_exact_match", "music_context_confidence", "music_context_evidence",
    "metadata_path", "lua_source_path", "songs_source_path", "source_test_line", "source_song_sha256",
)
STATE_NAME = re.compile(r"(?:^|[_.:])(init|start|title|menu|ending|gameover|state|level|room|scene)(?:$|[_:.])", re.I)
LEXEME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|0[xX][0-9a-fA-F]+|\d+(?:\.\d+)?|\S")
LONG_OPEN = re.compile(r"\[(=*)\[")


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def lua_tokens(code):
    """Lexical masking only: preserve positions, remove strings/comments, including [=[...]=]."""
    masked = list(code)
    i = 0
    while i < len(code):
        start = i
        comment = code.startswith(("--", "//"), i)
        if comment:
            i += 2
        long = LONG_OPEN.match(code, i)
        if long:
            close = "]" + long[1] + "]"
            end = code.find(close, long.end())
            i = len(code) if end < 0 else end + len(close)
        elif comment:
            end = code.find("\n", i)
            i = len(code) if end < 0 else end
        elif code[i] in "\"'":
            quote = code[i]
            i += 1
            while i < len(code):
                if code[i] == "\\":
                    i += 2
                elif code[i] == quote:
                    i += 1
                    break
                else:
                    i += 1
        else:
            i += 1
            continue
        masked[start:i] = ["\n" if c == "\n" else " " for c in code[start:i]]
    clean = "".join(masked)
    lines = [m.start() for m in re.finditer("\n", code)]
    return [(m[0], m.start(), m.end(), bisect.bisect_left(lines, m.start()) + 1)
            for m in LEXEME.finditer(clean)]


def integer_argument(tokens):
    values = [t[0] for t in tokens]
    sign = -1 if values[:1] == ["-"] else 1
    if values[:1] in (["-"], ["+"]):
        values = values[1:]
    if len(values) == 1 and re.fullmatch(r"(?:0[xX][0-9a-fA-F]+|\d+)", values[0]):
        return sign * int(values[0], 16 if values[0].lower().startswith("0x") else 10)
    return None


def analyze_lua(code, line_offset=0):
    """Recognize bare calls and a deliberately restricted direct function prefix.

    No arithmetic/dataflow, aliases, table decoding or reachability inference.
    A control-flow keyword ends a function's certifiable prefix permanently.
    Shorthand PICO-8 if and malformed scopes disable subsequent certification.
    Any possible shadowing of music disables strong/moderate throughout the file.
    """
    tokens = lua_tokens(code)
    values = [t[0] for t in tokens]
    shadowed = "_ENV" in values or any(
        v == "music" and (
            (i and values[i - 1] in ("local", "function"))
            or (i + 1 < len(values) and values[i + 1] == "="
                and (not i or values[i - 1] not in (".", ":")))
        ) for i, v in enumerate(values)
    )
    for i, value in enumerate(values):
        if value == "local":
            j = i + 1
            while j < len(values) and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*|,", values[j]):
                shadowed |= values[j] == "music"
                if j + 1 >= len(values) or values[j + 1] != ",":
                    break
                j += 2
        if value == "music" and (not i or values[i - 1] not in (".", ":")):
            j = i + 1
            while j + 1 < len(values) and values[j] == "," and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", values[j + 1]):
                j += 2
            if j < len(values) and values[j] == "=" and (j + 1 == len(values) or values[j + 1] != "="):
                shadowed = True
    # Parameter shadowing, including anonymous functions.
    for i, v in enumerate(values):
        if v == "function":
            j = i + 1
            while j < len(values) and values[j] not in ("(", "end", "function"):
                j += 1
            if j < len(values) and values[j] == "(":
                k = j + 1
                while k < len(values) and values[k] != ")":
                    shadowed |= values[k] == "music"
                    k += 1
    # A locally hidden/replaced callback cannot certify a global state entry.
    init_definitions = sum(values[i:i + 2] == ["function", "_init"] for i in range(len(values)))
    init_replaced = any(
        v == "_init" and ((i and values[i - 1] == "local")
                          or (i + 1 < len(values) and values[i + 1] == "="))
        for i, v in enumerate(values)
    )
    stack, calls = [], []
    uncertain = False
    parameter_end = -1
    for i, (value, start, _, line) in enumerate(tokens):
        if i <= parameter_end:
            continue
        if value == "function":
            for scope in stack:
                scope["prefix"] = False
            j = i + 1
            while j < len(tokens) and re.fullmatch(r"[A-Za-z_][\w]*|[.:]", values[j]):
                j += 1
            name = "".join(values[i + 1:j])
            if j >= len(tokens) or values[j] != "(":
                uncertain = True
                continue
            k = j + 1
            while k < len(tokens) and values[k] != ")":
                k += 1
            parameter_end = k
            stack.append({"kind": "function", "name": name,
                          "prefix": not stack and not uncertain and (not i or values[i - 1] != "local")})
        elif value in ("if", "for", "while", "repeat", "do", "goto", "return", "and", "or"):
            for scope in stack:
                scope["prefix"] = False
            if value == "if":
                # Multiline if is conservatively uncertified as well.
                tail, following = [], i + 1
                while following < len(tokens) and tokens[following][3] == line:
                    tail.append(values[following])
                    following += 1
                if "then" not in tail:
                    uncertain = True
            if value == "do" and stack and stack[-1]["kind"] in ("for", "while"):
                continue
            if value not in ("goto", "return", "and", "or"):
                stack.append({"kind": value, "name": "", "prefix": False})
        elif value in ("end", "until"):
            if stack:
                stack.pop()
            else:
                uncertain = True
        if value != "music" or i + 1 >= len(tokens) or values[i + 1] != "(":
            continue
        if i and values[i - 1] in (".", ":", "function"):
            continue
        j, depth, first_end = i + 2, 1, None
        while j < len(tokens):
            v = values[j]
            if v == "(":
                depth += 1
            elif v == ")":
                depth -= 1
                if depth == 0:
                    break
            elif v == "," and depth == 1 and first_end is None:
                first_end = j
            j += 1
        complete = j < len(tokens)
        arg_end = first_end if first_end is not None else j
        arg_tokens = tokens[i + 2:arg_end]
        constant = integer_argument(arg_tokens) if complete else None
        state = stack[-1]["name"] if stack and stack[-1]["kind"] == "function" else ""
        direct = (len(stack) == 1 and bool(state) and stack[0]["prefix"]
                  and not uncertain and not shadowed and complete)
        if state == "_init" and (init_definitions != 1 or init_replaced):
            direct = False
        calls.append({
            "line": line + line_offset, "lua_line": line,
            "call": code[start:tokens[j][2] if complete else len(code)],
            "argument": " ".join(t[0] for t in arg_tokens), "constant_pattern": constant,
            "state_hint": state, "certified_direct_prefix": direct,
            "shadowing_possible": shadowed,
        })
    return calls


def song_evidence(calls, start_pattern, end_pattern):
    exact = [c for c in calls if c["constant_pattern"] == start_pattern]
    dynamic = [c for c in calls if c["constant_pattern"] is None]
    relevant = [c for c in calls if c["constant_pattern"] is not None
                and start_pattern <= c["constant_pattern"] <= end_pattern]
    confidence = "weak" if relevant else "none"
    for c in exact:
        if c["certified_direct_prefix"]:
            if c["state_hint"] == "_init":
                confidence = "strong"
                break
            if STATE_NAME.search(c["state_hint"]):
                confidence = "moderate"
    kinds = {"dynamic" if c["constant_pattern"] is None else "constant" for c in calls}
    return {
        "music_call_found": bool(calls),
        "music_call_kind": "mixed" if len(kinds) > 1 else next(iter(kinds), "none"),
        "music_call_count": len(calls), "music_exact_call_count": len(exact),
        "music_dynamic_call_count": len(dynamic),
        "music_start_pattern_exact_match": bool(exact),
        "music_context_confidence": confidence,
        "music_context_evidence": packed([
            {**c, "exact_start_match": c["constant_pattern"] == start_pattern,
             "within_segment": c["constant_pattern"] is not None
             and start_pattern <= c["constant_pattern"] <= end_pattern}
            for c in calls
        ]),
    }


# Small, explicit EN/PT cue families. These are lexical proxies, not semantic validation.
DESCRIPTION_CUES = {
    "role": r"\b(?:you (?:are|play as|control)|voc[eê] (?:[eé]|controla)|player (?:is|controls)|jogador)\b",
    "goal": r"\b(?:collect|catch|rescue|defeat|fight|destroy|survive|escape|reach|find|protect|solve|deliver|burn|slay|coletar|pescar|resgatar|derrotar|sobreviver|escapar|encontrar|proteger|resolver|help .*?before)\b",
    "mechanic": r"\b(?:one.button|single.button|um bot[aã]o|jump|bounce|dash|shoot|platforming|rotate|build|stack|spin|shapeshift|pular|saltar|atirar|construir|empilhar)\b",
    "genre": r"\b(?:fishing|pesca|puzzle|platformer|plataforma|roguelike|racing|corrida|shooter|shoot.?em.?up|tetris|snake|breakout|asteroids|simulation|simulator|simulador)\b",
}


# Copula phrases need local corroboration; explicit play/control wording already
# names player agency. These lexical lists are deliberately small and auditable.
ROLE_CONTEXT = re.compile(
    r"\b(?:you (?:play as|control)|voc[eê] controla|player controls|jogador)\b|"
    r"\b(?:you are|voc[eê] [eé]|player is)\s+(?:(?:a|an|the|um|uma)\s+)?"
    r"(?:[a-z]+\s+){0,2}(?:robot|fisherman|raincloud|knight|hero|ghost|cat|slime|"
    r"wizard|pilot|ninja|soldier|detective|adventurer|explorer|archdemon|farmer|teacher|cop|officer|bad guy|rob[oô]|pescador|personagem)\b|"
    r"\b(?:you are|voc[eê] [eé])\s+(?:in|inside|trapped in|em)\s+(?:a|an|the|um|uma)?\s*"
    r"(?:maze|dungeon|forest|castle|world|labirinto|floresta|castelo)\b|"
    r"\b(?:player character|your character|farm|hive|nectar|bee|crosshairs|rocket|ball|"
    r"boost|brake|power.ups|bombs|detonate|enemy stats|seed|caverns)\b", re.I)
META_ADDRESS = r"(?:teacher|professor|instructor)\s*[,!:]?\s+(?:if|please|can you|could you|give|grade|se|por favor)\b"
META_REQUEST = re.compile(
    r"\b" + META_ADDRESS + r"|\b(?:give|grant|award)\s+me\s+(?:a\s+)?(?:100|grade|mark|score)\b|"
    r"\b(?:please|por favor)\b.{0,60}\b(?:grade|evaluate my|assess my|give me|minha nota)\b|"
    r"\b(?:school|class|course)\s+(?:assignment|submission)\b", re.I)
TECHNICAL_PLANNING = re.compile(
    r"\b(?:cart(?:ridge)? loading|map loading|load(?:ing|er)? (?:the )?(?:map|cart)|"
    r"load .{0,25}into .{0,15}memory|map (?:data|memory)|cartdata|multi.cart|"
    r"token limit|testing (?:cartridge|map)|planejamento t[eé]cnico)\b", re.I)


def description_quality(description):
    text = re.sub(r"\s+", " ", html.unescape(description)).strip()
    body = re.sub(r"https?://\S+", " ", text, flags=re.I)
    # Thread headers can repeat before the actual post. Preserve text after the cart header.
    body = re.sub(r"[^.!?]*?\bBBS\s*>.*?Cart\s*#\s*[\w-]+(?:\s*\|)*", " ", body, flags=re.I)
    body = re.sub(r"Cart\s*#\s*[\w-]+|License:\s*\S+", " ", body, flags=re.I)
    body = re.sub(r"(?:^|(?<=[.!?]))\s*(?:controls|controles|credits|cr[eé]ditos)\s*:?.*?(?=\b(?:about|goals|gameplay|description|objetivo|sobre)\b|$)", " ", body, flags=re.I)
    chunks = re.split(r"[.!?\n]+|\b(?:changelog|updates?|versions?|v\d+\.\d+|added|fixed|changed|atualizaç[aã]o)\b", body, flags=re.I)
    # Split obvious meta blocks from explicit player-role clauses in the same
    # sentence; do not reject a whole description because it addresses a teacher.
    fragments = []
    for chunk in chunks:
        for part in chunk.split(";"):
            if META_REQUEST.search(part):
                fragments.extend(re.split(r"(?=\b(?:you (?:are (?:a|an|the)\b|play as|control)|" + META_ADDRESS + r"))", part, flags=re.I))
            else:
                fragments.append(part)
    kept, seen, families = [], set(), set()
    for chunk in fragments:
        chunk = re.sub(r"\s+", " ", chunk).strip(" |:-0123456789")
        if META_REQUEST.search(chunk):
            continue
        cues = {k for k, pattern in DESCRIPTION_CUES.items() if re.search(pattern, chunk, re.I)}
        # A generic "you are" is not a role without a character/environment,
        # explicit agency, or another local gameplay cue.
        if "role" in cues and not ROLE_CONTEXT.search(chunk) and not (cues - {"role"}):
            cues.remove("role")
        # A game-name/genre keyword alone cannot rescue a technical planning block.
        if TECHNICAL_PLANNING.search(chunk) and (cues <= {"genre"} or
                (cues <= {"genre", "role"} and not ROLE_CONTEXT.search(chunk))):
            continue
        # Pure implementation/tutorial discussion must not count as game context.
        development = re.search(r"\b(?:learn(?:ing|ed)? lua|used lua|tracks?|channels?|programming|source code|tutorial|built.in text editor)\b", chunk, re.I)
        if not cues or (development and not cues.intersection({"role", "goal"})):
            continue
        normalized = chunk.casefold()
        if normalized not in seen:
            kept.append(chunk); seen.add(normalized); families.update(cues)
    core = ". ".join(kept)
    words = re.findall(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", core, re.UNICODE)
    letters = sum(c.isalpha() for c in core)
    # A corroborated short role (e.g. "You are a robot") need not reach 15 letters.
    informative = len(words) >= 3 and bool(families) and (letters >= 15 or
                    (letters >= 10 and "role" in families and bool(ROLE_CONTEXT.search(core))))
    ratio = round(max(0.0, 1 - len(core) / max(1, len(text))), 6)
    quality = (3 if informative and len(words) >= 75 and len(families) >= 3 and ratio < .5
               else 2 if informative and len(words) >= 20 and len(families) >= 2
               else 1 if informative else 0)
    return {"description": description, "description_present": bool(text), "description_length": len(text),
            "description_informative": informative, "description_quality": quality,
            "description_core": core, "description_core_words": len(words), "description_core_length": len(core),
            "description_cue_families": packed(sorted(families)), "description_boilerplate_ratio": ratio}


def header_identity(code, cartridge_name):
    """Only initial comment headers with explicit by/from, or title/cart-stem agreement."""
    headers = []
    for line in code.splitlines()[:8]:
        if not line.strip():
            continue
        if not line.lstrip().startswith("--") or line.lstrip().startswith("--["):
            break
        text = line.strip()[2:].strip().strip(" ~=|-*")
        if text:
            headers.append(text)
    title, author, evidence = "", "", ""
    for i, text in enumerate(headers):
        inline = re.fullmatch(r"(.+?)\s+by\s+(.+)", text, re.I)
        by = re.fullmatch(r"(?:by|from)\s+(.+)", text, re.I)
        if inline and i == 0 and inline[1].lower().strip() not in {"made", "music", "code", "created", "programmed"}:
            title, author = inline.groups()
        elif by and i == 1:
            title, author = headers[0], by[1]
        if title:
            author = re.sub(r"\s*[><][oO0][><].*$", "", author).strip()
            if len(title) <= 100 and len(author) <= 100 and re.fullmatch(r"[\w @&+.,'/-]+", author):
                if re.search(r"\b(?:music by|for the|code by)\b", author, re.I):
                    author = ""
                evidence = "explicit_initial_lua_comment_header"
                return {"game_title": title, "author": author, "identity_evidence": evidence,
                        "game_title_confidence": "header_only", "author_confidence": "header_only" if author else "unknown"}
            title, author = "", ""
    stem = re.sub(r"-\d+$", "", cartridge_name.removesuffix(".p8.png"))
    normalize = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    if headers and len(headers[0]) <= 100 and normalize(headers[0]) == normalize(stem) and not stem.isdigit():
        title, evidence = headers[0], "initial_lua_title_agrees_with_cartridge_filename"
    return {"game_title": title, "author": author, "identity_evidence": evidence or "unknown",
            "game_title_confidence": "header_only" if title else "unknown", "author_confidence": "unknown"}


class Sources:
    def __init__(self, root):
        self.root = Path(root)
        self.files = {}
        self.missing = set()

    def read(self, relative, optional=False):
        path = self.root / relative
        if not path.is_file():
            if not optional:
                raise ValueError(f"missing required source: {path}")
            self.missing.add(relative)
            return None
        raw = path.read_bytes()
        self.files[relative] = digest(raw)
        return raw

    def png(self, relative):
        raw = self.read(relative, optional=True)
        dimensions = None
        if raw and len(raw) >= 33 and raw[:8] == b"\x89PNG\r\n\x1a\n" and raw[12:16] == b"IHDR":
            width, height = struct.unpack(">II", raw[16:24])
            if width > 0 and height > 0:
                dimensions = (width, height)
        return raw is not None, dimensions


def pixel_section_evidence(p8, section):
    match = re.search(r"^__" + section + r"__\s*\n(.*?)(?=^__\w+__\s*$|\Z)", p8, re.M | re.S)
    lines = match[1].strip().splitlines() if match else []
    # PICO label pixels may also use g..v for extended palette indices.
    valid = bool(lines) and len(lines) <= 128 and all(re.fullmatch(r"[0-9a-v]{1,128}", l) for l in lines)
    if section == "label":
        valid = valid and len(lines) == 128 and all(len(l) == 128 for l in lines)
    pixels = "".join(l.ljust(128, "0") for l in lines).ljust(128 * 128, "0") if valid else ""
    counts = {c: pixels.count(c) for c in set(pixels)}
    minority = len(pixels) - max(counts.values(), default=0)
    return {"present": bool(match), "valid": valid, "colors": len(counts),
            "minority_pixels": minority, "informative": valid and len(counts) >= 2 and minority >= 16}


def visual_context(source, tid, p8):
    carts = sorted(p.relative_to(source.root).as_posix()
                   for p in (source.root / "cartridges" / tid).glob("*.p8.png"))
    label, gfx = pixel_section_evidence(p8, "label"), pixel_section_evidence(p8, "gfx")
    evidence = {"label": label, "gfx": gfx, "png_validation": "header_only"}
    preferred, path = "none", ""
    for cart in carts:
        _, size = source.png(cart)
        evidence[cart] = size
        if size == (160, 205) and label["informative"] and not path:
            preferred, path = "cartridge", cart
    sprites = f"sprites/{tid}/spritesheet.png"
    sp_present, sp_size = source.png(sprites)
    map_path = f"maps/{tid}/rendered_map.png"
    map_present, map_size = source.png(map_path)
    evidence[sprites], evidence[map_path] = sp_size, map_size
    if not path and sp_size == (128, 128) and gfx["informative"]:
        preferred, path = "spritesheet", sprites
    present_map = source.read(f"maps/{tid}/map.json", optional=True) is not None
    return {"cartridge_present": bool(carts), "cartridge_visual_informative": label["informative"] and any(evidence[c] == (160,205) for c in carts),
            "spritesheet_present": sp_present, "map_json_present": present_map, "rendered_map_present": map_present,
            "preferred_visual_type": preferred, "preferred_visual_path": path, "visual_informative": bool(path),
            "visual_evidence_level": {"cartridge": "cartridge_pixels", "spritesheet": "asset_pixels", "none": "none"}[preferred],
            "visual_representativeness": "unverified", "visual_evidence": packed(evidence)}, Path(carts[0]).name if carts else ""


def evidence_count(row):
    return sum((bool(row["description_informative"]), bool(row["visual_informative"]),
                row["music_context_confidence"] in ("strong", "moderate")))


def classify_tier(row):
    return "DCBA"[evidence_count(row)]


def priority_key(row):
    return ("ABCD".index(row["context_tier"]), -row["description_quality"],
            -CONFIDENCE[row["music_context_confidence"]],
            -{"none": 0, "asset_pixels": 1, "cartridge_pixels": 2}[row["visual_evidence_level"]],
            -min(row["description_core_words"], 100), int(row["tid"]), row["song_id"])


def assign_serve_order(rows):
    """Tier-first rounds of queue heads. Does not change context_priority."""
    from collections import defaultdict, deque
    position = 0
    for tier in "ABCD":
        queues = defaultdict(deque)
        for row in sorted(rows, key=lambda r: r["context_priority"]):
            if row["context_tier"] == tier:
                queues[row["tid"]].append(row)
        while queues:
            active = sorted(queues, key=lambda t: queues[t][0]["context_priority"])
            for tid in active:
                position += 1
                queues[tid].popleft()["serve_order"] = position
                if not queues[tid]:
                    del queues[tid]
    return sorted(rows, key=lambda r: r["serve_order"])


def split_hashes(splits_dir):
    result = {}
    for line in (splits_dir / "split_hashes.sha256").read_text().splitlines():
        expected, name = line.split()
        if name not in ("train.jsonl", "val.jsonl", "test.jsonl") or name in result:
            raise ValueError("unexpected/duplicate filename in split_hashes.sha256")
        with (splits_dir / name).open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"split hash mismatch: {name}")
        result[name] = actual
    if len(result) != 3:
        raise ValueError("expected hashes for all three splits")
    return result


def check_identities(rows, expected):
    actual = [(str(r["tid"]), int(r["song_id"])) for r in rows]
    if len(actual) != len(set(actual)):
        raise ValueError("duplicate (tid, song_id)")
    if set(actual) != set(expected) or len(actual) != len(expected):
        raise ValueError("pool identities differ from TEST")
    if any(r["split"] != "test" for r in rows):
        raise ValueError("non-test row")


def split_identities(path):
    identities = set()
    with path.open() as handle:
        for line in handle:
            song = json.loads(line)
            identity = str(song["tid"]), int(song["song_id"])
            if identity in identities:
                raise ValueError(f"duplicate identity in {path.name}: {identity}")
            identities.add(identity)
    return identities


def prepare_pool(root, splits_dir, expected_count=1854, expected_tids=873, progress=None):
    root, splits_dir = Path(root), Path(splits_dir)
    before = split_hashes(splits_dir)
    test_raw = (splits_dir / "test.jsonl").read_bytes()
    selected = {}
    for line_number, line in enumerate(test_raw.splitlines(), 1):
        song = json.loads(line)
        tid, song_id = str(song.pop("tid")), song["song_id"]
        if not tid.isdecimal() or not isinstance(song_id, int) or song_id < 0:
            raise ValueError("invalid TEST identity")
        identity = tid, song_id
        if identity in selected:
            raise ValueError("duplicate TEST identity")
        selected[identity] = {"tid": tid, "song_id": song_id, "split": "test",
                              "start_pattern": song["start_pattern"], "end_pattern": song["end_pattern"],
                              "source_test_line": line_number, "source_song_sha256": digest(packed(song).encode())}
    if len(selected) != expected_count or len({t for t, _ in selected}) != expected_tids:
        raise ValueError(f"unexpected TEST size: {len(selected)} songs")
    manifest_raw = (splits_dir / "split_manifest.csv").read_bytes()
    manifest = list(csv.DictReader(io.StringIO(manifest_raw.decode())))
    manifest_test = [(r["tid"], int(r["song_id"])) for r in manifest if r["split"] == "test"]
    if len(manifest_test) != len(set(manifest_test)) or set(manifest_test) != set(selected):
        raise ValueError("manifest TEST identities differ")
    other = {(r["tid"], int(r["song_id"])) for r in manifest if r["split"] != "test"}
    if other & selected.keys():
        raise ValueError("TEST identity also occurs in another split")
    for name in ("train", "val"):
        actual = split_identities(splits_dir / f"{name}.jsonl")
        declared = {(r["tid"], int(r["song_id"])) for r in manifest if r["split"] == name}
        if actual != declared or actual & selected.keys():
            raise ValueError(f"{name}: manifest mismatch or TEST overlap")
    source = Sources(root)
    master = {}
    for line in source.read("dataset.jsonl").splitlines():
        item = json.loads(line)
        tid = str(item["tid"])
        if tid in master:
            raise ValueError(f"duplicate master TID: {tid}")
        master[tid] = item
    games = {}
    game_tids = sorted({t for t, _ in selected}, key=int)
    for game_number, tid in enumerate(game_tids, 1):
        metadata_path, songs_path = f"metadata/{tid}/game_info.json", f"audio/{tid}/songs.json"
        metadata = json.loads(source.read(metadata_path))
        songs = json.loads(source.read(songs_path))
        if str(metadata["tid"]) != tid or str(songs["tid"]) != tid:
            raise ValueError(f"source TID mismatch: {tid}")
        song_ids = [s["song_id"] for s in songs["songs"]]
        if len(song_ids) != len(set(song_ids)):
            raise ValueError(f"duplicate source song: {tid}")
        for song in songs["songs"]:
            identity = tid, song["song_id"]
            if identity in selected and digest(packed(song).encode()) != selected[identity]["source_song_sha256"]:
                raise ValueError(f"source music differs from TEST: {identity}")
        if any(s not in song_ids for t, s in selected if t == tid):
            raise ValueError(f"missing source song: {tid}")
        lua_path = f"p8/{tid}/{tid}.p8"
        p8 = source.read(lua_path).decode("utf-8")
        match = re.search(r"^__lua__\s*\n(.*?)(?=^__\w+__\s*$|\Z)", p8, re.M | re.S)
        if not match:
            raise ValueError(f"missing __lua__: {tid}")
        code = match[1]
        visuals, cartridge_name = visual_context(source, tid, p8)
        comments = source.read(f"metadata/{tid}/comments.jsonl", optional=True)
        stars = metadata.get("total_stars")
        if stars is not None and (type(stars) is not int or stars < 0):
            raise ValueError(f"invalid total_stars: {tid}")
        media = metadata.get("media_urls", [])
        game = {
            **header_identity(code, cartridge_name), **description_quality(metadata.get("description", "")),
            **visuals, "total_stars": stars if stars is not None else "",
            "comments_present": bool(comments and comments.strip()),
            "tags": packed(metadata.get("tags", [])), "license": metadata.get("license", ""),
            "media_urls_count": len(media), "media_urls": packed(media),
            "original_pid": master.get(tid, {}).get("pid_found", ""),
            "original_cart_url": master.get(tid, {}).get("cart_url_remote", ""),
            "metadata_path": metadata_path, "lua_source_path": lua_path, "songs_source_path": songs_path,
        }
        games[tid] = (game, analyze_lua(code, line_offset=p8[:match.start(1)].count("\n")))
        if progress and (game_number % 100 == 0 or game_number == len(game_tids)):
            progress(game_number, len(game_tids))
    rows = []
    for identity, selected_song in selected.items():
        game, calls = games[identity[0]]
        row = {**selected_song, **game, **song_evidence(calls, selected_song["start_pattern"], selected_song["end_pattern"])}
        row["context_evidence_count"] = evidence_count(row)
        row["context_tier"] = classify_tier(row)
        row["priority_key"] = packed(priority_key(row))
        rows.append(row)
    rows.sort(key=priority_key)
    for rank, row in enumerate(rows, 1):
        row["context_priority"] = rank
    rows = assign_serve_order(rows)
    check_identities(rows, list(selected))
    after = split_hashes(splits_dir)
    if before != after or digest(test_raw) != before["test.jsonl"]:
        raise ValueError("splits changed during preparation")
    summary = {
        "total": len(rows), "tids": len(games),
        "tiers": {k: sum(r["context_tier"] == k for r in rows) for k in "ABCD"},
        "music_context_confidence": {k: sum(r["music_context_confidence"] == k for r in rows) for k in CONFIDENCE},
        "coverage_songs": {k: sum(bool(r[k]) for r in rows) for k in (
            "description_present", "description_informative", "comments_present", "cartridge_present",
            "spritesheet_present", "map_json_present", "rendered_map_present", "media_urls_count", "visual_informative")},
        "preferred_visual_types": {k: sum(r["preferred_visual_type"] == k for r in rows)
                                   for k in ("cartridge", "rendered_map", "spritesheet", "none")},
        "with_preferred_visual": sum(r["preferred_visual_type"] != "none" for r in rows),
        "without_text": sum(not r["description_present"] for r in rows),
        "without_informative_text": sum(not r["description_informative"] for r in rows),
        "without_visual": sum(r["preferred_visual_type"] == "none" for r in rows),
        "without_informative_visual": sum(not r["visual_informative"] for r in rows),
    }
    summary["coverage_tids"] = {
        k: sum(bool(game[k]) for game, _ in games.values())
        for k in summary["coverage_songs"]
    }
    summary["description_quality"] = {str(q): sum(r["description_quality"] == q for r in rows) for q in range(4)}
    summary["cartridge_visual_informative"] = sum(r["cartridge_visual_informative"] for r in rows)
    summary["serve_prefixes"] = {}
    from collections import Counter
    from itertools import groupby
    for n in (50, 100, 200, 400, len(rows)):
        prefix = rows[:n]
        frequencies = Counter(r["tid"] for r in prefix)
        summary["serve_prefixes"][str(n)] = {
            "unique_tids": len(frequencies), "max_per_tid": max(frequencies.values(), default=0),
            "longest_run": max((len(list(g)) for _, g in groupby(r["tid"] for r in prefix)), default=0),
        }
    example_fields = ("context_priority", "tid", "song_id", "game_title", "context_tier",
                      "description_quality", "preferred_visual_type", "music_context_confidence", "total_stars")
    audit = {
        "protocol": PROTOCOL, "rules": RULES, "roots": {"dataset": "dataset", "splits": "splits", "repository": "repository"},
        "python_version": sys.version.split()[0], "physical_csv_order": "serve_order", "split_hashes_before": before, "split_hashes_after": after,
        "test_manifest_sha256": digest(manifest_raw), "implementation_sha256": digest(Path(__file__).read_bytes()),
        "source_files_sha256": dict(sorted(source.files.items())), "missing_optional_sources": sorted(source.missing),
        "summary": summary,
        "sanity_checks": {"unique_identities": True, "exact_test_identity_set": True,
                          "train_val_overlap": 0, "source_songs_equal_test": True, "split_hashes_unchanged": True,
                          "serve_order_unique": len({r["serve_order"] for r in rows}) == len(rows)},
        "ranking_inputs": "TEST identity/pattern bounds; game metadata, original cartridge, map/sprite files, cartridge Lua only",
        "limitations": [
            "Static heuristic, not a Lua parser or proof of reachability/playback/frame association.",
            "Dynamic calls are inventoried only, never evidence for a specific segment.",
            "PNG headers/dimensions are checked; pixel decoding, semantic representativeness and live URLs are not validated.",
            "Lexical core cues are conservative EN/PT proxies; no semantic or runtime validation.",
            "Titles/authors are blank unless supported by restricted initial Lua headers.",
            "Stars, comments, tags, title and author never affect priority or serve_order.",
            "Nontrivial label/gfx pixels signal information only, never gameplay representativeness; maps inventory only.",
        ],
        "examples_per_tier": {tier: [{k: r[k] for k in example_fields} for r in rows if r["context_tier"] == tier][:10]
                              for tier in "ABCD"},
    }
    return rows, audit


def serialize_outputs(rows, audit):
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_text = csv_buffer.getvalue()
    audit = {**audit, "context_pool_csv_sha256": digest(csv_text.encode("utf-8"))}
    return csv_text, json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=REPO.parent / "dataset_pico8_v2" / "dataset_pico8_v2")
    parser.add_argument("--splits-dir", type=Path, default=REPO / "data" / "splits")
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs" / "subjective" / PROTOCOL)
    args = parser.parse_args()
    rows, audit = prepare_pool(
        args.dataset_root, args.splits_dir,
        progress=lambda n, total: print(f"Inspecionados {n}/{total} TIDs", file=sys.stderr),
    )
    csv_text, audit_text = serialize_outputs(rows, audit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "context_pool.csv").write_text(csv_text, encoding="utf-8", newline="")
    (args.output_dir / "context_pool_config.json").write_text(audit_text, encoding="utf-8")
    print(json.dumps(audit["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
