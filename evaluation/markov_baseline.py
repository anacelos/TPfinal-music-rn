#!/usr/bin/env python3

"""
Baseline de Markov adaptado ao pipeline canônico do TP final.

Baseado no gerador original de João Silveira:
https://github.com/jcssilveira/TP1-DCC831

Adaptações para o TP final:
- treinamento exclusivamente no split de treino;
- entrada/saída no JSONL canônico do projeto;
- amostragem de estruturas apenas do treino;
- temperatura configurável;
- seed fixa para reprodutibilidade;
- saída simbólica compatível com o pipeline de avaliação.

A cadeia permanece de primeira ordem e separada por canal, como no baseline
original. Cada estado é (pitch, instrument, volume, effect).
"""

import argparse
import copy
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


CHANNELS = ("ch1", "ch2", "ch3", "ch4")
SILENCE = (0, 0, 0, 0)


def note_to_state(note):
    """Converte uma nota canônica para o estado utilizado pelo Markov."""
    if not note or note.get("volume", 0) <= 0:
        return SILENCE

    return (
        int(note.get("pitch", 0)),
        int(note.get("instrument", 0)),
        int(note.get("volume", 0)),
        int(note.get("effect", 0)),
    )


def state_to_note(state):
    pitch, instrument, volume, effect = state

    return {
        "pitch": pitch,
        "instrument": instrument,
        "volume": volume,
        "effect": effect,
    }


class ChannelMarkov:
    def __init__(self):
        self.transitions = defaultdict(Counter)
        self.sound_states = set()
        self.transition_count = 0

    def observe(self, notes):
        states = [note_to_state(note) for note in notes]

        for state in states:
            if state[2] > 0:
                self.sound_states.add(state)

        for current_state, next_state in zip(states, states[1:]):
            self.transitions[current_state][next_state] += 1
            self.transition_count += 1

    def sample_next(self, current_state, temperature, rng):
        counter = self.transitions.get(current_state)

        if not counter:
            return SILENCE

        items = sorted(counter)
        inv_temp = 1.0 / max(0.01, temperature)
        weights = [counter[item] ** inv_temp for item in items]

        return rng.choices(items, weights=weights, k=1)[0]

    def generate(self, length, temperature, reactivation_prob, rng):
        if length <= 0:
            return []

        sound_states = sorted(self.sound_states)

        if sound_states:
            current = rng.choice(sound_states)
        else:
            current = SILENCE

        states = [current]

        for _ in range(length - 1):
            current = self.sample_next(current, temperature, rng)

            # Mantém a heurística do baseline original:
            # ao chegar ao silêncio, há uma probabilidade fixa de reativação.
            if (
                current == SILENCE
                and sound_states
                and rng.random() < reactivation_prob
            ):
                current = rng.choice(sound_states)

            states.append(current)

        return [state_to_note(state) for state in states]


def train_and_sample_templates(train_path, num_templates, rng):
    """
    Treina os quatro modelos e seleciona, por reservoir sampling,
    estruturas de músicas exclusivamente do conjunto de treino.
    """
    models = {channel: ChannelMarkov() for channel in CHANNELS}

    reservoir = []
    song_count = 0
    active_channel_sequences = 0

    with Path(train_path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            song = json.loads(line)
            song_count += 1

            for pattern in song.get("patterns", []):
                channels = pattern.get("channels", {})

                for channel_name in CHANNELS:
                    channel = channels.get(channel_name, {})

                    if not channel.get("active", False):
                        continue

                    notes = channel.get("notes_unrolled", [])

                    if not notes:
                        continue

                    models[channel_name].observe(notes)
                    active_channel_sequences += 1

            # Reservoir sampling: permite escolher templates uniformemente
            # sem carregar os ~380 MB do train inteiro na memória.
            if len(reservoir) < num_templates:
                reservoir.append(copy.deepcopy(song))
            else:
                replacement = rng.randrange(song_count)

                if replacement < num_templates:
                    reservoir[replacement] = copy.deepcopy(song)

    if song_count == 0:
        raise RuntimeError(f"Nenhuma música encontrada em {train_path}")

    if num_templates > song_count:
        raise ValueError(
            f"--num-songs={num_templates} é maior que o número de músicas "
            f"de treino ({song_count})."
        )

    return models, reservoir, song_count, active_channel_sequences


def generate_song(
    template,
    models,
    generated_index,
    temperature,
    reactivation_prob,
    seed,
    rng,
):
    """
    Preserva apenas a estrutura temporal de uma música do treino.
    O conteúdo das notas é completamente regenerado pelo Markov.
    """
    song = copy.deepcopy(template)

    song["tid"] = f"markov_seed{seed}"
    song["song_id"] = generated_index
    song["wav_file"] = None
    song["uses_custom_instruments"] = False

    # O Markov não modela identidade/reutilização de blocos.
    song["blocks_dictionary"] = {}

    for pattern in song.get("patterns", []):
        channels = pattern.get("channels", {})

        for channel_name in CHANNELS:
            channel = channels.get(channel_name)

            if channel is None:
                continue

            if not channel.get("active", False):
                channel["active"] = False
                channel["original_sfx_id"] = None
                channel["block_id"] = None
                channel["is_repeat"] = False
                channel["speed"] = None
                channel["notes_unrolled"] = []
                continue

            length = len(channel.get("notes_unrolled", []))

            channel["original_sfx_id"] = None
            channel["block_id"] = None
            channel["is_repeat"] = False

            channel["notes_unrolled"] = models[channel_name].generate(
                length=length,
                temperature=temperature,
                reactivation_prob=reactivation_prob,
                rng=rng,
            )

    return song


def write_generated_songs(
    output_path,
    templates,
    models,
    temperature,
    reactivation_prob,
    seed,
    rng,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for index, template in enumerate(templates):
            generated = generate_song(
                template=template,
                models=models,
                generated_index=index,
                temperature=temperature,
                reactivation_prob=reactivation_prob,
                seed=seed,
                rng=rng,
            )

            handle.write(
                json.dumps(generated, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Baseline Markov de primeira ordem para músicas PICO-8."
    )

    parser.add_argument(
        "--train",
        default="data/splits/train.jsonl",
        help="Split de treino canônico.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="JSONL canônico que receberá as músicas geradas.",
    )
    parser.add_argument(
        "--num-songs",
        type=int,
        default=100,
        help="Número de músicas sintéticas a gerar.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperatura da amostragem das transições.",
    )
    parser.add_argument(
        "--reactivation-prob",
        type=float,
        default=0.25,
        help="Probabilidade de reativação após estado de silêncio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para seleção dos templates e geração.",
    )

    args = parser.parse_args()

    if args.num_songs <= 0:
        parser.error("--num-songs deve ser positivo.")

    if args.temperature <= 0:
        parser.error("--temperature deve ser positiva.")

    if not 0.0 <= args.reactivation_prob <= 1.0:
        parser.error("--reactivation-prob deve estar entre 0 e 1.")

    rng = random.Random(args.seed)

    print(f"Treino: {args.train}")
    print(f"Temperatura: {args.temperature}")
    print(f"Prob. reativação: {args.reactivation_prob}")
    print(f"Seed: {args.seed}")
    print(f"Músicas a gerar: {args.num_songs}")
    print()

    models, templates, train_song_count, sequence_count = (
        train_and_sample_templates(
            train_path=args.train,
            num_templates=args.num_songs,
            rng=rng,
        )
    )

    print(f"Músicas lidas do treino: {train_song_count}")
    print(f"Sequências de canal observadas: {sequence_count}")

    for channel_name in CHANNELS:
        model = models[channel_name]

        print(
            f"{channel_name}: "
            f"{len(model.transitions)} estados com saída | "
            f"{len(model.sound_states)} estados audíveis | "
            f"{model.transition_count} transições"
        )

    write_generated_songs(
        output_path=args.output,
        templates=templates,
        models=models,
        temperature=args.temperature,
        reactivation_prob=args.reactivation_prob,
        seed=args.seed,
        rng=rng,
    )

    print()
    print(f"Saída: {args.output}")
    print(f"Músicas geradas: {len(templates)}")


if __name__ == "__main__":
    main()
