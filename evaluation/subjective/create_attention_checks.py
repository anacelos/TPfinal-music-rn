#!/usr/bin/env python3
"""Cria attention checks a partir do manifest de synthesize_jsonl.py.

Exemplo, a partir da raiz do repositório:
    python evaluation/subjective/create_attention_checks.py \
        --manifest <manifest-real.csv> --output-dir <checks> --snr-db 0 --seed 831

Aceita apenas WAV PCM mono de 16 bits. Caminhos relativos de wav_file são
resolvidos em relação ao manifest de entrada. Os originais nunca são escritos.
O manifest attention_checks_manifest.csv usa attention_wav relativo à sua pasta
e source_wav absoluto. A seed usa SOMENTE global_seed, tid, song_id e o SHA-256
dos bytes do WAV real. Índices e caminhos são apenas proveniência.

O ruído é amostrado com PCG64/standard_normal e calibrado pelo RMS da realização
finita. Um único ganho atenua sinal e ruído juntos, se necessário. Não há
limitação individual de amostras. O arredondamento final é para o inteiro par
mais próximo (numpy.rint). RMSs são registrados em unidades de amplitude PCM.
SNRs aceitas: [-60, 60] dB. Após a quantização, o erro absoluto da SNR deve ser
no máximo 0.1 dB; entradas que não atendem ao protocolo são rejeitadas.

Todos os WAVs são preparados e relidos em temporários no diretório de destino.
Falhas de processamento preservam o lote anterior. Antes da publicação dos
WAVs, um manifest anterior é removido: uma falha de publicação pode deixar WAVs
parciais, mas nunca um manifest antigo descrevendo a nova execução. O manifest
novo é publicado atomicamente por último. Somente um writer por output-dir é
suportado; não há transação atômica de múltiplos arquivos nem garantia contra
concorrência maliciosa ou perda de energia. Temporários são limpos em exceções
normais, mas interrupções abruptas podem deixá-los no diretório.
"""

import argparse
from contextlib import ExitStack
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import wave

import numpy as np


PROTOCOL = "attention-check-v1"
SEED_DERIVATION = "sha256-first128-big-endian-json-utf8-[global_seed,tid,song_id,source_sha256]-v1"
MAX_ABSOLUTE_SNR_ERROR_DB = 0.1
INPUT_FIELDS = ("index", "song_id", "tid", "wav_file", "duration_sec", "num_patterns")
OUTPUT_FIELDS = (
    "source_index", "source_song_id", "source_tid", "source_wav_file",
    "source_wav", "attention_wav", "source_manifest", "source_manifest_sha256",
    "source_duration_sec", "source_num_patterns", "source_identity",
    "snr_db", "global_seed", "derived_seed", "sample_rate", "num_frames",
    "num_channels", "sample_width_bytes", "source_sha256", "attention_sha256",
    "signal_rms", "noise_rms", "post_mix_gain", "protocol", "rng",
    "seed_derivation", "numpy_version", "effective_snr_db", "absolute_snr_error_db",
    "max_absolute_snr_error_db", "python_version",
)
# One PCM unit of headroom below the positive rail, including after rounding.
PCM16_PEAK = 32766.0


def derive_seed(global_seed, tid, song_id, source_sha256):
    """SHA-256 do JSON compacto UTF-8 [int(seed), tid, song_id, source_sha256].

    tid e song_id são strings literais do CSV; o digest do WAV é hexadecimal
    minúsculo. ensure_ascii=False, sem espaços. Primeiros 16 bytes do digest
    viram inteiro não negativo big-endian, usado por PCG64.
    """
    payload = json.dumps(
        [int(global_seed), tid, song_id, source_sha256], ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def read_pcm16(path):
    """Lê e calcula o hash dos mesmos bytes; rejeita WAV inválido ou truncado."""
    raw = path.read_bytes()
    try:
        with wave.open(io.BytesIO(raw), "rb") as wav:
            if (wav.getcomptype(), wav.getnchannels(), wav.getsampwidth()) != ("NONE", 1, 2):
                raise ValueError(f"{path}: formato incompatível; esperado WAV PCM mono 16-bit")
            sample_rate = wav.getframerate()
            num_frames = wav.getnframes()
            pcm = wav.readframes(num_frames)
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"{path}: WAV inválido ou incompatível: {exc}") from exc
    if num_frames == 0:
        raise ValueError(f"{path}: WAV vazio (zero frames)")
    if sample_rate <= 0:
        raise ValueError(f"{path}: sample rate inválido")
    if len(pcm) != num_frames * 2:
        raise ValueError(f"{path}: WAV truncado; frames incompletos")
    signal = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    return signal, sample_rate, hashlib.sha256(raw).hexdigest()


def validate_snr(snr_db):
    if not math.isfinite(snr_db):
        raise ValueError("--snr-db deve ser finito")
    if not -60.0 <= snr_db <= 60.0:
        raise ValueError("--snr-db deve estar entre -60 e +60 dB")


def mix_noise(signal, snr_db, seed):
    """Mistura em float64, calibra o RMS e quantiza sem saturação ou wraparound."""
    validate_snr(snr_db)
    signal_rms = float(np.sqrt(np.mean(signal * signal)))
    if signal_rms == 0.0:
        raise ValueError("áudio completamente silencioso; SNR não definida")
    noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))

    noise = np.random.Generator(np.random.PCG64(seed)).standard_normal(signal.size)
    sampled_rms = float(np.sqrt(np.mean(noise * noise)))
    if not math.isfinite(sampled_rms) or sampled_rms <= 0:
        raise ValueError("RMS inválido na realização do ruído")
    noise *= noise_rms / sampled_rms
    noise += signal  # Reuse the noise buffer for the mix, one WAV at a time.
    peak = float(np.max(np.abs(noise)))
    gain = min(1.0, PCM16_PEAK / peak) if peak else 1.0
    noise *= gain
    np.rint(noise, out=noise)
    pcm = noise.astype("<i2")
    return pcm, signal_rms, noise_rms, gain


def measure_effective_snr(signal, quantized, gain, requested_snr_db):
    """SNR = 20 log10(RMS(g*x) / RMS(y_pcm16 - g*x)), incluindo quantização."""
    if np.array_equal(signal, quantized):
        raise ValueError("resultado idêntico ao original após quantização PCM16")
    scaled_signal = gain * signal
    residual = quantized.astype(np.float64) - scaled_signal
    residual_rms = float(np.sqrt(np.mean(residual * residual)))
    if residual_rms == 0.0:
        raise ValueError("ruído efetivo desapareceu após quantização PCM16")
    scaled_rms = float(np.sqrt(np.mean(scaled_signal * scaled_signal)))
    if not math.isfinite(residual_rms) or not math.isfinite(scaled_rms) or scaled_rms <= 0:
        raise ValueError("SNR efetiva não finita após quantização PCM16")
    effective_snr_db = 20.0 * math.log10(scaled_rms / residual_rms)
    if not math.isfinite(effective_snr_db):
        raise ValueError("SNR efetiva não finita após quantização PCM16")
    absolute_error = abs(effective_snr_db - requested_snr_db)
    if absolute_error > MAX_ABSOLUTE_SNR_ERROR_DB:
        raise ValueError(
            f"erro absoluto de SNR {absolute_error:.6g} dB excede "
            f"{MAX_ABSOLUTE_SNR_ERROR_DB} dB após quantização PCM16"
        )
    return effective_snr_db, absolute_error


def temporary_file(cleanup, directory, suffix):
    """Temporário no mesmo filesystem, removido mesmo se houver exceção."""
    handle = tempfile.NamedTemporaryFile(
        dir=directory, prefix=".attention-", suffix=suffix, delete=False,
    )
    cleanup.callback(Path(handle.name).unlink, missing_ok=True)
    return handle


def check_destinations(destinations, protected, overwrite):
    protected_inodes = {(p.stat().st_dev, p.stat().st_ino) for p in protected}
    for target in destinations:
        if target.resolve() in protected:
            raise ValueError(f"{target}: saída coincide com arquivo de origem")
        if target.exists():
            stat = target.stat()
            if (stat.st_dev, stat.st_ino) in protected_inodes:
                raise ValueError(f"{target}: saída é um alias de arquivo de origem")
            if not target.is_symlink() and not target.is_file():
                raise ValueError(f"{target}: destino não é um arquivo regular")
        if (target.exists() or target.is_symlink()) and not overwrite:
            raise FileExistsError(f"{target}: já existe; use --overwrite explicitamente")


def create_attention_checks(manifest_path, output_dir, snr_db=0.0, seed=831, overwrite=False):
    """Retorna o path do manifest de checks; falha antes de sobrescrever sem opt-in."""
    validate_snr(snr_db)
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_manifest = output_dir / "attention_checks_manifest.csv"
    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    reader = csv.DictReader(io.StringIO(manifest_bytes.decode("utf-8"), newline=""))
    missing = set(INPUT_FIELDS) - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"{manifest_path}: colunas ausentes: {', '.join(sorted(missing))}")

    items = []
    indices = set()
    logical_ids = set()
    for line_number, row in enumerate(reader, start=2):
        if any(row.get(field) is None for field in INPUT_FIELDS) or None in row:
            raise ValueError(f"{manifest_path}:{line_number}: linha CSV malformada")
        try:
            index = int(row["index"])
        except ValueError as exc:
            raise ValueError(f"{manifest_path}:{line_number}: index inválido") from exc
        if index < 0 or index in indices:
            raise ValueError(f"{manifest_path}:{line_number}: index negativo ou duplicado: {index}")
        for field in ("tid", "song_id"):
            if not row[field].strip():
                raise ValueError(f"{manifest_path}:{line_number}: {field} vazio")
        logical_id = (row["tid"], row["song_id"])
        if logical_id in logical_ids:
            raise ValueError(f"{manifest_path}:{line_number}: identidade (tid, song_id) duplicada: {logical_id}")
        if not row["wav_file"].strip():
            raise ValueError(f"{manifest_path}:{line_number}: wav_file vazio")
        indices.add(index)
        logical_ids.add(logical_id)
        source = (manifest_path.parent / row["wav_file"]).resolve()
        if not source.is_file():
            raise ValueError(f"{source}: WAV de origem não encontrado")
        identity = {key: row[key] for key in ("index", "tid", "song_id", "wav_file")}
        target = output_dir / f"attention_{index:04d}.wav"
        items.append((row, source, target, identity))
    if not items:
        raise ValueError(f"{manifest_path}: manifest sem músicas")

    # Protect every input, including aliases/hardlinks and the input manifest,
    # even with --overwrite. Check all destinations before writing any output.
    protected = {manifest_path} | {source for _, source, _, _ in items}
    destinations = [target for _, _, target, _ in items] + [output_manifest]
    check_destinations(destinations, protected, overwrite)

    output_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as cleanup:
        output_rows = []
        staged_wavs = []
        for row, source, target, identity in items:
            signal, sample_rate, source_hash = read_pcm16(source)
            derived_seed = derive_seed(seed, row["tid"], row["song_id"], source_hash)
            try:
                pcm, signal_rms, noise_rms, gain = mix_noise(signal, snr_db, derived_seed)
                with temporary_file(cleanup, output_dir, ".wav.tmp") as handle:
                    staged = Path(handle.name)
                    with wave.open(handle, "wb") as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(sample_rate)
                        wav.writeframes(pcm.tobytes())
                    handle.flush()
                    os.fsync(handle.fileno())
                rendered, rendered_rate, attention_hash = read_pcm16(staged)
                if rendered_rate != sample_rate or not np.array_equal(rendered, pcm):
                    raise ValueError("WAV temporário diverge do áudio preparado")
                effective_snr, absolute_error = measure_effective_snr(signal, rendered, gain, snr_db)
            except (ValueError, wave.Error) as exc:
                raise ValueError(f"{source}: {exc}") from exc
            staged_wavs.append((staged, target))
            output_rows.append({
                "source_index": row["index"],
                "source_song_id": row["song_id"],
                "source_tid": row["tid"],
                "source_wav_file": row["wav_file"],
                "source_wav": str(source),
                "attention_wav": target.name,
                "source_manifest": str(manifest_path),
                "source_manifest_sha256": manifest_hash,
                "source_duration_sec": row["duration_sec"],
                "source_num_patterns": row["num_patterns"],
                "source_identity": json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                "snr_db": snr_db,
                "global_seed": seed,
                "derived_seed": derived_seed,
                "sample_rate": sample_rate,
                "num_frames": signal.size,
                "num_channels": 1,
                "sample_width_bytes": 2,
                "source_sha256": source_hash,
                "attention_sha256": attention_hash,
                "signal_rms": signal_rms,
                "noise_rms": noise_rms,
                "post_mix_gain": gain,
                "protocol": PROTOCOL,
                "rng": "numpy.random.PCG64.standard_normal",
                "seed_derivation": SEED_DERIVATION,
                "numpy_version": np.__version__,
                "effective_snr_db": effective_snr,
                "absolute_snr_error_db": absolute_error,
                "max_absolute_snr_error_db": MAX_ABSOLUTE_SNR_ERROR_DB,
                "python_version": platform.python_version(),
            })

        with temporary_file(cleanup, output_dir, ".csv.tmp") as handle:
            staged_manifest = Path(handle.name)
            # TextIOWrapper is detached so the temporary handle keeps ownership.
            text_handle = io.TextIOWrapper(handle, encoding="utf-8", newline="")
            try:
                writer = csv.DictWriter(text_handle, fieldnames=OUTPUT_FIELDS)
                writer.writeheader()
                writer.writerows(output_rows)
                text_handle.flush()
            finally:
                text_handle.detach()
            handle.flush()
            os.fsync(handle.fileno())

        check_destinations(destinations, protected, overwrite)
        # No old success manifest may describe newly replaced WAVs if publication
        # fails. Validation failures above leave the previous batch untouched.
        if overwrite:
            output_manifest.unlink(missing_ok=True)
        for staged, target in staged_wavs:
            os.replace(staged, target)
        os.replace(staged_manifest, output_manifest)
    return output_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path, help="Manifest CSV real de synthesize_jsonl.py")
    parser.add_argument("--output-dir", required=True, type=Path, help="Pasta exclusiva dos attention checks")
    parser.add_argument("--snr-db", type=float, default=0.0, help="SNR em dB, de -60 a +60 (padrão: 0.0)")
    parser.add_argument("--seed", type=int, default=831, help="Seed global (padrão: 831)")
    parser.add_argument("--overwrite", action="store_true", help="Permite substituir saídas existentes")
    args = parser.parse_args(argv)
    try:
        manifest = create_attention_checks(args.manifest, args.output_dir, args.snr_db, args.seed, args.overwrite)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Erro: {exc}\n")
    print(f"Manifest dos attention checks: {manifest}")


if __name__ == "__main__":
    main()
