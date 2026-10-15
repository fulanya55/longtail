"""Generate the fresh no-fit Route-A H0n four-PRI MFT intervention source."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"C:\LJQ_code\干扰数据")
# The un-suffixed directory was left with a partial HDF5 payload by the first
# write attempt.  The r1 source was then rejected before scoring because all
# receiver identities overlapped H1m.  Preserve both forensic artifacts and
# use this new immutable source for the first auditable formal run.
OUTPUT = DATA_ROOT / "h0n_mft_crosspulse_pairs_20260803_r2"
N_PAIRS = 640
BASE_SEED = 880_000_000
NOISE_SEED_OFFSET = 208_677_743
NUM_PULSES = 4
SAMPLES_PER_PRI = 3000
PULSE_SAMPLES = 1000
DELAY_START = 400
DELAY_STOP = 999


def _reference() -> np.ndarray:
    time = np.arange(PULSE_SAMPLES, dtype=np.float64) / 10e6
    return np.exp(1j * np.pi * (5e6 / 100e-6) * np.square(time)).astype(np.complex128)


def _write_arm(path: Path, payload: dict[str, np.ndarray]) -> None:
    with h5py.File(path, "w") as handle:
        iq = payload["iq"]
        handle.create_dataset("iq_real", data=iq.real, compression="gzip", compression_opts=4)
        handle.create_dataset("iq_imag", data=iq.imag, compression="gzip", compression_opts=4)
        for name, value in payload.items():
            if name == "iq":
                continue
            handle.create_dataset(name, data=value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(*, execute: bool) -> dict[str, object]:
    receipt_root = ROOT / "outputs" / "tapc" / "route_a_h0n_mft_crosspulse_generation"
    receipt_root.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, object] = {
        "status": "planned" if not execute else "completed",
        "executed": bool(execute),
        "experiment": "RouteA_H0n_MFTCrossPulseCoherence",
        "evidence_role": "new_observation_physical_identifiability_only",
        "sample_count_per_arm": N_PAIRS,
        "base_pair_seed": BASE_SEED,
        "receiver_noise_seed_offset": NOISE_SEED_OFFSET,
        "receiver_noise_seed_range": [BASE_SEED + NOISE_SEED_OFFSET, BASE_SEED + NOISE_SEED_OFFSET + N_PAIRS - 1],
        "num_pulses": NUM_PULSES,
        "samples_per_pri": SAMPLES_PER_PRI,
        "mft_delay_samples": [DELAY_START, DELAY_STOP],
        "mft_num_targets": 8,
        "mft_total_jnr_db": [0.0, 8.0],
        "intervention": "whole_PRI_unit_phase_on_MFT_component_only",
        "test_split_generated": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "test_files_opened": False,
    }
    if not execute:
        (receipt_root / "dry_run_generation_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return receipt
    if OUTPUT.exists():
        # A prior pre-write failure may have created the directory itself. It
        # is safe to reuse only when it contains no payload or receipt; any
        # material content remains a strict no-overwrite condition.
        if any(OUTPUT.iterdir()):
            raise FileExistsError(f"refusing to overwrite existing source: {OUTPUT}")
    else:
        OUTPUT.mkdir(parents=True)
    rng_phase = np.random.default_rng(BASE_SEED + 31_337)
    shape = (N_PAIRS, NUM_PULSES * SAMPLES_PER_PRI)
    coherent = {"iq": np.zeros(shape, np.complex64)}
    scrambled = {"iq": np.zeros(shape, np.complex64)}
    common = {
        "pair_seed": np.zeros(N_PAIRS, np.int64),
        "receiver_noise_seed": np.zeros(N_PAIRS, np.int64),
        "mft_total_jnr_db": np.zeros(N_PAIRS, np.float64),
        "arrival_delay_samples": np.zeros(N_PAIRS, np.int64),
        "delay_pool_samples": np.zeros((N_PAIRS, 8), np.int64),
        "amplitude_pool": np.zeros((N_PAIRS, 8), np.float64),
        "scramble_phase_rad": np.zeros((N_PAIRS, NUM_PULSES), np.float64),
    }
    reference = _reference()
    for index in range(N_PAIRS):
        pair_seed = BASE_SEED + index
        rng = np.random.default_rng(pair_seed)
        # This H0 isolates cross-PRI coherence only.  A random common arrival
        # would require a second, fitted alignment stage and would confound the
        # pre-registered fixed delay dictionary, so it is intentionally zero.
        arrival = 0
        delays = np.sort(rng.choice(np.arange(DELAY_START, DELAY_STOP + 1), size=8, replace=False))
        amplitudes = rng.uniform(0.5, 1.5, size=8)
        jnr_db = float(rng.uniform(0.0, 8.0))
        one_pri = np.zeros(SAMPLES_PER_PRI, np.complex128)
        for delay, amplitude in zip(delays, amplitudes, strict=True):
            start = arrival + int(delay)
            if start + PULSE_SAMPLES > SAMPLES_PER_PRI:
                raise RuntimeError("locked MFT support left its PRI")
            one_pri[start : start + PULSE_SAMPLES] += amplitude * reference
        train = np.tile(one_pri, NUM_PULSES)
        target_power = 10.0 ** (jnr_db / 10.0)
        train *= np.sqrt(target_power / max(np.mean(np.abs(train) ** 2), np.finfo(float).eps))
        noise_seed = pair_seed + NOISE_SEED_OFFSET
        noise_rng = np.random.default_rng(noise_seed)
        noise = (noise_rng.standard_normal(shape[1]) + 1j * noise_rng.standard_normal(shape[1])) / np.sqrt(2.0)
        phases = rng_phase.uniform(-np.pi, np.pi, size=NUM_PULSES)
        # Anchor the first PRI phase; the remaining three are independent and
        # neither arm changes per-PRI power or any intra-PRI delay statistic.
        phases[0] = 0.0
        phase_train = np.repeat(np.exp(1j * phases), SAMPLES_PER_PRI)
        coherent["iq"][index] = (train + noise).astype(np.complex64)
        scrambled["iq"][index] = (train * phase_train + noise).astype(np.complex64)
        common["pair_seed"][index] = pair_seed
        common["receiver_noise_seed"][index] = noise_seed
        common["mft_total_jnr_db"][index] = jnr_db
        common["arrival_delay_samples"][index] = arrival
        common["delay_pool_samples"][index] = delays
        common["amplitude_pool"][index] = amplitudes
        common["scramble_phase_rad"][index] = phases
    for name, source in (("mft_coherent.mat", coherent), ("mft_scrambled.mat", scrambled)):
        payload = {key: value.copy() for key, value in common.items()}
        payload.update(source)
        _write_arm(OUTPUT / name, payload)
    receipt.update(
        {
            "coherent_path": str(OUTPUT / "mft_coherent.mat"),
            "scrambled_path": str(OUTPUT / "mft_scrambled.mat"),
            "coherent_sha256": _sha256(OUTPUT / "mft_coherent.mat"),
            "scrambled_sha256": _sha256(OUTPUT / "mft_scrambled.mat"),
        }
    )
    (receipt_root / "formal_generation_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    print(json.dumps(generate(execute=parser.parse_args().execute), ensure_ascii=False, indent=2))
