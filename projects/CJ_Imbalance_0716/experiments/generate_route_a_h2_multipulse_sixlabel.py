"""Generate role-isolated local staggered-PRI six-label H2 sources.

Formal roles use the preregistered counts; ``--smoke`` creates a separately
named small source and is never admissible development evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"C:\LJQ_code\干扰数据")
FS, PULSE, PRI = 10e6, 1000, 3000
INTERVALS = np.asarray((3000, 3310, 3000, 3310, 3000, 3310, 3000), dtype=np.int64)
STARTS = np.r_[0, np.cumsum(INTERVALS)].astype(np.int64)
LENGTH = int(STARTS[-1] + PRI)
LABELS = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
FORMAL = {"train-mp-v1": (5_040, 2_900_000_000), "development-mp-v1": (2_520, 3_000_000_000), "external-mp-v1": (2_520, 3_100_000_000)}
NOISE_SALT = 2 * 2_246_822_519


def receiver_seed(seed: int) -> int:
    return int((np.uint64(seed) + np.uint64(NOISE_SALT)) % np.uint64(2**32))


def _rng(seed: int, salt: int) -> np.random.Generator:
    return np.random.default_rng((np.uint64(seed) + np.uint64(salt)) % np.uint64(2**32))


def _scale(signal: np.ndarray, jnr_db: float) -> np.ndarray:
    return signal * np.sqrt(10.0 ** (jnr_db / 10.0) / max(float(np.mean(np.abs(signal) ** 2)), np.finfo(np.float64).eps))


def _delay(signal: np.ndarray, amount: int) -> np.ndarray:
    result = np.zeros_like(signal)
    if amount < signal.size:
        result[amount:] = signal[:-amount] if amount else signal
    return result


def _lfm_train(fd_hz: float, arrival: int) -> np.ndarray:
    local_t = np.arange(PULSE) / FS
    lfm = np.exp(1j * np.pi * (5e6 / 100e-6) * local_t**2)
    result = np.zeros(LENGTH, dtype=np.complex128)
    for start in STARTS:
        index = int(start) + arrival + np.arange(PULSE)
        absolute_t = (int(start) + arrival + np.arange(PULSE)) / FS
        result[index] = lfm * np.exp(2j * np.pi * fd_hz * absolute_t)
    return result


def _fmb(seed: int) -> np.ndarray:
    rng = _rng(seed, 747_796_405); bandwidth = min((1.5 + 1.5 * rng.random()) * 5e6, FS * (1 - 1e-12))
    white = rng.standard_normal(LENGTH); frequency = np.fft.fftfreq(LENGTH, 1 / FS)
    colored = np.fft.ifft(np.fft.fft(white) * (np.abs(frequency) <= bandwidth / 2)).real
    colored = (colored - colored.mean()) / max(np.max(np.abs(colored - colored.mean())), np.finfo(float).eps)
    return np.roll(np.exp(2j * np.pi * np.cumsum((bandwidth / 2) * colored) / FS), int(rng.integers(100, 501)))


def _saw(seed: int) -> np.ndarray:
    rng = _rng(seed, 2_891_336_453); bandwidth = (2 + 3 * rng.random()) * 1e6; period = (20 + 60 * rng.random()) * 1e-6
    time = np.arange(LENGTH) / FS; frequency = -bandwidth / 2 + bandwidth * np.mod(time, period) / period
    return np.roll(np.exp(2j * np.pi * np.cumsum(frequency) / FS), int(rng.integers(100, 501)))


def _sin(seed: int) -> np.ndarray:
    rng = _rng(seed, 1_181_783_497); center = (.5 + rng.random()) * 1e6; deviation = (1 + 2 * rng.random()) * 1e6; modulation = (10 + 10 * rng.random()) * 1e3
    frequency = center + deviation * np.cos(2 * np.pi * modulation * np.arange(LENGTH) / FS)
    return np.roll(np.exp(2j * np.pi * np.cumsum(frequency) / FS), int(rng.integers(100, 501)))


def _mft(seed: int) -> np.ndarray:
    rng = _rng(seed, 3_160_543_729); count = int(rng.integers(3, 9)); delays = rng.choice(np.arange(300, 501), size=count, replace=False); amplitude = .5 + rng.random(count)
    base = _lfm_train(0.0, 0); return sum((float(a) * _delay(base, int(d))) for a, d in zip(amplitude, delays))


def _isrj(seed: int) -> np.ndarray:
    rng = _rng(seed, 1_013_904_223); width = int(rng.integers(50, 201)); replicas = int(rng.integers(1, 6)); base = _lfm_train(0.0, 0)
    window = (np.arange(LENGTH) % ((replicas + 1) * width) < width).astype(np.float64)
    return _delay(sum((_delay(base * window, item * width) for item in range(replicas + 1))), int(rng.integers(100, 501)))


def _sample_pair(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = _rng(seed, 91_827_364); background = rng.random(5) < .55
    if not background.any(): background[int(rng.integers(0, 5))] = True
    weak = rng.random(5) < .35; powers = np.where(weak, 8.0, 20.0)
    components = (_fmb, _saw, _sin, _mft, _isrj); signal = np.zeros(LENGTH, dtype=np.complex128)
    labels = np.zeros(6, dtype=np.uint8); component_jnr = np.full(6, np.nan, dtype=np.float32)
    label_index = (0, 1, 2, 3, 5)
    for index, active in enumerate(background):
        if active:
            target = label_index[index]
            signal += _scale(components[index](seed), float(powers[index])); labels[target] = 1; component_jnr[target] = powers[index]
    noise_rng = np.random.default_rng(receiver_seed(seed)); noise = (noise_rng.standard_normal(LENGTH) + 1j * noise_rng.standard_normal(LENGTH)) / np.sqrt(2)
    absent = signal + noise; present = absent.copy(); vft_rng = _rng(seed, 1_664_525); present += _scale(_lfm_train(1000 + 6000 * vft_rng.random(), int(vft_rng.integers(100, 501))), 8.0)
    present_labels = labels.copy(); present_labels[4] = 1; present_jnr = component_jnr.copy(); present_jnr[4] = 8.0
    return absent, present, np.stack((labels, present_labels)), np.stack((component_jnr, present_jnr))


def generate(role: str, count: int, base_seed: int, *, smoke: bool, output: Path, receipt: Path) -> dict[str, object]:
    if count % 2: raise ValueError("count must be even because rows are VFT counterfactual pairs")
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True); receipt.parent.mkdir(parents=True, exist_ok=True)
    meta = {"status": "completed", "experiment": "RouteA_H2_MultiPulseSixLabel", "role": role, "evidence_role": "smoke_only" if smoke else "formal_role_isolated_train_development_external", "source_scope": "project_local_staggered_pri_physics_extension_not_official_uniform_pri_0716", "sample_count": count, "pair_count": count // 2, "base_pair_seed": base_seed, "sample_seed_range": [base_seed, base_seed + count // 2 - 1], "record_samples": LENGTH, "pulse_starts": STARTS.tolist(), "labels": LABELS, "test_split_generated": False, "test_files_opened": False, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    with h5py.File(output, "w") as h5:
        real = h5.create_dataset("iq_real", (count, LENGTH), dtype="f4", chunks=(1, LENGTH), compression="gzip", compression_opts=1)
        imag = h5.create_dataset("iq_imag", (count, LENGTH), dtype="f4", chunks=(1, LENGTH), compression="gzip", compression_opts=1)
        labels = h5.create_dataset("labels", (count, 6), dtype="u1"); power = h5.create_dataset("component_jnr_db", (count, 6), dtype="f4")
        pairs = h5.create_dataset("pair_seed", (count,), dtype="u8"); receiver = h5.create_dataset("receiver_noise_seed", (count,), dtype="u8"); arm = h5.create_dataset("vft_counterfactual_arm", (count,), dtype="u1")
        for pair_index in range(count // 2):
            seed = base_seed + pair_index; absent, present, pair_labels, pair_power = _sample_pair(seed)
            for local, value in enumerate((absent, present)):
                row = pair_index * 2 + local; real[row] = value.real.astype(np.float32); imag[row] = value.imag.astype(np.float32); labels[row] = pair_labels[local]; power[row] = pair_power[local]; pairs[row] = seed; receiver[row] = receiver_seed(seed); arm[row] = local
    digest = hashlib.sha256()
    with output.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    meta["source_path"] = str(output); meta["sha256"] = digest.hexdigest()
    receipt.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--role", choices=tuple(FORMAL) + ("smoke",), default="smoke"); p.add_argument("--execute", action="store_true"); p.add_argument("--count", type=int, default=None); p.add_argument("--tag", default="r1"); return p.parse_args()


if __name__ == "__main__":
    args = parse_args(); formal = args.role in FORMAL; default_count, base = FORMAL[args.role] if formal else (32, 3_200_000_000); count = args.count or default_count
    root = ROOT / "outputs/tapc/route_a_h2_multipulse_generation"; name = f"{args.role.replace('-', '_')}_{args.tag}"
    if not args.execute:
        payload = {"status": "planned", "role": args.role, "sample_count": count, "execute": False, "test_v3_used": False}; (root / f"{name}_dry_run_receipt.json").parent.mkdir(parents=True, exist_ok=True); (root / f"{name}_dry_run_receipt.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"); print(json.dumps(payload));
    else:
        source_root = DATA_ROOT / (f"h2_multipulse_sixlabel_smoke_20260803_{args.tag}" if not formal else f"h2_multipulse_sixlabel_{args.role}_20260803_{args.tag}")
        value = generate(args.role, count, base, smoke=not formal, output=source_root / "closedset_6class.mat", receipt=root / f"{name}_generation_receipt.json"); print(json.dumps({"status": value["status"], "role": value["role"], "sha256": value["sha256"]}, ensure_ascii=False))
