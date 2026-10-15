"""Generate P1's no-fit, observability-adequate counterfactual triplets.

The source is deliberately project-local and is never an original-0716 test
role.  Every formal identity produces absent, weak, and strong target views.
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
FS = 10e6
PULSE = 1_000
PRI = 5_000
PULSE_COUNT = 6
LENGTH = PRI * PULSE_COUNT
K = 5e6 / 100e-6
LABELS = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
TARGETS = ("MFT", "VFT", "ISRJ")
FORMAL_IDENTITIES = 900
FORMAL_BASE_SEED = 4_200_000_000
SMOKE_IDENTITIES = 9
SMOKE_BASE_SEED = 4_250_000_000


def _rng(seed: int, salt: int) -> np.random.Generator:
    return np.random.default_rng((np.uint64(seed) + np.uint64(salt)) % np.uint64(2**32))


def _receiver_seed(seed: int) -> int:
    return int((np.uint64(seed) + np.uint64(1_793_021_117)) % np.uint64(2**32))


def _scale(x: np.ndarray, jnr_db: float) -> np.ndarray:
    power = max(float(np.mean(np.abs(x) ** 2)), np.finfo(np.float64).eps)
    return x * np.sqrt(10.0 ** (jnr_db / 10.0) / power)


def _delay(x: np.ndarray, samples: int) -> np.ndarray:
    out = np.zeros_like(x)
    if samples == 0:
        out[:] = x
    elif samples < x.size:
        out[samples:] = x[:-samples]
    return out


def _reference(fd_hz: float = 0.0) -> np.ndarray:
    local = np.arange(PULSE, dtype=np.float64) / FS
    chirp = np.exp(1j * np.pi * K * local**2)
    out = np.zeros(LENGTH, dtype=np.complex128)
    for pulse in range(PULSE_COUNT):
        start = pulse * PRI
        absolute = (start + np.arange(PULSE, dtype=np.float64)) / FS
        out[start : start + PULSE] = chirp * np.exp(2j * np.pi * fd_hz * absolute)
    return out


REFERENCE = _reference()


def _fmb(seed: int) -> np.ndarray:
    rng = _rng(seed, 11_923_771)
    bandwidth = (1.5 + 1.5 * rng.random()) * 1e6
    white = rng.standard_normal(LENGTH)
    frequency = np.fft.fftfreq(LENGTH, 1 / FS)
    coloured = np.fft.ifft(np.fft.fft(white) * (np.abs(frequency) <= bandwidth / 2)).real
    coloured = (coloured - coloured.mean()) / max(np.max(np.abs(coloured - coloured.mean())), np.finfo(float).eps)
    return np.exp(2j * np.pi * np.cumsum((bandwidth / 2) * coloured) / FS)


def _saw(seed: int) -> np.ndarray:
    rng = _rng(seed, 22_947_193)
    bandwidth = (2.0 + 3.0 * rng.random()) * 1e6
    period = (20.0 + 60.0 * rng.random()) * 1e-6
    time = np.arange(LENGTH, dtype=np.float64) / FS
    frequency = -bandwidth / 2 + bandwidth * np.mod(time, period) / period
    return np.exp(2j * np.pi * np.cumsum(frequency) / FS)


def _sin(seed: int) -> np.ndarray:
    rng = _rng(seed, 31_571_909)
    centre = (0.5 + rng.random()) * 1e6
    deviation = (1.0 + 2.0 * rng.random()) * 1e6
    modulation = (10.0 + 10.0 * rng.random()) * 1e3
    time = np.arange(LENGTH, dtype=np.float64) / FS
    frequency = centre + deviation * np.cos(2 * np.pi * modulation * time)
    return np.exp(2j * np.pi * np.cumsum(frequency) / FS)


def _mft(seed: int) -> np.ndarray:
    rng = _rng(seed, 41_729_301)
    amplitude = 0.5 + rng.random(3)
    delays = (1_200, 2_300, 3_400)
    return sum(float(a) * _delay(REFERENCE, delay) for a, delay in zip(amplitude, delays))


def _vft(seed: int) -> np.ndarray:
    frequency = float(_rng(seed, 51_818_441).uniform(20_000.0, 60_000.0))
    return _reference(frequency)


def _isrj(seed: int) -> np.ndarray:
    rng = _rng(seed, 61_991_701)
    width = int(rng.integers(50, 201))
    replicas = int(rng.integers(1, 6))
    period = (replicas + 1) * width
    window = (np.arange(LENGTH) % period < width).astype(np.float64)
    repeated = sum((_delay(REFERENCE * window, item * width) for item in range(replicas + 1)))
    return _delay(repeated, int(rng.integers(100, 501)))


COMPONENTS = {"FMB": _fmb, "SawFM": _saw, "SinFM": _sin, "MFT": _mft, "VFT": _vft, "ISRJ": _isrj}


def _triplet(seed: int, target: str) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    rng = _rng(seed, 71_231_007)
    labels = np.zeros(6, dtype=np.uint8)
    powers = np.full(6, np.nan, dtype=np.float32)
    base = np.zeros(LENGTH, dtype=np.complex128)
    for index, name in enumerate(LABELS):
        if name == target:
            continue
        if rng.random() < 0.45:
            jnr = 8.0 if rng.random() < 0.35 else 20.0
            base += _scale(COMPONENTS[name](seed), jnr)
            labels[index] = 1; powers[index] = jnr
    if not labels.any():
        fallback = "FMB" if target != "FMB" else "SawFM"
        index = LABELS.index(fallback)
        base += _scale(COMPONENTS[fallback](seed), 20.0)
        labels[index] = 1; powers[index] = 20.0
    noise_rng = np.random.default_rng(_receiver_seed(seed))
    noise = (noise_rng.standard_normal(LENGTH) + 1j * noise_rng.standard_normal(LENGTH)) / np.sqrt(2)
    component = COMPONENTS[target](seed)
    target_index = LABELS.index(target)
    signals, all_labels, all_power = [], [], []
    for arm, jnr in enumerate((np.nan, 8.0, 20.0)):
        value = base + noise
        local_labels, local_power = labels.copy(), powers.copy()
        if np.isfinite(jnr):
            value = value + _scale(component, float(jnr))
            local_labels[target_index] = 1; local_power[target_index] = float(jnr)
        signals.append(value); all_labels.append(local_labels); all_power.append(local_power)
    return tuple(signals), np.asarray(all_labels), np.asarray(all_power)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(*, smoke: bool, output: Path, receipt: Path) -> dict:
    identities = SMOKE_IDENTITIES if smoke else FORMAL_IDENTITIES
    base_seed = SMOKE_BASE_SEED if smoke else FORMAL_BASE_SEED
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True); receipt.parent.mkdir(parents=True, exist_ok=True)
    views = identities * 3
    meta = {
        "status": "completed", "experiment": "RouteA_P1_ObservabilityAdequate",
        "evidence_role": "smoke_only" if smoke else "external_protocol_h0_development_only",
        "source_scope": "project_local_new_physical_protocol_not_original_0716_or_test_v3",
        "identity_count": identities, "view_count": views, "base_seed": base_seed,
        "labels": LABELS, "fs_hz": FS, "pulse_samples": PULSE, "pri_samples": PRI,
        "pulse_count": PULSE_COUNT, "record_samples": LENGTH,
        "mft_delays_samples": [1200, 2300, 3400], "vft_hz_range": [20_000, 60_000],
        "arms": ["absent", "weak", "strong"], "arm_jnr_db": [None, 8.0, 20.0],
        "test_split_generated": False, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    with h5py.File(output, "w") as handle:
        real = handle.create_dataset("iq_real", (views, LENGTH), dtype="f4", chunks=(1, LENGTH), compression="gzip", compression_opts=1)
        imag = handle.create_dataset("iq_imag", (views, LENGTH), dtype="f4", chunks=(1, LENGTH), compression="gzip", compression_opts=1)
        labels = handle.create_dataset("labels", (views, 6), dtype="u1")
        power = handle.create_dataset("component_jnr_db", (views, 6), dtype="f4")
        pair = handle.create_dataset("pair_seed", (views,), dtype="u8")
        receiver = handle.create_dataset("receiver_noise_seed", (views,), dtype="u8")
        target = handle.create_dataset("target_index", (views,), dtype="u1")
        arm = handle.create_dataset("visibility_arm", (views,), dtype="u1")
        for identity in range(identities):
            target_name = TARGETS[identity % len(TARGETS)]
            seed = base_seed + identity
            signals, triplet_labels, triplet_power = _triplet(seed, target_name)
            for local, signal in enumerate(signals):
                row = identity * 3 + local
                real[row] = signal.real.astype(np.float32); imag[row] = signal.imag.astype(np.float32)
                labels[row] = triplet_labels[local]; power[row] = triplet_power[local]
                pair[row] = seed; receiver[row] = _receiver_seed(seed); target[row] = LABELS.index(target_name); arm[row] = local
    meta["source_path"] = str(output); meta["sha256"] = _sha256(output)
    receipt.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true"); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tag", default="r1")
    args = parser.parse_args()
    base_name = "p1_observability_adequate_smoke_20260803" if args.smoke else "p1_observability_adequate_20260803"
    name = f"{base_name}_{args.tag}"
    output = DATA_ROOT / name / "closedset_6class.mat"
    receipt = ROOT / "outputs/tapc/route_a_p1_observability_adequate_generation" / (f"smoke_receipt_{args.tag}.json" if args.smoke else f"formal_receipt_{args.tag}.json")
    if not args.execute:
        payload = {"status": "planned", "experiment": "RouteA_P1_ObservabilityAdequate", "smoke": args.smoke, "test_v3_used": False}
        receipt.parent.mkdir(parents=True, exist_ok=True); receipt.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"); print(json.dumps(payload));
    else:
        print(json.dumps(generate(smoke=args.smoke, output=output, receipt=receipt), ensure_ascii=False))
