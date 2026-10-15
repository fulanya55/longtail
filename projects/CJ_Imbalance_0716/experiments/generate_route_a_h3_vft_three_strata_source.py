"""Generate fresh three-strata VFT counterfactual sources for H3.

Every identity yields a VFT-absent, weak-VFT and strong-VFT view under the
same non-VFT compound components and receiver noise.  This source is a local
staggered-PRI physics protocol, not an original-0716 test role.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np

from experiments.generate_route_a_h2_multipulse_sixlabel import (
    DATA_ROOT, LABELS, LENGTH, _fmb, _isrj, _lfm_train, _mft, _rng, _saw,
    _scale, _sin, receiver_seed,
)


FORMAL = {
    "train-3strata-v1": (6_048, 3_200_000_000),
    "development-3strata-v1": (3_024, 3_300_000_000),
}
ARMS = ("absent", "weak", "strong")
VFT_JNR = (np.nan, 8.0, 20.0)


def _sample_triplet(seed: int) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """Return three same-identity views plus labels and component powers."""
    rng = _rng(seed, 91_827_364)
    background = rng.random(5) < 0.55
    if not background.any():
        background[int(rng.integers(0, 5))] = True
    background_weak = rng.random(5) < 0.35
    powers = np.where(background_weak, 8.0, 20.0)
    components = (_fmb, _saw, _sin, _mft, _isrj)
    indices = (0, 1, 2, 3, 5)
    signal = np.zeros(LENGTH, dtype=np.complex128)
    labels = np.zeros(6, dtype=np.uint8)
    component_jnr = np.full(6, np.nan, dtype=np.float32)
    for item, active in enumerate(background):
        if active:
            label = indices[item]
            signal += _scale(components[item](seed), float(powers[item]))
            labels[label] = 1
            component_jnr[label] = powers[item]

    noise_rng = np.random.default_rng(receiver_seed(seed))
    noise = (noise_rng.standard_normal(LENGTH) + 1j * noise_rng.standard_normal(LENGTH)) / np.sqrt(2)
    vft_rng = _rng(seed, 1_664_525)
    vft = _lfm_train(1000 + 6000 * vft_rng.random(), int(vft_rng.integers(100, 501)))
    signals: list[np.ndarray] = []
    all_labels = np.repeat(labels[None, :], 3, axis=0)
    all_power = np.repeat(component_jnr[None, :], 3, axis=0)
    for arm, jnr in enumerate(VFT_JNR):
        value = signal + noise
        if np.isfinite(jnr):
            value = value + _scale(vft, float(jnr))
            all_labels[arm, 4] = 1
            all_power[arm, 4] = float(jnr)
        signals.append(value)
    return tuple(signals), all_labels, all_power


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(role: str, count: int, base_seed: int, *, smoke: bool, output: Path, receipt: Path) -> dict[str, object]:
    if count % 3:
        raise ValueError("H3 count must be divisible by three counterfactual arms")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    pair_count = count // 3
    meta: dict[str, object] = {
        "status": "completed",
        "experiment": "RouteA_H3_VFTThreeStrataIdentifiability",
        "role": role,
        "evidence_role": "smoke_only" if smoke else "formal_role_isolated_train_development",
        "source_scope": "project_local_staggered_pri_physics_extension_not_official_uniform_pri_0716",
        "sample_count": count,
        "pair_count": pair_count,
        "base_pair_seed": base_seed,
        "sample_seed_range": [base_seed, base_seed + pair_count - 1],
        "record_samples": LENGTH,
        "labels": LABELS,
        "vft_arms": list(ARMS),
        "vft_jnr_db": [None, 8.0, 20.0],
        "test_split_generated": False,
        "test_files_opened": False,
        "external_used": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    with h5py.File(output, "w") as h5:
        real = h5.create_dataset("iq_real", (count, LENGTH), dtype="f4", chunks=(1, LENGTH), compression="gzip", compression_opts=1)
        imag = h5.create_dataset("iq_imag", (count, LENGTH), dtype="f4", chunks=(1, LENGTH), compression="gzip", compression_opts=1)
        labels = h5.create_dataset("labels", (count, 6), dtype="u1")
        power = h5.create_dataset("component_jnr_db", (count, 6), dtype="f4")
        pairs = h5.create_dataset("pair_seed", (count,), dtype="u8")
        receiver = h5.create_dataset("receiver_noise_seed", (count,), dtype="u8")
        arm = h5.create_dataset("vft_visibility_arm", (count,), dtype="u1")
        for pair_index in range(pair_count):
            seed = base_seed + pair_index
            signals, pair_labels, pair_power = _sample_triplet(seed)
            for local, value in enumerate(signals):
                row = pair_index * 3 + local
                real[row] = value.real.astype(np.float32)
                imag[row] = value.imag.astype(np.float32)
                labels[row] = pair_labels[local]
                power[row] = pair_power[local]
                pairs[row] = seed
                receiver[row] = receiver_seed(seed)
                arm[row] = local
    meta["source_path"] = str(output)
    meta["sha256"] = _sha256(output)
    receipt.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=tuple(FORMAL) + ("smoke",), default="smoke")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--tag", default="r1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    formal = args.role in FORMAL
    default_count, base = FORMAL[args.role] if formal else (30, 3_400_000_000)
    count = args.count or default_count
    output_root = ROOT / "outputs" / "tapc" / "route_a_h3_vft_three_strata_generation"
    receipt_name = f"{args.role.replace('-', '_')}_{args.tag}"
    if not args.execute:
        value = {"status": "planned", "experiment": "RouteA_H3_VFTThreeStrataIdentifiability", "role": args.role, "sample_count": count, "execute": False, "test_v3_used": False}
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / f"{receipt_name}_dry_run_receipt.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(value, ensure_ascii=False))
    else:
        source_name = f"h3_vft_three_strata_smoke_20260803_{args.tag}" if not formal else f"h3_vft_three_strata_{args.role}_20260803_{args.tag}"
        value = generate(args.role, count, base, smoke=not formal, output=DATA_ROOT / source_name / "closedset_6class.mat", receipt=output_root / f"{receipt_name}_generation_receipt.json")
        print(json.dumps({"status": value["status"], "role": value["role"], "sha256": value["sha256"]}, ensure_ascii=False))
