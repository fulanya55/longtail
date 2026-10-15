"""Generate P2 external-physics absent/weak/strong counterfactual triplets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.generate_route_a_p1_observability_adequate_pairs import (
    COMPONENTS as P1_COMPONENTS, DATA_ROOT, FORMAL_IDENTITIES, LABELS, LENGTH,
    REFERENCE, SMOKE_IDENTITIES, TARGETS, _delay, _receiver_seed, _rng, _scale,
)


FORMAL_BASE_SEED = 4_150_000_000
SMOKE_BASE_SEED = 4_160_000_000


def _isrj_high_replica(seed: int) -> np.ndarray:
    rng = _rng(seed, 91_230_517)
    width = int(rng.integers(250, 301)); replicas = 9
    period = (replicas + 1) * width
    window = (np.arange(LENGTH) % period < width).astype(np.float64)
    repeated = sum((_delay(REFERENCE * window, item * width) for item in range(replicas + 1)))
    return _delay(repeated, int(rng.integers(100, 501)))


COMPONENTS = {**P1_COMPONENTS, "ISRJ": _isrj_high_replica}


def _triplet(seed: int, target: str) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    rng = _rng(seed, 71_231_007)
    labels = np.zeros(6, dtype=np.uint8); powers = np.full(6, np.nan, dtype=np.float32); base = np.zeros(LENGTH, dtype=np.complex128)
    for index, name in enumerate(LABELS):
        if name != target and rng.random() < .45:
            jnr = 8.0 if rng.random() < .35 else 20.0
            base += _scale(COMPONENTS[name](seed), jnr); labels[index] = 1; powers[index] = jnr
    if not labels.any():
        fallback = "FMB" if target != "FMB" else "SawFM"; index = LABELS.index(fallback)
        base += _scale(COMPONENTS[fallback](seed), 20.0); labels[index] = 1; powers[index] = 20.0
    noise_rng = np.random.default_rng(_receiver_seed(seed)); noise = (noise_rng.standard_normal(LENGTH) + 1j * noise_rng.standard_normal(LENGTH)) / np.sqrt(2)
    target_component, target_index = COMPONENTS[target](seed), LABELS.index(target)
    signals, all_labels, all_power = [], [], []
    for jnr in (np.nan, 8.0, 20.0):
        value = base + noise; current_labels, current_power = labels.copy(), powers.copy()
        if np.isfinite(jnr):
            value += _scale(target_component, float(jnr)); current_labels[target_index] = 1; current_power[target_index] = float(jnr)
        signals.append(value); all_labels.append(current_labels); all_power.append(current_power)
    return tuple(signals), np.asarray(all_labels), np.asarray(all_power)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(*, smoke: bool, output: Path, receipt: Path) -> dict:
    identities = SMOKE_IDENTITIES if smoke else FORMAL_IDENTITIES; base_seed = SMOKE_BASE_SEED if smoke else FORMAL_BASE_SEED
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True); receipt.parent.mkdir(parents=True, exist_ok=True)
    views = identities * 3
    meta = {"status":"completed","experiment":"RouteA_P2_FullTargetObservability","evidence_role":"smoke_only" if smoke else "external_protocol_h0_development_only","source_scope":"project_local_new_physical_protocol_not_p1_original_0716_or_test_v3","identity_count":identities,"view_count":views,"base_seed":base_seed,"labels":LABELS,"record_samples":LENGTH,"mft_delays_samples":[1200,2300,3400],"vft_hz_range":[20_000,60_000],"isrj_replicas":9,"isrj_width_samples":[250,300],"arms":["absent","weak","strong"],"arm_jnr_db":[None,8.0,20.0],"test_split_generated":False,"locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    with h5py.File(output, "w") as handle:
        real = handle.create_dataset("iq_real", (views,LENGTH), dtype="f4", chunks=(1,LENGTH), compression="gzip", compression_opts=1)
        imag = handle.create_dataset("iq_imag", (views,LENGTH), dtype="f4", chunks=(1,LENGTH), compression="gzip", compression_opts=1)
        labels = handle.create_dataset("labels", (views,6), dtype="u1"); power = handle.create_dataset("component_jnr_db", (views,6), dtype="f4")
        pair = handle.create_dataset("pair_seed", (views,), dtype="u8"); receiver = handle.create_dataset("receiver_noise_seed", (views,), dtype="u8"); target = handle.create_dataset("target_index", (views,), dtype="u1"); arm = handle.create_dataset("visibility_arm", (views,), dtype="u1")
        for identity in range(identities):
            target_name = TARGETS[identity % len(TARGETS)]; seed = base_seed + identity; signals, triplet_labels, triplet_power = _triplet(seed, target_name)
            for local, signal in enumerate(signals):
                row=identity*3+local; real[row]=signal.real.astype(np.float32); imag[row]=signal.imag.astype(np.float32); labels[row]=triplet_labels[local]; power[row]=triplet_power[local]; pair[row]=seed; receiver[row]=_receiver_seed(seed); target[row]=LABELS.index(target_name); arm[row]=local
    meta["source_path"] = str(output); meta["sha256"] = _sha256(output); receipt.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); return meta


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--execute",action="store_true"); parser.add_argument("--smoke",action="store_true"); parser.add_argument("--tag",default="r1"); args=parser.parse_args()
    base="p2_full_target_observability_smoke_20260803" if args.smoke else "p2_full_target_observability_20260803"; output=DATA_ROOT/f"{base}_{args.tag}"/"closedset_6class.mat"; receipt=ROOT/"outputs/tapc/route_a_p2_full_target_generation"/(f"smoke_receipt_{args.tag}.json" if args.smoke else f"formal_receipt_{args.tag}.json")
    if args.execute: print(json.dumps(generate(smoke=args.smoke,output=output,receipt=receipt),ensure_ascii=False))
    else:
        payload={"status":"planned","experiment":"RouteA_P2_FullTargetObservability","smoke":args.smoke,"test_v3_used":False}; receipt.parent.mkdir(parents=True,exist_ok=True); receipt.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); print(json.dumps(payload))
