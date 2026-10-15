"""Audit one H2 local multi-pulse source without opening any test set."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np

from cjlt.models.vft_staggered_pri import record_samples
from experiments.find_route_a_h0o_crosspulse_identity_range import historic_receiver_ids

def _receiver_ids(path: Path) -> set[int]:
    with h5py.File(path, "r") as h5:
        return set(map(int, np.asarray(h5["receiver_noise_seed"]).reshape(-1).astype(np.uint64)))


def audit(path: Path, receipt: Path, output: Path, *, h0q: Path | None = None, h1a: Path | None = None, h1b: Path | None = None, other_sources: tuple[Path, ...] = ()) -> dict[str, object]:
    manifest = json.loads(receipt.read_text(encoding="utf-8"))
    if manifest.get("test_v3_used") or manifest.get("test_files_opened"):
        raise ValueError("receipt violates test boundary")
    with h5py.File(path, "r") as h5:
        names = {"iq_real", "iq_imag", "labels", "component_jnr_db", "pair_seed", "receiver_noise_seed", "vft_counterfactual_arm"}
        if not names.issubset(h5.keys()): raise ValueError("H2 source fields missing")
        real, imag, labels, power = (np.asarray(h5[n]) for n in ("iq_real", "iq_imag", "labels", "component_jnr_db"))
        pair, receiver, arm = (np.asarray(h5[n]).reshape(-1) for n in ("pair_seed", "receiver_noise_seed", "vft_counterfactual_arm"))
    size = labels.shape[0]; groups = [np.flatnonzero(pair == value) for value in np.unique(pair)]
    historic, sources = historic_receiver_ids(); current = set(map(int, receiver.astype(np.uint64)))
    checks = {
        "payload_shape": real.shape == imag.shape == (size, record_samples()),
        "label_shape": labels.shape == (size, 6) and power.shape == (size, 6),
        "finite_iq": bool(np.isfinite(real).all() and np.isfinite(imag).all()),
        "receiver_per_pair_shared": all(np.unique(receiver[index]).size == 1 for index in groups),
        "counterfactual_pair_cardinality": all(index.size == 2 for index in groups),
        "counterfactual_pair_background_labels_shared": all(np.array_equal(labels[index[0], [0,1,2,3,5]], labels[index[1], [0,1,2,3,5]]) for index in groups),
        "counterfactual_vft_absent_present_exact": all(np.array_equal(np.sort(labels[index, 4]), [0,1]) and np.array_equal(np.sort(arm[index]), [0,1]) for index in groups),
        "vft_present_is_8db_weak": bool(np.allclose(power[labels[:,4] == 1, 4], 8.0)),
        "historical_train_identity_overlap_zero": not bool(current.intersection(historic)),
        "h0q_identity_overlap_zero": h0q is None or not bool(current.intersection(_receiver_ids(h0q))),
        "h1a_identity_overlap_zero": h1a is None or not bool(current.intersection(_receiver_ids(h1a))),
        "h1b_identity_overlap_zero": h1b is None or not bool(current.intersection(_receiver_ids(h1b))),
        "declared_other_role_identity_overlap_zero": all(not bool(current.intersection(_receiver_ids(item))) for item in other_sources),
        "test_v3_still_unused": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    if not all(checks.values()): raise ValueError(f"H2 source audit failed: {checks}")
    result = {"status": "completed", "role": manifest["role"], "evidence_role": manifest["evidence_role"], "source_sha256": manifest["sha256"], "contracts": checks, "historic_train_sources_scanned": sources, "sample_count": size, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--source", type=Path, required=True); p.add_argument("--receipt", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--h0q",type=Path);p.add_argument("--h1a",type=Path);p.add_argument("--h1b",type=Path);p.add_argument("--other-source",type=Path,action="append",default=[]); args=p.parse_args(); print(json.dumps(audit(args.source,args.receipt,args.output,h0q=args.h0q,h1a=args.h1a,h1b=args.h1b,other_sources=tuple(args.other_source)),ensure_ascii=False))
