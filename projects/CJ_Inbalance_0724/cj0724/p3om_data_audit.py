"""Pre-training P3o-M identity and MFT-count contract audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.p2w_isrj_width_oracle_ood import POWERS
from cj0724.p3c_mft_sparse_deconvolution import COUNT_COLUMN, MFT_CLASS_INDEX, REGIMES
from cj0724.p3o_low_power_attribution import audit_identities


def role_paths(root: Path, role: str, seed: int, suffix: str, prefix: str = "p3om_sparse_peak") -> dict[str, dict[int, Path]]:
    stem = f"{prefix}_{role}_seed{seed}"
    # Pure path construction keeps planning/tests independent of archived
    # D-drive availability.  read_role is the validation boundary and will
    # raise if a requested role is actually executed without its files.
    canonical = list(range(10, 51, 5))
    def expected(root_path: Path) -> dict[int, Path]:
        return {
            power: root_path / f"level_{canonical.index(power) + 1:02d}__jnr_{power}dB"
            / "closedset_6class_train.mat"
            for power in POWERS
        }
    return {regime: expected(root / f"{stem}_{regime}{suffix}") for regime in REGIMES}


def read_role(paths: dict[str, dict[int, Path]]) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    reference_identity = reference_labels = None
    rows = []
    for regime, files in paths.items():
        alignment = audit_power_alignment(files)
        identity, labels = read_identity_metadata(files[25])
        with h5py.File(files[25], "r") as handle:
            values = np.asarray(handle["parameter_values"][COUNT_COLUMN], dtype=np.float64)
            mask = np.asarray(handle["parameter_mask"][COUNT_COLUMN], dtype=np.uint8).astype(bool)
        valid = (labels[:, MFT_CLASS_INDEX] > 0) & mask
        if len(identity) != 2079 or not bool(valid.any()) or bool((valid & ~np.isfinite(values)).any()):
            raise RuntimeError(f"P3o-M malformed MFT count contract: {regime}")
        if bool((~(labels[:, MFT_CLASS_INDEX] > 0) & mask).any()):
            raise RuntimeError(f"P3o-M MFT count mask is active on a negative label: {regime}")
        if reference_identity is None:
            reference_identity, reference_labels = identity, labels
        elif not np.array_equal(reference_identity, identity) or not np.array_equal(reference_labels, labels):
            raise RuntimeError(f"P3o-M roles are not paired across regimes: {regime}")
        rows.append({"regime": regime, "identity_count": int(len(identity)), "valid_mft_count": int(valid.sum()), "alignment": alignment})
    assert reference_identity is not None and reference_labels is not None
    return reference_identity, reference_labels, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rechunk-root", required=True); parser.add_argument("--project", default="."); parser.add_argument("--output", required=True); parser.add_argument("--suffix", default="_rechunked_v2"); parser.add_argument("--prefix", default="p3om_sparse_peak"); parser.add_argument("--train-seed", type=int, default=1570000000); parser.add_argument("--development-seed", type=int, default=1580000000)
    args = parser.parse_args(); root, project = Path(args.rechunk_root), Path(args.project).resolve()
    train_paths = role_paths(root, "train", args.train_seed, args.suffix, args.prefix); development_paths = role_paths(root, "development", args.development_seed, args.suffix, args.prefix)
    train_identity, train_labels, train_rows = read_role(train_paths); development_identity, development_labels, development_rows = read_role(development_paths)
    if not np.array_equal(train_labels, development_labels):
        raise RuntimeError("P3o-M train/development label contract differs")
    reference = json.loads((project / "outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156/protocol.json").read_text(encoding="utf-8"))
    audit = audit_identities(project, train_identity, development_identity, {args.train_seed, args.development_seed}, Path(reference["data_file"]))
    for directory in sorted(root.glob("*_rechunked*")):
        if f"{args.prefix}_train_seed{args.train_seed}" in directory.name or f"{args.prefix}_development_seed{args.development_seed}" in directory.name:
            continue
        prior_file = next(directory.rglob("closedset_6class_train.mat"), None)
        if prior_file is None:
            continue
        prior_identity, _ = read_identity_metadata(prior_file)
        train_overlap = int(np.intersect1d(train_identity, prior_identity).size)
        development_overlap = int(np.intersect1d(development_identity, prior_identity).size)
        audit["rows"].append({"path": str(prior_file.resolve()), "train_overlap": train_overlap, "development_overlap": development_overlap})
        audit["maximum_prior_overlap"] = max(audit["maximum_prior_overlap"], train_overlap, development_overlap)
    audit["prior_files_checked"] = len(audit["rows"])
    if audit["maximum_prior_overlap"] or audit["cross_role_overlap"]:
        raise RuntimeError("P3o-M identities overlap an earlier role")
    audit.update({"prefix": args.prefix, "train_seed": args.train_seed, "development_seed": args.development_seed, "train_contract": train_rows, "development_contract": development_rows, "power_alignment_verified": True, "label_contract_verified": True, "rechunk_root": str(root.resolve()), "suffix": args.suffix, "test_generated": False, "test_opened": False, "locked_test_used": False})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
