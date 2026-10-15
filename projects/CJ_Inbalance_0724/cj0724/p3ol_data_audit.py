"""Audit P3o-L D-drive roots before any model reads development data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.p2w_isrj_width_oracle_ood import POWERS
from cj0724.p3c_mft_sparse_deconvolution import REGIMES
from cj0724.p3o_low_power_attribution import audit_identities


def _role_paths(root: Path, stem: str) -> dict[str, dict[int, Path]]:
    # This helper is a path constructor.  Do not require historical D-drive
    # files merely to construct an audit plan; _read_role performs the actual
    # existence/alignment validation before any data are opened.
    canonical = list(range(10, 51, 5))
    def expected(root_path: Path) -> dict[int, Path]:
        return {
            power: root_path / f"level_{canonical.index(power) + 1:02d}__jnr_{power}dB"
            / "closedset_6class_train.mat"
            for power in POWERS
        }
    return {regime: expected(root / f"{stem}_{regime}_rechunked") for regime in REGIMES}


def _read_role(paths: dict[str, dict[int, Path]]) -> tuple[np.ndarray, np.ndarray]:
    identity = labels = None
    for regime, files in paths.items():
        audit_power_alignment(files); current_identity, current_labels = read_identity_metadata(files[25])
        if identity is None: identity, labels = current_identity, current_labels
        elif not np.array_equal(identity, current_identity) or not np.array_equal(labels, current_labels):
            raise RuntimeError(f"unpaired identity or label contract: {regime}")
    assert identity is not None and labels is not None
    return identity, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rechunk-root", required=True); parser.add_argument("--project", default="."); parser.add_argument("--output", required=True)
    args = parser.parse_args(); root, project = Path(args.rechunk_root), Path(args.project).resolve()
    train_paths = _role_paths(root, "p3ol_component_conditioned_train_seed1530000000"); dev_paths = _role_paths(root, "p3ol_component_conditioned_dev_seed1540000000")
    train_identity, train_labels = _read_role(train_paths); development_identity, development_labels = _read_role(dev_paths)
    if not np.array_equal(train_labels, development_labels): raise RuntimeError("train/development label contract mismatch")
    # The historical project scan is supplemented by the explicit D-drive roots.
    old = json.loads((project / "outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156/protocol.json").read_text(encoding="utf-8"))
    base = audit_identities(project, train_identity, development_identity, {1530000000, 1540000000}, Path(old["data_file"]))
    extra = []
    for directory in sorted(root.glob("*_rechunked")):
        if "p3ol_component_conditioned_train_seed1530000000" in directory.name or "p3ol_component_conditioned_dev_seed1540000000" in directory.name: continue
        path = next(directory.rglob("closedset_6class_train.mat"), None)
        if path is not None: extra.append(path)
    maximum = 0
    for path in extra:
        old_identity, _ = read_identity_metadata(path); train_overlap = int(np.intersect1d(train_identity, old_identity).size); dev_overlap = int(np.intersect1d(development_identity, old_identity).size)
        maximum = max(maximum, train_overlap, dev_overlap); base["rows"].append({"path": str(path.resolve()), "train_overlap": train_overlap, "development_overlap": dev_overlap})
    base["prior_files_checked"] = len(base["rows"]); base["maximum_prior_overlap"] = max(base["maximum_prior_overlap"], maximum)
    if maximum: raise RuntimeError("P3o-L identities overlap D-drive historical role")
    base.update({"train_identity_count": int(len(train_identity)), "development_identity_count": int(len(development_identity)), "power_alignment_verified": True, "label_contract_verified": True, "rechunk_root": str(root.resolve()), "test_generated": False, "test_opened": False, "locked_test_used": False})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(base, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
