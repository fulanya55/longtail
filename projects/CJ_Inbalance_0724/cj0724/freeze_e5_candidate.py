"""Create an auditable frozen E5 development-candidate bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import h5py
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/vft_physics_e5_frozen_candidate_20260725"),
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    seeds = (20260724, 20260725, 20260726, 271828, 314159)
    rows = []

    for seed in seeds:
        if seed in (20260724, 20260725, 20260726):
            baseline = Path(
                f"outputs/vft_d11_backbone_loss_20260725/VFT/D11/b1_bce/seed_{seed}"
            )
            physics = (
                Path("outputs/vft_physics_screen_20260725")
                if seed == 20260724
                else Path(f"outputs/vft_physics_screen_20260725/seed_{seed}")
            )
            fusion = Path(f"outputs/vft_physics_fusion_20260725/seed_{seed}")
        else:
            baseline = Path(
                f"outputs/vft_d11_e5_confirmation_20260725/VFT/D11/b1_bce/seed_{seed}"
            )
            physics = Path(f"outputs/vft_physics_confirmation_20260725/seed_{seed}")
            fusion = Path(f"outputs/vft_physics_confirmation_fusion_20260725/seed_{seed}")

        protocol = json.loads((fusion / "protocol.json").read_text(encoding="utf-8"))
        manifest = json.loads((baseline / "longtail_manifest.json").read_text(encoding="utf-8"))
        audit = json.loads((baseline / "data_audit.json").read_text(encoding="utf-8"))
        features = np.load(physics / "features_and_predictions.npz")
        train_indices = features["train_indices"].astype(np.int64)
        with h5py.File(audit["levels"]["10"]["path"], "r") as handle:
            target = np.asarray(handle["label_selected"][4, train_indices], dtype=np.uint8)
        classifier = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=7,
            l2_regularization=1.0,
            random_state=seed,
        )
        classifier.fit(features["features_train"], target)
        seed_dir = args.output_root / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        classifier_path = seed_dir / "vft_physics_classifier.joblib"
        joblib.dump(classifier, classifier_path, compress=3)
        frozen_protocol = {
            "seed": seed,
            "neural_checkpoint": str((baseline / "best.pt").resolve()),
            "neural_checkpoint_sha256": sha256(baseline / "best.pt"),
            "physics_classifier": str(classifier_path.resolve()),
            "physics_classifier_sha256": sha256(classifier_path),
            "physics_weight": float(protocol["selected_physics_weight"]),
            "thresholds": protocol["thresholds"],
            "manifest_sha256": manifest["manifest_sha256"],
            "split_sha256": protocol["split_sha256"],
            "frequency_grid_hz": np.arange(-8000.0, 8000.1, 500.0).tolist(),
            "feature_dimension": 72,
            "classifier_parameters": {
                "max_iter": 200,
                "learning_rate": 0.05,
                "max_leaf_nodes": 7,
                "l2_regularization": 1.0,
                "random_state": seed,
            },
            "locked_test_used": False,
            "locked_test_authorized": False,
        }
        protocol_path = seed_dir / "frozen_protocol.json"
        protocol_path.write_text(
            json.dumps(frozen_protocol, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rows.append(
            {
                "seed": seed,
                "protocol_path": str(protocol_path.resolve()),
                "protocol_sha256": sha256(protocol_path),
                "neural_checkpoint_sha256": frozen_protocol["neural_checkpoint_sha256"],
                "physics_classifier_sha256": frozen_protocol[
                    "physics_classifier_sha256"
                ],
                "manifest_sha256": frozen_protocol["manifest_sha256"],
                "split_sha256": frozen_protocol["split_sha256"],
            }
        )

    bundle = {
        "method": "e5_e0_plus_vft_matched_filter",
        "status": "frozen_development_candidate",
        "seeds": list(seeds),
        "confirmation_protocol": str(
            Path("configs/e5_confirmation_20260725.json").resolve()
        ),
        "confirmation_protocol_sha256": sha256(
            Path("configs/e5_confirmation_20260725.json")
        ),
        "five_seed_result": str(
            Path("outputs/vft_physics_e5_five_seed_20260725/result.json").resolve()
        ),
        "five_seed_result_sha256": sha256(
            Path("outputs/vft_physics_e5_five_seed_20260725/result.json")
        ),
        "locked_test_used": False,
        "locked_test_authorized": False,
        "members": rows,
    }
    bundle_path = args.output_root / "bundle_manifest.json"
    bundle_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_root / "bundle_manifest.sha256").write_text(
        f"{sha256(bundle_path)}  bundle_manifest.json\n", encoding="ascii"
    )
    print(json.dumps(bundle, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
