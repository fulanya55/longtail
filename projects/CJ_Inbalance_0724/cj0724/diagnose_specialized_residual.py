"""Inspect whether the two VFT residual heads learn distinct evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from cj0724.dual_axis import PowerViewDataset, read_identity_metadata
from cj0724.models import build_benchmark_model


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "outputs/vft_d11_hard_negative_20260725/VFT/D11/"
            "e3_specialized_residual_moe"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    rows = []
    for run_dir in sorted(args.run_root.glob("seed_*")):
        seed = int(run_dir.name.removeprefix("seed_"))
        audit = json.loads((run_dir / "data_audit.json").read_text(encoding="utf-8"))
        files = {int(level): Path(item["path"]) for level, item in audit["levels"].items()}
        _, labels = read_identity_metadata(files[30])
        predictions = np.load(run_dir / "development_predictions.npz")
        indices = predictions["sample_indices_10"]
        dataset = PowerViewDataset(files, labels, indices, (10,), seed)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = build_benchmark_model(
            "efficientnet_b1_vft_residual_moe_1d",
            num_classes=6,
            dropout=0.2,
            head_dim=256,
        ).to(device)
        checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        collected = {
            key: []
            for key in (
                "logits",
                "general_logits",
                "weak_positive_score",
                "hard_negative_score",
                "target_residual",
            )
        }
        with torch.no_grad():
            for batch in loader:
                components = model.forward_components(batch["iq"].to(device))
                for key in collected:
                    collected[key].append(components[key].float().cpu().numpy())
        dataset.close()
        values = {key: np.concatenate(parts) for key, parts in collected.items()}
        target = labels[indices, 4]
        general_aupr = float(
            average_precision_score(target, sigmoid(values["general_logits"][:, 4]))
        )
        fused_aupr = float(average_precision_score(target, sigmoid(values["logits"][:, 4])))
        residual = values["target_residual"]
        rows.append(
            {
                "seed": seed,
                "general_target_aupr": general_aupr,
                "fused_target_aupr": fused_aupr,
                "fused_minus_general_aupr": fused_aupr - general_aupr,
                "weak_positive_head_aupr": float(
                    average_precision_score(target, values["weak_positive_score"])
                ),
                "hard_negative_head_negative_ap": float(
                    average_precision_score(1 - target, values["hard_negative_score"])
                ),
                "specialist_score_correlation": float(
                    np.corrcoef(
                        values["weak_positive_score"], values["hard_negative_score"]
                    )[0, 1]
                ),
                "residual_positive_mean": float(residual[target == 1].mean()),
                "residual_negative_mean": float(residual[target == 0].mean()),
                "residual_abs_max": float(np.abs(residual).max()),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No E3 runs found under {args.run_root}")
    csv_path = args.run_root.parent.parent.parent / "specialized_residual_diagnostics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metric_names = [name for name in rows[0] if name != "seed"]
    result = {
        "locked_test_used": False,
        "runs": rows,
        "means": {
            name: float(np.mean([row[name] for row in rows])) for name in metric_names
        },
    }
    json_path = args.run_root.parent.parent.parent / "specialized_residual_diagnostics.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
