"""Audit whether the learned low/neutral/high side router collapsed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cj0724.dual_axis import POWER_LEVELS, PowerViewDataset, discover_power_train_files, read_identity_metadata
from cj0724.models import SideRoutedPhysicsResidualHead
from cj0724.multitask import build_multitask_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--residual-root", default="outputs/side_routed_physics_full_20260725")
    parser.add_argument("--profile", default="side_r03")
    parser.add_argument("--power-root", default=r"C:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1")
    parser.add_argument("--output", default="outputs/side_routed_physics_full_20260725/router_audit.json")
    parser.add_argument("--batch-size", type=int, default=126)
    args = parser.parse_args()
    run_root = Path(args.run_root)
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    roles_npz = np.load(run_root / "role_indices.npz")
    confirm = roles_npz["confirm"].astype(np.int64)
    files = discover_power_train_files(args.power_root)
    _, labels = read_identity_metadata(files[30])
    selected = {
        int(row["seed"]): row for row in summary
        if row["model"] == "mlcnn_iq_grouped_task_routed" and row.get("checkpoint_objective") == "parameter"
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed, row in selected.items():
        base = build_multitask_model(
            row["model"], protocol["width"], protocol["dropout"], protocol["head_dim"],
            protocol["parameter_hidden_dim"], protocol["legacy_attention"],
        ).to(device)
        base.load_state_dict(torch.load(row["checkpoint"], map_location=device, weights_only=False)["model_state"])
        base.eval()
        checkpoint = torch.load(
            Path(args.residual_root) / args.profile / f"seed_{seed}" / "best.pt",
            map_location=device, weights_only=False,
        )
        head = SideRoutedPhysicsResidualHead(
            width=8, hidden_dim=64, dropout=0.1,
            residual_bound=float(checkpoint["profile_config"]["residual_bound"]),
        ).to(device)
        head.load_state_dict(checkpoint["model_state"])
        head.eval()
        for level in POWER_LEVELS:
            dataset = PowerViewDataset(files, labels, confirm, (level,), seed)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            probabilities = []
            with torch.no_grad():
                for batch in loader:
                    iq = batch["iq"].to(device)
                    output = base(iq)
                    logits = output[0] if isinstance(output, tuple) else output
                    probabilities.append(head.forward_components(iq, logits)["route_probability"].cpu().numpy())
            probability = np.concatenate(probabilities)
            target = 2 if level < 30 else 1 if level == 30 else 0
            rows.append({
                "seed": seed,
                "power_level_db": level,
                "target_route": target,
                "mean_fixed_weak_probability": float(probability[:, 0].mean()),
                "mean_neutral_probability": float(probability[:, 1].mean()),
                "mean_swept_weak_probability": float(probability[:, 2].mean()),
                "route_accuracy": float((probability.argmax(1) == target).mean()),
                "route_entropy": float((-(probability * np.log(np.clip(probability, 1e-8, 1.0))).sum(1)).mean()),
            })
            dataset.close()
        del base, head
        if device.type == "cuda":
            torch.cuda.empty_cache()
    aggregate = {
        "mean_route_accuracy": float(np.mean([row["route_accuracy"] for row in rows])),
        "low_side_accuracy": float(np.mean([row["route_accuracy"] for row in rows if row["power_level_db"] < 30])),
        "neutral_accuracy": float(np.mean([row["route_accuracy"] for row in rows if row["power_level_db"] == 30])),
        "high_side_accuracy": float(np.mean([row["route_accuracy"] for row in rows if row["power_level_db"] > 30])),
        "mean_entropy": float(np.mean([row["route_entropy"] for row in rows])),
        "collapsed": bool(max(
            np.mean([row["mean_fixed_weak_probability"] for row in rows]),
            np.mean([row["mean_neutral_probability"] for row in rows]),
            np.mean([row["mean_swept_weak_probability"] for row in rows]),
        ) > 0.95),
    }
    payload = {"aggregate": aggregate, "per_seed_level": rows, "locked_test_used": False}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
