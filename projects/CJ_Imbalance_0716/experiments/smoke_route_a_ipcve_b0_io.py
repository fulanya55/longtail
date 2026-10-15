"""Real-MAT B0 forward/backward/artifact smoke for IP-CVE Stage 0."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.models.backbone_adapter import build_backbone  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _complex(dataset) -> np.ndarray:
    raw = np.asarray(dataset)
    return raw["real"].astype(np.float32) + 1j * raw["imag"].astype(np.float32)


def _load(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(path, "r") as handle:
        values = _complex(handle["iq_complex"])
        labels = np.asarray(handle["label_selected"], dtype=np.float32)
    iq = np.stack([values.real, values.imag], axis=1).astype(np.float32)
    rms = np.sqrt(np.mean(np.sum(iq * iq, axis=1), axis=1, keepdims=True))
    iq = iq / np.maximum(rms[:, :, None], 1e-8)
    return torch.from_numpy(iq), torch.from_numpy(labels)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("E:/LJQ_code")
        / "干扰数据"
        / "route_a_ipcve_development_cf_v1_20260807"
        / "ipcve_triples_smoke.mat",
    )
    parser.add_argument("--seed", type=int, default=2039)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/route_a_ipcve/stage0/b0_io_smoke",
    )
    args = parser.parse_args()

    torch.manual_seed(20260807)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = (
        PROJECT_ROOT
        / "outputs/tapc/route_a_backbone_b0_development_v1"
        / "seed_{}".format(args.seed)
        / "best_model.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_backbone("dfms_iq", 6)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    values, labels = _load(args.data_path)
    values = values.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        maps = model.forward_structure_maps(values)
        logits = model(values)
    probe = nn.Linear(int(maps["global"].shape[1]), 3).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    output = probe(maps["global"].detach())
    loss = nn.functional.binary_cross_entropy_with_logits(output, labels[:, 3:6])
    loss.backward()
    gradient_norm = float(
        torch.sqrt(
            sum(
                (parameter.grad.detach() ** 2).sum()
                for parameter in probe.parameters()
                if parameter.grad is not None
            )
        ).item()
    )
    optimizer.step()

    args.output_root.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_root / "throwaway_probe_after_one_step.pth"
    torch.save({"state_dict": probe.state_dict(), "smoke_only": True}, artifact_path)
    receipt = {
        "status": "completed_real_mat_forward_backward_artifact_smoke",
        "evidence_role": "smoke_only_not_model_evidence",
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "data_path": str(args.data_path),
        "data_sha256": _sha256(args.data_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "input_shape": list(values.shape),
        "logits_shape": list(logits.shape),
        "global_shape": list(maps["global"].shape),
        "temporal_shape": list(maps["temporal"].shape),
        "loss": float(loss.detach().item()),
        "gradient_norm": gradient_norm,
        "finite": bool(
            torch.isfinite(logits).all()
            and torch.isfinite(loss)
            and np.isfinite(gradient_norm)
        ),
        "artifact_path": str(artifact_path),
        "artifact_sha256": _sha256(artifact_path),
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    receipt_path = args.output_root / "run_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["finite"] and gradient_norm > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
