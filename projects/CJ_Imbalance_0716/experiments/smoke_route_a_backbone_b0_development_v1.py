"""Real-MAT CUDA smoke for the fresh Route-A B0 backbone contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.v73_dataset import V73SignalDataset  # noqa: E402
from cjlt.losses.classification import MultiLabelClassificationLoss  # noqa: E402
from cjlt.models.backbone_adapter import build_backbone  # noqa: E402


NON_SAW = (0, 2, 3, 4, 5)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def weighted_focal_loss(
    criterion: MultiLabelClassificationLoss,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    element = criterion.elementwise(logits, targets)
    non_saw = element[:, torch.as_tensor(NON_SAW, device=logits.device)].mean()
    saw = element[:, 1].mean()
    return non_saw + 0.2 * saw


def _seal_flags(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    flags = {
        key: payload.get("test_v3", {}).get(key)
        for key in ("access_authorized", "analysis_allowed", "used")
    }
    if flags != {
        "access_authorized": False,
        "analysis_allowed": False,
        "used": False,
    }:
        raise ValueError("test-v3 seal is not intact")
    return flags


def _stage_shapes(model, values: torch.Tensor) -> dict:
    maps = model.forward_structure_maps(values)
    return {name: list(tensor.shape) for name, tensor in maps.items()}


def run(args):
    output_root = args.output_root.resolve()
    receipt_path = output_root / "smoke_receipt.json"
    started = time.time()
    base_receipt = {
        "experiment": "RouteA_BackboneDevelopmentV1_B0Smoke",
        "evidence_level": "smoke_only_not_formal_model_evidence",
        "data_path": str(args.data_path.resolve()),
        "role_indices_path": str(args.role_indices.resolve()),
        "seed": int(args.seed),
        "test_v3_seal": _seal_flags(args.seal_manifest),
        "test_files_opened": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the registered B0 smoke")
        saved = np.load(args.role_indices)
        fit_ids = saved["fit"].astype(np.int64)
        if len(fit_ids) < args.batch_size * args.steps:
            raise ValueError("fit role is too small for the requested smoke")
        selected = fit_ids[: args.batch_size * args.steps]
        dataset = V73SignalDataset(
            args.data_path,
            selected,
            class_mode="six",
            normalization="complex_rms",
        )
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=0,
        )
        device = torch.device("cuda")
        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))
        torch.cuda.manual_seed_all(int(args.seed))
        torch.cuda.reset_peak_memory_stats(device)

        model = build_backbone(
            "dfms_iq", num_classes=6, width=16, dropout=0.2
        ).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != 257511:
            raise AssertionError(f"B0 parameter drift: {parameter_count}")
        classifier_ids = {id(parameter) for parameter in model.classifier.parameters()}
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        parameter
                        for parameter in model.parameters()
                        if id(parameter) not in classifier_ids
                    ],
                    "lr": 5e-5,
                },
                {"params": list(model.classifier.parameters()), "lr": 5e-4},
            ],
            weight_decay=1e-4,
        )
        criterion = MultiLabelClassificationLoss(
            "focal", gamma_pos=0.0, gamma_neg=3.0
        ).to(device)

        history = []
        first_values = None
        first_shapes = None
        input_rms = None
        model.train()
        for step, batch in enumerate(loader, start=1):
            values = batch["signal"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)
            if first_values is None:
                first_values = values.detach().clone()
                input_rms = torch.sqrt(
                    values[:, 0].square().mean(dim=1)
                    + values[:, 1].square().mean(dim=1)
                ).detach().cpu().numpy()
                model.eval()
                with torch.no_grad():
                    first_shapes = _stage_shapes(model, first_values)
                model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, features = model(values, return_features=True)
            loss = weighted_focal_loss(criterion, logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite smoke loss")
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            if not gradients or not all(torch.isfinite(grad).all() for grad in gradients):
                raise FloatingPointError("missing or non-finite smoke gradients")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            history.append(
                {
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "gradient_norm_before_clip": float(grad_norm.detach().cpu()),
                    "logits_shape": list(logits.shape),
                    "features_shape": list(features.shape),
                }
            )
            if step >= int(args.steps):
                break
        dataset.close()

        checkpoint = output_root / "b0_smoke_checkpoint.pth"
        output_root.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "seed": int(args.seed),
                "steps": int(args.steps),
            },
            checkpoint,
        )
        model.eval()
        with torch.no_grad():
            reference_logits, reference_features = model(
                first_values, return_features=True
            )
        reloaded = build_backbone(
            "dfms_iq", num_classes=6, width=16, dropout=0.2
        ).to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        reloaded.load_state_dict(payload["model_state_dict"], strict=True)
        reloaded.eval()
        with torch.no_grad():
            reload_logits, reload_features = reloaded(
                first_values, return_features=True
            )
        maximum_reload_error = max(
            float((reference_logits - reload_logits).abs().max().cpu()),
            float((reference_features - reload_features).abs().max().cpu()),
        )
        if maximum_reload_error > 1e-6:
            raise AssertionError("checkpoint reload changed B0 outputs")

        receipt = {
            **base_receipt,
            "status": "completed_real_mat_cuda_smoke",
            "device": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "parameter_count": int(parameter_count),
            "data_sha256": _sha256(args.data_path),
            "role_indices_sha256": _sha256(args.role_indices),
            "model_source_sha256": _sha256(
                PROJECT_ROOT / "cjlt" / "models" / "phase_multiscale.py"
            ),
            "loss_source_sha256": _sha256(
                PROJECT_ROOT / "cjlt" / "losses" / "classification.py"
            ),
            "batch_size": int(args.batch_size),
            "steps": int(args.steps),
            "history": history,
            "stage_shapes": first_shapes,
            "input_complex_rms_min": float(np.min(input_rms)),
            "input_complex_rms_max": float(np.max(input_rms)),
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_reload_max_abs_error": maximum_reload_error,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "elapsed_seconds": float(time.time() - started),
        }
        _write(receipt_path, receipt)
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return receipt
    except Exception as exception:
        receipt = {
            **base_receipt,
            "status": "aborted_engineering_failure",
            "error_type": type(exception).__name__,
            "error": str(exception),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": float(time.time() - started),
        }
        _write(receipt_path, receipt)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(
            r"E:\LJQ_code\干扰数据\route_a_backbone-development-v1_20260806"
        )
        / "closedset_6class_train.mat",
    )
    parser.add_argument(
        "--role-indices",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "tapc"
        / "route_a_backbone_development_v1"
        / "roles"
        / "seed_2039"
        / "role_indices.npz",
    )
    parser.add_argument(
        "--seal-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "frozen"
        / "20260731_route_a_test_v3_sealed"
        / "test_v3_sealed_manifest.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "tapc"
        / "route_a_backbone_development_v1"
        / "b0_smoke",
    )
    parser.add_argument("--seed", type=int, default=2039)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
