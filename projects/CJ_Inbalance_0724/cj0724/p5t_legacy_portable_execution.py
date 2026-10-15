"""P5t source-manifest preflight, then execute the frozen P5s runner."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from cj0724.models import LEGACY_SOURCE_ROOT
from cj0724.p5b1_jnr_physical_oracle import sha256_file
from cj0724.p5b3_bounded_jnr_attachment import load_base
from cj0724.p5s_detached_chunked_execution import validate_protocol as validate_p5s_protocol


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(project: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project / path).resolve()


def _atomic_write_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise RuntimeError(f"P5t immutable receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def validate_legacy_source_manifest(project: Path, protocol: dict) -> dict[str, object]:
    contract = protocol["legacy_source_contract"]
    manifest_path = _resolve(project, contract["manifest_path"])
    if sha256_file(manifest_path) != contract["manifest_sha256"]:
        raise RuntimeError("P5t legacy-source manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "P5t-legacy-source-manifest-v1":
        raise RuntimeError("P5t legacy-source manifest schema mismatch")
    root = LEGACY_SOURCE_ROOT.resolve()
    declared_root = Path(manifest["legacy_source_root"]).resolve()
    if root != declared_root:
        raise RuntimeError(f"P5t runtime legacy root differs from hash-pinned manifest: {root} != {declared_root}")
    forbidden = tuple(str(item).replace("/", "\\").lower() for item in manifest["forbidden_roots"])
    if any(str(root).lower().startswith(prefix) for prefix in forbidden):
        raise RuntimeError(f"P5t legacy root resolves to forbidden historical drive: {root}")
    rows = []
    for item in manifest["p5e_actual_required_files"]:
        path = root / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"P5t required legacy source missing: {path}")
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"P5t required legacy source hash mismatch: {path.name}")
        rows.append({"path": str(path), "sha256": actual})
    if list(contract["required_by_P5e"]) != [Path(row["path"]).name for row in rows]:
        raise RuntimeError("P5t required legacy-source contract changed")
    return {"manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path), "legacy_source_root": str(root), "required_files": rows}


def validate_protocol(project: Path, preregister: Path, freeze_path: Path) -> dict[str, object]:
    parent = validate_p5s_protocol(project, preregister, freeze_path)
    source = validate_legacy_source_manifest(project, parent["protocol"])
    return {**parent, "legacy_source_validation": source}


def validate_five_frozen_models(project: Path, protocol: dict) -> list[dict[str, object]]:
    freeze_path = _resolve(project, protocol["base_candidate"]["freeze_path"])
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    rows = []
    for seed_value in protocol["base_candidate"]["source_model_seeds"]:
        seed = int(seed_value); key = str(seed)
        model, _normalizer, _thresholds, source = load_base(project, seed, freeze["frozen_candidate"]["base_roots"][key])
        state = model.state_dict()
        if not state or not all(value.numel() > 0 for value in state.values()):
            raise RuntimeError(f"P5t loaded invalid frozen state for source seed {seed}")
        rows.append({"source_seed": seed, "checkpoint": source["checkpoint"], "checkpoint_sha256": source["checkpoint_sha256"], "state_key_count": len(state), "state_element_count": int(sum(value.numel() for value in state.values()))})
        del model
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", required=True)
    parser.add_argument("--p5e-freeze", required=True)
    parser.add_argument("--validate-protocol", action="store_true")
    parser.add_argument("--self-test-frozen-models", action="store_true")
    parser.add_argument("--output-root")
    parser.add_argument("--data-roots", nargs="+")
    parser.add_argument("--audits", nargs="+")
    args, passthrough = parser.parse_known_args()
    project = Path(__file__).resolve().parents[1]
    preregister, freeze_path = _resolve(project, args.preregister), _resolve(project, args.p5e_freeze)
    validated = validate_protocol(project, preregister, freeze_path)
    if args.validate_protocol:
        print(json.dumps({"protocol_valid": True, "legacy_source_validation": validated["legacy_source_validation"]}, indent=2)); return
    if args.self_test_frozen_models:
        if args.output_root or args.data_roots or args.audits or passthrough:
            raise RuntimeError("P5t frozen-model self-test takes no execution arguments")
        rows = validate_five_frozen_models(project, validated["protocol"])
        print(json.dumps({"frozen_model_self_test_valid": True, "models": rows, "data_opened": False, "cuda_opened": False}, indent=2)); return
    if not args.output_root or not args.data_roots or not args.audits:
        raise RuntimeError("P5t execution requires the P5s execution arguments")
    if passthrough:
        raise RuntimeError(f"P5t received unexpected arguments: {passthrough}")
    rows = validate_five_frozen_models(project, validated["protocol"])
    output_root = _resolve(project, args.output_root)
    _atomic_write_once(output_root / "p5t_legacy_source_preflight.json", {
        "receipt_schema": "P5t-v1", "created_utc": _utc_now(), "pid": os.getpid(),
        "preregister_sha256": sha256_file(preregister), "legacy_source_validation": validated["legacy_source_validation"],
        "frozen_models": rows, "data_opened": False, "cuda_opened": False,
    })
    delegated = ["-m", "cj0724.p5s_detached_chunked_execution", "--preregister", str(preregister), "--p5e-freeze", str(freeze_path), "--data-roots", *args.data_roots, "--audits", *args.audits, "--output-root", args.output_root]
    os.execv(sys.executable, [sys.executable, *delegated])


if __name__ == "__main__":
    main()
