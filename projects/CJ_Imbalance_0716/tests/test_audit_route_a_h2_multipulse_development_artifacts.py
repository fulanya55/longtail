from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "h2_artifact_audit", ROOT / "experiments" / "audit_route_a_h2_multipulse_development_artifacts.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_artifact_audit_hashes_every_formal_run_and_checks_capacity(tmp_path: Path) -> None:
    for seed in MODULE.SEEDS:
        for mode in MODULE.MODES:
            short = mode.lower().replace("-mp", "")
            root = tmp_path / f"run_{short}_seed{seed}"
            artifact_root = root / f"{mode}_seed{seed}"
            artifact_root.mkdir(parents=True)
            for name in MODULE.REQUIRED_RUN_FILES:
                (artifact_root / name).write_bytes(name.encode("utf-8"))
            count = 28104 if mode in {"C3-mp", "C4-mp"} else 26000
            receipt = {
                "status": "completed_development_unaggregated",
                "smoke": False,
                "locked_test_used": False,
                "test_v2_used": False,
                "test_v3_used": False,
                "modes": [mode],
                "seeds": [seed],
                "source": {"train": "train-mp-v1", "development": "development-mp-v1"},
                "results": [{"mode": mode, "seed": seed, "best_epoch": 3, "parameter_count": count}],
            }
            (root / "run_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    output = tmp_path / "audit.json"
    generation_root = tmp_path / "generation"
    generation_root.mkdir()
    for role, name in MODULE.SOURCE_RECEIPTS.items():
        source_receipt = {
            "status": "completed",
            "role": role,
            "test_files_opened": False,
            "locked_test_used": False,
            "test_v2_used": False,
            "test_v3_used": False,
            "sha256": "a" * 64,
            "source_path": f"{role}.mat",
            "sample_count": 10,
            "pair_count": 5,
            "sample_seed_range": [1, 5],
        }
        (generation_root / name).write_text(json.dumps(source_receipt), encoding="utf-8")
    value = MODULE.audit(str(tmp_path / "run_{mode}_seed{seed}"), output, generation_root)
    assert value["record_count"] == 25
    assert value["c3_c4_equal_parameter_count"] is True
    assert set(value["source_provenance"]) == {"train-mp-v1", "development-mp-v1"}
    assert json.loads(output.read_text(encoding="utf-8"))["locked_test_used"] is False
