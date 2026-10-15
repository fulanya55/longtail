import json
from argparse import Namespace

from experiments.audit_route_a_h3_vft_three_strata_artifacts import run


def _source(role, sha):
    return {"status": "completed", "role": role, "source_path": f"{role}.mat", "sha256": sha, "sample_count": 3,
            "external_used": False, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}


def test_h3_artifact_audit_requires_the_same_formal_source_and_matched_t1_t2_capacity(tmp_path):
    generation = tmp_path / "generation"; generation.mkdir()
    train_sha, dev_sha = "a" * 64, "b" * 64
    (generation / "train_3strata_v1_r1_generation_receipt.json").write_text(json.dumps(_source("train-3strata-v1", train_sha)), encoding="utf-8")
    (generation / "development_3strata_v1_r1_generation_receipt.json").write_text(json.dumps(_source("development-3strata-v1", dev_sha)), encoding="utf-8")
    seed = 7
    for mode, count in (("T0-h3", 10), ("T1-h3", 11), ("T2-h3", 11)):
        root = tmp_path / f"route_{mode.replace('-', '_')}_seed{seed}"; artifact = root / f"{mode}_seed{seed}"; artifact.mkdir(parents=True)
        receipt = {"status": "completed_development_unaggregated", "smoke": False, "modes": [mode], "seeds": [seed],
                   "external_used": False, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
                   "source": {"train_sha256": train_sha, "development_sha256": dev_sha},
                   "results": [{"mode": mode, "seed": seed, "best_epoch": 2, "parameter_count": count}]}
        (root / "run_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        (artifact / "checkpoint.pt").write_bytes(b"checkpoint")
        (artifact / "development_predictions.npz").write_bytes(b"predictions")
    output = tmp_path / "audit.json"
    value = run(Namespace(root_template=str(tmp_path / "route_{mode}_seed{seed}"), seeds=[seed], generation_root=generation, output=output))
    assert value["record_count"] == 3
    assert value["t1_t2_equal_parameter_count"] is True
    assert json.loads(output.read_text(encoding="utf-8"))["locked_test_used"] is False
