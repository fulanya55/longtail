"""Hash audit for all formal H3 model artifacts; never opens any test role."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MODES = ("T0-h3", "T1-h3", "T2-h3")
SEEDS = (20261003, 20261011, 20261019, 20261027, 20261104)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(generation_root: Path, name: str, role: str) -> dict[str, object]:
    path = generation_root / name
    item = json.loads(path.read_text(encoding="utf-8"))
    if item.get("status") != "completed" or item.get("role") != role or any(item.get(k, True) for k in ("external_used", "locked_test_used", "test_v2_used", "test_v3_used")):
        raise ValueError(f"source boundary or role failure: {path}")
    return {"receipt": str(path), "receipt_sha256": _sha(path), "source_path": item["source_path"], "source_sha256": item["sha256"], "sample_count": item["sample_count"]}


def run(args: argparse.Namespace) -> dict[str, object]:
    provenance = {"train": _source(args.generation_root, "train_3strata_v1_r1_generation_receipt.json", "train-3strata-v1"), "development": _source(args.generation_root, "development_3strata_v1_r1_generation_receipt.json", "development-3strata-v1")}
    records, counts = [], {mode: [] for mode in MODES}
    for mode in MODES:
        for seed in args.seeds:
            root = Path(args.root_template.format(mode=mode.replace("-", "_"), seed=seed)); receipt_path = root / "run_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("status") != "completed_development_unaggregated" or receipt.get("smoke") or receipt.get("modes") != [mode] or receipt.get("seeds") != [seed] or any(receipt.get(k, True) for k in ("external_used", "locked_test_used", "test_v2_used", "test_v3_used")):
                raise ValueError(f"formal receipt invalid: {receipt_path}")
            source = receipt["source"]
            if source.get("train_sha256") != provenance["train"]["source_sha256"] or source.get("development_sha256") != provenance["development"]["source_sha256"]:
                raise ValueError(f"source SHA mismatch: {receipt_path}")
            artifact = root / f"{mode}_seed{seed}"; result = receipt["results"][0]
            if result.get("mode") != mode or int(result.get("seed")) != seed: raise ValueError(f"result identity mismatch: {receipt_path}")
            files = {}
            for path in (receipt_path, artifact / "checkpoint.pt", artifact / "development_predictions.npz"):
                if not path.is_file(): raise ValueError(f"missing artifact: {path}")
                files[path.name] = {"bytes": path.stat().st_size, "sha256": _sha(path)}
            counts[mode].append(int(result["parameter_count"]))
            records.append({"mode": mode, "seed": seed, "root": str(root), "best_epoch": int(result["best_epoch"]), "parameter_count": int(result["parameter_count"]), "files": files})
    equal = counts["T1-h3"] == counts["T2-h3"]
    if not equal: raise ValueError("T1/T2 capacity mismatch")
    value = {"status": "completed_artifact_audit", "experiment": "RouteA_H3_VFTThreeStrataIdentifiability", "record_count": len(records), "source_provenance": provenance, "records": records, "parameter_counts_per_mode": counts, "t1_t2_equal_parameter_count": equal, "external_used": False, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-template", default="outputs/tapc/route_a_h3_vft_{mode}_seed{seed}")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--generation-root", type=Path, default=Path("outputs/tapc/route_a_h3_vft_three_strata_generation"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tapc/route_a_h3_vft_three_strata_artifact_audit.json"))
    value = run(parser.parse_args())
    print(json.dumps({"status": value["status"], "record_count": value["record_count"]}, ensure_ascii=False))
