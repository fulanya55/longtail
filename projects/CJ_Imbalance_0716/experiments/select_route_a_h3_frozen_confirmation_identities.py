"""Pre-generation identity audit for the one-shot H3 frozen-candidate source."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"C:\LJQ_code\干扰数据")
COUNT, BASE_SEED, RECEIVER_OFFSET = 25_200, 134_000_000, 2 * 2_246_822_519


def _receiver(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)


def run(args: argparse.Namespace) -> dict[str, object]:
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 seal is not intact")
    samples = np.arange(BASE_SEED + 1, BASE_SEED + COUNT + 1, dtype=np.uint64)
    planned = ((samples + np.uint64(RECEIVER_OFFSET)) % np.uint64(2**32)).astype(np.int64)
    rows=[]
    for path in DATA_ROOT.rglob("*train*.mat"):
        try: overlap=int(np.intersect1d(planned,_receiver(path)).size)
        except (OSError,KeyError): continue
        rows.append({"path":str(path),"receiver_overlap_count":overlap})
    total=int(sum(row["receiver_overlap_count"] for row in rows))
    result={"status":"completed_pre_generation_identity_selection","experiment":"RouteA_H3_FrozenUnifiedExternalConfirmation","base_sample_seed":BASE_SEED,"sample_seed_range":[BASE_SEED+1,BASE_SEED+COUNT],"receiver_seed_range":[int(planned.min()),int(planned.max())],"sample_count":COUNT,"identity_overlap_count":total,"all_historic_train_receiver_overlap_zero":total==0,"audited_train_sources":rows,"test_files_opened":False,"locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    if total: raise ValueError(f"H3 planned receiver identities overlap historic train data: {total}")
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--test-v3-manifest",type=Path,default=ROOT/"frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json"); parser.add_argument("--output",type=Path,default=ROOT/"outputs/tapc/route_a_h3_frozen_confirmation_generation/pre_generation_identity_selection.json"); print(json.dumps(run(parser.parse_args()),ensure_ascii=False,indent=2))
