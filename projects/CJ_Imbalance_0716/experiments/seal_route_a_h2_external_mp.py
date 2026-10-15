"""Create a no-overwrite unused guard for the H2 external multi-pulse role."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def seal(receipt: Path, audit: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"external guard already exists: {output}")
    generated=json.loads(receipt.read_text(encoding="utf-8")); checked=json.loads(audit.read_text(encoding="utf-8"))
    if generated["role"] != "external-mp-v1" or checked["role"] != "external-mp-v1":
        raise ValueError("not the H2 external source")
    if not all(checked["contracts"].values()):
        raise ValueError("external source audit is incomplete")
    value={"status":"sealed_unused_external_role","experiment":"RouteA_H2_MultiPulseSixLabel","role":"external-mp-v1","source_path":generated["source_path"],"source_sha256":generated["sha256"],"source_audit":str(audit),"external_access_authorized":False,"external_analysis_allowed":False,"external_used":False,"rerun_allowed_after_execution":False,"locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");return value


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--receipt",type=Path,default=ROOT/"outputs/tapc/route_a_h2_multipulse_generation/external_mp_v1_r1_generation_receipt.json");p.add_argument("--audit",type=Path,default=ROOT/"outputs/tapc/route_a_h2_multipulse_generation/external_mp_v1_r1_source_audit.json");p.add_argument("--output",type=Path,default=ROOT/"frozen/20260803_route_a_h2_external_mp_v1/external_mp_sealed_manifest.json");a=p.parse_args();print(json.dumps(seal(a.receipt,a.audit,a.output),ensure_ascii=False))
