"""H4-labelled wrapper around the audited H3 three-arm source contract."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_route_a_h3_vft_three_strata_source import audit as audit_triplets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--source", type=Path, required=True); parser.add_argument("--receipt", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--other-source", type=Path, action="append", default=[])
    args = parser.parse_args(); result = audit_triplets(args.source, args.receipt, args.output, other_sources=tuple(args.other_source))
    result["experiment"] = "RouteA_H4_VFTBeamspaceMapP0"; result["role"] = json.loads(args.receipt.read_text(encoding="utf-8"))["role"]; result["p0_model_training_authorized"] = False
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, ensure_ascii=False))
