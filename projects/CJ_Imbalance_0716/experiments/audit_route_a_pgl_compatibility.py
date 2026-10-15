"""Read-only, pinned-commit compatibility audit for the authors' PGL code."""
from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


REPOSITORY = "Kunmonkey/PGLoss"
COMMIT = "d11a6520b0241946672364d974a5ebbd43c554e0"
PATH = "mllt/models/losses/pg_loss.py"


def fetch() -> bytes:
    url = f"https://api.github.com/repos/{REPOSITORY}/contents/{PATH}?ref={COMMIT}"
    request = Request(url, headers={"User-Agent": "CJLT-reproducibility-audit"})
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    return base64.b64decode(payload["content"])


def audit(output: Path) -> dict[str, object]:
    content = fetch(); text = content.decode("utf-8")
    try:
        ast.parse(text); syntax = {"compiles": True, "error": None}
    except SyntaxError as exc:
        syntax = {"compiles": False, "error": f"{exc.__class__.__name__}: {exc.msg} at line {exc.lineno}"}
    findings = {
        "uses_project_independent_image_framework_mmcv": "import mmcv" in text,
        "hardcodes_cuda_in_loss_constructor": ".cuda()" in text,
        "undefined_apaf_reference": "self.apaf = apaf" in text,
        "undefined_neg_scale1_reference": "neg_scale1" in text,
    }
    exact_reproduction_admissible = bool(syntax["compiles"] and not any(findings.values()))
    result = {
        "status": "completed_read_only_audit", "experiment": "RouteA_PGL_OfficialCodeCompatibility",
        "paper_doi": "10.1609/aaai.v37i2.25244", "repository": REPOSITORY, "commit": COMMIT,
        "path": PATH, "blob_sha256": hashlib.sha256(content).hexdigest(), "syntax": syntax,
        "findings": findings, "exact_equal_protocol_reproduction_admissible": exact_reproduction_admissible,
        "decision": "do_not_patch_or_rename_as_pgl" if not exact_reproduction_admissible else "requires_separate_protocol_registration",
        "external_used": False, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = audit(Path(__file__).resolve().parents[1] / "outputs" / "tapc" / "route_a_pgl_compatibility_audit" / "summary.json")
    print(json.dumps(result, ensure_ascii=False))
