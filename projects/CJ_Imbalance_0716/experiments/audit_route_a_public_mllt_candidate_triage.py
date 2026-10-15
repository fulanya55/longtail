"""Read-only provenance and interface triage for public recent MLLT repositories.

This audit does not download a repository into the project and does not inspect
the 0716 data.  A public GitHub repository is not treated as a fair numerical
baseline merely because its README calls itself "official".  Admission requires
both traceable method provenance and a compatible, exact protocol boundary.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GITHUB_API = "https://api.github.com"
CANDIDATES = {
    "tec_graph": {
        "repository": "ZhuJiaYucode/TEC-Graph",
        "claimed_method": "Typed Error-Correction Graph for Long-Tailed Multi-Label Recognition",
        "interface_markers": ["model/visual_anchor_adapter.py", "data/voc.py", "model/typed_signed_unified_graph.py"],
    },
    "talc": {
        "repository": "JiajunChen223/RuralTail-MLR",
        "claimed_method": "Tail-Aware Label Conversion for Long-Tailed Multi-Label Remote-Sensing Recognition",
        "interface_markers": ["src/ruraltail_mlr/losses/decision.py", "data/processed/images_index.csv", "configs/protocol/paper_talc.yaml"],
    },
}


def _get_json(url: str) -> dict:
    request = Request(
        url,
        headers={"User-Agent": "RouteA-Reproducibility-Audit/1.0", "Accept": "application/vnd.github+json"},
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _content(repository: str, path: str, ref: str) -> dict | None:
    try:
        payload = _get_json(f"{GITHUB_API}/repos/{repository}/contents/{path}?ref={ref}")
    except Exception:
        return None
    encoded = payload.get("content")
    if not isinstance(encoded, str):
        return None
    raw = base64.b64decode(encoded.encode("ascii"))
    return {
        "github_blob_sha": payload.get("sha"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "text": raw.decode("utf-8", errors="replace"),
    }


def _has_doi(text: str) -> bool:
    return bool(re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, flags=re.IGNORECASE))


def _triage_one(name: str, spec: dict) -> dict:
    repository = str(spec["repository"])
    metadata = _get_json(f"{GITHUB_API}/repos/{repository}")
    branch = str(metadata["default_branch"])
    tree = _get_json(f"{GITHUB_API}/repos/{repository}/git/trees/{branch}?recursive=1")
    paths = {str(item["path"]) for item in tree.get("tree", [])}
    citation = _content(repository, "CITATION.cff", branch)
    readme = _content(repository, "README.md", branch)
    markers = {path: path in paths for path in spec["interface_markers"]}
    citation_text = citation["text"] if citation is not None else ""
    license_id = ((metadata.get("license") or {}).get("spdx_id"))
    primary_traceable = bool(citation is not None and _has_doi(citation_text))
    exact_interface_candidate = bool(all(markers.values()))
    # A source cannot become an exact comparator when the publication identity
    # is unavailable, even if it exposes a technically inspectable loss.
    admissible = bool(primary_traceable and exact_interface_candidate)
    return {
        "name": name,
        "repository": repository,
        "repository_html_url": metadata.get("html_url"),
        "default_branch": branch,
        "created_at": metadata.get("created_at"),
        "updated_at": metadata.get("updated_at"),
        "license_spdx": license_id,
        "claimed_method": spec["claimed_method"],
        "citation": None if citation is None else {k: citation[k] for k in ("github_blob_sha", "sha256")},
        "citation_has_doi": _has_doi(citation_text),
        "readme": None if readme is None else {k: readme[k] for k in ("github_blob_sha", "sha256")},
        "interface_marker_presence": markers,
        "primary_method_provenance_traceable": primary_traceable,
        "exact_interface_candidate": exact_interface_candidate,
        "exact_equal_protocol_reproduction_admissible": admissible,
        "decision": (
            "requires_full_method_and_protocol_audit"
            if admissible
            else "do_not_run_or_rename_as_equal_protocol_recent_mllt_baseline"
        ),
    }


def run(output: Path) -> dict:
    rows = [_triage_one(name, spec) for name, spec in CANDIDATES.items()]
    payload = {
        "status": "completed_read_only_public_candidate_triage",
        "experiment": "RouteA_PublicMLLTRecentCandidateTriage",
        "candidates": rows,
        "any_exact_equal_protocol_reproduction_admissible": any(
            row["exact_equal_protocol_reproduction_admissible"] for row in rows
        ),
        "decision": "no_new_recent_mllt_baseline_admitted" if not any(
            row["exact_equal_protocol_reproduction_admissible"] for row in rows
        ) else "candidate_requires_separate_preregistered_protocol",
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/tapc/route_a_public_mllt_candidate_triage/public_candidate_triage.json",
    )
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
