"""Refresh auditable GEDFD metadata and public-code availability.

The audit deliberately distinguishes an absent public repository from an
absent private author implementation.  It only records public, reproducible
evidence and never licenses an approximate implementation under the GEDFD
name.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOI = "10.1109/TMM.2025.3535395"
TITLE = "Towards Gradient Equalization and Feature Diversification for Long-Tailed Multi-Label Image Recognition"
GITHUB_QUERY_TERMS = {
    "exact_title": TITLE,
    "doi": DOI,
    "acronym_and_topic": "GEDFD long-tailed multi-label",
    "author_family_and_topic": "Chen Cui Zhang Deng Xia Lu long-tailed multi-label",
}


def _get_json(url: str, github: bool = False) -> dict:
    headers = {"User-Agent": "RouteA-Reproducibility-Audit/1.0 (research reproducibility check)"}
    if github:
        headers["Accept"] = "application/vnd.github+json"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _github_search(term: str) -> tuple[str, dict]:
    url = f"https://api.github.com/search/repositories?q={quote(term)}&per_page=20"
    return url, _get_json(url, github=True)


def run(output: Path) -> dict:
    crossref = _get_json(f"https://api.crossref.org/works/{DOI}")["message"]
    searches = {}
    for label, term in GITHUB_QUERY_TERMS.items():
        url, response = _github_search(term)
        searches[label] = {
            "term": term,
            "url": url,
            "repository_count": int(response["total_count"]),
            "repositories": [item["full_name"] for item in response.get("items", [])],
        }
    exact = searches["exact_title"]
    any_public_hit = any(item["repository_count"] > 0 for item in searches.values())
    payload = {
        "status": "completed",
        "experiment": "RouteA_LatestMLLTPublicCodeRefresh",
        "doi": DOI,
        "expected_title": TITLE,
        "crossref_title": crossref["title"][0],
        "crossref_authors": [item["family"] for item in crossref.get("author", [])],
        "github_exact_title_query": exact["url"],
        "github_exact_title_repository_count": exact["repository_count"],
        "github_exact_title_repositories": exact["repositories"],
        "github_cross_query_searches": searches,
        "official_or_auditable_code_located": False,
        "decision": "do_not_implement_or_numerically_compare_gedfd_without_full_algorithm_or_auditable_author_code",
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    if any_public_hit:
        payload["official_or_auditable_code_located"] = None
        payload["decision"] = "manual_repository_authorship_and_algorithm_audit_required_before_any_implementation"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_latest_mllt_reproducibility_audit/public_code_refresh_20260801.json")
    args = parser.parse_args(); run(args.output)


if __name__ == "__main__":
    main()
