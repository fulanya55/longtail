"""Audit whether LabelCoRank is an admissible Route-A external comparator.

This is a provenance and interface audit only. It never opens 0716 I/Q,
labels, predictions, checkpoints, or any sealed test manifest. It separates an
author-linked implementation from an exact equal-protocol reproduction: the
released LabelCoRank model is a BERT text model and relies on a dataset-specific
precomputed label co-occurrence matrix.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOI = "10.1613/jair.1.16968"
TITLE = "LabelCoRank: Revolutionizing Long Tail Multi-Label Classification with Co-Occurrence Reranking"
REPOSITORY = "821code/LabelCoRank"
GITHUB_API = "https://api.github.com"
REQUIRED_PATHS = {
    "model": "train/model6.py",
    "training": "train/model_train.py",
    "text_loader": "train/load_data.py",
}


def _get_json(url: str, github: bool = False) -> dict:
    headers = {"User-Agent": "RouteA-Reproducibility-Audit/1.0 (research reproducibility check)"}
    if github:
        headers["Accept"] = "application/vnd.github+json"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _content(repository: str, path: str, ref: str) -> dict:
    payload = _get_json(f"{GITHUB_API}/repos/{repository}/contents/{path}?ref={ref}", github=True)
    raw = base64.b64decode(str(payload["content"]).encode("ascii"))
    return {
        "github_blob_sha": payload.get("sha"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "text": raw.decode("utf-8", errors="replace"),
    }


def run(output: Path) -> dict:
    crossref = _get_json(f"https://api.crossref.org/works/{DOI}")["message"]
    metadata = _get_json(f"{GITHUB_API}/repos/{REPOSITORY}", github=True)
    branch = str(metadata["default_branch"])
    tree = _get_json(f"{GITHUB_API}/repos/{REPOSITORY}/git/trees/{branch}?recursive=1", github=True)
    paths = {str(item["path"]) for item in tree.get("tree", [])}
    contents = {name: _content(REPOSITORY, path, branch) for name, path in REQUIRED_PATHS.items()}
    model_text = contents["model"]["text"]
    training_text = contents["training"]["text"]
    loader_text = contents["text_loader"]["text"]

    text_backbone = "transformers.AutoModel.from_pretrained" in model_text
    tokenized_text_input = "tokenizer.encode_plus" in loader_text
    precomputed_cooccurrence_matrix = "matrix_labels_" in model_text and "np.load" in model_text
    two_stage_logits = "return cls_pre, final_output" in model_text and "loss=0.8*loss1+0.2*loss2" in training_text
    source_paths_present = {name: path in paths for name, path in REQUIRED_PATHS.items()}

    payload = {
        "status": "completed_read_only_provenance_and_transfer_audit",
        "experiment": "RouteA_LabelCoRank2025TransferAudit",
        "paper": {
            "doi": DOI,
            "expected_title": TITLE,
            "crossref_title": crossref["title"][0],
            "publisher": crossref.get("publisher"),
            "authors": [f"{item.get('given', '')} {item.get('family', '')}".strip() for item in crossref.get("author", [])],
        },
        "repository": {
            "name": REPOSITORY,
            "html_url": metadata.get("html_url"),
            "default_branch": branch,
            "tree_sha": tree.get("sha"),
            "license_spdx": (metadata.get("license") or {}).get("spdx_id"),
            "paths_present": source_paths_present,
            "source_blobs": {name: {key: value for key, value in content.items() if key != "text"} for name, content in contents.items()},
        },
        "verified_source_mechanism": {
            "two_stage_initial_logits_and_reranked_logits": two_stage_logits,
            "top_ranked_labels_expanded_by_precomputed_cooccurrence": precomputed_cooccurrence_matrix,
            "label_query_attention_over_input_sequence": "atten_bert_multi" in model_text and "torch.matmul(query_layer, key_layer.transpose(-1, -2))" in model_text,
        },
        "interface_audit": {
            "uses_text_transformer_backbone": text_backbone,
            "uses_tokenized_text_input": tokenized_text_input,
            "cooccurrence_matrix_construction_released_in_repository": False,
            "current_route_a_f2_global_iq_interface_matches_source_exactly": False,
            "exact_equal_protocol_reproduction_admissible": False,
        },
        "admission": {
            "paper_and_author_linked_public_code_verified": bool(crossref["title"][0] == TITLE and all(source_paths_present.values())),
            "decision": "do_not_claim_or_run_as_an_exact_labelcorank_reproduction",
            "allowed_next_step": "A separately preregistered LabelCoRank-adapted external-mechanism comparator may be designed only with a training-role-only cooccurrence matrix, an identity-isolated fresh development source, matched capacity/optimizer/seeds, and no visibility, test, or frozen-logit inputs.",
            "prohibited_shortcuts": [
                "Do not copy the unlicensed source into this project.",
                "Do not construct the cooccurrence matrix from validation, threshold, OOD, locked-test, test-v2, or test-v3 labels.",
                "Do not rename a custom F2 adaptation as an exact LabelCoRank reproduction.",
            ],
        },
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
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_labelcorank_transfer_audit/receipt.json")
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
