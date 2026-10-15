from __future__ import annotations

import base64

from experiments import audit_route_a_labelcorank_transfer_audit as audit


def test_text_specific_source_is_not_admitted_as_exact_iq_reproduction(tmp_path, monkeypatch) -> None:
    source = {
        "train/model6.py": "transformers.AutoModel.from_pretrained(x)\nmatrix_labels_ = np.load(x)\natten_bert_multi\ntorch.matmul(query_layer, key_layer.transpose(-1, -2))\nreturn cls_pre, final_output\n",
        "train/model_train.py": "loss=0.8*loss1+0.2*loss2\n",
        "train/load_data.py": "tokenizer.encode_plus(x)\n",
    }

    def fake_get_json(url: str, github: bool = False) -> dict:
        if "api.crossref.org" in url:
            return {"message": {"title": [audit.TITLE], "publisher": "AI Access Foundation", "author": [{"given": "Yan", "family": "Yan"}]}}
        if "/git/trees/" in url:
            return {"sha": "tree", "tree": [{"path": path} for path in source]}
        if "/contents/" in url:
            path = url.split("/contents/")[1].split("?")[0]
            return {"sha": "blob-" + path, "content": base64.b64encode(source[path].encode()).decode()}
        return {"default_branch": "main", "html_url": "https://example.invalid/LabelCoRank", "license": None}

    monkeypatch.setattr(audit, "_get_json", fake_get_json)
    payload = audit.run(tmp_path / "receipt.json")

    assert payload["admission"]["paper_and_author_linked_public_code_verified"] is True
    assert payload["verified_source_mechanism"]["two_stage_initial_logits_and_reranked_logits"] is True
    assert payload["interface_audit"]["uses_text_transformer_backbone"] is True
    assert payload["interface_audit"]["exact_equal_protocol_reproduction_admissible"] is False
    assert payload["locked_test_used"] is False
    assert payload["test_v3_used"] is False
