import torch

from cjlt.models.target_specific_complex_evidence import TargetSpecificComplexEvidenceClassifier


def test_zero_scale_recovers_global_path_and_safe_labels():
    torch.manual_seed(24)
    model = TargetSpecificComplexEvidenceClassifier(channels=4, topk=5).eval()
    values = model(torch.randn(3, 2, 3000), return_evidence=True)
    assert values["logits"].shape == (3, 6)
    assert values["local_evidence"].shape[:2] == (3, 3)
    assert torch.equal(values["logits"], values["global_logits"])


def test_source_control_has_same_parameter_count_and_global_path():
    torch.manual_seed(25)
    candidate = TargetSpecificComplexEvidenceClassifier(channels=4, topk=5).eval()
    torch.manual_seed(25)
    control = TargetSpecificComplexEvidenceClassifier(channels=4, topk=5, source_control=True).eval()
    signal = torch.randn(2, 2, 3000)
    c = candidate(signal, return_evidence=True)
    q = control(signal, return_evidence=True)
    assert sum(p.numel() for p in candidate.parameters()) == sum(p.numel() for p in control.parameters())
    assert torch.equal(c["global_logits"], q["global_logits"])
    assert torch.equal(c["logits"], q["logits"])
