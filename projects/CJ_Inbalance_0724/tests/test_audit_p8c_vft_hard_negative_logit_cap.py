import numpy as np

from cj0724.audit_p8c_vft_hard_negative_logit_cap import inspect


def test_p8c_audit_reports_any_historic_train_identity_overlap(tmp_path, monkeypatch):
    labels = np.asarray([[int(bit) for bit in format(value, "06b")] for value in range(1, 64)], dtype=np.uint8)
    labels = np.repeat(labels, 400, axis=0)
    fresh = np.arange(len(labels), dtype=np.int64) + 1000
    reference = tmp_path / "closedset_6class_train.mat"; reference.touch()
    import cj0724.audit_p8c_vft_hard_negative_logit_cap as module
    monkeypatch.setattr(module, "read_identity_metadata", lambda _: (np.asarray([1000, 999999]), labels[:2]))
    detail = inspect(fresh, labels, [reference])
    assert detail["reference_file_count"] == 1
    assert detail["identity_overlap_total"] == 1
    assert str(reference) in detail["identity_overlap_by_reference"]
