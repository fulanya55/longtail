import pytest
import torch

from cj0724.concurrent_task_route import ConcurrentTaskRouteRunner
from cj0724.multitask import build_multitask_model


def test_concurrent_runner_rejects_non_routed_model():
    with pytest.raises(TypeError):
        ConcurrentTaskRouteRunner(build_multitask_model("legacy_mlcnn_global", head_dim=32))


def test_concurrent_runner_cpu_fallback_matches_serial_forward():
    torch.manual_seed(9)
    model = build_multitask_model(
        "mlcnn_iq_grouped_task_routed_w4", head_dim=32, parameter_hidden_dim=16
    ).eval()
    sample = torch.randn(2, 2, 3000)
    runner = ConcurrentTaskRouteRunner(model)
    with torch.no_grad():
        serial = model(sample)
        fallback = runner(sample)
    assert torch.equal(serial[0], fallback[0])
    assert torch.equal(serial[1], fallback[1])
