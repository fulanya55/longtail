import pytest
import torch

from cj0724.cuda_graph_runner import FixedShapeCUDAGraphRunner
from cj0724.multitask import build_multitask_model


def test_cuda_graph_runner_rejects_cpu_capture():
    model = build_multitask_model("legacy_mlcnn_global", head_dim=32).eval()
    with pytest.raises(ValueError):
        FixedShapeCUDAGraphRunner(model, torch.randn(1, 2, 3000))
