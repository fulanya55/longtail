"""Fixed-shape CUDA Graph inference runner."""

from __future__ import annotations

import torch
from torch import nn


class FixedShapeCUDAGraphRunner:
    """Capture one model and one input shape for low-overhead replay."""

    def __init__(self, model: nn.Module, example: torch.Tensor, capture_warmup: int = 30):
        if not example.is_cuda:
            raise ValueError("FixedShapeCUDAGraphRunner requires a CUDA example tensor")
        if capture_warmup < 1:
            raise ValueError("capture_warmup must be positive")
        self.model = model.eval()
        self.shape = tuple(example.shape)
        self.dtype = example.dtype
        self.device = example.device
        self.static_input = torch.empty_like(example)

        current = torch.cuda.current_stream(self.device)
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(current)
        with torch.cuda.stream(warmup_stream), torch.inference_mode():
            self.static_input.copy_(example)
            for _ in range(int(capture_warmup)):
                self.model(self.static_input)
        current.wait_stream(warmup_stream)
        torch.cuda.synchronize(self.device)

        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.static_output = self.model(self.static_input)

    def __call__(self, value: torch.Tensor):
        if tuple(value.shape) != self.shape or value.dtype != self.dtype or value.device != self.device:
            raise ValueError("CUDA Graph input must match the captured shape, dtype and device")
        self.static_input.copy_(value)
        self.graph.replay()
        return self.static_output
