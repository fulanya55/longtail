"""Concurrent inference for independent classification and parameter branches."""

from __future__ import annotations

import torch

from cj0724.multitask import TaskRoutedMLCNNIQGroupedMultiTaskModel


class ConcurrentTaskRouteRunner:
    """Run the two frozen task branches on persistent CUDA streams.

    CPU execution deliberately falls back to the model's ordinary forward path so
    unit tests and non-CUDA diagnostics retain the same public contract.
    """

    def __init__(self, model: TaskRoutedMLCNNIQGroupedMultiTaskModel):
        if not isinstance(model, TaskRoutedMLCNNIQGroupedMultiTaskModel):
            raise TypeError("ConcurrentTaskRouteRunner requires a task-routed MLCNN/IQ-Grouped model")
        self.model = model
        self._device: torch.device | None = None
        self._classification_stream: torch.cuda.Stream | None = None
        self._parameter_stream: torch.cuda.Stream | None = None

    def _ensure_streams(self, device: torch.device) -> None:
        if self._device == device:
            return
        self._device = device
        self._classification_stream = torch.cuda.Stream(device=device)
        self._parameter_stream = torch.cuda.Stream(device=device)

    def __call__(self, iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not iq.is_cuda:
            return self.model(iq)
        self._ensure_streams(iq.device)
        if self._classification_stream is None or self._parameter_stream is None:
            raise RuntimeError("CUDA streams were not initialized")

        current = torch.cuda.current_stream(iq.device)
        self._classification_stream.wait_stream(current)
        self._parameter_stream.wait_stream(current)
        iq.record_stream(self._classification_stream)
        iq.record_stream(self._parameter_stream)

        with torch.cuda.stream(self._classification_stream):
            classification_feature = self.model.classification_adapter(
                self.model.classification_backbone(iq)
            )
            logits = self.model.classifier(classification_feature)

        with torch.cuda.stream(self._parameter_stream):
            parameter_feature = self.model.parameter_adapter(self.model.parameter_backbone(iq))
            parameters = self.model.parameter_head(parameter_feature)

        current.wait_stream(self._classification_stream)
        current.wait_stream(self._parameter_stream)
        logits.record_stream(current)
        parameters.record_stream(current)
        return logits, parameters
