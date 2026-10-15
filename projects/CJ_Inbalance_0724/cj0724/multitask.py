"""Common heads and masked regression utilities for fair backbone comparison."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cj0724 import CLASS_NAMES
from cj0724.models import (
    DEFAULT_UNIFIED_HEAD_DIM,
    LegacyConcatIQAdapter,
    UnifiedBackboneClassifier,
    build_benchmark_model,
)


MLCNN_PARAMETER_INPUT_MODELS = (
    "legacy_mlcnn_global",
    "legacy_mlcnn_global_compute_matched_control",
    "legacy_mlcnn_shallow_multiscale",
    "legacy_mlcnn_class_conditioned",
    "legacy_mlcnn_shallow_multiscale_detached",
    "legacy_mlcnn_class_conditioned_detached",
)
TASK_ROUTED_MULTITASK_MODELS = (
    "mlcnn_iq_grouped_task_routed",
    "mlcnn_iq_grouped_task_routed_w4",
    "mlcnn_iq_grouped_task_routed_w16",
    "mlcnn_shared_temporal_task_routed",
    "mlcnn_shared_temporal_detached_task_routed",
    "mlcnn_iq_grouped_task_routed_saw_loss2",
    "mlcnn_iq_grouped_task_routed_saw_safe",
    "mlcnn_iq_grouped_task_routed_saw_shallow_safe",
    "mlcnn_iq_grouped_physical_film_task_routed",
    "mlcnn_iq_grouped_spectral_film_task_routed",
    "mlcnn_iq_grouped_temporal_query_task_routed",
    "mlcnn_iq_grouped_temporal_global_control_task_routed",
)


@dataclass(frozen=True)
class ParameterSpec:
    class_name: str
    name: str
    column_index: int
    unit: str


PARAMETER_SPECS = (
    ParameterSpec("FMB", "JNR", 0, "dB"),
    ParameterSpec("FMB", "interference_bandwidth", 2, "Hz"),
    ParameterSpec("SawFM", "JNR", 11, "dB"),
    ParameterSpec("SawFM", "sweep_period", 14, "s"),
    ParameterSpec("SinFM", "JNR", 16, "dB"),
    ParameterSpec("SinFM", "frequency_deviation", 18, "Hz"),
    ParameterSpec("MFT", "JNR", 27, "dB"),
    ParameterSpec("MFT", "num_targets", 28, "count"),
    ParameterSpec("VFT", "JNR", 48, "dB"),
    ParameterSpec("VFT", "Doppler", 49, "Hz"),
    ParameterSpec("ISRJ", "JNR", 52, "dB"),
    ParameterSpec("ISRJ", "sample_width", 53, "s"),
)
PARAMETER_COLUMNS = tuple(spec.column_index for spec in PARAMETER_SPECS)


def parameter_schema() -> list[dict[str, object]]:
    return [asdict(spec) for spec in PARAMETER_SPECS]


def reshape_selected_parameters(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2 or values.shape[1] != len(PARAMETER_SPECS):
        raise ValueError(f"Expected selected parameters [B, {len(PARAMETER_SPECS)}]")
    return values.reshape(values.shape[0], len(CLASS_NAMES), 2)


class ParameterNormalizer:
    """Per-target z-score statistics fitted only on valid positive training targets."""

    def __init__(self, epsilon: float = 1.0e-6):
        self.epsilon = float(epsilon)
        self.mean: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None

    def fit(self, targets: torch.Tensor, valid_mask: torch.Tensor) -> "ParameterNormalizer":
        values = torch.as_tensor(targets).float()
        valid = torch.as_tensor(valid_mask).bool()
        if values.shape != valid.shape or values.ndim != 3:
            raise ValueError("targets and valid_mask must match [B, C, 2]")
        mean = torch.zeros(values.shape[1:], dtype=torch.float32)
        scale = torch.ones(values.shape[1:], dtype=torch.float32)
        for class_index in range(values.shape[1]):
            for parameter_index in range(values.shape[2]):
                selected = values[valid[:, class_index, parameter_index], class_index, parameter_index]
                if selected.numel() == 0 or not bool(torch.isfinite(selected).all()):
                    raise ValueError("every target requires finite valid training values")
                mean[class_index, parameter_index] = selected.mean()
                scale[class_index, parameter_index] = selected.std(unbiased=False).clamp_min(self.epsilon)
        self.mean, self.scale = mean, scale
        return self

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.scale is None:
            raise RuntimeError("ParameterNormalizer must be fitted first")
        tensor = torch.as_tensor(values).float()
        return (tensor - self.mean.to(tensor.device)) / self.scale.to(tensor.device)

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.scale is None:
            raise RuntimeError("ParameterNormalizer must be fitted first")
        tensor = torch.as_tensor(values).float()
        return tensor * self.scale.to(tensor.device) + self.mean.to(tensor.device)

    def state_dict(self) -> dict[str, list[list[float]]]:
        if self.mean is None or self.scale is None:
            raise RuntimeError("ParameterNormalizer must be fitted first")
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


class LabelParameterHead(nn.Module):
    """Identical independent two-parameter expert for each known interference type."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
        )
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(int(hidden_dim), 32),
                nn.GELU(),
                nn.Linear(32, 2),
            )
            for _ in CLASS_NAMES
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        shared = self.shared(features)
        return torch.stack([head(shared) for head in self.heads], dim=1)


class ShallowMultiScaleParameterHead(nn.Module):
    """Fuse the pooled I/Q activations from MLCNN stages 1-3 for regression."""

    def __init__(self, token_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        channel_dims = (32, 32, 64, 64, 128, 128)
        self.projectors = nn.ModuleList(
            nn.Sequential(nn.Linear(channels, token_dim), nn.LayerNorm(token_dim), nn.GELU())
            for channels in channel_dims
        )
        self.parameter_head = LabelParameterHead(len(channel_dims) * token_dim, hidden_dim)

    def forward(self, pooled_features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(pooled_features) != len(self.projectors):
            raise ValueError("MLCNN shallow head expects I/Q features from stages 1-3")
        tokens = [projector(feature) for projector, feature in zip(self.projectors, pooled_features)]
        return self.parameter_head(torch.cat(tokens, dim=-1))


class ClassConditionedParameterHead(nn.Module):
    """Use one learned class query per interference type over the same shallow tokens."""

    def __init__(self, token_dim: int = 64):
        super().__init__()
        channel_dims = (32, 32, 64, 64, 128, 128)
        self.token_dim = int(token_dim)
        self.projectors = nn.ModuleList(
            nn.Sequential(nn.Linear(channels, token_dim), nn.LayerNorm(token_dim), nn.GELU())
            for channels in channel_dims
        )
        self.class_queries = nn.Parameter(torch.empty(len(CLASS_NAMES), token_dim))
        nn.init.normal_(self.class_queries, std=0.02)
        self.heads = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 32), nn.GELU(), nn.Linear(32, 2))
            for _ in CLASS_NAMES
        )

    def forward(self, pooled_features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(pooled_features) != len(self.projectors):
            raise ValueError("MLCNN conditioned head expects I/Q features from stages 1-3")
        tokens = torch.stack(
            [projector(feature) for projector, feature in zip(self.projectors, pooled_features)], dim=1
        )
        attention = torch.einsum("btd,cd->bct", tokens, self.class_queries) / np.sqrt(self.token_dim)
        conditioned = torch.einsum("bct,btd->bcd", attention.softmax(dim=-1), tokens)
        return torch.stack([head(conditioned[:, index]) for index, head in enumerate(self.heads)], dim=1)


class MLCNNParameterInputMultiTaskModel(nn.Module):
    """Keep the MLCNN classifier fixed while varying only the parameter-head input."""

    def __init__(
        self,
        classifier: UnifiedBackboneClassifier,
        parameter_input: str,
        parameter_hidden_dim: int = 64,
    ):
        super().__init__()
        if not isinstance(classifier.backbone, LegacyConcatIQAdapter):
            raise TypeError("MLCNN parameter-input variants require the legacy concatenated-I/Q adapter")
        self.mlcnn = classifier.backbone.backbone
        self.feature_adapter = classifier.feature_adapter
        self.classifier = classifier.classifier
        self.parameter_input = str(parameter_input)
        if self.parameter_input == "global":
            self.parameter_head = LabelParameterHead(classifier.head_dim, parameter_hidden_dim)
        elif self.parameter_input in ("shallow_multiscale", "shallow_multiscale_detached"):
            self.parameter_head = ShallowMultiScaleParameterHead(64, parameter_hidden_dim)
        elif self.parameter_input in ("class_conditioned", "class_conditioned_detached"):
            self.parameter_head = ClassConditionedParameterHead(64)
        else:
            raise ValueError(f"Unknown MLCNN parameter input {parameter_input!r}")

    def _stage_pair(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
        convolution: nn.Module,
        normalization: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Preserve the legacy real-then-imag call order because the BatchNorm modules are shared.
        real = self.mlcnn.maxpool(self.mlcnn.relu(normalization(convolution(real))))
        imag = self.mlcnn.maxpool(self.mlcnn.relu(normalization(convolution(imag))))
        return real, imag

    def forward(self, iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if iq.ndim != 3 or iq.shape[1] != 2:
            raise ValueError(f"Expected [B, 2, L] I/Q input, got {tuple(iq.shape)}")
        real, imag = iq[:, 0:1], iq[:, 1:2]
        shallow = []
        for convolution, normalization in (
            (self.mlcnn.conv1, self.mlcnn.bn1),
            (self.mlcnn.conv2, self.mlcnn.bn2),
            (self.mlcnn.conv3, self.mlcnn.bn3),
        ):
            real, imag = self._stage_pair(real, imag, convolution, normalization)
            shallow.extend((real.mean(dim=-1), imag.mean(dim=-1)))
        real, imag = self._stage_pair(real, imag, self.mlcnn.conv4, self.mlcnn.bn4)
        real, imag = self.mlcnn.drop(real), self.mlcnn.drop(imag)
        raw_global = torch.cat(
            (self.mlcnn.fc1(torch.flatten(real, 1)), self.mlcnn.fc1(torch.flatten(imag, 1))), dim=1
        )
        global_feature = self.feature_adapter(raw_global)
        logits = self.classifier(global_feature)
        if self.parameter_input == "global":
            parameter_feature = global_feature
        elif self.parameter_input in ("shallow_multiscale_detached", "class_conditioned_detached"):
            parameter_feature = tuple(feature.detach() for feature in shallow)
        else:
            parameter_feature = tuple(shallow)
        return logits, self.parameter_head(parameter_feature)


class TaskRoutedMLCNNIQGroupedMultiTaskModel(nn.Module):
    """Route classification through MLCNN and parameter extraction through IQ-Grouped."""

    def __init__(
        self,
        classification_model: UnifiedBackboneClassifier,
        parameter_model: UnifiedBackboneClassifier,
        parameter_hidden_dim: int = 64,
    ):
        super().__init__()
        self.classification_backbone = classification_model.backbone
        self.classification_adapter = classification_model.feature_adapter
        self.classifier = classification_model.classifier
        self.parameter_backbone = parameter_model.backbone
        self.parameter_adapter = parameter_model.feature_adapter
        self.parameter_head = LabelParameterHead(parameter_model.head_dim, parameter_hidden_dim)

    def forward(self, iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        classification_feature = self.classification_adapter(self.classification_backbone(iq))
        parameter_feature = self.parameter_adapter(self.parameter_backbone(iq))
        return self.classifier(classification_feature), self.parameter_head(parameter_feature)


class ComputeMatchedGlobalControl(nn.Module):
    """A generic, non-grouped parameter branch matched to the routed model's cost.

    This is a capacity-and-MAC control, not a proposed expert.  It preserves the
    ordinary MLCNN classification path and gives parameter regression an
    independent raw-I/Q temporal branch without I/Q grouping or class-specific
    physical structure.  Its channel sequence is fixed by a pre-data static
    MAC screen in P7a.
    """

    def __init__(
        self,
        classifier: UnifiedBackboneClassifier,
        parameter_hidden_dim: int = 64,
        branch_channels: int = 67,
    ) -> None:
        super().__init__()
        self.classification_backbone = classifier.backbone
        self.classification_adapter = classifier.feature_adapter
        self.classifier = classifier.classifier
        channels = int(branch_channels)
        self.generic_parameter_branch = nn.Sequential(
            nn.Conv1d(2, channels, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(channels), nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(channels), nn.GELU(),
            nn.Conv1d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
        )
        self.parameter_head = LabelParameterHead(int(classifier.head_dim) + 64, parameter_hidden_dim)

    def forward(self, iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        classification_feature = self.classification_adapter(self.classification_backbone(iq))
        logits = self.classifier(classification_feature)
        generic = F.adaptive_avg_pool1d(self.generic_parameter_branch(iq.float()), 1).squeeze(-1)
        return logits, self.parameter_head(torch.cat((classification_feature, generic), dim=1))


def raw_iq_physical_features(iq: torch.Tensor) -> torch.Tensor:
    """Return the fixed 26-D time, spectrum and ACF descriptor used by P5b2.

    The descriptor has no nominal-power, label, or parameter input.  Keeping it
    in the model makes its execution path auditable and prevents train/eval
    feature-preprocessing drift.
    """
    if iq.ndim != 3 or iq.shape[1] != 2:
        raise ValueError(f"Expected [B, 2, L] I/Q input, got {tuple(iq.shape)}")
    real, imag = iq[:, 0], iq[:, 1]
    magnitude = torch.sqrt(real.square() + imag.square() + 1.0e-8)
    time = torch.stack((
        real.mean(dim=1), imag.mean(dim=1), real.std(dim=1), imag.std(dim=1),
        magnitude.mean(dim=1), magnitude.std(dim=1), magnitude.amax(dim=1),
        (real * imag).mean(dim=1),
    ), dim=1)
    spectrum = torch.fft.fft(torch.complex(real, imag), dim=1).abs().square()
    spectrum = spectrum / spectrum.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    bins = spectrum.shape[1]
    edges = torch.linspace(0, bins, 9, device=iq.device).round().long()
    bands = torch.stack(
        [spectrum[:, edges[index]:edges[index + 1]].sum(dim=1) for index in range(8)], dim=1
    )
    frequency = torch.arange(bins, device=iq.device, dtype=iq.dtype)[None, :]
    centroid = (spectrum * frequency).sum(dim=1) / max(bins - 1, 1)
    entropy = -(spectrum * spectrum.clamp_min(1.0e-12).log()).sum(dim=1) / math.log(max(bins, 2))
    autocorrelation = torch.stack([
        (real[:, :-lag] * real[:, lag:] + imag[:, :-lag] * imag[:, lag:]).mean(dim=1)
        for lag in (1, 2, 4, 8, 16, 32, 64, 128)
    ], dim=1)
    return torch.cat((time, bands, centroid[:, None], entropy[:, None], autocorrelation), dim=1)


def raw_iq_spectral_features(iq: torch.Tensor) -> torch.Tensor:
    """Return an auxiliary PSD descriptor computed only from the I/Q input.

    This is deliberately separate from ``raw_iq_physical_features`` so the
    R2 comparison can distinguish I/Q-only from I/Q+PSD conditioning without
    changing the six-class recognition head or the sample-level input shape.
    """
    if iq.ndim != 3 or iq.shape[1] != 2:
        raise ValueError(f"Expected [B, 2, L] I/Q input, got {tuple(iq.shape)}")
    real, imag = iq[:, 0].float(), iq[:, 1].float()
    complex_iq = torch.complex(real, imag)
    psd = torch.fft.fft(complex_iq, dim=1).abs().square()[:, : iq.shape[-1] // 2 + 1]
    psd = psd / psd.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    bins = psd.shape[1]
    edges = torch.linspace(0, bins, 16 + 1, device=iq.device).round().long()
    bands = torch.stack(
        [psd[:, edges[index]:edges[index + 1]].sum(dim=1) for index in range(16)], dim=1
    )
    frequency = torch.arange(bins, device=iq.device, dtype=psd.dtype)[None, :]
    centroid = (psd * frequency).sum(dim=1) / max(bins - 1, 1)
    entropy = -(psd * psd.clamp_min(1.0e-12).log()).sum(dim=1) / math.log(max(bins, 2))
    peak = psd.amax(dim=1)
    return torch.cat((bands, centroid[:, None], entropy[:, None], peak[:, None]), dim=1)


class PhysicalFiLMTaskRoutedMLCNNIQGroupedMultiTaskModel(TaskRoutedMLCNNIQGroupedMultiTaskModel):
    """Condition only the IQ-Grouped parameter representation on observable context.

    The MLCNN posterior is detached before conditioning.  Consequently the
    regression loss cannot modify the classification branch through this path;
    the classification output remains the ordinary MLCNN output.
    """

    def __init__(
        self,
        classification_model: UnifiedBackboneClassifier,
        parameter_model: UnifiedBackboneClassifier,
        parameter_hidden_dim: int = 64,
    ) -> None:
        super().__init__(classification_model, parameter_model, parameter_hidden_dim)
        condition_dim = 26 + len(CLASS_NAMES)
        feature_dim = int(parameter_model.head_dim)
        self.condition_encoder = nn.Sequential(
            nn.LayerNorm(condition_dim), nn.Linear(condition_dim, 64), nn.GELU(),
            nn.Linear(64, 2 * feature_dim),
        )
        # Start from the unmodulated parameter representation.  The sole
        # capacity change is learned, bounded affine conditioning.
        nn.init.zeros_(self.condition_encoder[-1].weight)
        nn.init.zeros_(self.condition_encoder[-1].bias)

    def forward(self, iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        classification_feature = self.classification_adapter(self.classification_backbone(iq))
        logits = self.classifier(classification_feature)
        parameter_feature = self.parameter_adapter(self.parameter_backbone(iq))
        observable_context = torch.cat(
            (raw_iq_physical_features(iq), torch.sigmoid(logits).detach()), dim=1
        )
        scale, bias = self.condition_encoder(observable_context).chunk(2, dim=1)
        conditioned = parameter_feature * (1.0 + 0.1 * torch.tanh(scale)) + 0.1 * bias
        return logits, self.parameter_head(conditioned)


class SpectralFiLMTaskRoutedMLCNNIQGroupedMultiTaskModel(TaskRoutedMLCNNIQGroupedMultiTaskModel):
    """Condition only the parameter representation on an I/Q-derived PSD bank."""

    def __init__(
        self,
        classification_model: UnifiedBackboneClassifier,
        parameter_model: UnifiedBackboneClassifier,
        parameter_hidden_dim: int = 64,
    ) -> None:
        super().__init__(classification_model, parameter_model, parameter_hidden_dim)
        condition_dim = 19 + len(CLASS_NAMES)
        feature_dim = int(parameter_model.head_dim)
        self.condition_encoder = nn.Sequential(
            nn.LayerNorm(condition_dim), nn.Linear(condition_dim, 64), nn.GELU(),
            nn.Linear(64, 2 * feature_dim),
        )
        nn.init.zeros_(self.condition_encoder[-1].weight)
        nn.init.zeros_(self.condition_encoder[-1].bias)

    def forward(self, iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        classification_feature = self.classification_adapter(self.classification_backbone(iq))
        logits = self.classifier(classification_feature)
        parameter_feature = self.parameter_adapter(self.parameter_backbone(iq))
        context = torch.cat((raw_iq_spectral_features(iq), torch.sigmoid(logits).detach()), dim=1)
        scale, bias = self.condition_encoder(context).chunk(2, dim=1)
        conditioned = parameter_feature * (1.0 + 0.1 * torch.tanh(scale)) + 0.1 * bias
        return logits, self.parameter_head(conditioned)


class ClassSpecificTemporalParameterHead(nn.Module):
    """Six target queries over a common IQ-Grouped temporal token sequence.

    ``global_control`` executes the identical adapter and query-score map but
    reduces it to a global modulation instead of using the locations as an
    attention distribution.  This makes it a capacity- and main-compute-
    matched control for whether *localisation*, rather than query capacity,
    supplies a regression benefit.
    """

    def __init__(self, token_dim: int, hidden_dim: int = 64, global_control: bool = False) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(len(CLASS_NAMES), int(token_dim)))
        nn.init.normal_(self.queries, std=0.02)
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(int(token_dim)), nn.Linear(int(token_dim), int(hidden_dim)),
                nn.GELU(), nn.Linear(int(hidden_dim), 2),
            )
            for _ in CLASS_NAMES
        )
        self.global_control = bool(global_control)
        self.scale = float(np.sqrt(int(token_dim)))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[2] != self.queries.shape[1]:
            raise ValueError(
                f"Expected temporal tokens [B, T, {self.queries.shape[1]}], got {tuple(tokens.shape)}"
            )
        scores = torch.einsum("btd,cd->bct", tokens, self.queries) / self.scale
        if self.global_control:
            global_feature = tokens.mean(dim=1, keepdim=True)
            global_score = scores.mean(dim=-1, keepdim=True)
            features = global_feature + 0.1 * torch.tanh(global_score) * self.queries[None, :, :]
        else:
            weights = scores.softmax(dim=-1)
            features = torch.einsum("bct,btd->bcd", weights, tokens)
        return torch.stack([head(features[:, index]) for index, head in enumerate(self.heads)], dim=1)


class TemporalQueryTaskRoutedMLCNNIQGroupedMultiTaskModel(nn.Module):
    """Preserve MLCNN classification while replacing only parameter pooling."""

    def __init__(
        self,
        classification_model: UnifiedBackboneClassifier,
        parameter_model: UnifiedBackboneClassifier,
        parameter_hidden_dim: int = 64,
        global_control: bool = False,
    ) -> None:
        super().__init__()
        if not all(hasattr(parameter_model.backbone, name) for name in ("iq_paths", "fuse_iq")):
            raise TypeError("Temporal-query route requires the IQ-Grouped parameter backbone")
        self.classification_backbone = classification_model.backbone
        self.classification_adapter = classification_model.feature_adapter
        self.classifier = classification_model.classifier
        self.parameter_backbone = parameter_model.backbone
        raw_dim = int(parameter_model.raw_feature_dim)
        self.token_adapter = nn.Sequential(
            nn.Linear(raw_dim, int(parameter_model.head_dim)),
            nn.LayerNorm(int(parameter_model.head_dim)),
            nn.GELU(),
        )
        self.parameter_head = ClassSpecificTemporalParameterHead(
            int(parameter_model.head_dim), parameter_hidden_dim, global_control,
        )
        self.global_control = bool(global_control)

    def forward(self, iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        classification_feature = self.classification_adapter(self.classification_backbone(iq))
        logits = self.classifier(classification_feature)
        temporal = self.parameter_backbone.fuse_iq(self.parameter_backbone.iq_paths(iq))
        tokens = self.token_adapter(temporal.transpose(1, 2))
        return logits, self.parameter_head(tokens)


class SharedTemporalMLCNNMultiTaskModel(nn.Module):
    """Share MLCNN stages 1-2 and preserve ordered I/Q maps for regression."""

    def __init__(
        self,
        classification_model: UnifiedBackboneClassifier,
        parameter_hidden_dim: int = 64,
        dropout: float = 0.2,
        detach_parameter_input: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(classification_model.backbone, LegacyConcatIQAdapter):
            raise TypeError("Shared temporal routing requires the legacy MLCNN adapter")
        self.mlcnn = classification_model.backbone.backbone
        self.classification_adapter = classification_model.feature_adapter
        self.classifier = classification_model.classifier
        self.parameter_temporal = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=5, stride=2, padding=2, groups=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, groups=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.parameter_adapter = nn.Sequential(
            nn.Linear(64, classification_model.head_dim),
            nn.LayerNorm(classification_model.head_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.parameter_head = LabelParameterHead(
            classification_model.head_dim, parameter_hidden_dim
        )
        self.detach_parameter_input = bool(detach_parameter_input)

    def _stage_pair(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
        convolution: nn.Module,
        normalization: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Preserve the legacy real-then-imag BatchNorm update order.
        real = self.mlcnn.maxpool(self.mlcnn.relu(normalization(convolution(real))))
        imag = self.mlcnn.maxpool(self.mlcnn.relu(normalization(convolution(imag))))
        return real, imag

    def forward(self, iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if iq.ndim != 3 or iq.shape[1] != 2:
            raise ValueError(f"Expected [B, 2, L] I/Q input, got {tuple(iq.shape)}")
        real, imag = iq[:, 0:1], iq[:, 1:2]
        real, imag = self._stage_pair(real, imag, self.mlcnn.conv1, self.mlcnn.bn1)
        real, imag = self._stage_pair(real, imag, self.mlcnn.conv2, self.mlcnn.bn2)

        ordered_iq = torch.cat((real, imag), dim=1)
        if self.detach_parameter_input:
            ordered_iq = ordered_iq.detach()
        parameter_feature = self.parameter_temporal(ordered_iq).squeeze(-1)
        parameter_feature = self.parameter_adapter(parameter_feature)

        real, imag = self._stage_pair(real, imag, self.mlcnn.conv3, self.mlcnn.bn3)
        real, imag = self._stage_pair(real, imag, self.mlcnn.conv4, self.mlcnn.bn4)
        real, imag = self.mlcnn.drop(real), self.mlcnn.drop(imag)
        raw_global = torch.cat(
            (self.mlcnn.fc1(torch.flatten(real, 1)), self.mlcnn.fc1(torch.flatten(imag, 1))), dim=1
        )
        classification_feature = self.classification_adapter(raw_global)
        return self.classifier(classification_feature), self.parameter_head(parameter_feature)


class SawSafeTaskRoutedMLCNNIQGroupedMultiTaskModel(nn.Module):
    """Use a detached MLCNN fallback only for the replicated SawFM-period failure."""

    def __init__(
        self,
        classification_model: UnifiedBackboneClassifier,
        parameter_model: UnifiedBackboneClassifier,
        parameter_hidden_dim: int = 64,
    ):
        super().__init__()
        self.classification_backbone = classification_model.backbone
        self.classification_adapter = classification_model.feature_adapter
        self.classifier = classification_model.classifier
        self.parameter_backbone = parameter_model.backbone
        self.parameter_adapter = parameter_model.feature_adapter
        self.parameter_head = LabelParameterHead(parameter_model.head_dim, parameter_hidden_dim)
        self.saw_period_safe_head = nn.Sequential(
            nn.Linear(classification_model.head_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        classification_feature = self.classification_adapter(self.classification_backbone(iq))
        parameter_feature = self.parameter_adapter(self.parameter_backbone(iq))
        prediction = self.parameter_head(parameter_feature)
        safe_period = self.saw_period_safe_head(classification_feature.detach()).squeeze(-1)
        prediction = prediction.clone()
        prediction[:, 1, 1] = safe_period
        return self.classifier(classification_feature), prediction


class ShallowSawPeriodSafeHead(nn.Module):
    """Compact scalar head over detached MLCNN stages 1--3."""

    def __init__(self, token_dim: int = 16):
        super().__init__()
        channel_dims = (32, 32, 64, 64, 128, 128)
        self.projectors = nn.ModuleList(
            nn.Sequential(nn.Linear(channels, token_dim), nn.LayerNorm(token_dim), nn.GELU())
            for channels in channel_dims
        )
        self.regressor = nn.Sequential(
            nn.Linear(len(channel_dims) * token_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, pooled_features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(pooled_features) != len(self.projectors):
            raise ValueError("SawFM shallow safe head expects six pooled I/Q features")
        encoded = [
            projector(feature.detach())
            for projector, feature in zip(self.projectors, pooled_features)
        ]
        return self.regressor(torch.cat(encoded, dim=-1)).squeeze(-1)


class ShallowSawSafeTaskRoutedMLCNNIQGroupedMultiTaskModel(nn.Module):
    """Fallback to detached local MLCNN texture only for SawFM sweep period."""

    def __init__(
        self,
        classification_model: UnifiedBackboneClassifier,
        parameter_model: UnifiedBackboneClassifier,
        parameter_hidden_dim: int = 64,
    ):
        super().__init__()
        if not isinstance(classification_model.backbone, LegacyConcatIQAdapter):
            raise TypeError("Shallow SawFM fallback requires the legacy MLCNN adapter")
        self.mlcnn = classification_model.backbone.backbone
        self.classification_adapter = classification_model.feature_adapter
        self.classifier = classification_model.classifier
        self.parameter_backbone = parameter_model.backbone
        self.parameter_adapter = parameter_model.feature_adapter
        self.parameter_head = LabelParameterHead(parameter_model.head_dim, parameter_hidden_dim)
        self.saw_period_safe_head = ShallowSawPeriodSafeHead(token_dim=16)

    def _stage_pair(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
        convolution: nn.Module,
        normalization: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        real = self.mlcnn.maxpool(self.mlcnn.relu(normalization(convolution(real))))
        imag = self.mlcnn.maxpool(self.mlcnn.relu(normalization(convolution(imag))))
        return real, imag

    def forward(self, iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if iq.ndim != 3 or iq.shape[1] != 2:
            raise ValueError(f"Expected [B, 2, L] I/Q input, got {tuple(iq.shape)}")
        real, imag = iq[:, 0:1], iq[:, 1:2]
        shallow = []
        for convolution, normalization in (
            (self.mlcnn.conv1, self.mlcnn.bn1),
            (self.mlcnn.conv2, self.mlcnn.bn2),
            (self.mlcnn.conv3, self.mlcnn.bn3),
        ):
            real, imag = self._stage_pair(real, imag, convolution, normalization)
            shallow.extend((real.mean(dim=-1), imag.mean(dim=-1)))
        real, imag = self._stage_pair(real, imag, self.mlcnn.conv4, self.mlcnn.bn4)
        real, imag = self.mlcnn.drop(real), self.mlcnn.drop(imag)
        raw_global = torch.cat(
            (self.mlcnn.fc1(torch.flatten(real, 1)), self.mlcnn.fc1(torch.flatten(imag, 1))), dim=1
        )
        classification_feature = self.classification_adapter(raw_global)
        parameter_feature = self.parameter_adapter(self.parameter_backbone(iq))
        prediction = self.parameter_head(parameter_feature).clone()
        prediction[:, 1, 1] = self.saw_period_safe_head(tuple(shallow))
        return self.classifier(classification_feature), prediction


class UnifiedBackboneMultiTaskModel(nn.Module):
    """One raw backbone, one common embedding, and common classification/regression heads."""

    def __init__(self, classifier: UnifiedBackboneClassifier, parameter_hidden_dim: int = 64):
        super().__init__()
        if not isinstance(classifier, UnifiedBackboneClassifier):
            raise TypeError("The first screen accepts only single backbones with the common adapter")
        self.backbone = classifier.backbone
        self.raw_feature_dim = classifier.raw_feature_dim
        self.head_dim = classifier.head_dim
        self.feature_adapter = classifier.feature_adapter
        self.classifier = classifier.classifier
        self.parameter_head = LabelParameterHead(self.head_dim, parameter_hidden_dim)

    def forward(self, iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.backbone(iq)
        if raw.ndim != 2 or raw.shape[1] != self.raw_feature_dim:
            raise RuntimeError(f"Expected raw features [B, {self.raw_feature_dim}], got {tuple(raw.shape)}")
        features = self.feature_adapter(raw)
        return self.classifier(features), self.parameter_head(features)


def build_multitask_model(
    name: str,
    width: int = 24,
    dropout: float = 0.2,
    head_dim: int = DEFAULT_UNIFIED_HEAD_DIM,
    parameter_hidden_dim: int = 64,
    legacy_attention: str = "se",
) -> UnifiedBackboneMultiTaskModel:
    if name == "legacy_mlcnn_global_compute_matched_control":
        classifier = build_benchmark_model(
            "legacy_mlcnn", num_classes=len(CLASS_NAMES), width=width, dropout=dropout,
            head_dim=head_dim, legacy_attention=legacy_attention,
        )
        return ComputeMatchedGlobalControl(classifier, parameter_hidden_dim)
    if name in TASK_ROUTED_MULTITASK_MODELS:
        classification_model = build_benchmark_model(
            "legacy_mlcnn", num_classes=len(CLASS_NAMES), width=width, dropout=dropout,
            head_dim=head_dim, legacy_attention=legacy_attention,
        )
        if name in (
            "mlcnn_shared_temporal_task_routed",
            "mlcnn_shared_temporal_detached_task_routed",
        ):
            return SharedTemporalMLCNNMultiTaskModel(
                classification_model,
                parameter_hidden_dim,
                dropout,
                detach_parameter_input=name.endswith("_detached_task_routed"),
            )
        parameter_width = {
            "mlcnn_iq_grouped_task_routed_w4": 4,
            "mlcnn_iq_grouped_task_routed_w16": 16,
        }.get(name, width)
        parameter_model = build_benchmark_model(
            "iq_grouped", num_classes=len(CLASS_NAMES), width=parameter_width, dropout=dropout,
            head_dim=head_dim, legacy_attention=legacy_attention,
        )
        if name == "mlcnn_iq_grouped_task_routed_saw_safe":
            return SawSafeTaskRoutedMLCNNIQGroupedMultiTaskModel(
                classification_model, parameter_model, parameter_hidden_dim
            )
        if name == "mlcnn_iq_grouped_task_routed_saw_shallow_safe":
            return ShallowSawSafeTaskRoutedMLCNNIQGroupedMultiTaskModel(
                classification_model, parameter_model, parameter_hidden_dim
            )
        if name == "mlcnn_iq_grouped_physical_film_task_routed":
            return PhysicalFiLMTaskRoutedMLCNNIQGroupedMultiTaskModel(
                classification_model, parameter_model, parameter_hidden_dim
            )
        if name == "mlcnn_iq_grouped_spectral_film_task_routed":
            return SpectralFiLMTaskRoutedMLCNNIQGroupedMultiTaskModel(
                classification_model, parameter_model, parameter_hidden_dim
            )
        if name in (
            "mlcnn_iq_grouped_temporal_query_task_routed",
            "mlcnn_iq_grouped_temporal_global_control_task_routed",
        ):
            return TemporalQueryTaskRoutedMLCNNIQGroupedMultiTaskModel(
                classification_model, parameter_model, parameter_hidden_dim,
                global_control=name.endswith("temporal_global_control_task_routed"),
            )
        return TaskRoutedMLCNNIQGroupedMultiTaskModel(
            classification_model, parameter_model, parameter_hidden_dim
        )
    if name in MLCNN_PARAMETER_INPUT_MODELS:
        parameter_input = name.removeprefix("legacy_mlcnn_")
        classifier = build_benchmark_model(
            "legacy_mlcnn",
            num_classes=len(CLASS_NAMES),
            width=width,
            dropout=dropout,
            head_dim=head_dim,
            legacy_attention=legacy_attention,
        )
        return MLCNNParameterInputMultiTaskModel(classifier, parameter_input, parameter_hidden_dim)
    if name == "efficientnet_b1_moe3_1d":
        raise ValueError("The common multi-task wrapper accepts single backbones, not the three-expert MoE")
    classifier = build_benchmark_model(
        name,
        num_classes=len(CLASS_NAMES),
        width=width,
        dropout=dropout,
        head_dim=head_dim,
        legacy_attention=legacy_attention,
    )
    return UnifiedBackboneMultiTaskModel(classifier, parameter_hidden_dim)


def masked_parameter_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
    parameter_mask: torch.Tensor,
    target_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape != parameter_mask.shape:
        raise ValueError("prediction, target, and mask must have the same shape")
    if labels.shape != prediction.shape[:2]:
        raise ValueError("labels must match [B, C]")
    valid = parameter_mask.bool() & (labels[:, :, None] > 0.5)
    if bool((valid & ~torch.isfinite(target)).any()):
        raise ValueError("valid parameter targets must be finite")
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    selected_prediction = prediction[valid]
    selected_target = target[valid]
    element_loss = F.smooth_l1_loss(selected_prediction, selected_target, reduction="none")
    if target_weights is None:
        return element_loss.mean()
    weights = torch.as_tensor(target_weights, dtype=element_loss.dtype, device=element_loss.device)
    if weights.shape != prediction.shape[1:]:
        raise ValueError(f"target_weights must have shape {tuple(prediction.shape[1:])}")
    if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
        raise ValueError("target_weights must be finite and strictly positive")
    expanded_weights = weights.unsqueeze(0).expand_as(prediction)
    selected_weights = expanded_weights[valid]
    return (element_loss * selected_weights).sum() / selected_weights.sum()


def parameter_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    normalizer: ParameterNormalizer,
) -> dict[str, object]:
    if normalizer.mean is None or normalizer.scale is None:
        raise RuntimeError("normalizer must be fitted")
    scale = normalizer.scale.numpy().astype(np.float64)
    rows, all_normalized, correlations = [], [], []
    for flat_index, spec in enumerate(PARAMETER_SPECS):
        class_index, parameter_index = divmod(flat_index, 2)
        selected = valid[:, class_index, parameter_index]
        error = np.abs(prediction[selected, class_index, parameter_index] - target[selected, class_index, parameter_index])
        normalized = error / scale[class_index, parameter_index]
        if error.size >= 2 and np.std(prediction[selected, class_index, parameter_index]) > 0 and np.std(
            target[selected, class_index, parameter_index]
        ) > 0:
            correlation = float(np.corrcoef(
                prediction[selected, class_index, parameter_index], target[selected, class_index, parameter_index]
            )[0, 1])
        else:
            correlation = float("nan")
        correlations.append(correlation)
        all_normalized.extend(normalized.tolist())
        rows.append({
            **asdict(spec),
            "count": int(selected.sum()),
            "mae": float(error.mean()) if error.size else float("nan"),
            "rmse": float(np.sqrt(np.square(error).mean())) if error.size else float("nan"),
            "normalized_mae": float(normalized.mean()) if normalized.size else float("nan"),
            "pearson_r": correlation,
        })
    return {
        "oracle_valid_values": int(valid.sum()),
        "oracle_macro_target_nmae": float(np.mean([row["normalized_mae"] for row in rows])),
        "oracle_micro_nmae": float(np.mean(all_normalized)),
        "oracle_macro_pearson_r": float(np.nanmean(correlations)),
        "per_target": rows,
    }
