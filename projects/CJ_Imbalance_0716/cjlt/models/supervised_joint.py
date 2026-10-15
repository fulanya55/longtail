"""Supervised multilabel recognition with conditional parameter heads."""

import torch
import torch.nn.functional as F
from torch import nn

from cjlt.models.label_conditioned import LabelConditionedFeatureBank
from cjlt.models.parameter_heads import LabelGatedParameterHeads


def temporal_local_distribution(temporal, topk=8, lags=(1, 4, 16)):
    """Translation-tolerant local activation statistics from ``[B,D,T]``."""

    if temporal.ndim != 3:
        raise ValueError("temporal features must have shape [B, D, T]")
    length = int(temporal.shape[-1])
    if length < 2:
        raise ValueError("temporal feature length must be at least two")
    count = min(int(topk), length)
    values = [
        temporal.mean(dim=-1),
        temporal.std(dim=-1, unbiased=False),
        temporal.amax(dim=-1),
        temporal.topk(count, dim=-1).values.mean(dim=-1),
    ]
    for lag in lags:
        lag_value = int(lag)
        if 0 < lag_value < length:
            values.append(
                (temporal[..., lag_value:] - temporal[..., :-lag_value])
                .abs()
                .mean(dim=-1)
            )
        else:
            values.append(torch.zeros_like(values[0]))
    return torch.cat(values, dim=1)


class LocalDistributionParameterHead(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        input_dim = int(feature_dim) * 7
        self.regressor = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, temporal):
        return self.regressor(temporal_local_distribution(temporal)).squeeze(-1)


class VFTSpectralParameterHead(nn.Module):
    """Estimate normalized VFT Doppler from a compact full-spectrum summary."""

    def __init__(self, pooled_bins=64, hidden_dim=64):
        super().__init__()
        self.pooled_bins = int(pooled_bins)
        self.regressor = nn.Sequential(
            nn.LayerNorm(self.pooled_bins + 3),
            nn.Linear(self.pooled_bins + 3, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, signal):
        # FFT on a length-3000 half tensor is unsupported on several CUDA paths.
        # Keep only this lightweight branch in float32 while the rest may use AMP.
        iq = signal.float()
        complex_signal = torch.complex(iq[:, 0], iq[:, 1])
        spectrum = torch.fft.fftshift(torch.fft.fft(complex_signal, dim=-1), dim=-1)
        power = spectrum.abs().square()
        distribution = power / power.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        coordinate = torch.linspace(
            -1.0,
            1.0,
            distribution.shape[-1],
            device=distribution.device,
            dtype=distribution.dtype,
        )
        centroid = (distribution * coordinate[None, :]).sum(dim=-1)
        spread = (
            distribution * (coordinate[None, :] - centroid[:, None]).square()
        ).sum(dim=-1).sqrt()
        peak = coordinate[distribution.argmax(dim=-1)]
        log_profile = torch.log1p(distribution * float(distribution.shape[-1]))
        pooled = F.adaptive_avg_pool1d(
            log_profile[:, None, :], self.pooled_bins
        ).squeeze(1)
        features = torch.cat(
            [pooled, centroid[:, None], spread[:, None], peak[:, None]], dim=1
        )
        return self.regressor(features).squeeze(-1)


class SupervisedJointParameterModel(nn.Module):
    """Keep the deployable classifier path and attach supervised parameter heads."""

    def __init__(
        self,
        backbone,
        relation_dim=64,
        parameters_per_label=2,
        enable_parameter_heads=True,
        relation_dropout=0.0,
        use_mft_count_classifier=False,
        use_vft_spectral_head=False,
        mft_class_index=3,
        vft_class_index=4,
        local_parameter_targets=(),
    ):
        super().__init__()
        if not hasattr(backbone, "feature_dim") or not hasattr(backbone, "num_classes"):
            raise ValueError("backbone must expose feature_dim and num_classes")
        self.backbone = backbone
        self.num_classes = int(backbone.num_classes)
        self.feature_dim = int(backbone.feature_dim)
        self.relation_dim = int(relation_dim)
        self.parameters_per_label = int(parameters_per_label)
        self.enable_parameter_heads = bool(enable_parameter_heads)
        self.use_mft_count_classifier = bool(use_mft_count_classifier)
        self.use_vft_spectral_head = bool(use_vft_spectral_head)
        self.mft_class_index = int(mft_class_index)
        self.vft_class_index = int(vft_class_index)
        self.local_parameter_targets = tuple(
            (int(class_index), int(parameter_index))
            for class_index, parameter_index in local_parameter_targets
        )
        self.label_feature_bank = None
        self.parameter_heads = None
        self.mft_count_head = None
        self.vft_spectral_head = None
        self.local_parameter_heads = nn.ModuleDict()
        if self.enable_parameter_heads:
            self.label_feature_bank = LabelConditionedFeatureBank(
                feature_dim=self.feature_dim,
                num_labels=self.num_classes,
                relation_dim=self.relation_dim,
                dropout=float(relation_dropout),
                conditioning_mode="independent",
            )
            self.parameter_heads = LabelGatedParameterHeads(
                num_labels=self.num_classes,
                embedding_dim=self.relation_dim,
                parameters_per_label=self.parameters_per_label,
            )
            if self.use_mft_count_classifier:
                self.mft_count_head = nn.Linear(self.relation_dim, 6)
            if self.use_vft_spectral_head:
                self.vft_spectral_head = VFTSpectralParameterHead()

            for class_index, parameter_index in self.local_parameter_targets:
                if not 0 <= class_index < self.num_classes:
                    raise ValueError("local parameter class index out of range")
                if not 0 <= parameter_index < self.parameters_per_label:
                    raise ValueError("local parameter index out of range")
                self.local_parameter_heads[
                    "{}_{}".format(class_index, parameter_index)
                ] = LocalDistributionParameterHead(self.feature_dim)

    def forward(self, signal, return_aux=False, detach_parameter_backbone=False):
        needs_temporal = bool(self.local_parameter_heads)
        temporal = None
        if needs_temporal and hasattr(self.backbone, "forward_temporal_features"):
            temporal = self.backbone.forward_temporal_features(signal)
            features = self.backbone.global_pool(temporal).squeeze(-1)
            logits = self.backbone.forward_classifier(features)
        elif needs_temporal:
            raise ValueError("local parameter heads require a temporal backbone interface")
        else:
            logits, features = self.backbone(signal, return_features=True)
        if not return_aux:
            return logits
        parameter_predictions = None
        label_embeddings = None
        mft_count_logits = None
        if self.enable_parameter_heads:
            parameter_features = (
                features.detach() if bool(detach_parameter_backbone) else features
            )
            label_embeddings = self.label_feature_bank(parameter_features)
            parameter_predictions = self.parameter_heads(label_embeddings)
            if self.mft_count_head is not None:
                mft_count_logits = self.mft_count_head(
                    label_embeddings[:, self.mft_class_index]
                )
            if self.vft_spectral_head is not None:
                parameter_predictions = parameter_predictions.clone()
                parameter_predictions[:, self.vft_class_index, 1] = (
                    self.vft_spectral_head(signal)
                )
            if self.local_parameter_heads:
                parameter_predictions = parameter_predictions.clone()
                local_temporal = (
                    temporal.detach()
                    if bool(detach_parameter_backbone)
                    else temporal
                )
                for class_index, parameter_index in self.local_parameter_targets:
                    key = "{}_{}".format(class_index, parameter_index)
                    parameter_predictions[:, class_index, parameter_index] = (
                        self.local_parameter_heads[key](local_temporal)
                    )
        return {
            "logits": logits,
            "features": features,
            "label_embeddings": label_embeddings,
            "parameter_predictions": parameter_predictions,
            "mft_count_logits": mft_count_logits,
        }
