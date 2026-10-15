"""Label-conditioned local evidence readout and early SawFM specialist."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from cjlt.models.complex_layers import complex_magnitude


class LabelConditionedEvidenceReadout(nn.Module):
    """Residual label-query statistics with an optional top-k local path."""

    def __init__(
        self,
        channels=128,
        num_labels=6,
        query_dim=64,
        evidence_dim=64,
        attention_floor=0.2,
        topk=8,
        dropout=0.1,
        enable_local=False,
        initial_scale=0.05,
    ):
        super().__init__()
        self.channels = int(channels)
        self.num_labels = int(num_labels)
        self.query_dim = int(query_dim)
        self.evidence_dim = int(evidence_dim)
        self.topk = int(topk)
        self.enable_local = bool(enable_local)
        floor = float(attention_floor)
        if floor < 0.0 or floor >= 1.0:
            raise ValueError("attention_floor must be in [0, 1)")
        if self.topk <= 0:
            raise ValueError("topk must be positive")
        self.attention_floor = floor
        self.key_projection = nn.Conv1d(
            self.channels, self.query_dim, kernel_size=1, bias=False
        )
        self.queries = nn.Parameter(
            torch.empty(self.num_labels, self.query_dim)
        )
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        self.query_projection = nn.Sequential(
            nn.Linear(self.channels * 2, self.evidence_dim),
            nn.LayerNorm(self.evidence_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.query_weight = nn.Parameter(
            torch.empty(self.num_labels, self.evidence_dim)
        )
        self.query_bias = nn.Parameter(torch.zeros(self.num_labels))
        self.query_scale = nn.Parameter(
            torch.full((self.num_labels,), float(initial_scale))
        )
        nn.init.xavier_uniform_(self.query_weight)
        if self.enable_local:
            self.local_projection = nn.Sequential(
                nn.Linear(self.channels * 2, self.evidence_dim),
                nn.LayerNorm(self.evidence_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            )
            self.local_weight = nn.Parameter(
                torch.empty(self.num_labels, self.evidence_dim)
            )
            self.local_bias = nn.Parameter(torch.zeros(self.num_labels))
            self.local_scale = nn.Parameter(
                torch.full((self.num_labels,), float(initial_scale))
            )
            nn.init.xavier_uniform_(self.local_weight)
        else:
            self.local_projection = None
            self.register_parameter("local_weight", None)
            self.register_parameter("local_bias", None)
            self.register_parameter("local_scale", None)

    def attention_scores(self, values):
        if values.ndim != 3 or values.shape[1] != self.channels:
            raise ValueError(
                "temporal values must have shape [B, {}, T]".format(
                    self.channels
                )
            )
        keys = self.key_projection(values)
        return torch.einsum("bdt,kd->bkt", keys, self.queries) / math.sqrt(
            float(self.query_dim)
        )

    def attention_weights(self, values):
        attention = torch.softmax(self.attention_scores(values), dim=-1)
        if self.attention_floor:
            uniform = 1.0 / float(values.shape[-1])
            attention = self.attention_floor * uniform + (
                1.0 - self.attention_floor
            ) * attention
        return attention

    @staticmethod
    def _label_logits(features, weight, bias):
        return torch.einsum("bkd,kd->bk", features, weight) + bias[None, :]

    def _query_features(self, values, attention):
        mean = torch.einsum("bkt,bct->bkc", attention, values)
        second = torch.einsum("bkt,bct->bkc", attention, values.pow(2))
        std = torch.sqrt(torch.clamp(second - mean.pow(2), min=1e-5))
        return self.query_projection(torch.cat([mean, std], dim=-1))

    def _local_features(self, values, scores):
        count = min(self.topk, int(values.shape[-1]))
        indices = torch.topk(scores, k=count, dim=-1).indices
        temporal = values.transpose(1, 2)
        batch = torch.arange(values.shape[0], device=values.device)[:, None, None]
        batch = batch.expand(-1, self.num_labels, count)
        selected = temporal[batch, indices]
        local_mean = selected.mean(dim=2)
        local_max = selected.max(dim=2).values
        return self.local_projection(torch.cat([local_mean, local_max], dim=-1))

    def forward(self, values, base_logits, return_diagnostics=False):
        if base_logits.ndim != 2 or base_logits.shape[1] != self.num_labels:
            raise ValueError("base_logits must have shape [B, num_labels]")
        scores = self.attention_scores(values)
        attention = torch.softmax(scores, dim=-1)
        if self.attention_floor:
            uniform = 1.0 / float(values.shape[-1])
            attention = self.attention_floor * uniform + (
                1.0 - self.attention_floor
            ) * attention
        query_features = self._query_features(values, attention)
        query_logits = self._label_logits(
            query_features, self.query_weight, self.query_bias
        )
        logits = base_logits + torch.tanh(self.query_scale)[None, :] * query_logits
        local_logits = None
        indices = None
        if self.enable_local:
            count = min(self.topk, int(values.shape[-1]))
            indices = torch.topk(scores, k=count, dim=-1).indices
            local_features = self._local_features(values, scores)
            local_logits = self._label_logits(
                local_features, self.local_weight, self.local_bias
            )
            logits = logits + torch.tanh(self.local_scale)[None, :] * local_logits
        if not return_diagnostics:
            return logits
        normalized_entropy = -(
            attention * torch.log(attention.clamp_min(1e-12))
        ).sum(dim=-1) / math.log(float(values.shape[-1]))
        diagnostics = {
            "attention_entropy": normalized_entropy,
            "query_scale": torch.tanh(self.query_scale),
            "local_scale": (
                torch.tanh(self.local_scale)
                if self.local_scale is not None
                else None
            ),
            "topk_indices": indices,
            "query_logits": query_logits,
            "local_logits": local_logits,
        }
        return logits, diagnostics


class BoundedLocalContrastReadout(nn.Module):
    """Zero-centred label-local evidence with a hard residual-logit bound.

    Unlike a free label-query classifier, this module can only respond to
    temporal contrast: every label heatmap is centred over time and there is
    no learnable class bias.  The frozen global classifier remains the safety
    path, while the local branch is limited to ``residual_cap`` logits.
    """

    def __init__(
        self,
        channels=128,
        num_labels=6,
        hidden_channels=32,
        topk=8,
        residual_cap=0.35,
    ):
        super().__init__()
        self.channels = int(channels)
        self.num_labels = int(num_labels)
        self.hidden_channels = int(hidden_channels)
        self.topk = int(topk)
        self.residual_cap = float(residual_cap)
        if self.topk <= 0:
            raise ValueError("topk must be positive")
        if self.residual_cap <= 0.0:
            raise ValueError("residual_cap must be positive")
        groups = 8 if self.hidden_channels % 8 == 0 else 1
        self.projection = nn.Sequential(
            nn.Conv1d(self.channels, self.hidden_channels, 1, bias=False),
            nn.GroupNorm(groups, self.hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.label_direction = nn.Parameter(
            torch.empty(self.num_labels, self.hidden_channels)
        )
        # [top-k mean, top-k max, top-k std, full-map std]. No bias is
        # intentionally provided, preventing a label-prior-only residual.
        self.statistic_weight = nn.Parameter(torch.empty(self.num_labels, 4))
        nn.init.normal_(self.label_direction, mean=0.0, std=0.02)
        nn.init.normal_(self.statistic_weight, mean=0.0, std=0.02)

    def evidence(self, temporal):
        if temporal.ndim != 3 or temporal.shape[1] != self.channels:
            raise ValueError(
                "temporal must have shape [B, {}, T]".format(self.channels)
            )
        encoded = self.projection(temporal)
        direction = F.normalize(self.label_direction, p=2, dim=-1)
        heatmap = torch.einsum("bdt,kd->bkt", encoded, direction)
        return heatmap - heatmap.mean(dim=-1, keepdim=True)

    def forward(self, temporal, base_logits, return_diagnostics=False):
        if base_logits.ndim != 2 or base_logits.shape[1] != self.num_labels:
            raise ValueError("base_logits must have shape [B, num_labels]")
        heatmap = self.evidence(temporal)
        count = min(self.topk, int(heatmap.shape[-1]))
        local = torch.topk(heatmap, k=count, dim=-1)
        local_values = local.values
        statistics = torch.stack(
            [
                local_values.mean(dim=-1),
                local_values.max(dim=-1).values,
                local_values.std(dim=-1, unbiased=False),
                heatmap.std(dim=-1, unbiased=False),
            ],
            dim=-1,
        )
        raw_residual = torch.einsum(
            "bkf,kf->bk", statistics, self.statistic_weight
        )
        residual = self.residual_cap * torch.tanh(raw_residual)
        logits = base_logits + residual
        if not return_diagnostics:
            return logits
        return logits, {
            "residual": residual,
            "raw_residual": raw_residual,
            "topk_indices": local.indices,
            "heatmap_mean": heatmap.mean(dim=-1),
            "heatmap_std": heatmap.std(dim=-1, unbiased=False),
            "statistics": statistics,
        }


def _previous(values):
    return torch.cat([values[..., :1], values[..., :-1]], dim=-1)


class EarlySawFMEnvelopeIFExpert(nn.Module):
    """Independent stage2 envelope, phase-increment, and reset specialist."""

    def __init__(
        self,
        complex_channels=32,
        hidden_channels=32,
        topk=8,
        dropout=0.1,
        initial_scale=0.05,
    ):
        super().__init__()
        complex_channels = int(complex_channels)
        hidden_channels = int(hidden_channels)
        self.complex_channels = complex_channels
        self.topk = int(topk)
        descriptor_channels = complex_channels * 6
        self.stem = nn.Sequential(
            nn.Conv1d(descriptor_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        kernels = (5, 15, 31)
        dilations = (1, 2, 4)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size,
                        padding=dilation * (kernel_size // 2),
                        dilation=dilation,
                        groups=hidden_channels,
                        bias=False,
                    ),
                    nn.BatchNorm1d(hidden_channels),
                    nn.SiLU(inplace=True),
                )
                for kernel_size, dilation in zip(kernels, dilations)
            ]
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_channels * 3, hidden_channels, 1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.downsample = nn.AvgPool1d(kernel_size=4, stride=4)
        statistics_dim = hidden_channels * 4
        self.classifier = nn.Sequential(
            nn.Linear(statistics_dim, 64),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(64, 1),
        )
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def descriptors(self, real, imag):
        if real.shape != imag.shape or real.ndim != 3:
            raise ValueError("real and imag must have matching [B,C,T] shapes")
        if real.shape[1] != self.complex_channels:
            raise ValueError("unexpected complex channel count")
        magnitude = complex_magnitude(real, imag)
        normalized_real = real / magnitude
        normalized_imag = imag / magnitude
        previous_real = _previous(normalized_real)
        previous_imag = _previous(normalized_imag)
        phase_cos = normalized_real * previous_real + normalized_imag * previous_imag
        phase_sin = normalized_imag * previous_real - normalized_real * previous_imag
        envelope_delta = torch.abs(magnitude - _previous(magnitude))
        phase_cos_delta = torch.abs(phase_cos - _previous(phase_cos))
        phase_sin_delta = torch.abs(phase_sin - _previous(phase_sin))
        return torch.cat(
            [
                magnitude,
                envelope_delta,
                phase_cos,
                phase_sin,
                phase_cos_delta,
                phase_sin_delta,
            ],
            dim=1,
        )

    def forward(self, real, imag, return_diagnostics=False):
        values = self.stem(self.descriptors(real, imag))
        values = self.fusion(torch.cat([branch(values) for branch in self.branches], dim=1))
        values = self.downsample(values)
        mean = values.mean(dim=-1)
        std = torch.sqrt(
            torch.clamp(values.pow(2).mean(dim=-1) - mean.pow(2), min=1e-5)
        )
        maximum = values.max(dim=-1).values
        count = min(self.topk, int(values.shape[-1]))
        topk_mean = torch.topk(values, k=count, dim=-1).values.mean(dim=-1)
        expert_logit = self.classifier(
            torch.cat([mean, std, maximum, topk_mean], dim=1)
        )
        scaled = torch.tanh(self.scale) * expert_logit
        if return_diagnostics:
            return scaled, {
                "scale": torch.tanh(self.scale),
                "expert_logit": expert_logit,
            }
        return scaled


class RawStage1SawFMFrontend(nn.Module):
    """Small SawFM specialist using raw I/Q and frozen stage1 complex maps."""

    def __init__(
        self,
        mode="raw_stage1",
        hidden_channels=16,
        topk=16,
        dropout=0.1,
        descriptor_mode="full",
    ):
        super().__init__()
        if mode not in {"raw", "stage1", "raw_stage1"}:
            raise ValueError("mode must be raw, stage1, or raw_stage1")
        self.mode = mode
        if descriptor_mode not in {"full", "envelope", "phase"}:
            raise ValueError("descriptor_mode must be full, envelope, or phase")
        self.descriptor_mode = descriptor_mode
        self.hidden_channels = int(hidden_channels)
        self.topk = int(topk)
        if self.hidden_channels <= 0 or self.topk <= 0:
            raise ValueError("hidden_channels and topk must be positive")
        self.raw_branch = self._branch() if mode in {"raw", "raw_stage1"} else None
        self.stage1_branch = self._branch() if mode in {"stage1", "raw_stage1"} else None
        branch_dim = self.hidden_channels * 4 + 24
        input_dim = branch_dim * (2 if mode == "raw_stage1" else 1)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(32, 1),
        )

    def _branch(self):
        kernels = (7, 31, 63)
        return nn.ModuleDict(
            {
                "stem": nn.Sequential(
                    nn.Conv1d(6, self.hidden_channels, 1, bias=False),
                    nn.GroupNorm(4, self.hidden_channels),
                    nn.SiLU(inplace=True),
                ),
                "multiscale": nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Conv1d(
                                self.hidden_channels,
                                self.hidden_channels,
                                kernel,
                                padding=kernel // 2,
                                groups=self.hidden_channels,
                                bias=False,
                            ),
                            nn.GroupNorm(4, self.hidden_channels),
                            nn.SiLU(inplace=True),
                        )
                        for kernel in kernels
                    ]
                ),
                "fusion": nn.Sequential(
                    nn.Conv1d(self.hidden_channels * 3, self.hidden_channels, 1, bias=False),
                    nn.GroupNorm(4, self.hidden_channels),
                    nn.SiLU(inplace=True),
                ),
            }
        )

    def _descriptors(self, real, imag):
        if real.shape != imag.shape or real.ndim != 3:
            raise ValueError("complex inputs must be aligned [B,C,T]")
        magnitude = complex_magnitude(real, imag)
        previous_magnitude = _previous(magnitude)
        normalized_real = real / magnitude
        normalized_imag = imag / magnitude
        previous_real = _previous(normalized_real)
        previous_imag = _previous(normalized_imag)
        phase_cos = normalized_real * previous_real + normalized_imag * previous_imag
        phase_sin = normalized_imag * previous_real - normalized_real * previous_imag
        descriptors = torch.stack(
            [
                magnitude.mean(dim=1),
                magnitude.max(dim=1).values,
                magnitude.std(dim=1, unbiased=False),
                torch.abs(magnitude - previous_magnitude).mean(dim=1),
                phase_cos.mean(dim=1),
                phase_sin.mean(dim=1),
            ],
            dim=1,
        )
        if self.descriptor_mode == "envelope":
            descriptors = torch.cat(
                [descriptors[:, :4], torch.zeros_like(descriptors[:, 4:])], dim=1
            )
        elif self.descriptor_mode == "phase":
            descriptors = torch.cat(
                [torch.zeros_like(descriptors[:, :4]), descriptors[:, 4:]], dim=1
            )
        return descriptors

    def _statistics(self, values):
        mean = values.mean(dim=-1)
        std = values.std(dim=-1, unbiased=False)
        maximum = values.max(dim=-1).values
        count = min(self.topk, int(values.shape[-1]))
        topk = torch.topk(values, k=count, dim=-1).values.mean(dim=-1)
        return torch.cat([mean, std, maximum, topk], dim=1)

    def _encode(self, branch, real, imag):
        descriptors = self._descriptors(real, imag)
        values = branch["stem"](descriptors)
        values = branch["fusion"](
            torch.cat([module(values) for module in branch["multiscale"]], dim=1)
        )
        values = F.avg_pool1d(values, kernel_size=4, stride=4)
        return torch.cat([self._statistics(values), self._statistics(descriptors)], dim=1)

    def forward(self, raw_real, raw_imag, stage1_real=None, stage1_imag=None, return_features=False):
        features = []
        if self.raw_branch is not None:
            features.append(self._encode(self.raw_branch, raw_real, raw_imag))
        if self.stage1_branch is not None:
            if stage1_real is None or stage1_imag is None:
                raise ValueError("stage1 maps are required for the selected mode")
            features.append(self._encode(self.stage1_branch, stage1_real, stage1_imag))
        fused = torch.cat(features, dim=1)
        logits = self.classifier(fused).squeeze(1)
        if return_features:
            return logits, fused
        return logits


class SawStage2LowRankEvidence(nn.Module):
    """Small residual adapter and temporally normalized SawFM evidence map."""

    def __init__(
        self,
        channels=64,
        rank=8,
        hidden_channels=32,
        topk=16,
        adapter_limit=0.20,
    ):
        super().__init__()
        channels = int(channels)
        rank = int(rank)
        hidden_channels = int(hidden_channels)
        self.channels = channels
        self.topk = int(topk)
        self.adapter_limit = float(adapter_limit)
        if channels <= 0 or rank <= 0 or hidden_channels <= 0:
            raise ValueError("channels, rank, and hidden_channels must be positive")
        if self.topk <= 0:
            raise ValueError("topk must be positive")
        if self.adapter_limit <= 0.0:
            raise ValueError("adapter_limit must be positive")
        self.input_norm = nn.GroupNorm(8, channels, affine=False)
        self.adapter_down = nn.Conv1d(channels, rank, kernel_size=1, bias=False)
        self.adapter_up = nn.Conv1d(rank, channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.adapter_up.weight)
        self.adapter_alpha = nn.Parameter(torch.tensor(0.05))
        self.evidence_head = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=15,
                padding=7,
                groups=hidden_channels,
                bias=False,
            ),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden_channels, 1, kernel_size=1, bias=False),
        )
        self.logit_scale_raw = nn.Parameter(torch.tensor(0.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, values, return_map=False):
        if values.ndim != 3 or values.shape[1] != self.channels:
            raise ValueError(
                "stage2 values must have shape [B, {}, T]".format(self.channels)
            )
        normalized = self.input_norm(values)
        adjustment = self.adapter_up(F.gelu(self.adapter_down(normalized)))
        adapter_scale = self.adapter_limit * torch.tanh(self.adapter_alpha)
        adapted = values + adapter_scale * adjustment
        evidence = self.evidence_head(self.input_norm(adapted)).squeeze(1)
        evidence_mean = evidence.mean(dim=-1, keepdim=True)
        evidence_std = torch.sqrt(
            torch.clamp(
                evidence.pow(2).mean(dim=-1, keepdim=True)
                - evidence_mean.pow(2),
                min=1e-5,
            )
        )
        normalized_map = (evidence - evidence_mean) / evidence_std
        count = min(self.topk, int(normalized_map.shape[-1]))
        topk_mean = torch.topk(normalized_map, k=count, dim=-1).values.mean(dim=-1)
        maximum = normalized_map.max(dim=-1).values
        pooled = topk_mean + 0.25 * maximum
        logit_scale = F.softplus(self.logit_scale_raw) + 1e-4
        logits = logit_scale * pooled + self.logit_bias
        if return_map:
            return logits, normalized_map, {
                "adapter_scale": adapter_scale,
                "logit_scale": logit_scale,
                "pooled_evidence": pooled,
            }
        return logits


class MultiTargetStage2LowRankEvidence(nn.Module):
    """Rank-limited stage2 adapter with label-specific normalized evidence maps."""

    def __init__(
        self,
        channels=64,
        targets=3,
        rank=8,
        hidden_channels=32,
        topk=16,
        adapter_limit=0.20,
    ):
        super().__init__()
        self.channels = int(channels)
        self.targets = int(targets)
        self.topk = int(topk)
        self.adapter_limit = float(adapter_limit)
        if min(self.channels, self.targets, int(rank), int(hidden_channels), self.topk) <= 0:
            raise ValueError("channels, targets, rank, hidden_channels, and topk must be positive")
        if self.channels % 8 or int(hidden_channels) % 8:
            raise ValueError("channels and hidden_channels must be divisible by 8")
        if self.adapter_limit <= 0:
            raise ValueError("adapter_limit must be positive")
        self.input_norm = nn.GroupNorm(8, self.channels, affine=False)
        self.adapter_down = nn.Conv1d(self.channels, int(rank), 1, bias=False)
        self.adapter_up = nn.Conv1d(int(rank), self.channels, 1, bias=False)
        nn.init.zeros_(self.adapter_up.weight)
        self.adapter_alpha = nn.Parameter(torch.tensor(0.05))
        self.evidence_head = nn.Sequential(
            nn.Conv1d(self.channels, int(hidden_channels), 1, bias=False),
            nn.GroupNorm(8, int(hidden_channels)),
            nn.SiLU(inplace=True),
            nn.Conv1d(int(hidden_channels), int(hidden_channels), 15, padding=7,
                      groups=int(hidden_channels), bias=False),
            nn.SiLU(inplace=True),
            nn.Conv1d(int(hidden_channels), self.targets, 1, bias=False),
        )
        self.logit_scale_raw = nn.Parameter(torch.zeros(self.targets))
        self.maximum_weight_raw = nn.Parameter(torch.full((self.targets,), -1.2586915))
        self.logit_bias = nn.Parameter(torch.zeros(self.targets))

    def forward(self, values, return_map=False):
        if values.ndim != 3 or values.shape[1] != self.channels:
            raise ValueError(f"stage2 values must have shape [B, {self.channels}, T]")
        normalized = self.input_norm(values)
        adjustment = self.adapter_up(F.gelu(self.adapter_down(normalized)))
        adapter_scale = self.adapter_limit * torch.tanh(self.adapter_alpha)
        adapted = values + adapter_scale * adjustment
        evidence = self.evidence_head(self.input_norm(adapted))
        mean = evidence.mean(dim=-1, keepdim=True)
        variance = torch.clamp(evidence.pow(2).mean(dim=-1, keepdim=True) - mean.pow(2), min=1e-5)
        normalized_map = (evidence - mean) / torch.sqrt(variance)
        count = min(self.topk, int(normalized_map.shape[-1]))
        topk_mean = torch.topk(normalized_map, k=count, dim=-1).values.mean(dim=-1)
        maximum = normalized_map.max(dim=-1).values
        maximum_weight = torch.sigmoid(self.maximum_weight_raw)[None, :]
        pooled = topk_mean + maximum_weight * maximum
        logit_scale = F.softplus(self.logit_scale_raw)[None, :] + 1e-4
        logits = logit_scale * pooled + self.logit_bias[None, :]
        if return_map:
            return logits, normalized_map, {
                "adapter_scale": adapter_scale,
                "logit_scale": logit_scale.squeeze(0),
                "maximum_weight": maximum_weight.squeeze(0),
                "pooled_evidence": pooled,
            }
        return logits


class SawEvidenceRouter(nn.Module):
    """Nine-parameter convex gate between the frozen H2 and SawFM expert."""

    def __init__(self, feature_dim=8):
        super().__init__()
        feature_dim = int(feature_dim)
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.feature_dim = feature_dim
        self.gate = nn.Linear(feature_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, features, h2_saw_logit, expert_logit, return_gate=False):
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                "router features must have shape [B, {}]".format(self.feature_dim)
            )
        h2_saw_logit = h2_saw_logit.reshape(-1)
        expert_logit = expert_logit.reshape(-1)
        if (
            features.shape[0] != h2_saw_logit.shape[0]
            or features.shape[0] != expert_logit.shape[0]
        ):
            raise ValueError("router inputs must share the batch dimension")
        gate = torch.sigmoid(self.gate(features)).reshape(-1)
        fused = (1.0 - gate) * h2_saw_logit + gate * expert_logit
        if return_gate:
            return fused, gate
        return fused


class VisibilityAffineCalibrator(nn.Module):
    """Low-capacity positive-scale affine calibration conditioned on evidence."""

    def __init__(self, feature_dim=10):
        super().__init__()
        feature_dim = int(feature_dim)
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.feature_dim = feature_dim
        self.scale = nn.Linear(feature_dim, 1)
        self.bias = nn.Linear(feature_dim, 1)
        nn.init.zeros_(self.scale.weight)
        nn.init.constant_(self.scale.bias, 0.541324854612918)
        nn.init.zeros_(self.bias.weight)
        nn.init.zeros_(self.bias.bias)

    def forward(self, c2_logit, features, return_parameters=False):
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                "visibility features must have shape [B, {}]".format(
                    self.feature_dim
                )
            )
        c2_logit = c2_logit.reshape(-1)
        if c2_logit.shape[0] != features.shape[0]:
            raise ValueError("c2_logit and features must share batch size")
        scale = F.softplus(self.scale(features)).reshape(-1) + 1e-4
        bias = self.bias(features).reshape(-1)
        calibrated = scale * c2_logit + bias
        if return_parameters:
            return calibrated, scale, bias
        return calibrated


class OrdinalVisibilityAnchorCalibrator(nn.Module):
    """Sixteen-parameter ordinal visibility coordinate with three anchors."""

    def __init__(self, feature_dim=10):
        super().__init__()
        feature_dim = int(feature_dim)
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.feature_dim = feature_dim
        self.visibility = nn.Linear(feature_dim, 1)
        nn.init.zeros_(self.visibility.weight)
        nn.init.zeros_(self.visibility.bias)
        self.anchor_delta_mid = nn.Parameter(torch.tensor(-1.0))
        self.anchor_delta_low = nn.Parameter(torch.tensor(-1.0))
        self.global_scale_raw = nn.Parameter(torch.tensor(0.541324854612918))
        self.global_bias = nn.Parameter(torch.tensor(0.0))
        self.anchor_temperature_raw = nn.Parameter(torch.tensor(-0.4327521295671885))
        self.register_buffer("anchor_centers", torch.tensor([1.0, 0.0, -1.0]))

    def _anchor_values(self):
        middle = F.softplus(self.anchor_delta_mid)
        low = middle + F.softplus(self.anchor_delta_low)
        return torch.stack([middle.new_zeros(()), middle, low])

    def _anchor_weights(self, visibility):
        temperature = F.softplus(self.anchor_temperature_raw) + 1e-4
        distance = (visibility.unsqueeze(1) - self.anchor_centers.unsqueeze(0)).pow(2)
        return torch.softmax(-distance / temperature, dim=1), temperature

    def forward(
        self,
        c2_logit,
        features,
        oracle_visibility=None,
        return_diagnostics=False,
    ):
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                "visibility features must have shape [B, {}]".format(
                    self.feature_dim
                )
            )
        c2_logit = c2_logit.reshape(-1)
        if c2_logit.shape[0] != features.shape[0]:
            raise ValueError("c2_logit and features must share batch size")
        learned_visibility = self.visibility(features).reshape(-1)
        if oracle_visibility is None:
            visibility = learned_visibility
        else:
            visibility = oracle_visibility.reshape(-1).to(
                device=features.device, dtype=features.dtype
            )
            if visibility.shape[0] != features.shape[0]:
                raise ValueError("oracle_visibility and features must share batch size")
        anchor_weights, temperature = self._anchor_weights(visibility)
        anchor_values = self._anchor_values()
        anchor_bias = anchor_weights @ anchor_values
        scale = F.softplus(self.global_scale_raw) + 1e-4
        calibrated = scale * c2_logit + self.global_bias + anchor_bias
        if return_diagnostics:
            return calibrated, {
                "visibility": visibility,
                "learned_visibility": learned_visibility,
                "anchor_weights": anchor_weights,
                "anchor_values": anchor_values,
                "anchor_bias": anchor_bias,
                "scale": scale,
                "temperature": temperature,
            }
        return calibrated
