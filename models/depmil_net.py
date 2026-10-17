#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Core Stage-I encoder and Stage-II ProtoDual modules for DepMIL-Net.

The frame backbone intentionally follows ExpADA's released ``VatBackbone``:
TorchVision ShuffleNet V2 x2.0 with ImageNet-1K V1 weights and the final
classification layer removed.  The EAT neck is not copied.  It is replaced by
feature-map motion differences and multi-scale temporal convolutions.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


FEATURE_DIM = 1024
FRAME_FEATURE_DIM = 2048
SEVERITY_LEVELS = 4
ORDINAL_THRESHOLDS = (14.0, 20.0, 29.0)


def cumulative_ordinal_targets(
    labels: Tensor,
    thresholds: Tuple[float, float, float] = ORDINAL_THRESHOLDS,
) -> Tensor:
    labels = labels.reshape(-1, 1)
    boundary = labels.new_tensor(thresholds).reshape(1, -1)
    return (labels >= boundary).to(dtype=labels.dtype)


def cumulative_to_class_probabilities(cumulative: Tensor) -> Tensor:
    """Convert P(y>=k), k=1..3, to four mutually exclusive probabilities."""
    if cumulative.ndim != 2 or cumulative.shape[1] != 3:
        raise ValueError(f"Expected (B,3), got {tuple(cumulative.shape)}")
    p1, p2, p3 = cumulative.unbind(dim=1)
    probabilities = torch.stack((1.0 - p1, p1 - p2, p2 - p3, p3), dim=1)
    probabilities = probabilities.clamp_min(0.0)
    return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)


class ExpADAShuffleNetBackbone(nn.Module):
    """The released ExpADA per-frame ShuffleNet interface."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        import torchvision

        try:
            weights = (
                torchvision.models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )
            shufflenet = torchvision.models.shufflenet_v2_x2_0(weights=weights)
        except AttributeError:  # TorchVision < 0.13 compatibility
            shufflenet = torchvision.models.shufflenet_v2_x2_0(
                pretrained=pretrained
            )
        # This is the same extraction rule as ExpADA mmpretrain/backbones/vat.py.
        self.base = nn.Sequential(*list(shufflenet.children())[:-1])
        self.out_channels = FRAME_FEATURE_DIM

    def forward(self, clips: Tensor) -> Tuple[Tensor, int, int]:
        if clips.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(clips.shape)}")
        batch, frames, channels, height, width = clips.shape
        if frames != 32 or channels != 3:
            raise ValueError(
                f"Motion32 expects T=32,C=3, got T={frames},C={channels}"
            )
        maps = self.base(clips.reshape(batch * frames, channels, height, width))
        if maps.ndim != 4 or maps.shape[1] != self.out_channels:
            raise RuntimeError(
                f"Unexpected ShuffleNet output {tuple(maps.shape)}; expected (*,2048,H,W)"
            )
        return maps, batch, frames


class SpatialAttentionPool(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(32, channels // 4)
        self.score = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )

    def forward(self, maps: Tensor) -> Tuple[Tensor, Tensor]:
        batch_frames, channels, height, width = maps.shape
        logits = self.score(maps).reshape(batch_frames, 1, height * width)
        weights = logits.softmax(dim=-1)
        values = maps.reshape(batch_frames, channels, height * width)
        pooled = torch.bmm(values, weights.transpose(1, 2)).squeeze(-1)
        return pooled, weights.reshape(batch_frames, height, width)


class DepthwiseTemporalBranch(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class TemporalAttentionPool(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        hidden = max(32, channels // 4)
        self.score = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, sequence: Tensor) -> Tuple[Tensor, Tensor]:
        # sequence: [B,T,C]
        weights = self.score(sequence).squeeze(-1).softmax(dim=1)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class MotionMultiScaleInstanceEncoder(nn.Module):
    """Turn one 32-frame segment into one 1024-D instance representation."""

    def __init__(
        self,
        pretrained_backbone: bool = True,
        hidden_dim: int = 512,
        output_dim: int = FEATURE_DIM,
        dropout: float = 0.2,
        use_motion: bool = True,
        temporal_kernels: Sequence[int] = (3, 5, 7),
        spatial_pool_mode: str = "sap",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.use_motion = bool(use_motion)
        if spatial_pool_mode not in ("sap", "gap"):
            raise ValueError("spatial_pool_mode must be 'sap' or 'gap'")
        self.spatial_pool_mode = spatial_pool_mode
        self.temporal_kernels = tuple(int(value) for value in temporal_kernels)
        if not self.temporal_kernels or any(
            value < 1 or value % 2 == 0 for value in self.temporal_kernels
        ):
            raise ValueError("temporal_kernels must contain positive odd integers")
        self.backbone = ExpADAShuffleNetBackbone(pretrained=pretrained_backbone)
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(FRAME_FEATURE_DIM, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.appearance_motion_fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.spatial_pool = (
            SpatialAttentionPool(hidden_dim)
            if self.spatial_pool_mode == "sap"
            else None
        )
        self.temporal_branches = nn.ModuleList(
            DepthwiseTemporalBranch(hidden_dim, kernel, dropout)
            for kernel in self.temporal_kernels
        )
        self.temporal_fusion = nn.Sequential(
            nn.Conv1d(
                hidden_dim * len(self.temporal_kernels),
                hidden_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.temporal_pool = TemporalAttentionPool(hidden_dim, dropout)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim, bias=False),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self, clips: Tensor, return_details: bool = False
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        maps, batch, frames = self.backbone(clips)
        appearance = self.channel_reduce(maps)
        _, channels, height, width = appearance.shape
        appearance = appearance.reshape(batch, frames, channels, height, width)

        motion = torch.zeros_like(appearance)
        if self.use_motion:
            motion[:, 1:] = appearance[:, 1:] - appearance[:, :-1]
        fused = torch.cat((appearance, motion), dim=2).reshape(
            batch * frames, channels * 2, height, width
        )
        fused = self.appearance_motion_fusion(fused)
        if self.spatial_pool is None:
            frame_features = fused.mean(dim=(-2, -1))
            spatial_weights = fused.new_full(
                (batch * frames, height, width), 1.0 / (height * width)
            )
        else:
            frame_features, spatial_weights = self.spatial_pool(fused)
        frame_features = frame_features.reshape(batch, frames, channels)

        temporal_input = frame_features.transpose(1, 2)
        branch_features = [branch(temporal_input) for branch in self.temporal_branches]
        temporal = self.temporal_fusion(torch.cat(branch_features, dim=1))
        temporal = self.temporal_norm(
            (temporal + temporal_input).transpose(1, 2)
        )
        pooled, temporal_weights = self.temporal_pool(temporal)
        embedding = self.output_projection(pooled)

        if not return_details:
            return embedding
        details = {
            "spatial_weights": spatial_weights.reshape(
                batch, frames, height, width
            ),
            "temporal_weights": temporal_weights,
            "motion_energy": motion.square().mean(dim=(2, 3, 4)),
        }
        return embedding, details


class OrderedOrdinalHead(nn.Module):
    """ExpADA-style scalar eta with three sorted learnable thresholds."""

    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 512) -> None:
        super().__init__()
        self.eta = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.theta = nn.Parameter(torch.tensor([-1.0, 0.0, 1.0]))

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        eta = self.eta(features)
        theta = torch.sort(self.theta)[0].reshape(1, -1)
        logits = eta - theta
        cumulative = torch.sigmoid(logits)
        probabilities = cumulative_to_class_probabilities(cumulative)
        return logits, cumulative, probabilities


class Motion32Stage1Model(nn.Module):
    def __init__(
        self,
        pretrained_backbone: bool = True,
        hidden_dim: int = 512,
        feature_dim: int = FEATURE_DIM,
        dropout: float = 0.2,
        use_motion: bool = True,
        temporal_kernels: Sequence[int] = (3, 5, 7),
        spatial_pool_mode: str = "sap",
    ) -> None:
        super().__init__()
        self.encoder = MotionMultiScaleInstanceEncoder(
            pretrained_backbone=pretrained_backbone,
            hidden_dim=hidden_dim,
            output_dim=feature_dim,
            dropout=dropout,
            use_motion=use_motion,
            temporal_kernels=temporal_kernels,
            spatial_pool_mode=spatial_pool_mode,
        )
        self.regression_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )
        self.ordinal_head = OrderedOrdinalHead(feature_dim, 512)

    def forward(self, clips: Tensor, return_details: bool = False):
        encoded = self.encoder(clips, return_details=return_details)
        if return_details:
            features, details = encoded
        else:
            features, details = encoded, None
        regression = self.regression_head(features).squeeze(-1)
        ordinal_logits, cumulative, probabilities = self.ordinal_head(features)
        output = {
            "features": features,
            "regression": regression,
            "ordinal_logits": ordinal_logits,
            "ordinal_cumulative": cumulative,
            "ordinal_probs": probabilities,
        }
        if details is not None:
            output["details"] = details
        return output


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dimension: int, max_length: int = 2048) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.float32)
            * (-math.log(10000.0) / dimension)
        )
        encoding = torch.zeros(max_length, dimension, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] > self.encoding.shape[1]:
            raise ValueError(
                f"Sequence length {x.shape[1]} exceeds positional limit {self.encoding.shape[1]}"
            )
        return x + self.encoding[:, : x.shape[1]].to(dtype=x.dtype)


class SegmentTransformer(nn.Module):
    """Three-layer Transformer over the sequence of 32-frame instances."""

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        model_dim: int = 256,
        layers: int = 3,
        heads: int = 4,
        feedforward_dim: int = 1024,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.layers = int(layers)
        if self.layers < 0:
            raise ValueError("layers must be non-negative")
        if self.layers == 0:
            self.input_projection = nn.Identity()
            self.position = nn.Identity()
            self.encoder = nn.Identity()
            self.output_projection = nn.Identity()
            self.output_norm = nn.LayerNorm(feature_dim)
            return
        self.input_projection = nn.Linear(feature_dim, model_dim)
        self.position = SinusoidalPositionEncoding(model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.layers)
        self.output_projection = nn.Linear(model_dim, feature_dim)
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError(f"Expected [B,T,D], got {tuple(features.shape)}")
        if self.layers == 0:
            return self.output_norm(features)
        hidden = self.position(self.input_projection(features))
        hidden = self.encoder(hidden)
        return self.output_norm(features + self.output_projection(hidden))


class SeverityAwareProtoDual(nn.Module):
    """Transformer-refined ProtoDual with ExpADA-style severity selection.

    Stage 2 receives only the extracted 1024-D instance features.  As in
    ExpADA's released VideoAnchor neck, two internal heads predict instance
    depression category support and instance relevance.  Their product forms
    the weak video-level severity distribution used to adjust ProtoDual key
    weights.  No Stage-1 predictions are passed into this module.
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        transformer_dim: int = 256,
        transformer_layers: int = 3,
        transformer_heads: int = 4,
        transformer_ffn: int = 1024,
        dropout: float = 0.2,
        key_temperature: float = 0.1,
        context_temperature: float = 1.0,
        severity_strength: float = 1.0,
        use_severity: bool = True,
        fusion_mode: str = "fixed",
        fixed_key_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if key_temperature <= 0 or context_temperature <= 0:
            raise ValueError("temperatures must be positive")
        if fusion_mode not in ("fixed", "adaptive"):
            raise ValueError("fusion_mode must be fixed or adaptive")
        if not 0 <= fixed_key_weight <= 1:
            raise ValueError("fixed_key_weight must be in [0,1]")
        self.key_temperature = float(key_temperature)
        self.context_temperature = float(context_temperature)
        self.severity_strength = float(severity_strength)
        self.use_severity = bool(use_severity)
        self.fusion_mode = fusion_mode
        self.fixed_key_weight = float(fixed_key_weight)
        self.transformer = SegmentTransformer(
            feature_dim=feature_dim,
            model_dim=transformer_dim,
            layers=transformer_layers,
            heads=transformer_heads,
            feedforward_dim=transformer_ffn,
            dropout=dropout,
        )
        self.key_scorer = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        # ExpADA VideoAnchor-style category and sample-relevance heads.  Both
        # are learned again at video level from the extracted feature bag.
        self.severity_classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, SEVERITY_LEVELS),
        )
        self.severity_relevance = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, SEVERITY_LEVELS),
        )

        def regressor():
            return nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, 1),
            )

        self.key_regressor = regressor()
        self.context_regressor = regressor()
        if fusion_mode == "adaptive":
            self.fusion_gate = nn.Sequential(
                nn.Linear(feature_dim * 3, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )
        else:
            self.fusion_gate = None

    def forward(self, features: Tensor):
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError("features must be [B,T,D] or [T,D]")

        refined = self.transformer(features)
        key_logits = self.key_scorer(refined).squeeze(-1)
        base_weights = (key_logits / self.key_temperature).softmax(dim=1)

        severity_logits = self.severity_classifier(refined)
        relevance_logits = self.severity_relevance(refined)
        # These two softmax dimensions follow ExpADA's on_cls_pred and
        # on_smp_pred respectively.
        # Compute the ExpADA-style product-and-sum aggregation in log space.
        # This is mathematically equivalent to multiplying the two softmax
        # distributions and normalizing their temporal sum, but it avoids
        # probability underflow and the resulting gradient explosion when the
        # distributions become sharp during training.
        log_instance_severity = F.log_softmax(severity_logits, dim=-1)
        log_instance_relevance = F.log_softmax(relevance_logits, dim=1)
        instance_severity = log_instance_severity.exp()
        instance_relevance = log_instance_relevance.exp()
        severity_evidence = instance_severity * instance_relevance
        log_video_evidence = torch.logsumexp(
            log_instance_severity + log_instance_relevance, dim=1
        )
        video_severity = F.softmax(log_video_evidence, dim=-1)
        consistency = torch.sum(
            instance_severity * video_severity.unsqueeze(1), dim=-1
        ).clamp_min(1e-8)
        if self.use_severity:
            adjusted_logits = (
                key_logits / self.key_temperature
                + self.severity_strength * consistency.log()
            )
            key_weights = adjusted_logits.softmax(dim=1)
        else:
            key_weights = base_weights

        prototype = torch.sum(key_weights.unsqueeze(-1) * refined, dim=1)
        similarity = F.cosine_similarity(
            refined, prototype.unsqueeze(1), dim=-1, eps=1e-8
        )
        context_weights = (similarity / self.context_temperature).softmax(dim=1)
        context = torch.sum(context_weights.unsqueeze(-1) * refined, dim=1)

        if self.fusion_gate is None:
            gate = prototype.new_full((prototype.shape[0],), self.fixed_key_weight)
        else:
            gate_input = torch.cat(
                (prototype, context, torch.abs(prototype - context)), dim=-1
            )
            gate = self.fusion_gate(gate_input).squeeze(-1)

        key_prediction = self.key_regressor(prototype).squeeze(-1)
        context_prediction = self.context_regressor(context).squeeze(-1)
        prediction = gate * key_prediction + (1.0 - gate) * context_prediction

        return {
            "prediction": prediction,
            "key_prediction": key_prediction,
            "context_prediction": context_prediction,
            "fusion_gate_key": gate,
            "key_weights": key_weights,
            "base_key_weights": base_weights,
            "context_weights": context_weights,
            "severity_logits": severity_logits,
            "severity_relevance_logits": relevance_logits,
            "instance_severity_probs": instance_severity,
            "instance_relevance_probs": instance_relevance,
            "severity_evidence": severity_evidence,
            "severity_consistency": consistency,
            "video_severity_probs": video_severity,
            "prototype": prototype,
            "context": context,
            "refined_features": refined,
        }
