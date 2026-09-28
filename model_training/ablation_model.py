"""Configurable model variants for ripple/IED architecture ablations.

The baseline configuration intentionally preserves the module names and forward
operations of :class:`model_training.model.SharedModalityModel`.  Ablated
variants can therefore be compared without modifying the production model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import (
    ChannelSpecEncoder,
    ChannelWaveEncoder,
    CrossChannelAttention,
    ModalityAdapter,
    SharedBackbone,
    SpikeOscillationHead,
    TemporalConvBlock,
)


@dataclass(frozen=True)
class AblationConfig:
    """Switches defining one architecture ablation."""

    name: str = "baseline"
    modality: str = "both"
    fusion: str = "learned"
    use_global_features: bool = True
    use_temporal_conv: bool = True
    use_spike_oscillation: bool = True
    use_channel_attention: bool = True

    def __post_init__(self) -> None:
        if self.modality not in {"both", "wave", "spec"}:
            raise ValueError("modality must be one of: both, wave, spec")
        if self.fusion not in {"learned", "average"}:
            raise ValueError("fusion must be one of: learned, average")
        if self.modality != "both" and self.fusion != "learned":
            raise ValueError("fusion is only configurable when modality='both'")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


BASELINE_CONFIG = AblationConfig()

ABLATION_CONFIGS = {
    "baseline": BASELINE_CONFIG,
    "wave_only": replace(BASELINE_CONFIG, name="wave_only", modality="wave"),
    "spec_only": replace(BASELINE_CONFIG, name="spec_only", modality="spec"),
    "fixed_fusion": replace(
        BASELINE_CONFIG,
        name="fixed_fusion",
        fusion="average",
    ),
    "no_channel_attention": replace(
        BASELINE_CONFIG,
        name="no_channel_attention",
        use_channel_attention=False,
    ),
    "no_global_features": replace(
        BASELINE_CONFIG,
        name="no_global_features",
        use_global_features=False,
    ),
    "no_temporal_conv": replace(
        BASELINE_CONFIG,
        name="no_temporal_conv",
        use_temporal_conv=False,
    ),
    "no_spike_oscillation": replace(
        BASELINE_CONFIG,
        name="no_spike_oscillation",
        use_spike_oscillation=False,
    ),
}


def get_ablation_config(name: str) -> AblationConfig:
    """Return a named ablation configuration."""
    try:
        return ABLATION_CONFIGS[name]
    except KeyError as error:
        choices = ", ".join(ABLATION_CONFIGS)
        raise ValueError(f"Unknown ablation {name!r}; choose from: {choices}") from error


class TemporalAbsoluteMean(nn.Module):
    """Parameter-free replacement for the spike/oscillation summary head."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.abs().mean(dim=2)


class AblationModel(nn.Module):
    """Shared-modality classifier with independently removable major blocks."""

    def __init__(
        self,
        config: AblationConfig,
        feat_dim: int = 64,
        n_classes: int = 1,
        global_feat_dim: int = 5,
        global_emb_dim: int = 32,
    ):
        super().__init__()
        self.config = config
        self.use_global_features = config.use_global_features
        self.global_emb_dim = global_emb_dim if config.use_global_features else 0

        self.wave_enc = (
            ChannelWaveEncoder(feat_dim)
            if config.modality in {"wave", "both"}
            else None
        )
        self.spec_enc = (
            ChannelSpecEncoder(feat_dim)
            if config.modality in {"spec", "both"}
            else None
        )

        self.shared = SharedBackbone(feat_dim)
        self.wave_adapter = (
            ModalityAdapter(feat_dim)
            if config.modality in {"wave", "both"}
            else None
        )
        self.spec_adapter = (
            ModalityAdapter(feat_dim)
            if config.modality in {"spec", "both"}
            else None
        )

        self.fusion_gate = (
            nn.Linear(2 * feat_dim, feat_dim)
            if config.modality == "both" and config.fusion == "learned"
            else None
        )

        if config.use_global_features:
            self.cc_encoder = nn.Sequential(
                nn.Linear(global_feat_dim, global_emb_dim),
                nn.ReLU(),
                nn.Linear(global_emb_dim, global_emb_dim),
            )
        else:
            self.cc_encoder = None

        self.temp_conv = (
            TemporalConvBlock(feat_dim) if config.use_temporal_conv else nn.Identity()
        )
        self.spike_osc = (
            SpikeOscillationHead(feat_dim)
            if config.use_spike_oscillation
            else TemporalAbsoluteMean()
        )
        self.channel_attn = (
            CrossChannelAttention(feat_dim)
            if config.use_channel_attention
            else nn.Identity()
        )

        projection_input = feat_dim + self.global_emb_dim
        self.projection_head = nn.Sequential(
            nn.Linear(projection_input, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 64),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def _encode_wave(self, wave: torch.Tensor) -> torch.Tensor:
        if self.wave_enc is None or self.wave_adapter is None:
            raise ValueError("This ablation does not contain a waveform encoder")
        return self.wave_adapter(self.shared(self.wave_enc(wave)))

    def _encode_spec(self, spec: torch.Tensor) -> torch.Tensor:
        if self.spec_enc is None or self.spec_adapter is None:
            raise ValueError("This ablation does not contain a spectrogram encoder")
        return self.spec_adapter(self.shared(self.spec_enc(spec)))

    def forward(
        self,
        wave: torch.Tensor,
        spec: torch.Tensor,
        cc_feat: torch.Tensor | None = None,
        mode: str = "both",
    ):
        # ``evaluate`` passes the baseline default, while explicit non-default
        # modes remain useful for diagnostic inference on the baseline model.
        effective_mode = self.config.modality if mode == "both" else mode
        if effective_mode not in {"wave", "spec", "both"}:
            raise ValueError("mode must be one of: wave, spec, both")

        wave_features = (
            self._encode_wave(wave) if effective_mode in {"wave", "both"} else None
        )
        spec_features = (
            self._encode_spec(spec) if effective_mode in {"spec", "both"} else None
        )

        if effective_mode == "both":
            if wave_features is None or spec_features is None:
                raise ValueError("This ablation does not contain both modality encoders")
            if wave_features.shape != spec_features.shape:
                raise ValueError("Waveform and spectrogram encodings must have equal shapes")
            if self.config.fusion == "learned":
                gate = torch.sigmoid(
                    self.fusion_gate(torch.cat([wave_features, spec_features], dim=-1))
                )
                x = gate * wave_features + (1 - gate) * spec_features
            else:
                x = 0.5 * (wave_features + spec_features)
        elif effective_mode == "wave":
            x = wave_features
        else:
            x = spec_features

        x = self.temp_conv(x)
        x = self.spike_osc(x)
        x = self.channel_attn(x)
        x = torch.max(x, dim=1).values

        if self.use_global_features:
            if cc_feat is None:
                raise ValueError("cc_feat must be provided when global features are enabled")
            if cc_feat.ndim == 1:
                cc_feat = cc_feat.unsqueeze(1)
            x = torch.cat([x, self.cc_encoder(cc_feat)], dim=1)

        embeddings = F.normalize(self.projection_head(x), dim=1)
        logits = self.classifier(embeddings)
        return logits.squeeze(-1), embeddings, {}

