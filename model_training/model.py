"""CNN architecture for ripple/IED/noise event classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpikeOscillationHead(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.spike_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)

        self.osc_conv_short = nn.Conv1d(dim, dim, kernel_size=15, padding=7, groups=dim)
        self.osc_conv_long = nn.Conv1d(dim, dim, kernel_size=31, padding=15, groups=dim)

        self.fuse = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU()
        )

    def forward(self, x):
        B, C, T, D = x.shape

        x = x.reshape(B * C, T, D).permute(0, 2, 1)

        spike = self.spike_conv(x)
        osc_short = self.osc_conv_short(x)
        osc_long = self.osc_conv_long(x)

        spike_feat = spike.abs().mean(dim=2)
        osc_short_feat = osc_short.abs().mean(dim=2)
        osc_long_feat = osc_long.abs().mean(dim=2)

        feat = torch.cat(
            [spike_feat, osc_short_feat, osc_long_feat],
            dim=-1
        )

        feat = self.fuse(feat)

        return feat.view(B, C, D)

class TemporalConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv3 = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.conv7 = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.conv15 = nn.Conv1d(dim, dim, kernel_size=15, padding=7, groups=dim)
        self.conv31 = nn.Conv1d(dim, dim, kernel_size=31, padding=15, groups=dim)

        self.pointwise = nn.Conv1d(dim * 4, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, T, D = x.shape

        x = x.reshape(B * C, T, D).permute(0, 2, 1)

        res = x
        x1 = self.conv3(x)
        x2 = self.conv7(x)
        x3 = self.conv15(x)
        x4 = self.conv31(x)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.pointwise(x)

        x = x.permute(0, 2, 1)
        x = self.norm(x + res.permute(0, 2, 1))

        return x.reshape(B, C, T, D)

class SharedBackbone(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class ModalityAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.down = nn.Linear(dim, dim // 2)
        self.up = nn.Linear(dim // 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        r = x
        x = F.relu(self.down(x))
        x = self.up(x)
        return self.norm(x + r)


class CrossChannelAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.scale = dim ** -0.5

    def forward(self, x):
        """
        x: (B, C, D)
        """

        Q = self.q(x)  # (B, C, D)
        K = self.k(x)
        V = self.v(x)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, C, C)
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)  # (B, C, D)

        return out

# =========================
# ENCODERS
# =========================

class ChannelWaveEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, out_dim, 5, padding=2),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape

        x = x.reshape(B * C, 1, T)
        x = self.conv(x)

        x = x.permute(0, 2, 1)      # (B*C, T, D)
        x = x.reshape(B, C, T, -1)  # (B, C, T, D)

        return x


class ChannelSpecEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.proj = nn.Linear(32, out_dim)

    def forward(self, x):
        """
        x: (B, C, F, T)
        """
        B, C, F, T = x.shape

        x = x.reshape(B * C, 1, F, T)

        x = self.conv(x)  # (B*C, 32, F, T)

        # collapse frequency only
        x = x.mean(dim=2)          # (B*C, 32, T)
        x = x.permute(0, 2, 1)     # (B*C, T, 32)

        x = self.proj(x)           # (B*C, T, D)

        x = x.view(B, C, T, -1)

        return x


class SharedModalityModel(nn.Module):
    def __init__(
        self,
        feat_dim=64,
        n_classes=1,
        use_global_features=True,
        global_feat_dim=5,
        global_emb_dim=32,
    ):
        super().__init__()

        self.use_global_features = use_global_features
        self.global_emb_dim = global_emb_dim if use_global_features else 0

        self.wave_enc = ChannelWaveEncoder(feat_dim)
        self.spec_enc = ChannelSpecEncoder(feat_dim)

        self.shared = SharedBackbone(feat_dim)
        self.wave_adapter = ModalityAdapter(feat_dim)
        self.spec_adapter = ModalityAdapter(feat_dim)

        self.fusion_gate = nn.Linear(2 * feat_dim, feat_dim)

        if use_global_features:
            self.cc_encoder = nn.Sequential(
                nn.Linear(global_feat_dim, global_emb_dim),
                nn.ReLU(),
                nn.Linear(global_emb_dim, global_emb_dim)
            )
        else:
            self.cc_encoder = None

        self.temp_conv = TemporalConvBlock(feat_dim)
        self.spike_osc = SpikeOscillationHead(feat_dim)
        self.channel_attn = CrossChannelAttention(feat_dim)

        proj_in = feat_dim + self.global_emb_dim

        self.projection_head = nn.Sequential(
            nn.Linear(proj_in, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 64)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, wave, spec, cc_feat=None, mode="both"):
        if mode not in {"wave", "spec", "both"}:
            raise ValueError("mode must be one of: wave, spec, both")

        if mode in ["wave", "both"]:
            w = self.wave_enc(wave)
            w = self.shared(w)
            w = self.wave_adapter(w)
        else:
            w = None

        if mode in ["spec", "both"]:
            s = self.spec_enc(spec)
            s = self.shared(s)
            s = self.spec_adapter(s)
        else:
            s = None

        if mode == "both":
            assert w.shape == s.shape
            gate = torch.sigmoid(self.fusion_gate(torch.cat([w, s], dim=-1)))
            x = gate * w + (1 - gate) * s
        elif mode == "wave":
            x = w
        else:
            x = s

        x = self.temp_conv(x)
        x = self.spike_osc(x)

        x = self.channel_attn(x)
        x = torch.max(x, dim=1)[0]

        if self.use_global_features:
            if cc_feat is None:
                raise ValueError("cc_feat must be provided when use_global_features=True")

            if cc_feat.ndim == 1:
                cc_feat = cc_feat.unsqueeze(1)

            c = self.cc_encoder(cc_feat)
            x = torch.cat([x, c], dim=1)

        z = self.projection_head(x)
        z = F.normalize(z, dim=1)

        logits = self.classifier(z)

        return logits.squeeze(-1), z, {}
