"""
GW Classifier: 1D CNN + Transformer Hybrid
==========================================
Innovates beyond simple ResNet by combining multi-scale convolutional
feature extraction with self-attention over temporal segments.

Architecture motivation:
  GW signals contain chirp structure spanning ~0.1–4 s (inspiral → merger).
  Multi-scale CNN captures local frequency evolution; Transformer captures
  long-range phase coherence — key discriminant vs noise.

Reference: Gabbard et al. 2018 (CNN GW), George & Huerta 2018 (DL GW),
           Attention is All You Need (Vaswani 2017).
DOI: 10.1103/PhysRevLett.120.141103
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiScaleConvBlock(nn.Module):
    """
    Parallel 1D convolutions at 3 different temporal scales,
    capturing inspiral (long), merger (medium), ringdown (short) features.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.branch_short  = self._branch(in_ch, out_ch, kernel=7)
        self.branch_medium = self._branch(in_ch, out_ch, kernel=31)
        self.branch_long   = self._branch(in_ch, out_ch, kernel=63)
        self.fuse = nn.Sequential(
            nn.Conv1d(out_ch * 3, out_ch, kernel_size=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def _branch(self, in_ch, out_ch, kernel):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        s = self.branch_short(x)
        m = self.branch_medium(x)
        lg = self.branch_long(x)
        return self.fuse(torch.cat([s, m, lg], dim=1))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer over time patches."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class GWCNNTransformer(nn.Module):
    """
    Hybrid 1D-CNN + Transformer for GW signal classification.

    Pipeline:
      1. Multi-scale CNN encoder: extract multi-resolution features
      2. Adaptive pooling to fixed sequence length for Transformer
      3. Transformer encoder: model long-range temporal dependencies
      4. Global average pooling + MLP classifier head

    Input: (batch, 1, n_time) raw strain
    Output: (batch, 2) logits [noise, signal]
    """

    def __init__(
        self,
        seq_len: int = 16384,
        n_conv_stages: int = 4,
        base_ch: int = 32,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        d_model: int = 128,
        n_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Multi-scale CNN backbone
        self.cnn_stages = nn.ModuleList()
        in_ch = 1
        for i in range(n_conv_stages):
            out_ch = base_ch * (2 ** min(i, 3))
            self.cnn_stages.append(
                nn.Sequential(
                    MultiScaleConvBlock(in_ch, out_ch),
                    nn.MaxPool1d(kernel_size=4, stride=4),
                    nn.Dropout1d(dropout),
                )
            )
            in_ch = out_ch

        self.proj = nn.Conv1d(in_ch, d_model, kernel_size=1)

        # Transformer encoder
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_transformer_layers)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T)
        for stage in self.cnn_stages:
            x = stage(x)
        x = self.proj(x)          # (B, d_model, T')
        x = x.permute(0, 2, 1)    # (B, T', d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)   # (B, T', d_model)
        x = x.mean(dim=1)         # global avg pooling
        return self.head(x)


def build_model(seq_len: int = 16384) -> GWCNNTransformer:
    """Instantiate the full-scale GW classifier (~5M parameters)."""
    return GWCNNTransformer(seq_len=seq_len)


def build_compact_model(seq_len: int = 16384) -> GWCNNTransformer:
    """
    Compact CNN+Transformer for CPU-scale training (~200k parameters).

    Trims base_ch, conv depth, and Transformer width vs `build_model`.
    Architectural skeleton is unchanged so results transfer to the
    full-scale model. Used for the laptop sweep.
    """
    return GWCNNTransformer(
        seq_len=seq_len,
        n_conv_stages=3,
        base_ch=12,
        n_heads=2,
        n_transformer_layers=1,
        d_model=32,
        dropout=0.1,
    )


# ---------------------------------------------------------------------------
# Second, established baseline: a 1D residual CNN (ResNet)
# ---------------------------------------------------------------------------
# Deep residual convolutional networks are the standard architecture for
# gravitational-wave detection on strain data (Gabbard et al. 2018; Nousi et
# al. 2023, PRD 108 024022). Providing this as an independent, more
# conventional baseline addresses the concern that the null result might be
# specific to the compact CNN+Transformer above.

class ResidualBlock1D(nn.Module):
    """Two-conv residual block with optional stride-based downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, kernel: int = 16):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if out.size(-1) != idt.size(-1):        # reconcile odd-length padding
            m = min(out.size(-1), idt.size(-1))
            out, idt = out[..., :m], idt[..., :m]
        return self.act(out + idt)


class ResNet1D(nn.Module):
    """
    1D residual CNN for GR-vs-massive-graviton classification.

    Input:  (batch, 1, n_time) whitened strain
    Output: (batch, 2) logits [GR, MG]
    """

    def __init__(self, seq_len: int = 16384, base_ch: int = 32, n_stages: int = 4,
                 blocks_per_stage: int = 2, kernel: int = 16, n_classes: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_ch, kernel * 2, stride=4, padding=kernel, bias=False),
            nn.BatchNorm1d(base_ch), nn.ReLU(), nn.MaxPool1d(4),
        )
        layers, in_ch = [], base_ch
        for s in range(n_stages):
            out_ch = base_ch * (2 ** s)
            for b in range(blocks_per_stage):
                stride = 4 if b == 0 else 1
                layers.append(ResidualBlock1D(in_ch, out_ch, stride=stride, kernel=kernel))
                in_ch = out_ch
        self.blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(in_ch, n_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        return self.head(x)


def build_resnet1d(seq_len: int = 16384) -> ResNet1D:
    """Full 1D ResNet baseline (~1M parameters); intended for GPU runs."""
    return ResNet1D(seq_len=seq_len, base_ch=32, n_stages=4,
                    blocks_per_stage=2, dropout=0.2)


def build_compact_resnet1d(seq_len: int = 16384) -> ResNet1D:
    """Compact 1D ResNet for CPU-scale smoke tests."""
    return ResNet1D(seq_len=seq_len, base_ch=16, n_stages=3,
                    blocks_per_stage=1, dropout=0.2)
