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
