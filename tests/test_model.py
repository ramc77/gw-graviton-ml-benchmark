"""
Forward-pass tests for both classifier architectures: the CNN+Transformer of
the main study and the 1D ResNet baseline added for the architecture ablation.
Offline (random input); checks output shape, finiteness, and a sane parameter
count.
"""
import torch

from step1_gw_classifier.model import (
    build_model, build_compact_model, build_resnet1d, build_compact_resnet1d,
)


def _forward_ok(model, seq_len=4096):
    x = torch.randn(2, 1, seq_len)
    y = model(x)
    assert y.shape == (2, 2), y.shape
    assert torch.isfinite(y).all()


def _nparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_cnn_transformer_forward():
    _forward_ok(build_compact_model(4096))
    _forward_ok(build_model(4096))


def test_resnet_forward():
    _forward_ok(build_compact_resnet1d(4096))
    _forward_ok(build_resnet1d(4096))


def test_param_counts_reasonable():
    assert 5e4 < _nparams(build_compact_model(4096)) < 5e5
    assert 1e5 < _nparams(build_resnet1d(4096)) < 5e6
