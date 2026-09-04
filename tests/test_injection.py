"""
Tests for step1_gw_classifier/data.py — injection dataset.

Physics correctness checks:
  - achieved_snr matches target_snr within tolerance
  - GR and MG waveforms agree at large lambda_g (~1e18 km)
  - GR and MG waveforms diverge at small lambda_g (~1e12 km)
  - Whitened strain has approximately unit variance in noise-only regions
  - Labels are balanced
"""
import numpy as np
import pytest

from step1_gw_classifier.data import (
    build_noise_reservoir,
    make_one_injection,
    build_injection_dataset,
)
from step1_gw_classifier.waveform import (
    gr_waveform_fd, massive_graviton_waveform_fd,
)
from step1_gw_classifier.matched_filter import optimal_snr


@pytest.fixture(scope="module")
def reservoir():
    """Shared reservoir for all tests — 128 s of GW150914 pre-event H1."""
    return build_noise_reservoir(
        event_name="GW150914",
        pre_seconds=128,
        sample_rate=4096.0,
        seg_duration_s=4.0,
    )


def test_reservoir_basic(reservoir):
    assert reservoir.sample_rate == 4096.0
    assert reservoir.n_samples_per_seg == 16384
    assert len(reservoir.strain) >= 128 * 4096
    assert len(reservoir.psd_freqs) == len(reservoir.psd_vals)
    # PSD strictly positive in band
    in_band = (reservoir.psd_freqs > 20) & (reservoir.psd_freqs < 1024)
    assert np.all(reservoir.psd_vals[in_band] > 0)


def test_gr_injection_snr_matches_target(reservoir):
    rng = np.random.default_rng(0)
    target = 15.0
    x, meta = make_one_injection(
        m1=36.0, m2=29.0, target_snr=target, lambda_g_km=None,
        reservoir=reservoir, rng=rng,
    )
    # Achieved SNR is computed before noise is added; should match target tightly
    assert abs(meta.achieved_snr - target) / target < 0.02
    assert meta.label == 0
    assert meta.lambda_g_km is None
    assert np.all(np.isfinite(x))


def test_mg_injection_snr_matches_target(reservoir):
    rng = np.random.default_rng(0)
    target = 15.0
    x, meta = make_one_injection(
        m1=36.0, m2=29.0, target_snr=target, lambda_g_km=1e15,
        reservoir=reservoir, rng=rng,
    )
    assert abs(meta.achieved_snr - target) / target < 0.02
    assert meta.label == 1
    assert meta.lambda_g_km == 1e15
    assert np.all(np.isfinite(x))


def test_gr_and_mg_converge_at_large_lambda_g():
    """At lambda_g = 1e19 km, dispersion is negligible -> waveforms agree."""
    freqs, h_gr = gr_waveform_fd(36.0, 29.0, 410.0, delta_f=0.25,
                                 f_lower=20.0, f_final=1024.0)
    _, h_mg = massive_graviton_waveform_fd(
        36.0, 29.0, 410.0, lambda_g_m=1e19 * 1e3,
        delta_f=0.25, f_lower=20.0, f_final=1024.0,
    )
    # Use a flat PSD here (just for inner product structure)
    psd = np.ones_like(freqs)
    # Mask out f<f_lower
    psd[freqs < 20.0] = np.inf
    from step1_gw_classifier.matched_filter import match
    M = match(h_gr, h_mg, psd, 0.25, freqs=freqs, f_low=20.0, f_high=1024.0)
    assert M > 0.999, f"large-lambda match should be ~1, got {M:.4f}"


def test_gr_and_mg_diverge_at_small_lambda_g():
    """At lambda_g = 1e12 km the corrected (chirp-mass-free) dispersion phase
    is many radians across the band -> match << 1."""
    freqs, h_gr = gr_waveform_fd(36.0, 29.0, 410.0, delta_f=0.25,
                                 f_lower=20.0, f_final=1024.0)
    _, h_mg = massive_graviton_waveform_fd(
        36.0, 29.0, 410.0, lambda_g_m=1e12 * 1e3,
        delta_f=0.25, f_lower=20.0, f_final=1024.0,
    )
    psd = np.ones_like(freqs)
    psd[freqs < 20.0] = np.inf
    from step1_gw_classifier.matched_filter import match
    M = match(h_gr, h_mg, psd, 0.25, freqs=freqs, f_low=20.0, f_high=1024.0)
    assert M < 0.5, f"small-lambda match should be ~ chance, got {M:.4f}"


def test_whitened_noise_is_approximately_unit_variance(reservoir):
    """Without an injection, whitened noise should have variance ~ 1."""
    from step1_gw_classifier.data import _whiten_td
    rng = np.random.default_rng(1)
    noise = reservoir.draw_segment(rng)
    wn = _whiten_td(noise, reservoir.psd_vals, reservoir.sample_rate)
    # Discard edges (whitening kernel ringing)
    n = len(wn)
    var = wn[n // 8 : 7 * n // 8].var()
    # Should be order 1; allow factor 3 either way (PSD edge effects)
    assert 0.3 < var < 3.0, f"whitened noise variance = {var:.3f}"


def test_balanced_dataset_small(reservoir):
    """Build a tiny dataset and check shape + labels."""
    X, Xraw, y, meta, _ = build_injection_dataset(
        n_per_class=8, target_snr=15.0, lambda_g_km=1e15,
        reservoir_seconds=128, reservoir_event="GW150914", seed=7,
    )
    assert X.shape == (16, reservoir.n_samples_per_seg)
    assert Xraw.shape == X.shape
    assert y.shape == (16,)
    assert (y == 0).sum() == 8 and (y == 1).sum() == 8
    assert len(meta) == 16
    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(Xraw))
    assert X.dtype == np.float32 and Xraw.dtype == np.float32
