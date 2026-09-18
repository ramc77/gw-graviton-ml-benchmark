"""
Tests for the colored-Gaussian noise reservoir (data.colored_gaussian_segment /
GaussianNoiseReservoir), added for the noise-ladder control.

The decisive property is that a unit-norm template projected onto noise-only
colored-Gaussian data has UNIT VARIANCE in the matched-filter inner-product
convention, i.e. the injected noise power matches the PSD the matched filter
assumes. These tests are offline (analytic PSD, no GWOSC fetch).
"""
import numpy as np

from step1_gw_classifier.data import (
    colored_gaussian_segment, GaussianNoiseReservoir,
)
from step1_gw_classifier.matched_filter import inner_product


def _grid(sr=1024.0, seg=4.0):
    n_t = int(round(sr * seg))
    n_f = n_t // 2 + 1
    df = 1.0 / seg
    freqs = np.fft.rfftfreq(n_t, d=1.0 / sr)
    return sr, seg, n_t, n_f, df, freqs


def test_unit_variance_projection():
    """Var(<noise | normalised template>) == 1 in the inner-product convention."""
    sr, seg, n_t, n_f, df, freqs = _grid()
    f_low, f_high = 20.0, 400.0
    # A steep-ish analytic one-sided PSD (mimics a detector wall + floor).
    psd = 1e-40 * (1.0 + (60.0 / np.maximum(freqs, 1.0)) ** 4) + 1e-46
    # A smooth in-band normalised template.
    h = np.zeros(n_f, dtype=np.complex128)
    band = (freqs >= f_low) & (freqs <= f_high)
    h[band] = freqs[band] ** (-7.0 / 6.0) * np.exp(1j * freqs[band])
    sig2 = inner_product(h, h, psd, df, freqs=freqs, f_low=f_low, f_high=f_high)
    h_n = h / np.sqrt(sig2)

    rng = np.random.default_rng(0)
    z = np.array([
        inner_product(np.fft.rfft(colored_gaussian_segment(psd, sr, n_t, rng)) / sr,
                      h_n, psd, df, freqs=freqs, f_low=f_low, f_high=f_high)
        for _ in range(800)
    ])
    assert abs(z.var() - 1.0) < 0.12, f"projection variance {z.var():.3f} != 1"
    assert abs(z.mean()) < 0.12, f"projection mean {z.mean():.3f} != 0"


def test_time_domain_variance_matches_integral():
    """Time-domain variance equals the integral of the one-sided PSD."""
    sr, seg, n_t, n_f, df, freqs = _grid()
    psd = np.full(n_f, 2.0e-42)          # flat PSD
    rng = np.random.default_rng(1)
    var_td = np.mean([colored_gaussian_segment(psd, sr, n_t, rng).var()
                      for _ in range(40)])
    expected = float(np.sum(psd) * df)   # integral S(f) df over one-sided band
    assert abs(var_td / expected - 1.0) < 0.05, (var_td, expected)


def test_reservoir_draw_segment_shape():
    sr, seg, n_t, n_f, df, freqs = _grid()
    res = GaussianNoiseReservoir(sample_rate=sr, seg_duration_s=seg,
                                 psd_freqs=freqs, psd_vals=np.full(n_f, 1e-42))
    x = res.draw_segment(np.random.default_rng(2))
    assert x.shape == (n_t,)
    assert np.isfinite(x).all()
    assert res.n_samples_per_seg == n_t
