"""
PSD estimation, whitening, and matched-filter SNR.
===================================================

Provides:
  - estimate_psd_welch(strain, ...)    Welch median PSD from off-source data
  - inner_product(a, b, psd, df)        4 Re int a* b / S_n df
  - matched_filter_snr(template, data, psd, df)  optimal MF SNR
  - match(h1, h2, psd, df)              normalised overlap in [0, 1]
                                        (after maximizing over time/phase)

Conventions:
  * All inputs are frequency-domain arrays h(f) at uniform delta_f
    starting at f=0.
  * The PSD must be one-sided S_n(f).
  * Frequency mask is applied by passing arrays already zeroed
    outside [f_low, f_high], or by setting psd = inf there.

References:
  Allen et al. 2012, PRD 85 122006 (matched-filter formalism)
  Cutler & Flanagan 1994, PRD 49 2658 (Fisher matrix, inner product)
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch


def estimate_psd_welch(
    strain: np.ndarray,
    sample_rate: float,
    fftlength_s: float = 4.0,
    overlap_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Median-Welch PSD estimate. Median is more robust to transient
    glitches than mean (standard GW pipeline choice).

    Returns (freqs [Hz], psd [strain^2/Hz]).
    """
    nperseg = int(round(fftlength_s * sample_rate))
    noverlap = int(round(nperseg * overlap_frac))
    f, psd = welch(
        strain, fs=sample_rate, window="hann",
        nperseg=nperseg, noverlap=noverlap,
        average="median", return_onesided=True,
    )
    return f, psd


def inner_product(
    a: np.ndarray,
    b: np.ndarray,
    psd: np.ndarray,
    df: float,
    f_low: float | None = None,
    f_high: float | None = None,
    freqs: np.ndarray | None = None,
) -> float:
    """
    (a | b) = 4 Re int_{f_low}^{f_high} a*(f) b(f) / S_n(f) df

    a, b, psd must be sampled at the same df. If freqs is supplied
    along with f_low/f_high, the integral is masked.
    """
    integrand = np.conjugate(a) * b / np.where(psd > 0, psd, np.inf)
    if freqs is not None and (f_low is not None or f_high is not None):
        mask = np.ones_like(freqs, dtype=bool)
        if f_low is not None:
            mask &= freqs >= f_low
        if f_high is not None:
            mask &= freqs <= f_high
        integrand = np.where(mask, integrand, 0.0)
    return float(4.0 * df * np.real(integrand.sum()))


def optimal_snr(
    h: np.ndarray,
    psd: np.ndarray,
    df: float,
    **mask_kwargs,
) -> float:
    """Optimal matched-filter SNR for template h in stationary noise: sqrt((h|h))."""
    return float(np.sqrt(max(inner_product(h, h, psd, df, **mask_kwargs), 0.0)))


def match(
    h1: np.ndarray,
    h2: np.ndarray,
    psd: np.ndarray,
    df: float,
    **mask_kwargs,
) -> float:
    """
    Normalised match (max over time, phase) in [0, 1]:
        M = max_{t0, phi0} Re < h1 | h2 e^{-2pi i f t0 - i phi0} >
            / sqrt((h1|h1)(h2|h2))
    Implemented as IFFT( h1 conj(h2) / S_n ) and taking the absolute max,
    normalised by sqrt((h1|h1)(h2|h2)).
    """
    norm1 = np.sqrt(max(inner_product(h1, h1, psd, df, **mask_kwargs), 0.0))
    norm2 = np.sqrt(max(inner_product(h2, h2, psd, df, **mask_kwargs), 0.0))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    # Build masked integrand for the time-domain correlation
    integrand = h1 * np.conjugate(h2) / np.where(psd > 0, psd, np.inf)
    if mask_kwargs:
        freqs = mask_kwargs.get("freqs")
        f_low = mask_kwargs.get("f_low")
        f_high = mask_kwargs.get("f_high")
        if freqs is not None:
            mask = np.ones_like(freqs, dtype=bool)
            if f_low is not None:
                mask &= freqs >= f_low
            if f_high is not None:
                mask &= freqs <= f_high
            integrand = np.where(mask, integrand, 0.0)
    # Make a 2-sided array for IFFT
    n_pos = len(integrand)
    two_sided = np.zeros(2 * (n_pos - 1), dtype=np.complex128)
    two_sided[:n_pos] = integrand
    corr = np.fft.ifft(two_sided) * len(two_sided) * df * 4.0
    max_corr = float(np.max(np.abs(corr)))
    return max_corr / (norm1 * norm2)
