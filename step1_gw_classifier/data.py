"""
Injection dataset for the dispersion classifier.
================================================

Builds (whitened strain, label) pairs where:
  - noise is real off-source LIGO H1 strain from GWOSC
  - signal is either GR (IMRPhenomD) or massive-graviton (GR + MY-2012 dispersion)
  - injection is rescaled to a target *optimal* matched-filter SNR
    on the measured PSD

Per (lambda_g, target SNR) configuration we draw N injections by sampling
component masses + distance from the real GWTC-4.0 BBH catalog
(step1_gw_classifier/gwtc4_bbh_events.csv); the distance is then rescaled
to hit the SNR target so we are not coupled to the catalog distance.

Returns dict suitable for direct conversion to torch.Tensor:
  X      : (N, n_samples) float32   whitened time-domain strain
  y      : (N,) int64               0 = GR, 1 = MG
  meta   : (N,) records             m1, m2, dist, target_snr, lambda_g, ...

This module replaces the original synthetic data.py.

References
----------
- Allen et al. 2012, PRD 85, 122006 — injection / matched-filter conventions
- Khan et al. 2016, PRD 93, 044007 — IMRPhenomD
- Mirshekari, Yunes & Will 2012, PRD 85, 024041 — dispersion phase
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from loguru import logger

from .waveform import gr_waveform_fd, massive_graviton_waveform_fd
from .matched_filter import estimate_psd_welch, optimal_snr


# ---------------------------------------------------------------------------
# Event catalog loader
# ---------------------------------------------------------------------------

_DEFAULT_CATALOG = Path(__file__).parent / "gwtc4_bbh_events.csv"


def load_bbh_catalog(path: Path | str = _DEFAULT_CATALOG) -> list[dict]:
    """Return the curated GWTC-4.0 BBH event list as a list of dicts."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "name": r["name"],
                "gps": float(r["gps"]),
                "m1": float(r["mass_1_source"]),
                "m2": float(r["mass_2_source"]),
                "dist_mpc": float(r["luminosity_distance_Mpc"]),
                "snr_cat": float(r["network_snr"]),
            })
    return rows


# ---------------------------------------------------------------------------
# Noise reservoir
# ---------------------------------------------------------------------------

@dataclass
class NoiseReservoir:
    """
    A long stretch of real, off-source H1 strain + its PSD estimate.

    We sample non-overlapping segments of length `seg_duration_s` for
    each injection. Using one PSD across all injections from a single
    reservoir is consistent with the assumption of stationarity
    over the reservoir window.
    """
    strain: np.ndarray
    sample_rate: float
    seg_duration_s: float
    psd_freqs: np.ndarray
    psd_vals: np.ndarray

    @property
    def n_samples_per_seg(self) -> int:
        return int(round(self.sample_rate * self.seg_duration_s))

    def draw_segment(self, rng: np.random.Generator) -> np.ndarray:
        n = self.n_samples_per_seg
        max_start = len(self.strain) - n
        if max_start <= 0:
            raise RuntimeError("Reservoir shorter than one segment.")
        i0 = int(rng.integers(0, max_start))
        return self.strain[i0:i0 + n].copy()


def build_noise_reservoir(
    event_name: str = "GW150914",
    pre_seconds: int = 1024,
    sample_rate: float = 4096.0,
    seg_duration_s: float = 4.0,
    psd_fft_s: float = 4.0,
) -> NoiseReservoir:
    """
    Pull a long pre-event H1 segment from GWOSC; build a reservoir + PSD.

    `pre_seconds` of strain is sampled ending 16 s before the event time.
    """
    from gwosc.datasets import event_gps
    from gwpy.timeseries import TimeSeries

    gps = event_gps(event_name)
    logger.info(f"Fetching {pre_seconds} s H1 strain ending {gps-16:.0f}…")
    ts = TimeSeries.fetch_open_data(
        "H1", gps - pre_seconds - 16, gps - 16,
        sample_rate=int(sample_rate), cache=True, verbose=False,
    )
    strain = np.asarray(ts.value, dtype=np.float64)
    pf, pv = estimate_psd_welch(strain, sample_rate=sample_rate,
                                fftlength_s=psd_fft_s)
    return NoiseReservoir(
        strain=strain, sample_rate=sample_rate,
        seg_duration_s=seg_duration_s,
        psd_freqs=pf, psd_vals=pv,
    )


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def _align_psd_to_waveform(psd_vals: np.ndarray, n_target: int) -> np.ndarray:
    """Stretch/truncate a PSD onto a waveform frequency grid of length n_target."""
    if len(psd_vals) == n_target:
        return psd_vals
    if len(psd_vals) > n_target:
        return psd_vals[:n_target]
    return np.concatenate([psd_vals, np.full(n_target - len(psd_vals), psd_vals[-1])])


def _rescale_distance_for_snr(
    snr_at_dref: float, target_snr: float, dref_mpc: float
) -> float:
    """Strain ∝ 1/D, so SNR ∝ 1/D. Pick D so optimal SNR = target_snr."""
    return dref_mpc * snr_at_dref / target_snr


def _time_domain_from_fd(h_fd: np.ndarray, delta_f: float, sample_rate: float,
                        seg_duration_s: float) -> np.ndarray:
    """
    Convert one-sided frequency-domain waveform h(f) (length n_f = n_t//2+1
    at delta_f = 1/seg_duration) to a real time-domain segment of length
    n_t = sample_rate * seg_duration via inverse rFFT.
    """
    n_t = int(round(sample_rate * seg_duration_s))
    n_f_expected = n_t // 2 + 1
    if len(h_fd) > n_f_expected:
        h_fd = h_fd[:n_f_expected]
    elif len(h_fd) < n_f_expected:
        h_fd = np.concatenate([h_fd, np.zeros(n_f_expected - len(h_fd),
                                              dtype=h_fd.dtype)])
    return np.fft.irfft(h_fd, n=n_t) * sample_rate


def _whiten_td(strain_td: np.ndarray, psd_vals: np.ndarray,
               sample_rate: float) -> np.ndarray:
    """
    Whiten a time-domain segment by 1/sqrt(S_n(f)), then standardise to
    zero mean and unit variance over the interior of the segment.

    The interior-only normalisation removes whitener edge ringing
    (first/last 1/8 of the segment) from the scale estimate; the same
    rescaling is applied to the whole segment. The absolute scale
    after frequency-domain whitening depends on FFT normalisation
    conventions in a way that does not affect the classifier (which
    standardises per-segment anyway) or the matched filter (which uses
    the PSD-weighted inner product directly, not the whitened series).
    """
    n = len(strain_td)
    delta_f = sample_rate / n
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    psd_aligned = _align_psd_to_waveform(psd_vals, len(freqs))
    psd_safe = np.where(psd_aligned > 0, psd_aligned, np.inf)
    spec = np.fft.rfft(strain_td)
    spec_w = spec / np.sqrt(psd_safe / (2.0 * delta_f))
    out = np.fft.irfft(spec_w, n=n)
    # Interior-only stats; apply to whole segment
    interior = out[n // 8 : 7 * n // 8]
    mu = interior.mean()
    sigma = interior.std() + 1e-30
    out_norm = (out - mu) / sigma
    return out_norm.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class InjectionMeta:
    m1: float
    m2: float
    dist_mpc_eff: float
    target_snr: float
    achieved_snr: float
    lambda_g_km: float | None     # None for GR
    label: int                    # 0 = GR, 1 = MG
    event_drawn: str


def make_one_injection(
    m1: float, m2: float, target_snr: float,
    lambda_g_km: float | None,
    reservoir: NoiseReservoir,
    rng: np.random.Generator,
    f_lower: float = 20.0,
    f_final: float = 1024.0,
    dref_mpc: float = 100.0,
    return_raw: bool = False,
    noise_scale: float = 1.0,
) -> tuple[np.ndarray, InjectionMeta] | tuple[np.ndarray, np.ndarray, InjectionMeta]:
    """
    Generate ONE injection: real noise segment + (GR or MG) signal rescaled
    to target SNR. Returns whitened time-domain strain + metadata.

    If `return_raw=True`, also returns the un-whitened time-domain strain
    (noise + signal) for matched-filter analysis. The un-whitened series
    is what the matched filter requires as input; whitening is only for
    the classifier.

    `noise_scale` multiplies the drawn noise segment before it is added to
    the signal. The default 1.0 reproduces the standard injection exactly.
    Setting `noise_scale=0.0` yields a *signal-only* (noise-free) example:
    the noise segment is still drawn (so the random-number stream, and hence
    the sequence of sampled source parameters, is identical to a noisy run
    with the same seed) but contributes nothing. This is the classifier
    analogue of the matched-filter signal-only control.
    """
    sr = reservoir.sample_rate
    seg_dur = reservoir.seg_duration_s
    delta_f = 1.0 / seg_dur

    # 1. Reference waveform at dref_mpc to compute SNR-vs-distance scaling
    freqs, h_ref = gr_waveform_fd(
        m1, m2, dref_mpc,
        delta_f=delta_f, f_lower=f_lower, f_final=f_final,
    )
    psd_aligned = _align_psd_to_waveform(reservoir.psd_vals, len(freqs))
    snr_ref = optimal_snr(h_ref, psd_aligned, delta_f,
                          freqs=freqs, f_low=f_lower, f_high=f_final)
    if snr_ref <= 0 or not np.isfinite(snr_ref):
        raise RuntimeError("Reference SNR is non-positive — bad PSD?")
    dist_eff = _rescale_distance_for_snr(snr_ref, target_snr, dref_mpc)

    # 2. Final signal (GR or MG) at rescaled distance
    if lambda_g_km is None:
        _, h_fd = gr_waveform_fd(m1, m2, dist_eff, delta_f=delta_f,
                                 f_lower=f_lower, f_final=f_final)
        label = 0
    else:
        _, h_fd = massive_graviton_waveform_fd(
            m1, m2, dist_eff, lambda_g_km * 1e3,
            delta_f=delta_f, f_lower=f_lower, f_final=f_final,
        )
        label = 1
    achieved = optimal_snr(h_fd, psd_aligned, delta_f,
                           freqs=freqs, f_low=f_lower, f_high=f_final)

    # 3. To time domain
    signal_td = _time_domain_from_fd(h_fd, delta_f, sr, seg_dur)

    # 4. Real noise + signal (noise_scale=0 -> signal-only control; the
    #    segment is still drawn so the RNG stream is identical across runs)
    noise_td = reservoir.draw_segment(rng)
    # In case noise length differs by a sample due to rounding
    n_min = min(len(signal_td), len(noise_td))
    strain_td = noise_scale * noise_td[:n_min] + signal_td[:n_min]

    # 5. Whiten
    whitened = _whiten_td(strain_td, reservoir.psd_vals, sr)

    meta = InjectionMeta(
        m1=m1, m2=m2, dist_mpc_eff=dist_eff,
        target_snr=target_snr, achieved_snr=achieved,
        lambda_g_km=lambda_g_km, label=label, event_drawn="(reservoir)",
    )
    if return_raw:
        return whitened, strain_td.astype(np.float32), meta
    return whitened, meta


def build_injection_dataset(
    n_per_class: int = 200,
    target_snr: float = 12.0,
    lambda_g_km: float = 1e15,
    f_lower: float = 20.0,
    f_final: float = 1024.0,
    seg_duration_s: float = 4.0,
    sample_rate: float = 4096.0,
    reservoir_event: str = "GW150914",
    reservoir_seconds: int = 1024,
    seed: int = 42,
    catalog_path: Path | str = _DEFAULT_CATALOG,
    noise_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, list[InjectionMeta], NoiseReservoir]:
    """
    Build a balanced binary dataset: n_per_class GR + n_per_class MG injections.

    Component masses are drawn (with replacement) from the catalog;
    distance is rescaled per-injection to hit the target SNR. This means
    individual injections have *different* effective distances but the
    *catalog* parameters are real.

    Returns
    -------
    X            : (2*n_per_class, n_samples) float32   whitened strain (classifier input)
    Xraw         : (2*n_per_class, n_samples) float32   un-whitened strain (matched filter input)
    y            : (2*n_per_class,) int64               labels {0,1}
    meta         : list of InjectionMeta                bookkeeping
    reservoir    : NoiseReservoir                       reused PSD/noise source
    """
    rng = np.random.default_rng(seed)
    catalog = load_bbh_catalog(catalog_path)
    if not catalog:
        raise RuntimeError(f"Empty catalog at {catalog_path}")

    reservoir = build_noise_reservoir(
        event_name=reservoir_event,
        pre_seconds=reservoir_seconds,
        sample_rate=sample_rate,
        seg_duration_s=seg_duration_s,
    )

    n_samples_per_seg = reservoir.n_samples_per_seg
    X = np.zeros((2 * n_per_class, n_samples_per_seg), dtype=np.float32)
    Xraw = np.zeros((2 * n_per_class, n_samples_per_seg), dtype=np.float32)
    y = np.zeros(2 * n_per_class, dtype=np.int64)
    meta: list[InjectionMeta] = []

    logger.info(
        f"Building {n_per_class}+{n_per_class} injections "
        f"(target SNR={target_snr:.1f}, λ_g={lambda_g_km:.2e} km)"
    )

    for class_label in (0, 1):
        lam = None if class_label == 0 else lambda_g_km
        for i in range(n_per_class):
            ev = catalog[int(rng.integers(0, len(catalog)))]
            try:
                x, x_raw, m = make_one_injection(
                    ev["m1"], ev["m2"], target_snr, lam,
                    reservoir, rng, f_lower=f_lower, f_final=f_final,
                    return_raw=True, noise_scale=noise_scale,
                )
            except Exception as e:
                logger.warning(f"Skipping injection ({ev['name']}): {e}")
                continue
            row = class_label * n_per_class + i
            n_min = min(len(x), n_samples_per_seg)
            X[row, :n_min] = x[:n_min]
            Xraw[row, :n_min] = x_raw[:n_min]
            y[row] = class_label
            m.event_drawn = ev["name"]
            meta.append(m)

    logger.success(f"Built dataset: X.shape={X.shape}, balanced labels.")
    return X, Xraw, y, meta, reservoir
