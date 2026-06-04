"""
Why single-event discrimination is noise-limited: a glitch illustration.
========================================================================

Panel (a): noise-free, whitened time series of a GR template and the
  corresponding massive-graviton template (lambda_g = 1e14 km, the deepest
  dispersion in our grid) for one GWTC-4.0-like source at optimal SNR = 20,
  together with their difference MG - GR on the *same* vertical scale. The
  small lower trace is the entire inter-class signal that any single-event
  discriminator -- matched filter or classifier -- has to detect.

Panel (b): a 4 s whitened H1 strain segment drawn from the same off-source
  reservoir near GW150914 -- the one with the largest crest factor, i.e. the
  loudest localised noise transient -- with the identical GR injection added.
  Both panels share a single detector-noise whitening scale, so the vertical
  axis is in common units of noise standard deviations. The noise transient
  (several sigma) and the pervasively non-Gaussian baseline dwarf the ~1.7 sigma
  inter-class difference of panel (a): this is the physical origin of the
  chance-level single-event AUC reported in the paper.

Outputs: results/fig_glitch_illustration.{pdf,png}

Usage:
  python -m scripts.make_glitch_figure
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from step1_gw_classifier.data import (
    build_noise_reservoir,
    make_one_injection,
    _time_domain_from_fd,
    _align_psd_to_waveform,
    load_bbh_catalog,
)
from step1_gw_classifier.waveform import gr_waveform_fd, massive_graviton_waveform_fd
from step1_gw_classifier.matched_filter import optimal_snr

LAMBDA_G_KM = 1e14
TARGET_SNR = 20.0
F_LOWER, F_FINAL = 20.0, 1024.0
DREF_MPC = 100.0
SEED = 7


def _whiten_shared(td: np.ndarray, psd_vals: np.ndarray, sr: float,
                   scale: float | None = None) -> tuple[np.ndarray, float]:
    """Whiten by 1/sqrt(S_n(f)); divide by a *shared* interior std `scale`
    (computed from the first trace) so two traces stay on one scale."""
    n = len(td)
    delta_f = sr / n
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    psd = _align_psd_to_waveform(psd_vals, len(freqs))
    psd_safe = np.where(psd > 0, psd, np.inf)
    spec_w = np.fft.rfft(td) / np.sqrt(psd_safe / (2.0 * delta_f))
    out = np.fft.irfft(spec_w, n=n)
    if scale is None:
        scale = out[n // 8:7 * n // 8].std() + 1e-30
    return out / scale, scale


def _signal_td(m1, m2, dist_mpc, reservoir, lam_km=None):
    sr, seg = reservoir.sample_rate, reservoir.seg_duration_s
    df = 1.0 / seg
    if lam_km is None:
        _, h = gr_waveform_fd(m1, m2, dist_mpc, delta_f=df,
                              f_lower=F_LOWER, f_final=F_FINAL)
    else:
        _, h = massive_graviton_waveform_fd(m1, m2, dist_mpc, lam_km * 1e3,
                                            delta_f=df, f_lower=F_LOWER, f_final=F_FINAL)
    return _time_domain_from_fd(h, df, sr, seg)


def _dist_for_snr(m1, m2, reservoir):
    sr, seg = reservoir.sample_rate, reservoir.seg_duration_s
    df = 1.0 / seg
    freqs, h_ref = gr_waveform_fd(m1, m2, DREF_MPC, delta_f=df,
                                  f_lower=F_LOWER, f_final=F_FINAL)
    psd = _align_psd_to_waveform(reservoir.psd_vals, len(freqs))
    snr_ref = optimal_snr(h_ref, psd, df, freqs=freqs, f_low=F_LOWER, f_high=F_FINAL)
    return DREF_MPC * snr_ref / TARGET_SNR


def _noise_sigma(reservoir, n_scan=60, seed=SEED + 1):
    """Detector-noise whitening scale sigma_0: the *median* interior std of
    whitened pure-noise segments. Whitening with this single sigma_0 puts
    every trace -- signals and strain alike -- into common units of
    'detector noise standard deviations', so panels (a) and (b) are directly
    comparable."""
    rng = np.random.default_rng(seed)
    stds = []
    for _ in range(n_scan):
        seg = reservoir.draw_segment(rng)
        w, _ = _whiten_shared(seg, reservoir.psd_vals, reservoir.sample_rate,
                              scale=1.0)
        stds.append(w[len(w) // 8:7 * len(w) // 8].std())
    return float(np.median(stds))


def _find_glitch_segment(reservoir, sigma0, n_scan=240, seed=SEED):
    """Return the off-source segment containing the cleanest loud transient:
    the one with the largest *crest factor* (peak / interior-RMS). Maximising
    peak-to-RMS (rather than the raw peak) selects a localised excursion on an
    otherwise near-Gaussian baseline -- the visual signature of a blip glitch --
    instead of a segment whose whole baseline is simply elevated. Amplitudes are
    reported on the shared detector-noise scale sigma0."""
    rng = np.random.default_rng(seed)
    best, best_amp, best_crest = None, -1.0, -1.0
    for _ in range(n_scan):
        seg = reservoir.draw_segment(rng)
        w, _ = _whiten_shared(seg, reservoir.psd_vals, reservoir.sample_rate,
                              scale=sigma0)
        interior = w[len(w) // 8:7 * len(w) // 8]
        peak = np.max(np.abs(interior))
        crest = peak / (interior.std() + 1e-30)
        if crest > best_crest:
            best_crest, best_amp, best = crest, peak, seg
    return best, best_amp


def main():
    print("Building noise reservoir (H1 near GW150914)...", flush=True)
    res = build_noise_reservoir(event_name="GW150914", pre_seconds=1024)
    sr, seg = res.sample_rate, res.seg_duration_s
    t = np.arange(int(round(sr * seg))) / sr

    # Single detector-noise whitening scale shared by BOTH panels, so the
    # vertical axis is "detector-noise sigma" everywhere.
    sigma0 = _noise_sigma(res)
    print(f"  detector-noise whitening scale sigma_0 = {sigma0:.3e}")

    # The frequency-domain waveforms have coalescence at t=0 (== t=T by
    # periodicity), so the merger lands on the segment boundary, exactly where
    # whitening produces edge ringing. Roll the signal to the segment centre so
    # the merger sits in the artifact-free interior, and restrict the view (and
    # all peak measurements) to that interior. This is purely a display choice;
    # the classifier/MF see the un-rolled segment with the same physics.
    n = int(round(sr * seg))
    lo, hi = n // 8, 7 * n // 8                       # artifact-free interior
    roll = n // 2                                     # merger -> segment centre

    # GW150914-like source for the templates
    m1, m2 = 36.0, 29.0
    dist = _dist_for_snr(m1, m2, res)
    gr_td = np.roll(_signal_td(m1, m2, dist, res, lam_km=None), roll)
    mg_td = np.roll(_signal_td(m1, m2, dist, res, lam_km=LAMBDA_G_KM), roll)
    gr_w, _ = _whiten_shared(gr_td, res.psd_vals, sr, scale=sigma0)
    mg_w, _ = _whiten_shared(mg_td, res.psd_vals, sr, scale=sigma0)
    diff = mg_w - gr_w
    print(f"  GR signal peak           = {np.max(np.abs(gr_w[lo:hi])):.1f} sigma")
    print(f"  inter-class peak |MG-GR| = {np.max(np.abs(diff[lo:hi])):.2f} sigma")

    # Glitchy strain segment + identical GR injection (same sigma0 scale)
    glitch_raw, gamp = _find_glitch_segment(res, sigma0)
    print(f"  glitch peak |whitened|   = {gamp:.1f} sigma")
    n_min = min(len(glitch_raw), len(gr_td))
    glitch_plus_inj = glitch_raw[:n_min] + gr_td[:n_min]
    gw_inj, _ = _whiten_shared(glitch_plus_inj, res.psd_vals, sr, scale=sigma0)
    i_glitch = lo + int(np.argmax(np.abs(gw_inj[lo:hi])))   # interior peak only

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.6, 4.4))
    ti = t[lo:hi]

    # ---- Panel (a) ----
    off = 1.25 * np.max(np.abs(np.concatenate([gr_w[lo:hi], mg_w[lo:hi]])))
    axa.plot(ti, gr_w[lo:hi], color="C0", lw=1.0, label="GR template")
    axa.plot(ti, mg_w[lo:hi], color="C3", lw=1.0, alpha=0.8,
             label=r"MG template ($\lambda_g=10^{14}\,$km)")
    axa.plot(ti, diff[lo:hi] - off, color="0.25", lw=1.0,
             label=r"difference (MG $-$ GR)")
    axa.axhline(-off, ls=":", color="0.6", lw=0.8)
    axa.set_xlim(ti[0], ti[-1])
    axa.set_xlabel("Time [s]")
    axa.set_ylabel(r"Whitened strain [$\sigma$]")
    axa.set_title("(a) Noise-free signal: the inter-class difference",
                  fontsize=10)
    axa.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

    # ---- Panel (b) ---- (same y-scale span context: glitch in detector sigma)
    axb.plot(ti, gw_inj[lo:hi], color="0.35", lw=0.8)
    axb.plot(t[i_glitch], gw_inj[i_glitch], "v", color="C1", markersize=9)
    axb.annotate("loudest noise\ntransient", xy=(t[i_glitch], gw_inj[i_glitch]),
                 xytext=(t[i_glitch] + 0.25, gw_inj[i_glitch] * 0.95),
                 fontsize=9, color="C1", ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color="C1", lw=1.1))
    axb.set_xlim(ti[0], ti[-1])
    axb.set_xlabel("Time [s]")
    axb.set_ylabel(r"Whitened strain [$\sigma$]")
    axb.set_title(r"(b) H1 strain + GR injection ($\rho_{\rm opt}=20$)",
                  fontsize=10)

    fig.tight_layout()
    out = Path("results")
    fig.savefig(out / "fig_glitch_illustration.pdf", dpi=300)
    fig.savefig(out / "fig_glitch_illustration.png", dpi=300)
    plt.close(fig)
    print("wrote fig_glitch_illustration.{pdf,png}")


if __name__ == "__main__":
    main()
