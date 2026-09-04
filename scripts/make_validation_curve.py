"""
Regenerate the dispersion-validation mismatch curve (Fig. 1) on the SAME
1024 s off-source H1 PSD used for the injection study.
=====================================================================

The original validation curve was computed on an ad-hoc off-source PSD whose
exact length was not recorded. To keep a single, reproducible noise model
throughout the paper, we recompute it here on the identical reservoir PSD that
build_noise_reservoir(pre_seconds=1024) provides -- the same PSD the matched
filter and classifier see -- and report the template's optimal SNR so the
detectability threshold 1/(2 rho^2) in the figure is self-consistent.

Method: normalised match M(h_GR, h_MG) maximised over time and phase, for a
GW150914-like template (m1=36, m2=29 Msun, D_L=410 Mpc), scanned over a
logarithmic grid in lambda_g. The mismatch 1 - M crosses the per-event
detectability threshold near lambda_g ~ 1e16 km, recovering the LVK O3 bound to
within the factor expected from single-event vs catalogue-stacked SNR.

Outputs: results/validation_mg_mismatch.csv  (lambda_g_km, 1_minus_match)

Usage:
  python -m scripts.make_validation_curve
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from step1_gw_classifier.data import build_noise_reservoir, _align_psd_to_waveform
from step1_gw_classifier.waveform import gr_waveform_fd, massive_graviton_waveform_fd
from step1_gw_classifier.matched_filter import optimal_snr, match

M1_MSUN, M2_MSUN = 36.0, 29.0
DL_MPC = 410.0
F_LOWER, F_FINAL = 20.0, 1024.0
LAMBDA_G_KM = np.logspace(12, 18, 49)
PRE_SECONDS = 1024


def main():
    print(f"Building noise reservoir (H1, {PRE_SECONDS} s off-source)...",
          flush=True)
    res = build_noise_reservoir(event_name="GW150914", pre_seconds=PRE_SECONDS)
    df = 1.0 / res.seg_duration_s

    freqs, h_gr = gr_waveform_fd(M1_MSUN, M2_MSUN, DL_MPC, delta_f=df,
                                 f_lower=F_LOWER, f_final=F_FINAL)
    psd = _align_psd_to_waveform(res.psd_vals, len(freqs))
    mask = dict(freqs=freqs, f_low=F_LOWER, f_high=F_FINAL)

    snr = optimal_snr(h_gr, psd, df, **mask)
    thresh = 1.0 / (2.0 * snr ** 2)
    print(f"  template optimal SNR on {PRE_SECONDS} s PSD = {snr:.2f}")
    print(f"  detectability threshold 1/(2 SNR^2)        = {thresh:.3e}")

    rows = []
    crossing = None
    prev = None
    for lam_km in LAMBDA_G_KM:
        _, h_mg = massive_graviton_waveform_fd(
            M1_MSUN, M2_MSUN, DL_MPC, lam_km * 1e3,
            delta_f=df, f_lower=F_LOWER, f_final=F_FINAL)
        mm = 1.0 - match(h_gr, h_mg, psd, df, **mask)
        rows.append((lam_km, mm))
        if prev is not None:
            (lp, mp) = prev
            if (mp - thresh) * (mm - thresh) <= 0 and mm != mp:
                # log-linear interpolation of the crossing in lambda_g
                f = (np.log(thresh) - np.log(mp)) / (np.log(mm) - np.log(mp))
                crossing = np.exp(np.log(lp) + f * (np.log(lam_km) - np.log(lp)))
        prev = (lam_km, mm)

    out = Path("results/validation_mg_mismatch.csv")
    with out.open("w") as fh:
        fh.write("lambda_g_km, 1_minus_match\n")
        for lam_km, mm in rows:
            fh.write(f"{lam_km:.18e},{mm:.18e}\n")
    print(f"  wrote {out} ({len(rows)} points)")
    if crossing is not None:
        print(f"  mismatch crosses threshold at lambda_g ~ {crossing:.3e} km")


if __name__ == "__main__":
    main()
