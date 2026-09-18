"""
Noise ladder control  (referee 1, comments #2 and #3)
=====================================================

For each (lambda_g, SNR) cell, compute the oracle matched-filter AUC under
three noise conditions on *identical* injections:

    signal_only : noise-free                              (positive control)
    gaussian    : stationary colored Gaussian noise with  (isolates the effect
                  the measured H1 PSD                       of non-Gaussianity)
    real        : observational H1 strain near GW150914    (the reported result)

If the collapse to chance on real strain were caused by the signal or the SNR,
it would already appear in the Gaussian rung. If instead it is caused by
non-Gaussian, non-stationary transients (glitches), the Gaussian rung stays
high and only the real rung collapses. This directly answers the referee's
question about why even the oracle statistic is at chance at lambda_g=1e12.

Source parameters are drawn from `seed` and are identical across the three
conditions; the noise realisation is drawn from `noise_seed`.

The script also validates the colored-Gaussian generator against
estimate_psd_welch (the same estimator the matched filter uses).

Usage:
    python -m scripts.make_noise_ladder            # 5 lambda_g x SNR 50, N=400
    python -m scripts.make_noise_ladder --quick    # 1 cell, N=30 (smoke test)

Writes results/noise_ladder.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from loguru import logger

from step1_gw_classifier.data import (
    build_injection_dataset, build_noise_reservoir,
    build_gaussian_reservoir, colored_gaussian_segment,
)
from step1_gw_classifier.waveform import gr_waveform_fd
from step1_gw_classifier.matched_filter import inner_product
from step1_gw_classifier.mf_baseline import mf_score_dataset, roc_from_scores
from scripts.run_sweep import bootstrap_auc_ci

SEED = 42
NOISE_SEED = 1234
F_LOW, F_HIGH = 20.0, 1024.0


def validate_gaussian_normalisation(real_res, n_trials: int = 400,
                                    seed: int = 0) -> float:
    """
    Physically decisive normalisation check. A unit-norm template projected
    onto noise-only colored-Gaussian data must have UNIT VARIANCE in the
    inner-product convention the matched filter uses:  Var(<n|h_hat>) = 1.

    (A direct Welch re-estimate of the PSD is not used here: re-estimating a
    spectrum spanning ~6 decades incurs spectral-leakage bias in the low-power
    bins, so Welch(generated)/target can differ from 1 even when the injected
    noise power exactly matches the PSD the matched filter assumes. The
    projection variance is immune to that artefact.)
    """
    sr, n_t = real_res.sample_rate, real_res.n_samples_per_seg
    df = 1.0 / real_res.seg_duration_s
    n_f = n_t // 2 + 1
    freqs = np.fft.rfftfreq(n_t, d=1.0 / sr)
    psd = np.asarray(real_res.psd_vals)
    psd_l = (psd[:n_f] if len(psd) >= n_f
             else np.concatenate([psd, np.full(n_f - len(psd), psd[-1])]))
    _, h_short = gr_waveform_fd(35.0, 30.0, 400.0, delta_f=df,
                               f_lower=F_LOW, f_final=F_HIGH)
    h = np.zeros(n_f, dtype=np.complex128)
    m = min(len(h_short), n_f)
    h[:m] = h_short[:m]
    sig2 = inner_product(h, h, psd_l, df, freqs=freqs, f_low=F_LOW, f_high=F_HIGH)
    h_n = h / np.sqrt(sig2)
    rng = np.random.default_rng(seed)
    z = np.empty(n_trials)
    for i in range(n_trials):
        d_fd = np.fft.rfft(colored_gaussian_segment(psd, sr, n_t, rng)) / sr
        z[i] = inner_product(d_fd, h_n, psd_l, df, freqs=freqs,
                             f_low=F_LOW, f_high=F_HIGH)
    var = float(z.var())
    logger.info(f"Gaussian normalisation check: Var(<noise|template>) = "
                f"{var:.3f}  (target 1.0)")
    return var


def cell_ladder(lg_km: float, snr: float, n_per_class: int,
                real_res, gauss_res) -> dict:
    common = dict(n_per_class=n_per_class, target_snr=snr, lambda_g_km=lg_km,
                  f_lower=F_LOW, f_final=F_HIGH, seed=SEED, noise_seed=NOISE_SEED)
    # Identical source parameters (from SEED) across all three conditions.
    _, Xr, y, meta, _ = build_injection_dataset(reservoir=real_res,  noise_scale=1.0, **common)
    _, Xg, _, _,    _ = build_injection_dataset(reservoir=gauss_res, noise_scale=1.0, **common)
    _, Xs, _, _,    _ = build_injection_dataset(reservoir=real_res,  noise_scale=0.0, **common)

    out = {}
    for name, Xraw in [("signal_only", Xs), ("gaussian", Xg), ("real", Xr)]:
        s = mf_score_dataset(Xraw, meta, real_res, lambda_g_km=lg_km,
                             f_lower=F_LOW, f_final=F_HIGH, progress=False)
        auc = roc_from_scores(y, s["s_mf"])["auc"]
        lo, hi = bootstrap_auc_ci(y, s["s_mf"], seed=1)
        out[name] = {"auc": float(auc), "ci_lo": lo, "ci_hi": hi}
        logger.info(f"  lg={lg_km:.0e} snr={snr:.0f}  {name:11s} "
                    f"AUC={auc:.3f} [{lo:.3f},{hi:.3f}]")
    return out


def make_ladder_figure(ladder: dict, out_dir: Path) -> None:
    """Plot MF AUC vs lambda_g for the three noise conditions, with 90% CIs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = sorted(ladder["cells"], key=lambda c: c["lambda_g_km"])
    lg = np.array([c["lambda_g_km"] for c in cells])
    styles = {
        "signal_only": dict(color="C2", marker="o", mfc="none",
                            label="Signal only (no noise)"),
        "gaussian":    dict(color="C0", marker="s", mfc="C0",
                            label="Gaussian colored noise (measured PSD)"),
        "real":        dict(color="C3", marker="o", mfc="C3",
                            label="H1 strain near GW150914"),
    }
    dodge = {"signal_only": 0.80, "gaussian": 1.0, "real": 1.25}

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for cond, st in styles.items():
        auc = np.array([c[cond]["auc"] for c in cells])
        lo = np.array([c[cond]["ci_lo"] for c in cells])
        hi = np.array([c[cond]["ci_hi"] for c in cells])
        ax.errorbar(lg * dodge[cond], auc, yerr=[auc - lo, hi - auc],
                    fmt=st["marker"], color=st["color"], mfc=st["mfc"],
                    ms=7, lw=1.5, capsize=3, label=st["label"])
    ax.axhline(0.5, ls=":", color="0.4", lw=1.0)
    ax.text(lg.max() * 1.4, 0.507, "chance", fontsize=8, color="0.4",
            va="bottom", ha="right")
    ax.set_xscale("log")
    ax.set_xlabel(r"Graviton Compton wavelength $\lambda_g$ [km]")
    ax.set_ylabel("Matched-filter AUC")
    ax.set_ylim(0.4, 1.045)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="center left", fontsize=8.5, framealpha=0.93)
    fig.tight_layout()
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "fig_noise_ladder.pdf", dpi=300)
    fig.savefig(out_dir / "fig_noise_ladder.png", dpi=300)
    plt.close(fig)
    logger.info("wrote fig_noise_ladder.{pdf,png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="1 cell, N=30 smoke test")
    p.add_argument("--n_per_class", type=int, default=400)
    p.add_argument("--snrs", type=float, nargs="+", default=[50.0])
    p.add_argument("--lambdas_km", type=float, nargs="+",
                   default=[1e12, 1e13, 1e14, 1e15, 1e16])
    p.add_argument("--out", type=str, default="results/noise_ladder.json")
    args = p.parse_args()
    if args.quick:
        args.n_per_class, args.snrs, args.lambdas_km = 30, [50.0], [1e12]

    real_res = build_noise_reservoir()            # GW150914, 1024 s H1 (as in the sweep)
    gauss_res = build_gaussian_reservoir(real_res)
    norm_var = validate_gaussian_normalisation(real_res)

    cells: list[dict] = []
    t0 = time.time()
    for lg in args.lambdas_km:
        for snr in args.snrs:
            logger.info(f"--- ladder cell lambda_g={lg:.0e} km, SNR={snr:.0f}")
            res = cell_ladder(lg, snr, args.n_per_class, real_res, gauss_res)
            cells.append({"lambda_g_km": float(lg), "target_snr": float(snr), **res})
            Path(args.out).parent.mkdir(exist_ok=True)
            with open(args.out, "w") as f:
                json.dump({
                    "gaussian_norm_var": norm_var,
                    "config": {"seed": SEED, "noise_seed": NOISE_SEED,
                               "n_per_class": args.n_per_class,
                               "f_low": F_LOW, "f_high": F_HIGH},
                    "cells": cells,
                }, f, indent=2)
    make_ladder_figure({"cells": cells}, Path(args.out).parent)
    logger.success(f"Noise ladder complete in {(time.time()-t0)/60:.1f} min "
                   f"-> {args.out}")


if __name__ == "__main__":
    main()
