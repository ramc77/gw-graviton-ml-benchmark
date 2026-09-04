"""
Generate the three manuscript figures from the sweep output and the
validation CSV.

Outputs:
    results/fig_classifier_vs_mf.{pdf,png}   (Fig. 2: AUC vs lambda_g)
    results/fig_roc_best_cell.{pdf,png}      (Fig. 3: ROC at most discriminative cell)
    results/validation_mg_mismatch.{pdf,png} (Fig. 1: re-plotted from CSV)
    results/sweep_auc.csv                     (consolidated AUC table)

The AUC figure plots, for both target optimal SNRs:
  * filled circles  -- CNN+Transformer on H1 strain,
  * filled squares  -- oracle matched filter on H1 strain,
  * open squares    -- oracle matched filter on signal-only data (control),
with bootstrap 90% confidence intervals. The signal-only control sits near
AUC = 1, demonstrating that the discriminator is bug-free; the H1-strain
points sit near the chance line, which is the physics result.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from step1_gw_classifier.mf_baseline import roc_from_scores

SNR_VALIDATION = 35.1   # optimal SNR of the GW150914-like validation template
LVK_O3_BOUND_KM = 9.8e13   # GWTC-3: m_g <= 1.27e-23 eV/c^2  (lambda_g > 9.8e13 km)


def load_sweep(path: str | Path = "results/sweep_results.json") -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 2 -- AUC vs lambda_g
# ---------------------------------------------------------------------------

def make_auc_figure(sweep: dict, out_dir: Path):
    cells = sweep["cells"]
    snrs = sorted({c["target_snr"] for c in cells})
    colours = {snrs[0]: "C0", snrs[-1]: "C3"}
    # log-space horizontal dodge so the two SNR series do not overlap
    dodge = {snrs[0]: 0.82, snrs[-1]: 1.22}
    have_control = all("mf_signal_only" in c for c in cells)

    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    def _series(row, key, sub=None):
        if sub is None:
            auc = np.array([c[key]["auc"] for c in row])
            lo = np.array([c[key]["auc_ci_lo"] for c in row])
            hi = np.array([c[key]["auc_ci_hi"] for c in row])
        else:
            auc = np.array([c[key][sub] for c in row])
            lo = np.array([c[key][sub + "_ci_lo"] for c in row])
            hi = np.array([c[key][sub + "_ci_hi"] for c in row])
        return auc, np.vstack([auc - lo, hi - auc])

    for snr in snrs:
        row = sorted([c for c in cells if c["target_snr"] == snr],
                     key=lambda c: c["lambda_g_km"])
        lg = np.array([c["lambda_g_km"] for c in row]) * dodge[snr]
        col = colours[snr]

        # CNN+Transformer on H1 strain (filled circles)
        cnn, cnn_err = _series(row, "classifier", "val_auc")
        ax.errorbar(lg * 0.97, cnn, yerr=cnn_err, fmt="o", color=col,
                    capsize=3, markersize=7, lw=1.4, zorder=4)
        # Oracle matched filter on H1 strain (filled squares)
        mf, mf_err = _series(row, "mf")
        ax.errorbar(lg * 1.03, mf, yerr=mf_err, fmt="s", color=col,
                    capsize=3, markersize=7, lw=1.4, zorder=4)
        # Oracle matched filter on signal-only data (open squares)
        if have_control:
            ctrl, ctrl_err = _series(row, "mf_signal_only")
            ax.errorbar(lg, ctrl, yerr=ctrl_err, fmt="s", color=col,
                        mfc="white", mec=col, markeredgewidth=1.6,
                        capsize=3, markersize=8, lw=1.4, zorder=3)

    # Chance line and the noise-free / glitch-limited annotations
    ax.axhline(0.5, ls=":", color="0.4", lw=1.0)
    ax.text(1.05e16, 0.505, "chance", fontsize=8, color="0.4", va="bottom", ha="right")
    if have_control:
        ax.text(1.1e13, 0.985, "signal-only control (no noise)",
                fontsize=8.5, color="0.25", va="top", style="italic")
        ax.text(1.1e13, 0.555, "H1 strain near GW150914",
                fontsize=8.5, color="0.25", va="bottom", style="italic")

    # LVK O3 exclusion band
    ax.axvspan(LVK_O3_BOUND_KM, 1.0e18, alpha=0.08, color="C2")
    ax.text(1.15e14, 0.62, "LVK catalogue 90% CL", fontsize=8, color="C2",
            rotation=90, va="bottom")

    ax.set_xscale("log")
    ax.set_xlim(8e12, 1.0e18)
    ax.set_ylim(0.4, 1.05)
    ax.set_xlabel(r"Graviton Compton wavelength $\lambda_g$ [km]")
    ax.set_ylabel("Validation AUC")
    ax.grid(alpha=0.3)

    # Two-axis legend: marker shape = method, colour = SNR.
    shape_handles = [
        Line2D([], [], marker="o", color="0.35", ls="none", markersize=7,
               label="CNN+Transformer (H1 strain)"),
        Line2D([], [], marker="s", color="0.35", ls="none", markersize=7,
               label="Matched filter (H1 strain)"),
        Line2D([], [], marker="s", color="0.35", ls="none", markersize=8,
               mfc="white", mec="0.35", markeredgewidth=1.6,
               label="Matched filter (signal-only)"),
    ]
    snr_handles = [
        Line2D([], [], marker="o", color=colours[snrs[0]], ls="none",
               markersize=7, label=fr"$\rho_{{\rm opt}}={snrs[0]:.0f}$"),
        Line2D([], [], marker="o", color=colours[snrs[-1]], ls="none",
               markersize=7, label=fr"$\rho_{{\rm opt}}={snrs[-1]:.0f}$"),
    ]
    leg1 = ax.legend(handles=shape_handles, loc="upper left",
                     bbox_to_anchor=(0.012, 0.80), fontsize=8.5,
                     framealpha=0.92, title="Method")
    leg1.get_title().set_fontsize(8.5)
    ax.add_artist(leg1)
    ax.legend(handles=snr_handles, loc="upper left",
              bbox_to_anchor=(0.012, 0.50), fontsize=8.5,
              framealpha=0.92, title="Target SNR").get_title().set_fontsize(8.5)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_classifier_vs_mf.pdf", dpi=300)
    fig.savefig(out_dir / "fig_classifier_vs_mf.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 -- ROC at the most discriminative cell, with bootstrap bands
# ---------------------------------------------------------------------------

def _bootstrap_roc_band(y: np.ndarray, score: np.ndarray, grid: np.ndarray,
                        n_boot: int = 500, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (median, lo, hi) TPR on a common FPR grid via the percentile bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(y)
    curves = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ys, ss = y[idx], score[idx]
        if len(set(ys.tolist())) < 2:
            continue
        roc = roc_from_scores(ys, ss)
        order = np.argsort(roc["fpr"])
        fpr_s, tpr_s = roc["fpr"][order], roc["tpr"][order]
        curves.append(np.interp(grid, fpr_s, tpr_s))
    curves = np.vstack(curves)
    lo, med, hi = np.percentile(curves, [5, 50, 95], axis=0)
    return med, lo, hi


def make_roc_figure(sweep: dict, out_dir: Path):
    cells = sweep["cells"]
    snr_hi = max(c["target_snr"] for c in cells)
    candidates = sorted([c for c in cells if c["target_snr"] == snr_hi],
                        key=lambda c: c["lambda_g_km"])
    cell = candidates[0]

    y = np.asarray(cell["mf"]["labels"])
    mf_s = np.asarray(cell["mf"]["scores"])
    cls_s = np.asarray(cell["classifier"]["val_scores"])
    cls_y = np.asarray(cell["classifier"]["val_labels"])

    mf_roc = roc_from_scores(y, mf_s)
    cls_roc = roc_from_scores(cls_y, cls_s)

    grid = np.linspace(0, 1, 201)
    mf_med, mf_lo, mf_hi = _bootstrap_roc_band(y, mf_s, grid, seed=1)
    cls_med, cls_lo, cls_hi = _bootstrap_roc_band(cls_y, cls_s, grid, seed=2)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.fill_between(grid, mf_lo, mf_hi, color="C3", alpha=0.18, lw=0)
    ax.fill_between(grid, cls_lo, cls_hi, color="C0", alpha=0.18, lw=0)
    ax.plot(grid, mf_med, lw=2, color="C3",
            label=f"Matched filter, AUC$={mf_roc['auc']:.2f}$")
    ax.plot(grid, cls_med, lw=2, color="C0",
            label=f"CNN+Transformer, AUC$={cls_roc['auc']:.2f}$")
    ax.plot([0, 1], [0, 1], "k:", lw=0.9, label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title(fr"ROC at $\lambda_g=10^{{{int(round(np.log10(cell['lambda_g_km'])))}}}\,$km, "
                 fr"$\rho_{{\rm opt}}={cell['target_snr']:.0f}$", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_roc_best_cell.pdf", dpi=300)
    fig.savefig(out_dir / "fig_roc_best_cell.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 -- validation mismatch curve, re-plotted from the stored CSV
# ---------------------------------------------------------------------------

def make_validation_figure(out_dir: Path):
    csv_path = out_dir / "validation_mg_mismatch.csv"
    if not csv_path.exists():
        print(f"  (skipping validation figure: {csv_path} not found)")
        return
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    lg, mismatch = data[:, 0], data[:, 1]
    thresh = 1.0 / (2.0 * SNR_VALIDATION ** 2)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.loglog(lg, mismatch, "o-", color="C0", markersize=4, lw=1.3,
              label=fr"GW150914-like ($\rho_{{\rm opt}}={SNR_VALIDATION:.1f}$)")
    ax.axhline(thresh, ls="--", color="0.4", lw=1.1,
               label=fr"detectability $1/(2\rho_{{\rm opt}}^2)={thresh:.1e}$")
    ax.axvline(LVK_O3_BOUND_KM, ls=":", color="C3", lw=1.3,
               label=r"LVK catalogue bound $\lambda_g>9.8\times10^{13}$ km")
    ax.set_xlabel(r"Graviton Compton wavelength $\lambda_g$ [km]")
    ax.set_ylabel(r"Mismatch $1-\mathcal{M}(\tilde{h}_{\rm GR},\tilde{h}_{\rm MG})$")
    ax.set_title(r"Validation: Mirshekari--Yunes dispersion on measured H1 PSD",
                 fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "validation_mg_mismatch.pdf", dpi=300)
    fig.savefig(out_dir / "validation_mg_mismatch.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Consolidated CSV
# ---------------------------------------------------------------------------

def make_csv(sweep: dict, out_dir: Path):
    cells = sweep["cells"]
    rows = []
    for c in cells:
        ctrl = c.get("mf_signal_only", {})
        rows.append([
            c["lambda_g_km"], c["target_snr"],
            c["classifier"]["val_auc"],
            c["classifier"]["val_auc_ci_lo"], c["classifier"]["val_auc_ci_hi"],
            c["mf"]["auc"], c["mf"]["auc_ci_lo"], c["mf"]["auc_ci_hi"],
            ctrl.get("auc", np.nan),
            ctrl.get("auc_ci_lo", np.nan), ctrl.get("auc_ci_hi", np.nan),
        ])
    arr = np.array(rows)
    np.savetxt(out_dir / "sweep_auc.csv", arr, delimiter=",",
               header="lambda_g_km, target_snr, cnn_auc, cnn_ci_lo, cnn_ci_hi, "
                      "mf_auc, mf_ci_lo, mf_ci_hi, mf_signalonly_auc, "
                      "mf_signalonly_ci_lo, mf_signalonly_ci_hi",
               comments="")


if __name__ == "__main__":
    sweep = load_sweep()
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    make_auc_figure(sweep, out_dir)
    make_roc_figure(sweep, out_dir)
    make_validation_figure(out_dir)
    make_csv(sweep, out_dir)
    print("Wrote fig_classifier_vs_mf, fig_roc_best_cell, "
          "validation_mg_mismatch, sweep_auc.csv")
