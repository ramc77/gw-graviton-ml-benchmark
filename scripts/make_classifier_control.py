"""
Classifier positive control + training-set-size scaling study.
==============================================================

Referee-facing control that the convolutional--transformer classifier *can*
learn the GR-vs-massive-graviton distinction when detector noise does not
swamp it, and that the null result on H1 strain is therefore set by the
noise, not by network capacity or training-set size.

Panel (a): validation AUC vs training epoch for the classifier trained on
  (i)  noise-free (signal-only) injections  -> approaches AUC = 1,
  (ii) H1-strain injections                 -> stays at chance,
  both at lambda_g = 1e13 km, target SNR = 20, N = 400 per class. This cell is
  in the resolvable-but-buried regime: noise-free the inter-class difference is
  order the signal amplitude (AUC -> 1), yet on strain it sits well below the
  ~8 sigma noise transients (AUC -> chance). This is the classifier analogue of
  the matched-filter signal-only control (Fig. 2).

Panel (b): best validation AUC vs training-set size N (per class) for the same
  two regimes. On strain, more data does not lift the AUC off chance; noise
  free, the network separates the classes at every N. This converts the
  "sensitivity saturates" limitation into evidence.

The noise-free regime is produced with the backward-compatible noise_scale=0
path added to data.build_injection_dataset / train.train_classifier.

Outputs:
  results/classifier_control.json
  results/fig_classifier_control.{pdf,png}

Usage:
  python -m scripts.make_classifier_control            # compute + plot
  python -m scripts.make_classifier_control --plot-only # re-plot stored JSON
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from step1_gw_classifier.train import train_classifier

LAMBDA_G_KM = 1e13          # resolvable-but-buried cell (corrected phase)
TARGET_SNR = 20.0
N_EPOCHS = 12
SEED = 42
N_GRID = [100, 200, 400]
OUT = Path("results/classifier_control.json")
SWEEP = Path("results/sweep_results.json")


def _run(noise_scale: float, n_per_class: int) -> dict:
    """Train one classifier and return its validation-AUC summary."""
    out = train_classifier(
        n_per_class=n_per_class, target_snr=TARGET_SNR,
        lambda_g_km=LAMBDA_G_KM, n_epochs=N_EPOCHS,
        batch_size=32, learning_rate=3e-4, train_frac=0.85,
        reservoir_seconds=1024, seed=SEED, compact=True,
        noise_scale=noise_scale,
    )
    hist = out["history"]["val_auc"]
    return {
        "best_val_auc": float(out["best_val_auc"]),
        "max_val_auc": float(np.nanmax(hist)),
        "history_val_auc": [float(x) for x in hist],
        "history_val_acc": [float(x) for x in out["history"]["val_acc"]],
    }


def _load() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text())
    return {"config": {"lambda_g_km": LAMBDA_G_KM, "target_snr": TARGET_SNR,
                       "n_epochs": N_EPOCHS, "seed": SEED, "n_grid": N_GRID},
            "noise_free": {}, "strain": {}}


def _save(res: dict) -> None:
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))


def _seed_strain400_from_sweep(res: dict) -> None:
    """Reuse the identical (seed, config) N=400 strain run already in the sweep."""
    if "400" in res["strain"] or not SWEEP.exists():
        return
    sweep = json.loads(SWEEP.read_text())
    for c in sweep["cells"]:
        same = (abs(c["lambda_g_km"] - LAMBDA_G_KM) < 1
                and abs(c["target_snr"] - TARGET_SNR) < 1
                and c["n_per_class"] == 400 and c["n_epochs"] == N_EPOCHS)
        if same:
            h = c["classifier"]["history"]
            res["strain"]["400"] = {
                "best_val_auc": float(c["classifier"]["val_auc"]),
                "max_val_auc": float(np.nanmax(h["val_auc"])),
                "history_val_auc": [float(x) for x in h["val_auc"]],
                "history_val_acc": [float(x) for x in h["val_acc"]],
                "from_sweep": True,
            }
            print("  seeded strain N=400 from sweep_results.json")
            return


def compute() -> dict:
    res = _load()
    _seed_strain400_from_sweep(res)
    _save(res)
    t0 = time.time()
    for n in N_GRID:
        key = str(n)
        if key not in res["noise_free"]:
            print(f"[noise-free] N={n} ...", flush=True)
            res["noise_free"][key] = _run(0.0, n)
            _save(res)
        if key not in res["strain"]:
            print(f"[strain]     N={n} ...", flush=True)
            res["strain"][key] = _run(1.0, n)
            _save(res)
    print(f"compute done in {(time.time() - t0) / 60:.1f} min")
    return res


def plot(res: dict, out_dir: Path) -> None:
    nf, st = res["noise_free"], res["strain"]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.4, 4.4))

    # ---- Panel (a): learning curves at N = 400 ----
    nf400 = nf["400"]["history_val_auc"]
    st400 = st["400"]["history_val_auc"]
    axa.plot(np.arange(1, len(nf400) + 1), nf400, "o-", color="C2", lw=1.8,
             markersize=5, label="Noise-free (signal only)")
    axa.plot(np.arange(1, len(st400) + 1), st400, "s-", color="C3", lw=1.8,
             markersize=5, label="H1 strain near GW150914")
    axa.axhline(0.5, ls=":", color="0.4", lw=1.0)
    axa.text(len(st400), 0.505, "chance", fontsize=8, color="0.4",
             va="bottom", ha="right")
    axa.set_xlabel("Training epoch")
    axa.set_ylabel("Validation AUC")
    axa.set_ylim(0.4, 1.03)
    axa.grid(alpha=0.3)
    axa.set_title(r"(a) Learning curves "
                  r"($\lambda_g=10^{13}\,$km, $\rho_{\rm opt}=20$, $N=400$)",
                  fontsize=10)
    axa.legend(loc="center right", fontsize=9, framealpha=0.92)

    # ---- Panel (b): best AUC vs training-set size ----
    ns = sorted(int(k) for k in nf.keys())
    nf_best = [nf[str(n)]["best_val_auc"] for n in ns]
    st_best = [st[str(n)]["best_val_auc"] for n in ns]
    axb.plot(ns, nf_best, "o-", color="C2", lw=1.8, markersize=6,
             label="Noise-free (signal only)")
    axb.plot(ns, st_best, "s-", color="C3", lw=1.8, markersize=6,
             label="H1 strain near GW150914")
    axb.axhline(0.5, ls=":", color="0.4", lw=1.0)
    axb.text(ns[-1], 0.505, "chance", fontsize=8, color="0.4",
             va="bottom", ha="right")
    axb.set_xscale("log")
    axb.set_xticks(ns)
    axb.set_xticklabels([str(n) for n in ns])
    import matplotlib.ticker as mticker
    axb.xaxis.set_minor_formatter(mticker.NullFormatter())  # no spurious 3x10^2 label
    axb.set_xlabel(r"Training-set size $N$ per class")
    axb.set_ylabel("Best validation AUC")
    axb.set_ylim(0.4, 1.03)
    axb.grid(alpha=0.3, which="both")
    axb.set_title("(b) Sensitivity vs training-set size", fontsize=10)
    axb.legend(loc="center right", fontsize=9, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_classifier_control.pdf", dpi=300)
    fig.savefig(out_dir / "fig_classifier_control.png", dpi=300)
    plt.close(fig)
    print("wrote fig_classifier_control.{pdf,png}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()
    res = _load() if args.plot_only else compute()
    plot(res, Path("results"))
