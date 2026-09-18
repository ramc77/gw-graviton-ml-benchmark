"""
Noise-reservoir robustness  (referee 1, comment #3)
===================================================

The main study uses one ~17-min H1 segment near GW150914 (O1) as the noise
reservoir. A referee asked how sensitive the reported AUC is to that choice.
This script repeats the oracle matched-filter benchmark on independent noise
reservoirs spanning several observing runs and both LIGO interferometers, at
fixed injections, and reports the AUC spread.

If the null (chance-level real-strain AUC) is a property of single-event
detector noise rather than of one atypical, glitch-rich segment, the AUC should
remain near 0.5 across all reservoirs.

The reservoirs use noise only; source parameters are still drawn from the
GWTC-4.0 catalog. Each reservoir uses its OWN measured PSD (oracle matched
filter matched to that reservoir).

Usage:
    python -m scripts.make_noise_robustness           # ~5 reservoirs x 2 cells
    python -m scripts.make_noise_robustness --quick    # 2 reservoirs, N=30

Writes results/noise_robustness.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from loguru import logger

from step1_gw_classifier.data import (
    build_injection_dataset, build_noise_reservoir, split_reservoir_into_windows,
)
from step1_gw_classifier.mf_baseline import mf_score_dataset, roc_from_scores
from scripts.run_sweep import bootstrap_auc_ci

SEED = 42
NOISE_SEED = 1234
F_LOW, F_HIGH = 20.0, 1024.0

# (label, event_name for the noise epoch, detector) — noise only. H1 across
# three observing runs (L1 open-data windows near these events are gappy and
# fetch unreliably, so we span runs with the reliably-served H1 strain).
# Extra events give redundancy: any reservoir whose GWOSC fetch fails is skipped.
RESERVOIRS = [
    ("GW150914 H1 (O1)",  "GW150914",        "H1"),
    ("GW151226 H1 (O1)",  "GW151226",        "H1"),
    ("GW170814 H1 (O2)",  "GW170814",        "H1"),
    ("GW170823 H1 (O2)",  "GW170823",        "H1"),
    ("GW190521 H1 (O3a)", "GW190521",        "H1"),
    ("GW190814 H1 (O3a)", "GW190814",        "H1"),
    ("GW200129 H1 (O3b)", "GW200129_065458", "H1"),
]
CELLS = [(1e12, 50.0), (1e14, 50.0)]   # strong (glitch-limited) + physically-relevant


def reservoir_auc(res, lg_km: float, snr: float, n_per_class: int) -> dict:
    _, Xraw, y, meta, _ = build_injection_dataset(
        n_per_class=n_per_class, target_snr=snr, lambda_g_km=lg_km,
        f_lower=F_LOW, f_final=F_HIGH, seed=SEED, noise_seed=NOISE_SEED,
        reservoir=res, noise_scale=1.0,
    )
    s = mf_score_dataset(Xraw, meta, res, lambda_g_km=lg_km,
                         f_lower=F_LOW, f_final=F_HIGH, progress=False)
    auc = roc_from_scores(y, s["s_mf"])["auc"]
    lo, hi = bootstrap_auc_ci(y, s["s_mf"], seed=1)
    return {"auc": float(auc), "ci_lo": lo, "ci_hi": hi}


def make_robustness_figure(robust: dict, out_dir: Path) -> None:
    """Real-strain MF AUC across noise reservoirs, with 90% CIs, per cell."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reservoirs = robust["reservoirs"]
    if not reservoirs:
        return
    labels = [r["label"] for r in reservoirs]
    x = np.arange(len(labels))
    keys = [f"{lg:.0e}" for lg, _ in robust["config"]["cells"]]
    style = {
        "1e+12": dict(color="C3", label=r"$\lambda_g=10^{12}$ km (strong dispersion)"),
        "1e+14": dict(color="C0", label=r"$\lambda_g=10^{14}$ km (near LVK bound)"),
    }
    dodge = {k: (i - (len(keys) - 1) / 2) * 0.16 for i, k in enumerate(keys)}

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for k in keys:
        auc = np.array([r["cells"][k]["auc"] for r in reservoirs])
        lo = np.array([r["cells"][k]["ci_lo"] for r in reservoirs])
        hi = np.array([r["cells"][k]["ci_hi"] for r in reservoirs])
        st = style.get(k, dict(color="C2", label=k))
        ax.errorbar(x + dodge[k], auc, yerr=[auc - lo, hi - auc], fmt="o",
                    color=st["color"], ms=6, lw=1.4, capsize=3, label=st["label"])
    ax.axhline(0.5, ls=":", color="0.4", lw=1.0, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Real-strain matched-filter AUC")
    ax.set_ylim(0.4, 0.62)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.93)
    fig.tight_layout()
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "fig_noise_robustness.pdf", dpi=300)
    fig.savefig(out_dir / "fig_noise_robustness.png", dpi=300)
    plt.close(fig)
    logger.info("wrote fig_noise_robustness.{pdf,png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n_per_class", type=int, default=400)
    p.add_argument("--windows", type=int, default=0,
                   help="Split the cached GW150914 H1 reservoir into this many "
                        "independent time-window sub-reservoirs instead of "
                        "fetching multiple events. Reliable when fresh GWOSC "
                        "fetches of other events/detectors are unavailable.")
    p.add_argument("--out", type=str, default="results/noise_robustness.json")
    args = p.parse_args()
    cells = [CELLS[0]] if args.quick else CELLS
    n_per = 30 if args.quick else args.n_per_class

    # Build the list of (label, reservoir) to benchmark.
    reservoir_list: list[tuple[str, object]] = []
    if args.windows:
        base = build_noise_reservoir(event_name="GW150914", detector="H1")
        subs = split_reservoir_into_windows(base, args.windows)
        reservoir_list = [(f"O1 H1 window {i+1}/{args.windows}", r)
                          for i, r in enumerate(subs)]
        logger.info(f"Split the O1 H1 reservoir into {args.windows} "
                    f"independent time windows.")
    else:
        for label, event, det in (RESERVOIRS[:2] if args.quick else RESERVOIRS):
            try:
                reservoir_list.append(
                    (label, build_noise_reservoir(event_name=event, detector=det)))
            except Exception as e:
                logger.warning(f"Reservoir '{label}' unavailable ({e}); skipping.")

    results: list[dict] = []
    t0 = time.time()
    for label, res in reservoir_list:
        entry = {"label": label, "cells": {}}
        for lg, snr in cells:
            a = reservoir_auc(res, lg, snr, n_per)
            entry["cells"][f"{lg:.0e}"] = a
            logger.info(f"{label:22s} lg={lg:.0e} snr={snr:.0f}  "
                        f"real-strain MF AUC={a['auc']:.3f} "
                        f"[{a['ci_lo']:.3f},{a['ci_hi']:.3f}]")
        results.append(entry)
        Path(args.out).parent.mkdir(exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"config": {"seed": SEED, "noise_seed": NOISE_SEED,
                                  "n_per_class": n_per, "cells": cells,
                                  "windows": args.windows},
                       "reservoirs": results}, f, indent=2)

    # Spread across reservoirs, per cell.
    for lg, snr in cells:
        key = f"{lg:.0e}"
        aucs = [r["cells"][key]["auc"] for r in results if key in r["cells"]]
        if aucs:
            logger.success(f"lambda_g={key} km: real-strain MF AUC across "
                           f"{len(aucs)} reservoirs = {np.mean(aucs):.3f} "
                           f"+/- {np.std(aucs):.3f}  (min {min(aucs):.3f}, "
                           f"max {max(aucs):.3f})")
    make_robustness_figure({"reservoirs": results,
                            "config": {"cells": cells}}, Path(args.out).parent)
    logger.success(f"Robustness study complete in {(time.time()-t0)/60:.1f} min "
                   f"-> {args.out}")


if __name__ == "__main__":
    main()
