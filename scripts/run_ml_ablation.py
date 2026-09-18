"""
Machine-learning ablation  (referee 1 #4, referee 2 #1)
=======================================================

Tests whether the single-event null is specific to the compact
CNN+Transformer or holds across architectures, random seeds, and
training-set size. For a representative resolvable-but-buried cell, this
trains two architectures on observational H1 strain---

    cnn_transformer : the compact CNN+Transformer of the main study
    resnet          : an established 1D residual CNN baseline (model.ResNet1D)

---over several random seeds and increasing N, and reports the mean and
spread of the best validation AUC. A signal-only run at the largest N
confirms each architecture retains capacity in the absence of noise.

This is the one heavy step and is intended for a GPU (e.g. Google Colab);
train_classifier() uses CUDA automatically when available.

Full run (GPU):
    python -m scripts.run_ml_ablation \
        --archs cnn_transformer resnet --seeds 0 1 2 --n_grid 400 1000 2000 \
        --epochs 30 --lambda_g_km 1e13 --target_snr 20
CPU smoke test:
    python -m scripts.run_ml_ablation --quick

Writes results/ml_ablation.json (resumable: existing (arch,N,condition) rows
are skipped).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from loguru import logger

from step1_gw_classifier.train import train_classifier


def _load(out: str) -> list[dict]:
    p = Path(out)
    if p.exists():
        try:
            return json.loads(p.read_text()).get("rows", [])
        except Exception:
            return []
    return []


def _save(out: str, rows: list[dict], config: dict) -> None:
    Path(out).parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump({"config": config, "rows": rows}, f, indent=2)


def _done(rows, arch, n, condition) -> bool:
    return any(r["arch"] == arch and r["n_per_class"] == n
               and r["condition"] == condition for r in rows)


def make_ablation_figure(rows: list[dict], out_dir: Path) -> None:
    """Strain AUC (mean +/- std over seeds) vs N for each architecture."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    archs = sorted({r["arch"] for r in rows})
    colors = {"cnn_transformer": "C0", "resnet": "C1"}
    names = {"cnn_transformer": "CNN+Transformer", "resnet": "1D ResNet"}
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for arch in archs:
        st = sorted([r for r in rows if r["arch"] == arch and r["condition"] == "strain"],
                    key=lambda r: r["n_per_class"])
        if st:
            n = np.array([r["n_per_class"] for r in st])
            mu = np.array([r["auc_mean"] for r in st])
            sd = np.array([r["auc_std"] for r in st])
            ax.errorbar(n, mu, yerr=sd, fmt="o-", color=colors.get(arch, "C2"),
                        capsize=3, lw=1.6, ms=6, label=f"{names.get(arch, arch)} (H1 strain)")
        so = [r for r in rows if r["arch"] == arch and r["condition"] == "signal_only"]
        if so:
            ax.scatter([so[0]["n_per_class"]], [so[0]["auc_mean"]], marker="*",
                       s=120, color=colors.get(arch, "C2"), edgecolor="k", zorder=5,
                       label=f"{names.get(arch, arch)} (signal only)")
    ax.axhline(0.5, ls=":", color="0.4", lw=1.0, label="chance")
    ax.set_xscale("log")
    ax.set_xlabel(r"Training-set size $N$ per class")
    ax.set_ylabel("Best validation AUC")
    ax.set_ylim(0.4, 1.045)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="center left", fontsize=8.5, framealpha=0.93)
    fig.tight_layout()
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "fig_ml_ablation.pdf", dpi=300)
    fig.savefig(out_dir / "fig_ml_ablation.png", dpi=300)
    plt.close(fig)
    logger.info("wrote fig_ml_ablation.{pdf,png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archs", nargs="+", default=["cnn_transformer", "resnet"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n_grid", type=int, nargs="+", default=[400, 1000, 2000])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lambda_g_km", type=float, default=1e13)
    p.add_argument("--target_snr", type=float, default=20.0)
    p.add_argument("--compact", action="store_true",
                   help="use the compact CPU-scale models instead of the full ones")
    p.add_argument("--quick", action="store_true", help="tiny CPU smoke test")
    p.add_argument("--out", type=str, default="results/ml_ablation.json")
    args = p.parse_args()
    if args.quick:
        args.seeds, args.n_grid, args.epochs, args.compact = [0], [40], 2, True

    config = vars(args).copy()
    rows = _load(args.out)
    logger.info(f"Resuming with {len(rows)} rows already in {args.out}" if rows
                else "Starting fresh")

    t0 = time.time()
    for arch in args.archs:
        # strain: full seed x N grid
        for n in args.n_grid:
            if _done(rows, arch, n, "strain"):
                logger.info(f"skip arch={arch} N={n} strain (done)")
                continue
            aucs = []
            for seed in args.seeds:
                o = train_classifier(
                    n_per_class=n, target_snr=args.target_snr,
                    lambda_g_km=args.lambda_g_km, n_epochs=args.epochs,
                    batch_size=args.batch_size, seed=seed,
                    compact=args.compact, arch=arch, noise_scale=1.0)
                aucs.append(float(o["best_val_auc"]))
                logger.info(f"[strain] arch={arch} N={n} seed={seed} "
                            f"AUC={o['best_val_auc']:.3f} ({o['n_params']:,} params)")
            rows.append({"arch": arch, "n_per_class": n, "condition": "strain",
                         "seeds": list(args.seeds), "aucs": aucs,
                         "auc_mean": float(np.mean(aucs)),
                         "auc_std": float(np.std(aucs))})
            _save(args.out, rows, config)
            logger.success(f"arch={arch} N={n} strain AUC = "
                           f"{np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
        # signal-only capacity check at the largest N (single seed)
        n = max(args.n_grid)
        if not _done(rows, arch, n, "signal_only"):
            o = train_classifier(
                n_per_class=n, target_snr=args.target_snr,
                lambda_g_km=args.lambda_g_km, n_epochs=args.epochs,
                batch_size=args.batch_size, seed=args.seeds[0],
                compact=args.compact, arch=arch, noise_scale=0.0)
            rows.append({"arch": arch, "n_per_class": n, "condition": "signal_only",
                         "seeds": [args.seeds[0]], "aucs": [float(o["best_val_auc"])],
                         "auc_mean": float(o["best_val_auc"]), "auc_std": 0.0})
            _save(args.out, rows, config)
            logger.success(f"arch={arch} N={n} signal-only AUC = {o['best_val_auc']:.3f}")

    make_ablation_figure(rows, Path(args.out).parent)
    logger.success(f"ML ablation complete in {(time.time()-t0)/60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
