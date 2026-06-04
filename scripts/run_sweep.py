"""
Sweep classifier + matched filter over a grid of (lambda_g, target_snr).

Runs train_classifier() and mf_score_dataset() on each cell, writes
per-cell results to results/sweep_results.json, and the consolidated
AUC table to results/sweep_auc.csv. Designed for laptop CPU runs.

Usage:
    python -m scripts.run_sweep              # default grid
    python -m scripts.run_sweep --quick      # 1x2 grid for smoke testing

Output JSON schema (one entry per cell):
    {
      "lambda_g_km": float,
      "target_snr":  float,
      "n_per_class": int,
      "n_epochs":    int,
      "classifier": {
         "val_acc": float,
         "val_auc": float,
         "val_auc_ci_lo": float,
         "val_auc_ci_hi": float,
         "val_scores": [..],
         "val_labels": [..],
         "history": {...},
      },
      "mf": {
         "auc": float,
         "auc_ci_lo": float,
         "auc_ci_hi": float,
         "scores": [..],
         "labels": [..],
      },
    }
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np
from loguru import logger

from step1_gw_classifier.train import train_classifier
from step1_gw_classifier.data import build_injection_dataset
from step1_gw_classifier.mf_baseline import mf_score_dataset, roc_from_scores


def bootstrap_auc_ci(y: np.ndarray, scores: np.ndarray, n_boot: int = 1000,
                     ci: float = 0.90, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ys = y[idx]; ss = scores[idx]
        if len(set(ys)) < 2:
            continue
        aucs.append(roc_from_scores(ys, ss)["auc"])
    aucs = np.array(aucs)
    lo, hi = np.quantile(aucs, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def run_one_cell(lambda_g_km: float, target_snr: float,
                 n_per_class: int, n_epochs: int,
                 reservoir_seconds: int = 1024,
                 seed: int = 42) -> dict:
    """One classifier training + MF scoring on the same dataset."""
    t0 = time.time()

    # 1. Train classifier (this also builds the dataset internally; we
    # then rebuild with the same seed for MF, since the training-only
    # split discards Xraw)
    cls_out = train_classifier(
        n_per_class=n_per_class, target_snr=target_snr,
        lambda_g_km=lambda_g_km, n_epochs=n_epochs,
        batch_size=32, learning_rate=3e-4, train_frac=0.85,
        reservoir_seconds=reservoir_seconds, seed=seed, compact=True,
    )
    t_cls = time.time() - t0
    logger.info(f"Cell ({lambda_g_km:.1e}, {target_snr:.1f}) classifier: "
                f"{t_cls/60:.1f} min, val_auc={cls_out['best_val_auc']:.3f}")

    # 2. Rebuild dataset with same seed for MF (need Xraw)
    X, Xraw, y, meta, res = build_injection_dataset(
        n_per_class=n_per_class, target_snr=target_snr,
        lambda_g_km=lambda_g_km, reservoir_seconds=reservoir_seconds, seed=seed,
    )
    # MF scores: compute on the *validation* slice only for fair compare.
    # Reconstruct val indices the same way train_classifier does.
    n_total = 2 * n_per_class
    n_tr = int(n_total * 0.85)
    n_val = n_total - n_tr
    g = np.random.default_rng(seed)  # not the same as torch generator
    import torch
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=gen).numpy()
    val_idx = perm[n_tr:]
    mf_scores = mf_score_dataset(Xraw[val_idx], [meta[i] for i in val_idx],
                                 res, lambda_g_km=lambda_g_km, progress=False)
    mf_roc = roc_from_scores(y[val_idx], mf_scores["s_mf"])
    mf_lo, mf_hi = bootstrap_auc_ci(y[val_idx], mf_scores["s_mf"], seed=seed + 1)
    t_mf = time.time() - t0 - t_cls
    logger.info(f"Cell ({lambda_g_km:.1e}, {target_snr:.1f}) MF: "
                f"{t_mf/60:.1f} min, auc={mf_roc['auc']:.3f} "
                f"[{mf_lo:.3f},{mf_hi:.3f}]")

    # Classifier CIs on validation
    cls_scores = np.asarray(cls_out["val_scores"])
    cls_labels = np.asarray(cls_out["val_labels"])
    cls_lo, cls_hi = bootstrap_auc_ci(cls_labels, cls_scores, seed=seed + 2)

    return {
        "lambda_g_km": float(lambda_g_km),
        "target_snr": float(target_snr),
        "n_per_class": n_per_class,
        "n_epochs": n_epochs,
        "classifier": {
            "val_acc": float(cls_out["best_val_acc"]),
            "val_auc": float(cls_out["best_val_auc"]),
            "val_auc_ci_lo": cls_lo,
            "val_auc_ci_hi": cls_hi,
            "val_scores": cls_scores.tolist(),
            "val_labels": cls_labels.tolist(),
            "history": cls_out["history"],
        },
        "mf": {
            "auc": float(mf_roc["auc"]),
            "auc_ci_lo": mf_lo,
            "auc_ci_hi": mf_hi,
            "scores": mf_scores["s_mf"].tolist(),
            "labels": y[val_idx].tolist(),
        },
        "wall_minutes": (time.time() - t0) / 60,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="Use a 1x2 (lambda_g, SNR) mini-grid for testing.")
    p.add_argument("--n_per_class", type=int, default=1000)
    p.add_argument("--n_epochs", type=int, default=25)
    p.add_argument("--out", type=str, default="results/sweep_results.json")
    args = p.parse_args()

    if args.quick:
        lambdas_km = [1e15]
        snrs = [20.0, 50.0]
    else:
        lambdas_km = [1e14, 1e15, 1e16]
        snrs = [20.0, 50.0]

    # Resume: load any existing cells and skip (lambda_g, snr) pairs already done.
    out_path = Path(args.out)
    cells: list[dict] = []
    if out_path.exists():
        try:
            cells = json.loads(out_path.read_text()).get("cells", [])
            logger.info(f"Resuming: {len(cells)} cells already in {out_path}")
        except Exception:
            cells = []
    done = {(round(c["lambda_g_km"], 3), round(c["target_snr"], 3)) for c in cells}

    t0 = time.time()
    for lg in lambdas_km:
        for snr in snrs:
            if (round(lg, 3), round(snr, 3)) in done:
                logger.info(f"--- skipping cell lambda_g={lg:.1e} km, SNR={snr:.1f} (already done)")
                continue
            logger.info(f"--- starting cell lambda_g={lg:.1e} km, SNR={snr:.1f}")
            cell = run_one_cell(lg, snr, args.n_per_class, args.n_epochs)
            cells.append(cell)
            out_path.parent.mkdir(exist_ok=True)
            with open(out_path, "w") as f:
                json.dump({"cells": cells, "args": vars(args)}, f, indent=2)
            logger.success(f"Saved {len(cells)} cells to {out_path}")
    logger.success(f"Sweep complete in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
