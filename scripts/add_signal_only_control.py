"""
Compute the signal-only (noise-free) matched-filter control for each sweep cell.

For every (lambda_g, SNR) cell already present in results/sweep_results.json
we rebuild the *same* injection dataset (identical seed, so the validation
partition is reproduced exactly), regenerate each validation injection's
signal WITHOUT adding strain noise, and score it through the same oracle
matched filter used for the H1-strain numbers. In the absence of noise the
discriminator s_MF separates the GR and MG classes essentially perfectly;
this is the code-validation control plotted as open symbols in Fig. 2 and
quoted in the text.

The per-cell field "mf_signal_only": {auc, auc_ci_lo, auc_ci_hi} is appended
in place and the JSON is rewritten.

Usage:
    python -m scripts.add_signal_only_control
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from step1_gw_classifier.data import (
    build_injection_dataset,
    _time_domain_from_fd,
)
from step1_gw_classifier.waveform import (
    gr_waveform_fd,
    massive_graviton_waveform_fd,
)
from step1_gw_classifier.mf_baseline import mf_score_one, roc_from_scores
from scripts.run_sweep import bootstrap_auc_ci

SWEEP = Path("results/sweep_results.json")
F_LOWER = 20.0
F_FINAL = 1024.0
SEED = 42


def _val_indices(n_total: int, train_frac: float = 0.85, seed: int = SEED) -> np.ndarray:
    """Reproduce the exact validation partition used by train_classifier."""
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=gen).numpy()
    n_tr = int(n_total * train_frac)
    return perm[n_tr:]


def signal_only_auc(lambda_g_km: float, target_snr: float, n_per_class: int,
                    reservoir_seconds: int = 1024) -> dict:
    """AUC of the oracle matched filter on noise-free injections (val split)."""
    X, Xraw, y, meta, res = build_injection_dataset(
        n_per_class=n_per_class, target_snr=target_snr,
        lambda_g_km=lambda_g_km, reservoir_seconds=reservoir_seconds, seed=SEED,
    )
    sr = res.sample_rate
    seg_dur = res.seg_duration_s
    delta_f = 1.0 / seg_dur

    val_idx = _val_indices(2 * n_per_class)
    s = np.zeros(len(val_idx), dtype=np.float64)
    yv = np.zeros(len(val_idx), dtype=np.int64)
    for k, i in enumerate(val_idx):
        m = meta[i]
        # Regenerate the signal alone (no strain noise added).
        if m.lambda_g_km is None:
            _, h_fd = gr_waveform_fd(m.m1, m.m2, m.dist_mpc_eff,
                                     delta_f=delta_f, f_lower=F_LOWER, f_final=F_FINAL)
        else:
            _, h_fd = massive_graviton_waveform_fd(
                m.m1, m.m2, m.dist_mpc_eff, m.lambda_g_km * 1e3,
                delta_f=delta_f, f_lower=F_LOWER, f_final=F_FINAL)
        signal_td = _time_domain_from_fd(h_fd, delta_f, sr, seg_dur)
        out = mf_score_one(
            signal_td, m.m1, m.m2, lambda_g_km, m.dist_mpc_eff,
            sample_rate=sr, psd_freqs=res.psd_freqs, psd_vals=res.psd_vals,
            f_lower=F_LOWER, f_final=F_FINAL,
        )
        s[k] = out["s_mf"]
        yv[k] = m.label

    auc = roc_from_scores(yv, s)["auc"]
    lo, hi = bootstrap_auc_ci(yv, s, seed=SEED + 7)
    return {"auc": float(auc), "auc_ci_lo": lo, "auc_ci_hi": hi}


def main():
    sweep = json.loads(SWEEP.read_text())
    cells = sweep["cells"]
    t0 = time.time()
    for c in cells:
        lg = c["lambda_g_km"]
        snr = c["target_snr"]
        npc = c["n_per_class"]
        logger.info(f"--- signal-only control: lambda_g={lg:.1e} km, SNR={snr:.1f}")
        ctrl = signal_only_auc(lg, snr, npc)
        c["mf_signal_only"] = ctrl
        logger.success(
            f"    lambda_g={lg:.1e}, SNR={snr:.0f}: signal-only AUC={ctrl['auc']:.3f} "
            f"[{ctrl['auc_ci_lo']:.3f},{ctrl['auc_ci_hi']:.3f}]")
        SWEEP.write_text(json.dumps(sweep, indent=2))
    logger.success(f"All signal-only controls computed in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
