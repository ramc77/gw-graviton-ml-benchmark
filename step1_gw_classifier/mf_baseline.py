"""
Matched-filter baseline classifier (oracle parameters)
=======================================================

For each (real noise + injection) sample we know the true source
parameters (m1, m2). The oracle matched filter computes two
likelihoods:

    L_GR  = max_phi,t  Re < d | h_GR(m1,m2)  >
    L_MG  = max_phi,t  Re < d | h_MG(m1,m2, lambda_g) >

and reports the log-likelihood-ratio statistic

    s_MF = rho_MG^2 - rho_GR^2

where rho is the normalised matched-filter SNR. In Gaussian noise this
is twice the log Bayes factor (Neyman-Pearson optimal). This is the
strongest possible matched-filter baseline — the classifier must
approach it on the ROC plane to demonstrate value.

References
----------
- Allen et al. 2012, PRD 85 122006
- LVK Collaboration 2021 PRD 103 122002 (tests of GR using ppE statistic)
"""
from __future__ import annotations

import numpy as np
from loguru import logger
from tqdm import tqdm

from .waveform import gr_waveform_fd, massive_graviton_waveform_fd
from .matched_filter import inner_product


def _normalised_template(
    freqs: np.ndarray, h: np.ndarray, psd: np.ndarray,
    f_low: float, f_high: float, df: float,
) -> tuple[np.ndarray, float]:
    """Return h / sqrt(<h|h>) and the normalisation."""
    sigma2 = inner_product(h, h, psd, df, freqs=freqs, f_low=f_low, f_high=f_high)
    sigma = np.sqrt(max(sigma2, 0))
    if sigma == 0:
        return h, 1.0
    return h / sigma, sigma


def _matched_filter_snr_max_over_time(
    d_fd: np.ndarray, h_normalised_fd: np.ndarray,
    psd: np.ndarray, df: float, freqs: np.ndarray,
    f_low: float, f_high: float,
) -> float:
    """
    Maximum over time-shift of the normalised matched-filter SNR:

        rho(t0) = Re { IFFT[ d(f) conj(h_norm(f)) / S_n(f) ] }(t0) * 4 * df

    Maximum over phi is implicit by taking |rho(t0)| complex magnitude
    (since the template is normalised); equivalently the absolute value
    of the IFFT.
    """
    safe_psd = np.where(psd > 0, psd, np.inf)
    integrand = d_fd * np.conjugate(h_normalised_fd) / safe_psd
    if freqs is not None:
        mask = (freqs >= f_low) & (freqs <= f_high)
        integrand = np.where(mask, integrand, 0.0)
    n_pos = len(integrand)
    two_sided = np.zeros(2 * (n_pos - 1), dtype=np.complex128)
    two_sided[:n_pos] = integrand
    corr = np.fft.ifft(two_sided) * len(two_sided) * df * 4.0
    return float(np.max(np.abs(corr)))


def mf_score_one(
    strain_raw_td: np.ndarray,
    m1: float, m2: float,
    lambda_g_km: float,
    dist_mpc: float,
    sample_rate: float,
    psd_freqs: np.ndarray, psd_vals: np.ndarray,
    f_lower: float = 20.0,
    f_final: float = 1024.0,
) -> dict:
    """
    Compute the oracle MF log-likelihood-ratio statistic for one segment.

    Returns dict {rho_gr, rho_mg, s_mf = rho_mg^2 - rho_gr^2}.

    The two templates use the *true* (m1, m2, D_L) so this is an oracle
    bound. The luminosity distance D_L is required because the
    Mirshekari–Yunes dispersion phase scales as δΨ_MG ∝ D_L; using a
    reference distance for the template would generate the wrong phase.
    (The absolute amplitude of the template is irrelevant — we normalise
    by ⟨h|h⟩ — only the phase needs to match the source.)

    Implementation: work on the full rfft grid of the strain (length N/2+1).
    The templates from pycbc are only populated up to f_final; we zero-pad
    them to the full rfft length so the IFFT-based time maximisation is
    sampled at the strain's native rate.
    """
    n_t = len(strain_raw_td)
    seg_dur_s = n_t / sample_rate
    delta_f = 1.0 / seg_dur_s

    # FD strain on the strain's native rfft grid
    d_fd = np.fft.rfft(strain_raw_td) / sample_rate
    freqs = np.fft.rfftfreq(n_t, d=1.0 / sample_rate)
    n_pos = len(freqs)

    # Templates at the true source distance (so dispersion phase is correct).
    # Amplitude is irrelevant — they are normalised.
    _, h_gr_short = gr_waveform_fd(
        m1, m2, dist_mpc,
        delta_f=delta_f, f_lower=f_lower, f_final=f_final,
    )
    _, h_mg_short = massive_graviton_waveform_fd(
        m1, m2, dist_mpc, lambda_g_km * 1e3,
        delta_f=delta_f, f_lower=f_lower, f_final=f_final,
    )
    h_gr = np.zeros(n_pos, dtype=np.complex128)
    h_mg = np.zeros(n_pos, dtype=np.complex128)
    m_short = min(len(h_gr_short), n_pos)
    h_gr[:m_short] = h_gr_short[:m_short]
    h_mg[:m_short] = h_mg_short[:m_short]

    # PSD on the strain grid
    if len(psd_vals) >= n_pos:
        psd_l = psd_vals[:n_pos]
    else:
        psd_l = np.concatenate(
            [psd_vals, np.full(n_pos - len(psd_vals), psd_vals[-1])]
        )

    # Normalise templates
    h_gr_n, _ = _normalised_template(freqs, h_gr, psd_l, f_lower, f_final, delta_f)
    h_mg_n, _ = _normalised_template(freqs, h_mg, psd_l, f_lower, f_final, delta_f)

    rho_gr = _matched_filter_snr_max_over_time(
        d_fd, h_gr_n, psd_l, delta_f, freqs, f_lower, f_final,
    )
    rho_mg = _matched_filter_snr_max_over_time(
        d_fd, h_mg_n, psd_l, delta_f, freqs, f_lower, f_final,
    )
    return {"rho_gr": rho_gr, "rho_mg": rho_mg, "s_mf": rho_mg ** 2 - rho_gr ** 2}


def mf_score_dataset(
    Xraw: np.ndarray, meta: list, reservoir,
    lambda_g_km: float,
    f_lower: float = 20.0,
    f_final: float = 1024.0,
    progress: bool = True,
) -> np.ndarray:
    """
    Compute the oracle MF discriminator s_mf for every injection in the
    dataset. Returns a length-N array of s_mf values.
    """
    sr = reservoir.sample_rate
    pf, pv = reservoir.psd_freqs, reservoir.psd_vals
    N = len(meta)
    s = np.zeros(N, dtype=np.float64)
    rho_gr = np.zeros(N, dtype=np.float64)
    rho_mg = np.zeros(N, dtype=np.float64)
    iterator = enumerate(meta)
    if progress:
        iterator = tqdm(list(iterator), desc="MF scoring", leave=False)
    for i, m in iterator:
        out = mf_score_one(
            Xraw[i], m.m1, m.m2, lambda_g_km, m.dist_mpc_eff,
            sample_rate=sr, psd_freqs=pf, psd_vals=pv,
            f_lower=f_lower, f_final=f_final,
        )
        s[i] = out["s_mf"]
        rho_gr[i] = out["rho_gr"]
        rho_mg[i] = out["rho_mg"]
    return {"s_mf": s, "rho_gr": rho_gr, "rho_mg": rho_mg}


def roc_from_scores(y: np.ndarray, score: np.ndarray, n_thresh: int = 200) -> dict:
    """
    Compute ROC: (FPR, TPR) for varying threshold on `score`.
    Convention: high score = predict MG (label 1).
    """
    thresh = np.linspace(score.min() - 1e-9, score.max() + 1e-9, n_thresh)
    fpr = np.zeros(n_thresh)
    tpr = np.zeros(n_thresh)
    npos = max((y == 1).sum(), 1)
    nneg = max((y == 0).sum(), 1)
    for j, t in enumerate(thresh):
        pred = score > t
        tpr[j] = ((pred) & (y == 1)).sum() / npos
        fpr[j] = ((pred) & (y == 0)).sum() / nneg
    # AUC by trapezoidal rule with FPR ascending
    order = np.argsort(fpr)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    auc = float(trapz(tpr[order], fpr[order]))
    return {"fpr": fpr, "tpr": tpr, "thresh": thresh, "auc": auc}
