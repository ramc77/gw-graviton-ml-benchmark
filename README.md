# gw-graviton-ml-benchmark

Reproducibility code for the manuscript

> **Benchmarking deep learning against matched filtering for single-event
> graviton-dispersion tests on LIGO data** — R. Chand.

## What this is

A methods/benchmark study, not a new physics bound. The question:

> On a single binary-black-hole event, how does a compact CNN+Transformer
> classifier compare with the *optimal* matched filter when discriminating a
> general-relativistic (massless-graviton) inspiral from a massive-graviton
> dispersed template (Mirshekari–Yunes 2012), injected into LIGO H1 strain
> noise?

**Headline result.** At signal-to-noise ratios typical of the current
catalogue, the learned classifier and the oracle matched filter *both* perform
at chance (AUC ≈ 0.5): single-event discrimination is noise-limited, not
method-limited. On noise-free signals both reach AUC = 1, which validates the
implementation. A learned representation offers no shortcut around the noise
floor that obliges LIGO–Virgo–KAGRA to stack ~20 events for a useful
graviton-mass constraint.

## Layout

```
step1_gw_classifier/        — the pipeline
  waveform.py               — IMRPhenomD (pycbc) + Mirshekari–Yunes dispersion phase
  matched_filter.py         — Welch PSD, PSD-weighted inner product, optimal SNR, match
  mf_baseline.py            — oracle matched-filter discriminator s_MF = rho_MG^2 - rho_GR^2
  data.py                   — injection dataset: H1 noise reservoir + GR/MG injections + labels
  model.py                  — compact (~1.7e5 param) CNN+Transformer classifier
  train.py                  — per-cell training loop
  gwtc4_bbh_events.csv      — curated 56 GWTC-4.0 BBH events (m1,m2 > 3 Msun, network SNR > 10)
scripts/                    — figure and sweep drivers
  make_validation_curve.py  — dispersion-validation mismatch curve (Fig. 1)
  run_sweep.py              — classifier + matched-filter sweep over (lambda_g, SNR)
  make_figures.py           — sensitivity / AUC / ROC figures
  make_classifier_control.py— signal-only positive-control / scaling figure
  make_glitch_figure.py     — noise-transient vs inter-class-signal illustration
results/                    — figures (PDF + PNG, 300 DPI), CSV/JSON result tables, references.bib
tests/                      — pytest unit tests
paper.tex / paper.pdf       — the manuscript
```

## Setup

```bash
python -m pip install -r requirements.txt
```

Python 3.11 is assumed. The gravitational-wave stack (`gwpy`, `gwosc`,
`pycbc`, `lalsuite`) is required, not optional.

## Reproduce

The figure drivers are run as modules from the repository root:

```bash
python -m scripts.make_validation_curve     # Fig. 1 mismatch curve + threshold crossing
python -m scripts.run_sweep                 # classifier vs matched-filter AUC sweep
python -m scripts.make_figures              # sensitivity / AUC / ROC figures
python -m scripts.make_classifier_control   # signal-only positive control
python -m scripts.make_glitch_figure        # noise-transient illustration
pytest tests/ -v                            # unit tests
```

All figures are written to `results/` as both PDF and PNG at 300 DPI.

## Data

GW strain data are public via the [LIGO Open Science Center](https://gwosc.org)
and are fetched at runtime with `gwpy`. The curated event metadata in
`step1_gw_classifier/gwtc4_bbh_events.csv` come from the `gwosc.api` event JSON
endpoint (not hand-entered).

## Key references

- Mirshekari, Yunes & Will 2012, *Phys. Rev. D* **85**, 024041 (arXiv:1110.2720) — massive-graviton dispersion phase
- Will 1998, *Phys. Rev. D* **57**, 2061 — graviton-mass bound formalism
- Khan et al. 2016, *Phys. Rev. D* **93**, 044007 (arXiv:1508.07253) — IMRPhenomD
- Allen et al. 2012, *Phys. Rev. D* **85**, 122006 — matched-filter formalism
- Abbott et al. 2021, *Phys. Rev. D* **103**, 122002 — LVK O3 tests of GR

## Note

This code accompanies the manuscript and is provided for peer review and
reproducibility.
