"""
Fill measured AUC numbers into paper.tex from results/sweep_results.json.

Replaces the remaining [VALUE_*] placeholders by reading the sweep
output, computing per-method AUC ranges, and substituting in-place.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
import numpy as np

PAPER = Path("paper.tex")
SWEEP = Path("results/sweep_results.json")


def main():
    text = PAPER.read_text()
    sweep = json.loads(SWEEP.read_text())
    cells = sweep["cells"]

    cnn_aucs = np.array([c["classifier"]["val_auc"] for c in cells])
    mf_aucs = np.array([c["mf"]["auc"] for c in cells])

    cnn_lo, cnn_hi = float(cnn_aucs.min()), float(cnn_aucs.max())
    mf_lo, mf_hi = float(mf_aucs.min()), float(mf_aucs.max())
    auc_floor = float(min(cnn_aucs.mean(), mf_aucs.mean()))

    # Range strings, two decimal places
    cnn_range_str = f"{cnn_lo:.2f}--{cnn_hi:.2f}"
    cnn_mid = f"{cnn_aucs.mean():.2f}"
    auc_floor_str = f"{auc_floor:.2f}"

    subs = {
        r"\[VALUE\\_AUC\\_CNN\\_RANGE\]": cnn_range_str,
        r"\[VALUE\\_AUC\\_CNN\]":         cnn_mid,
        r"\[VALUE\\_AUC\\_FLOOR\]":       auc_floor_str,
    }
    n_replaced = 0
    for pat, repl in subs.items():
        new_text, n = re.subn(pat, repl, text)
        if n:
            n_replaced += n
            text = new_text
            print(f"Replaced {n}x {pat} -> {repl}")

    PAPER.write_text(text)
    print(f"Total replacements: {n_replaced}")
    print(f"Classifier AUC across cells: {cnn_aucs}")
    print(f"Matched filter AUC across cells: {mf_aucs}")


if __name__ == "__main__":
    main()
