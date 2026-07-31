"""
exp_a4_summary_annotate.py — Task 7 §1.2: annotate exp_a4's dual computational
basis for the I2 certificate and mark which one the paper adopts.

No rerun: reads the already-committed results/data/exp_a4_adjacent_baselines.csv
(from the Task 5 rebuild) and writes a small summary.json alongside it,
labeling both bases explicitly:

  口径 O (I2_O): population C2/L, no sampling error (deterministic per game).
  口径 B (I2_B): finite-sample empirical-Bernstein bound / L (same estimation
                 basis as PDIS/DR), ADOPTED BY THE PAPER per Task 7 §1.2 --
                 口径 O and PDIS/DR are not the same kind of estimate (one is
                 an exact population quantity, the other two are finite-sample
                 estimates), so comparing them side by side in one table
                 requires using 口径 B for I2.

Self-asserting: the summary numbers are recomputed directly from the CSV
before being written.
"""

import json

import numpy as np
import pandas as pd

from common.io_utils import data_path

df = pd.read_csv(data_path("exp_a4_adjacent_baselines.csv"))
g = df.groupby("method").agg(coverage=("covered", "mean"), median_ratio_L=("ratio_L", "median"))

summary = {
    "source_csv": "results/data/exp_a4_adjacent_baselines.csv",
    "n_games": int(df["game"].nunique()),
    "repeats": int(df["repeat"].nunique()),
    "I2_dual_basis": {
        "口径_O_oracle": {
            "definition": "population C2 (common.certificates.cert_Ctilde) / exact-DP true_loss L; no sampling error",
            "coverage": float(g.loc["I2_O", "coverage"]),
            "median_ratio_L": float(g.loc["I2_O", "median_ratio_L"]),
            "adopted_by_paper": False,
        },
        "口径_B_finite_sample": {
            "definition": "common.certificates.empirical_bernstein on Pi2 estimator_samples (m=2000) / true_loss L; same finite-sample basis as PDIS/DR",
            "coverage": float(g.loc["I2_B", "coverage"]),
            "median_ratio_L": float(g.loc["I2_B", "median_ratio_L"]),
            "adopted_by_paper": True,
        },
        "decision_ref": "Task 7 §1.2: 口径 O's 1.0183 vs the paper's historical 1.019 is a 0.07% gap, judged coincidental "
                        "-- 口径 O is a population quantity while PDIS/DR are finite-sample estimates, not the same "
                        "evaluation basis. Paper table adopts 口径 B for a like-for-like comparison across all four methods.",
    },
    "other_methods": {
        m: {"coverage": float(g.loc[m, "coverage"]), "median_ratio_L": float(g.loc[m, "median_ratio_L"])}
        for m in ["PDIS", "DR", "robust_sim_lemma"]
    },
}

out_path = data_path("exp_a4_adjacent_baselines_summary.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# Self-assertion: reload and confirm it matches direct recomputation.
with open(out_path) as f:
    check = json.load(f)
assert np.isclose(check["I2_dual_basis"]["口径_O_oracle"]["median_ratio_L"], g.loc["I2_O", "median_ratio_L"])
assert np.isclose(check["I2_dual_basis"]["口径_B_finite_sample"]["median_ratio_L"], g.loc["I2_B", "median_ratio_L"])
print(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"\nwrote {out_path}")
