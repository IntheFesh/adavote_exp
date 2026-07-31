"""
exp_03_bound_types.py — Task 8 §3.3: bound-type comparison for the Theorem-3
finite-sample estimator (NOT to be confused with exp_12's bound_comparison,
which sweeps bound types for the Theorem-4 rollout-bridge estimator; see
that file's docstring vs this one).

Same 8 games / m=5000 / Pi2 sample construction as exp_03_estimator.py (same
game seeds, same common.certificates.estimator_samples draws), but instead
of only empirical_bernstein, applies all four common/baselines.py bound
formulas to the SAME drawn X to isolate the effect of the bound FORMULA
itself: empirical Bernstein, Hoeffding, clipped empirical Bernstein, naive
plug-in mean.

Outputs:
    results/data/exp_03_bound_types.csv
    one summary line -> results/summary.txt
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import certificates as C
from common.baselines import hoeffding_bound, emp_bernstein_bound, naive_plugin_mean, clipped_emp_bernstein_bound
from common.games import make_game
from common.io_utils import data_path, write_summary
from e1_tabular._common import build, make_alpha_fn, const_fn

DELTA = 0.05
M = 5000
BOUND_TYPES = [
    ("empirical_bernstein", emp_bernstein_bound),
    ("hoeffding", hoeffding_bound),
    ("clipped_empirical_bernstein", clipped_emp_bernstein_bound),
    ("naive_plugin", naive_plugin_mean),
]


def run(n_games: int, repeats: int, m: int, seed: int, S: int, A: int, H: int, N: int):
    rows = []
    master_rng = np.random.default_rng(seed)
    for gidx in tqdm(range(n_games), desc="exp_03_bound_types games"):
        gseed = int(master_rng.integers(0, 2**31))
        game = make_game(S=S, A=A, H=H, n=2, seed=gseed, transition="dependent")
        alpha_fn = make_alpha_fn(gseed)
        b = build(game, alpha_fn, const_fn(N), eta=0.0, psi_rule="ref")
        tl = b.true_loss
        b_range = game.n * game.H * C.max_W(b.table)

        for rep in range(repeats):
            rep_rng = np.random.default_rng([seed, gidx, m, rep])
            X = C.estimator_samples(game, b.table, b.rho, m, rep_rng, variant="Pi2")
            for name, fn in BOUND_TYPES:
                Bhat = fn(X, DELTA, b_range)
                rows.append(dict(
                    bound_type=name, game=gidx, repeat=rep, Bhat=Bhat,
                    true_loss=tl, covered=int(Bhat >= tl - 1e-9),
                    ratio_L=Bhat / tl if tl > 1e-9 else np.nan,
                ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="exp_03 bound-type comparison (Theorem 3 estimator, m=5000)")
    ap.add_argument("--n_games", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--m", type=int, default=M)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--S", type=int, default=4)
    ap.add_argument("--A", type=int, default=3)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--N", type=int, default=5)
    args = ap.parse_args()

    df = run(args.n_games, args.repeats, args.m, args.seed, args.S, args.A, args.H, args.N)
    csv_path = data_path("exp_03_bound_types.csv")
    df.to_csv(csv_path, index=False)

    g = df.groupby("bound_type").agg(coverage=("covered", "mean"),
                                      median_ratio_L=("ratio_L", "median"))
    print(g.round(4).to_string())

    # Self-assertion: recompute from the just-written CSV.
    df_check = pd.read_csv(csv_path)
    g_check = df_check.groupby("bound_type").agg(
        coverage=("covered", "mean"), median_ratio_L=("ratio_L", "median"))
    assert np.allclose(g_check["coverage"].values, g["coverage"].values)
    assert np.allclose(g_check["median_ratio_L"].values, g["median_ratio_L"].values)

    parts = "  ".join(f"{bt}:cov={g.loc[bt,'coverage']:.4f},B/L={g.loc[bt,'median_ratio_L']:.4f}"
                       for bt in g.index)
    write_summary(f"exp_03 bound_types [m={args.m}]  {parts}")


if __name__ == "__main__":
    main()
