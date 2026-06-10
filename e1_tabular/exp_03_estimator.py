"""
exp_03_estimator.py  —  Theorem 3: finite-sample empirical-Bernstein coverage
and Π_2 vs Π_1 tightness.

Claim verified:
  * Both Π_1 and Π_2 empirical-Bernstein bounds B̂ cover the true loss with
    probability ≥ 1-δ = 0.95 across (game, repeat) pairs at all m.
  * Π_2 (logged fallback) is strictly tighter than Π_1 (worst-case), i.e.
    median(B̂_Π2 / true_loss) < median(B̂_Π1 / true_loss) at m=5000.

PASS <=> coverage ≥ 0.95 for both variants at all m  AND
         median Π_2 tightness < median Π_1 tightness.

Outputs:
    results/data/exp_03_estimator.csv
    results/figs/exp_03_coverage.pdf
    results/figs/exp_03_tightness.pdf
    one PASS/FAIL line  ->  results/summary.txt
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.games import make_game
from common import certificates as C
from common.io_utils import data_path, fig_path, write_summary
from common.plotting import new_fig, save_pdf
from e1_tabular._common import build, make_alpha_fn, const_fn

DELTA = 0.05
MS = [500, 1500, 5000]
VARIANTS = ["Pi2", "Pi1"]


def run(n_games: int, repeats: int, seed: int, S: int, A: int, H: int, N: int):
    rows = []
    master_rng = np.random.default_rng(seed)

    for gidx in tqdm(range(n_games), desc="exp_03 games"):
        gseed = int(master_rng.integers(0, 2**31))
        game = make_game(S=S, A=A, H=H, n=2, seed=gseed,
                         transition="dependent")
        alpha_fn = make_alpha_fn(gseed)
        N_fn = const_fn(N)
        b = build(game, alpha_fn, N_fn, eta=0.0, psi_rule="ref")

        tl = b.true_loss
        # range bound for empirical Bernstein: X ∈ [0, b_max]
        b_range = game.n * game.H * C.max_W(b.table)

        for variant in VARIANTS:
            for m in MS:
                for rep in range(repeats):
                    rep_rng = np.random.default_rng(
                        [seed, gidx, VARIANTS.index(variant), m, rep])
                    X = C.estimator_samples(
                        game, b.table, b.rho, m, rep_rng, variant=variant)
                    Bhat = C.empirical_bernstein(X, DELTA, b_range)
                    covered = int(Bhat >= tl - 1e-12)
                    tightness = Bhat / tl if tl > 1e-4 else np.nan

                    rows.append(dict(
                        variant=variant,
                        m=m,
                        game=gidx,
                        repeat=rep,
                        Bhat=Bhat,
                        true_loss=tl,
                        covered=covered,
                        tightness=tightness,
                    ))

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="exp_03: empirical-Bernstein coverage and tightness")
    ap.add_argument("--n_games", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--S", type=int, default=4)
    ap.add_argument("--A", type=int, default=3)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--N", type=int, default=5)
    args = ap.parse_args()

    df = run(args.n_games, args.repeats, args.seed,
             args.S, args.A, args.H, args.N)

    # --- CSV ---
    csv_file = data_path("exp_03_estimator.csv")
    df.to_csv(csv_file, index=False)

    # --- Coverage stats ---
    cov_stats = {}
    for variant in VARIANTS:
        for m in MS:
            sub = df[(df.variant == variant) & (df.m == m)]
            cov_stats[(variant, m)] = sub.covered.mean()

    # --- PDF (a): coverage bar plot ---
    fig, ax = new_fig(figsize=(6.0, 3.5))
    x = np.arange(len(MS))
    width = 0.35
    for vi, variant in enumerate(VARIANTS):
        covs = [cov_stats[(variant, m)] for m in MS]
        offset = (vi - 0.5) * width
        bars = ax.bar(x + offset, covs, width, label=variant)
    ax.axhline(1.0 - DELTA, color="k", ls="--", lw=1.0,
               label=f"1-δ = {1-DELTA:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in MS])
    ax.set_xlabel("sample size  m")
    ax.set_ylabel("coverage")
    ax.set_ylim(0.8, 1.02)
    ax.set_title("exp_03  empirical-Bernstein coverage")
    ax.legend()
    save_pdf(fig, fig_path("exp_03_coverage.pdf"))

    # --- PDF (b): tightness boxplot at m=5000 ---
    tight_Pi2 = df[(df.variant == "Pi2") & (df.m == 5000) &
                   df.tightness.notna()].tightness.values
    tight_Pi1 = df[(df.variant == "Pi1") & (df.m == 5000) &
                   df.tightness.notna()].tightness.values

    fig, ax = new_fig()
    ax.boxplot([tight_Pi2, tight_Pi1],
               tick_labels=[r"$\Pi_2$  (logged fb)", r"$\Pi_1$  (worst-case)"],
               showfliers=False)
    ax.set_ylabel(r"$\hat{B}$ / true loss  (tightness, m=5000)")
    ax.set_title("exp_03  estimator tightness comparison")
    save_pdf(fig, fig_path("exp_03_tightness.pdf"))

    # --- PASS/FAIL ---
    cov_ok = all(cov_stats[(v, m)] >= 1.0 - DELTA
                 for v in VARIANTS for m in MS)

    med_pi2 = float(np.nanmedian(tight_Pi2))
    med_pi1 = float(np.nanmedian(tight_Pi1))
    tight_ok = med_pi2 < med_pi1

    status = "PASS" if (cov_ok and tight_ok) else "FAIL"

    # coverage summary string
    cov_str_parts = []
    for variant in VARIANTS:
        for m in MS:
            cov_str_parts.append(
                f"{variant}/m={m}: cov={cov_stats[(variant,m)]:.3f}")
    cov_str = "  ".join(cov_str_parts)

    write_summary(
        f"exp_03 estimator [{status}] "
        f"median_tightness Pi2={med_pi2:.3f} Pi1={med_pi1:.3f}  "
        + cov_str
    )


if __name__ == "__main__":
    main()
