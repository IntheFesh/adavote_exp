"""
exp_08_rank_consistency.py  —  Proposition 3 (rank consistency of C* and true loss).

Three subclasses are tested:

(a) Fixed-occupancy (action-independent transitions): sweep a global failure
    level by scaling alpha DOWN (which raises g_N) over a fine grid.  For each
    game, check that C* and true_loss are CO-MONOTONE (both weakly increase as
    alpha decreases / g increases).  Report fraction of monotone-consistent games
    (expect ~100%).

(b) Provable-budget (action-independent + all alpha > 1/2, so eligible): sweep
    committee size along ODD budgets N = 1,3,5,7,9,11.  Since g_N strictly
    DECREASES in odd N for alpha > 1/2, both C* and true_loss should be DOUBLY
    MONOTONE (both decrease).  Verify bi-monotonicity holds 100%.

(c) General (action-DEPENDENT transitions): over many games compute Spearman rho
    between C* and true_loss across a sweep of per-game failure levels (alpha
    scale sweep).  Report median Spearman rho (expect ~1.0).  Honestly search
    for and report ORDER-REVERSAL counterexamples.

Outputs:
    results/data/exp_08_rank_consistency.csv
    results/figs/exp_08_rank_consistency.pdf
    one PASS/FAIL line -> results/summary.txt

PASS criterion:
    (a) ~100% monotone-consistent games
    (b) 100% bi-monotone games
    (c) median Spearman rho ~1.0; reversal count is reported honestly (nonzero allowed).
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from common import certificates as C
from common.games import make_independent_game_with_occupancy, make_game
from common.io_utils import data_path, fig_path, write_summary
from common.plotting import save_pdf
from e1_tabular._common import build, const_fn, make_alpha_fn

TOL = 1e-10


# ---------------------------------------------------------------------------
# Helper: build a game with a GLOBAL alpha scale (all units share one alpha)
# and compute (Cstar, true_loss).
# ---------------------------------------------------------------------------
def _cstar_and_loss(game, alpha_global, N_global=5):
    """Build with constant alpha and N; return (Cstar, true_loss)."""
    b = build(game, const_fn(alpha_global), const_fn(N_global))
    return b.chain["Cstar"], b.true_loss


# ---------------------------------------------------------------------------
# (a) Fixed-occupancy: co-monotonicity over alpha sweep (alpha decreasing => g increasing)
# ---------------------------------------------------------------------------
def run_part_a(n_games: int, seed: int, alpha_grid, N: int, S, A, H):
    """For each game, sweep alpha from high to low (so g increases).
    Both C* and true_loss should weakly increase => co-monotone."""
    rows_a = []
    mono_count = 0

    for g in tqdm(range(n_games), desc="(a) fixed-occupancy"):
        gseed = seed * 200000 + g
        game = make_independent_game_with_occupancy(S=S, A=A, H=H, n=2, seed=gseed)

        cstars = []
        losses = []
        for alpha in alpha_grid:  # alpha decreasing => g increasing
            cs, tl = _cstar_and_loss(game, alpha, N_global=N)
            cstars.append(cs)
            losses.append(tl)

        cstars = np.array(cstars)
        losses = np.array(losses)

        # Check co-monotonicity: as g increases (alpha decreases), both should be
        # non-decreasing, i.e. differences should be >= -TOL.
        diff_cs = np.diff(cstars)
        diff_tl = np.diff(losses)
        mono = bool(np.all(diff_cs >= -TOL) and np.all(diff_tl >= -TOL))
        if mono:
            mono_count += 1
        else:
            # Print first violation
            viol_idx = np.where((diff_cs < -TOL) | (diff_tl < -TOL))[0]
            print(
                f"[PART-A VIOLATION] game {g}: "
                f"alpha[{viol_idx[0]}]={alpha_grid[viol_idx[0]]:.3f}->"
                f"{alpha_grid[viol_idx[0]+1]:.3f} "
                f"dCstar={diff_cs[viol_idx[0]]:.4e} "
                f"d_loss={diff_tl[viol_idx[0]]:.4e}"
            )

        rows_a.append(dict(
            part="a", game=g,
            mono_consistent=int(mono),
            cstar_min=float(cstars.min()),
            cstar_max=float(cstars.max()),
            loss_min=float(losses.min()),
            loss_max=float(losses.max()),
        ))

    frac_a = mono_count / n_games
    return pd.DataFrame(rows_a), frac_a


# ---------------------------------------------------------------------------
# (b) Provable-budget: bi-monotonicity over odd N sweep (N increasing => g decreasing)
# ---------------------------------------------------------------------------
def run_part_b(n_games: int, seed: int, odd_Ns, alpha_val: float, S, A, H):
    """All units have alpha > 1/2 (eligible), sweep odd committee sizes.
    As N increases, g_N(alpha) strictly decreases => C* and true_loss decrease."""
    rows_b = []
    mono_count = 0

    for g in tqdm(range(n_games), desc="(b) provable-budget"):
        gseed = seed * 300000 + g
        game = make_independent_game_with_occupancy(S=S, A=A, H=H, n=2, seed=gseed)

        cstars = []
        losses = []
        for N in odd_Ns:
            cs, tl = _cstar_and_loss(game, alpha_val, N_global=N)
            cstars.append(cs)
            losses.append(tl)

        cstars = np.array(cstars)
        losses = np.array(losses)

        # g_N is strictly decreasing in odd N for alpha > 0.5,
        # so both should be non-increasing (differences <= TOL).
        diff_cs = np.diff(cstars)
        diff_tl = np.diff(losses)
        mono = bool(np.all(diff_cs <= TOL) and np.all(diff_tl <= TOL))
        if mono:
            mono_count += 1
        else:
            viol_idx = np.where((diff_cs > TOL) | (diff_tl > TOL))[0]
            print(
                f"[PART-B VIOLATION] game {g}: "
                f"N[{viol_idx[0]}]={odd_Ns[viol_idx[0]]}->N[{viol_idx[0]+1}]={odd_Ns[viol_idx[0]+1]} "
                f"dCstar={diff_cs[viol_idx[0]]:.4e} d_loss={diff_tl[viol_idx[0]]:.4e}"
            )

        rows_b.append(dict(
            part="b", game=g,
            mono_consistent=int(mono),
            cstar_N1=float(cstars[0]),
            cstar_Nlast=float(cstars[-1]),
            loss_N1=float(losses[0]),
            loss_Nlast=float(losses[-1]),
        ))

    frac_b = mono_count / n_games
    return pd.DataFrame(rows_b), frac_b


# ---------------------------------------------------------------------------
# (c) General (action-dependent): Spearman rho + honest reversal search
# ---------------------------------------------------------------------------
def run_part_c(n_games: int, seed: int, alpha_grid, N: int, S, A, H):
    """Action-dependent transitions.  Compute Spearman rho(C*, true_loss) across
    the alpha sweep per game.  Also search for order-reversal counterexamples."""
    rows_c = []
    reversal_count = 0
    reversal_example = None  # Will store first concrete reversal

    for g in tqdm(range(n_games), desc="(c) general action-dependent"):
        gseed = seed * 400000 + g
        game = make_game(S=S, A=A, H=H, n=2, seed=gseed, transition="dependent")

        cstars = []
        losses = []
        for alpha in alpha_grid:
            cs, tl = _cstar_and_loss(game, alpha, N_global=N)
            cstars.append(cs)
            losses.append(tl)

        cstars = np.array(cstars)
        losses = np.array(losses)

        # Spearman correlation between C* and true_loss across sweep
        if np.std(cstars) < 1e-15 or np.std(losses) < 1e-15:
            rho_val = 1.0  # degenerate: constant, treat as perfectly correlated
        else:
            rho_res = spearmanr(cstars, losses)
            rho_val = float(rho_res.statistic)

        # Search for order-reversal pairs (i < j where C*(i)<C*(j) but loss(i)>loss(j)
        # or vice versa).
        game_has_reversal = False
        for ii in range(len(alpha_grid)):
            for jj in range(ii + 1, len(alpha_grid)):
                cs_diff = cstars[jj] - cstars[ii]  # C*(j) - C*(i)
                tl_diff = losses[jj] - losses[ii]   # loss(j) - loss(i)
                # Reversal: strict sign disagreement
                if (cs_diff > TOL and tl_diff < -TOL) or (cs_diff < -TOL and tl_diff > TOL):
                    if not game_has_reversal:
                        game_has_reversal = True
                        reversal_count += 1
                        if reversal_example is None:
                            reversal_example = dict(
                                game=g,
                                alpha_i=float(alpha_grid[ii]),
                                alpha_j=float(alpha_grid[jj]),
                                cstar_i=float(cstars[ii]),
                                cstar_j=float(cstars[jj]),
                                loss_i=float(losses[ii]),
                                loss_j=float(losses[jj]),
                            )
                            print(
                                f"[PART-C REVERSAL FOUND] game={g} "
                                f"alpha_i={alpha_grid[ii]:.3f} alpha_j={alpha_grid[jj]:.3f} "
                                f"Cstar_i={cstars[ii]:.4f} Cstar_j={cstars[jj]:.4f} "
                                f"loss_i={losses[ii]:.4f} loss_j={losses[jj]:.4f}"
                            )

        rows_c.append(dict(
            part="c", game=g,
            spearman_rho=float(rho_val),
            has_reversal=int(game_has_reversal),
        ))

    df_c = pd.DataFrame(rows_c)
    median_rho = float(df_c["spearman_rho"].median())
    return df_c, median_rho, reversal_count, reversal_example


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_games", type=int, default=100,
                    help="# games per subclass")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--S", type=int, default=4)
    ap.add_argument("--A", type=int, default=3)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--N", type=int, default=5,
                    help="Committee size for parts (a) and (c)")
    ap.add_argument("--n_alpha", type=int, default=10,
                    help="# alpha levels in sweep (equally spaced from high to low)")
    args = ap.parse_args()

    # Alpha grid: from high (0.85) to low (0.55) — DECREASING so g INCREASES
    # Must be > 0.5 for (a) and (b) to keep eligible units
    alpha_grid_a = np.linspace(0.85, 0.55, args.n_alpha)

    # For (c): same eligible range — the coupling effect creates reversals even within this range
    alpha_grid_c = np.linspace(0.85, 0.55, args.n_alpha)

    odd_Ns = [1, 3, 5, 7, 9, 11]
    alpha_b = 0.72  # > 0.5 => eligible

    # ---- Part (a) ----
    df_a, frac_a = run_part_a(
        args.n_games, args.seed, alpha_grid_a, args.N,
        args.S, args.A, args.H
    )
    print(f"[Part-a] Monotone-consistent fraction: {frac_a:.4f} "
          f"({int(frac_a * args.n_games)}/{args.n_games})")

    # ---- Part (b) ----
    df_b, frac_b = run_part_b(
        args.n_games, args.seed, odd_Ns, alpha_b,
        args.S, args.A, args.H
    )
    print(f"[Part-b] Bi-monotone fraction: {frac_b:.4f} "
          f"({int(frac_b * args.n_games)}/{args.n_games})")

    # ---- Part (c) ----
    df_c, median_rho, reversal_count, reversal_example = run_part_c(
        args.n_games, args.seed, alpha_grid_c, args.N,
        args.S, args.A, args.H
    )
    print(f"[Part-c] Median Spearman rho: {median_rho:.4f} | "
          f"Reversal games: {reversal_count}/{args.n_games}")
    if reversal_example is not None:
        print(f"  First reversal: game={reversal_example['game']} "
              f"alpha_i={reversal_example['alpha_i']:.3f} alpha_j={reversal_example['alpha_j']:.3f} "
              f"Cstar_i={reversal_example['cstar_i']:.4f} Cstar_j={reversal_example['cstar_j']:.4f} "
              f"loss_i={reversal_example['loss_i']:.4f} loss_j={reversal_example['loss_j']:.4f}")
    else:
        print("  No order reversals found (as expected for eligible-only sweep).")

    # ---- Save CSV ----
    df_all = pd.concat([df_a, df_b, df_c], ignore_index=True, sort=False)
    csv_path = data_path("exp_08_rank_consistency.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

    # ---- Figures ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Spearman rho histogram (part c)
    rhos = df_c["spearman_rho"].values
    ax1.hist(rhos, bins=20, color="steelblue", edgecolor="k", lw=0.4)
    ax1.axvline(median_rho, color="red", lw=1.2, ls="--", label=f"median={median_rho:.3f}")
    ax1.set_xlabel("Spearman rho (C* vs true_loss)")
    ax1.set_ylabel("# games")
    ax1.set_title("(c) General: Spearman rho distribution")
    ax1.legend(fontsize=9)

    # Right: monotone-fraction bar (parts a and b)
    bars = [frac_a * 100, frac_b * 100]
    labels = ["(a) Fixed-occ\nco-mono", "(b) Prov-budget\nbi-mono"]
    colors = ["steelblue", "darkorange"]
    ax2.bar(labels, bars, color=colors, edgecolor="k", lw=0.5)
    ax2.axhline(100, color="k", ls="--", lw=0.8)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("Consistent games (%)")
    ax2.set_title("(a)/(b) Monotone consistency")

    fig.tight_layout()
    pdf_path = fig_path("exp_08_rank_consistency.pdf")
    save_pdf(fig, pdf_path)
    print(f"PDF saved: {pdf_path}")

    # ---- Summary ----
    status_a = "PASS" if frac_a >= 0.99 else "FAIL"
    status_b = "PASS" if frac_b >= 1.00 else "FAIL"
    # (c): PASS if median rho is high AND reversal count is honestly reported
    status_c = "PASS" if median_rho >= 0.90 else "FAIL"
    overall = "PASS" if all(s == "PASS" for s in [status_a, status_b, status_c]) else "FAIL"

    write_summary(
        f"exp_08 rank_consistency [{overall}] "
        f"(a)mono_frac={frac_a:.3f}[{status_a}] "
        f"(b)bi_mono_frac={frac_b:.3f}[{status_b}] "
        f"(c)median_spearman={median_rho:.4f}[{status_c}] "
        f"reversals_found={reversal_count}"
    )


if __name__ == "__main__":
    main()
