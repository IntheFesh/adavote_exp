"""
exp_12_rollout_bridge.py — Theorem 4 conservative rollout value bound +
Theorem 3 finite-sample certificate, audited against exact-DP truth.

This reproduces the paper's Section 9.2 / Figure 5 "conservative rollout
bridge" result, which was NOT previously implemented anywhere in this repo
(no code computed W_fb^+ from resettable Monte-Carlo rollouts; the certificate
machinery in exp_03 uses the EXACT Delta_+ / W values). This experiment fills
that gap: exact tabular DP is used only to AUDIT the rollout-based bound
against ground truth, never to compute the bound itself.

Protocol per repeat (Definition 10 + Theorem 3 + Theorem 4), eta=0:
  1. Draw m i.i.d. certification episodes: each draws one unit U_j ~ mu
     (occupancy law) and one committee failure indicator F_j ~ Bernoulli(g(U_j))
     (equivalently Binomial(N,alpha)<=floor(N/2), matching Definition 4).
  2. For each LOGGED FAILED unit (F_j=1): draw the realized fallback action
     a_fb ~ fb(.|U_j); run K independent resettable Monte-Carlo rollouts of
     the reference tail and K of the fallback-then-reference tail (real
     stochastic simulation via e1_tabular.rollout_sim, NOT the exact DP
     value); form W_fb^+(U_j) via Theorem 4 with delta' = delta_G/(2 m_F).
  3. X_j = nH * W_fb^+(U_j) * F_j; Bhat_N = empirical_bernstein(X, delta_B, b0);
     capped Bhat_N^ = min(H*Delta_r, Bhat_N).
  4. AUDIT ONLY (uses exact DP, never feeds the certificate): does
     W_fb^+(U_j) >= exact Delta_+(U_j, a_fb) hold for every logged unit this
     repeat ("joint conservative event")? Does Bhat_N >= exact true loss L?

Claim verified:
  * The joint conservative event (Assumption 2) holds at rate >= 1-delta_G
    for EVERY K in the sweep (validity unconditional in K).
  * Loss-relative tightness (median Bhat_N / L) improves monotonically (or
    non-increasing) as K grows; Bhat_N / Rmax reported to show how much of
    the residual conservatism is the trivial range cap vs. genuine slack.

PASS <=> joint-conservative rate >= 1-delta_G-tol for all K in the sweep
         AND median tightness at the largest K <= median tightness at the
         smallest K (conservatism does not increase with more rollout budget).

Outputs:
    results/data/exp_12_rollout_bridge.csv        (per game x K x repeat)
    results/figs/exp_12_conservative_rate.pdf
    results/figs/exp_12_tightness.pdf
    one PASS/FAIL line -> results/summary.txt
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import certificates as C
from common.games import make_game
from common.io_utils import data_path, fig_path, write_summary
from common.plotting import new_fig, save_pdf
from e1_tabular._common import build, make_alpha_fn, const_fn
from e1_tabular.rollout_sim import mc_tail_returns_batch

DELTA_B = 0.05
DELTA_G = 0.05
K_GRID = [25, 50, 100, 200, 400]


def run_repeat(b, rng, m, K, delta_B, delta_G):
    """One certification repeat: m episodes, per-repeat delta' from m_F.

    Returns a dict of per-repeat results (Bhat, Bhat_capped, m_F,
    joint_conservative, ...).
    """
    game, table, rho = b.game, b.table, b.rho
    nH = game.n * game.H
    Rmax = game.H * game.delta_r
    B_Q = Rmax  # global tail-return range bound (Theorem 4 convention)

    keys = list(table.keys())
    probs = np.array([rho[k] for k in keys])
    probs = probs / probs.sum()
    idxs = rng.choice(len(keys), size=m, p=probs)

    # Pass 1: draw F_j and (for failures) the realized fallback action.
    logged = []  # list of (unit_key, a_fb, exact_delta_plus)
    F = np.zeros(m)
    for j in range(m):
        u = table[keys[idxs[j]]]
        endorsers = rng.binomial(u.N, u.alpha)
        f = 1.0 if endorsers <= np.floor(u.N / 2.0) else 0.0
        F[j] = f
        if f == 1.0:
            a_fb = int(rng.choice(game.A, p=u.fb))
            logged.append((j, keys[idxs[j]], a_fb, float(u.delta_plus[a_fb])))

    m_F = len(logged)
    delta_prime = C.delta_prime_union(delta_G, m_F)

    X = np.zeros(m)
    conservative_ok = True
    for (j, key, a_fb, exact_dplus) in logged:
        t, s, i, prefix = key
        ref_samples = mc_tail_returns_batch(game, rng, t, s, i, prefix, table[key].ref_i, K)
        fb_samples = mc_tail_returns_batch(game, rng, t, s, i, prefix, a_fb, K)
        Qref_hat = float(ref_samples.mean())
        Qfb_hat = float(fb_samples.mean())
        w_fb_plus = C.wfb_plus_from_rollouts(Qref_hat, Qfb_hat, K, delta_prime, B_Q)
        X[j] = nH * w_fb_plus * 1.0
        if w_fb_plus < exact_dplus - 1e-9:
            conservative_ok = False

    b0 = nH * B_Q
    Bhat = C.empirical_bernstein(X, delta_B, b0)
    Bhat_capped = min(Rmax, Bhat)

    return dict(
        m_F=m_F,
        Bhat=Bhat,
        Bhat_capped=Bhat_capped,
        true_loss=b.true_loss,
        Rmax=Rmax,
        joint_conservative=conservative_ok,
        ratio_L=Bhat / b.true_loss if b.true_loss > 1e-9 else np.nan,
        ratio_Rmax=Bhat / Rmax,
        ratio_L_capped=Bhat_capped / b.true_loss if b.true_loss > 1e-9 else np.nan,
    )


def run(n_games, repeats, m, seed, S, A, H, N, k_grid):
    rows = []
    master_rng = np.random.default_rng(seed)

    for gidx in tqdm(range(n_games), desc="exp_12 games"):
        gseed = int(master_rng.integers(0, 2**31))
        game = make_game(S=S, A=A, H=H, n=2, seed=gseed, transition="dependent")
        alpha_fn = make_alpha_fn(gseed)
        N_fn = const_fn(N)
        b = build(game, alpha_fn, N_fn, eta=0.0, psi_rule="ref")

        for K in k_grid:
            for rep in range(repeats):
                rep_rng = np.random.default_rng([seed, gidx, K, rep])
                res = run_repeat(b, rep_rng, m, K, DELTA_B, DELTA_G)
                res.update(game=gidx, K=K, repeat=rep)
                rows.append(res)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="exp_12: Theorem-4 conservative rollout bridge, audited against exact-DP truth")
    ap.add_argument("--n_games", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--m", type=int, default=2000, help="certification episodes per repeat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--S", type=int, default=4)
    ap.add_argument("--A", type=int, default=3)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--k_grid", default=",".join(str(k) for k in K_GRID))
    args = ap.parse_args()

    k_grid = [int(x) for x in args.k_grid.split(",")]

    df = run(args.n_games, args.repeats, args.m, args.seed,
              args.S, args.A, args.H, args.N, k_grid)

    csv_path = data_path("exp_12_rollout_bridge.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

    # --- Aggregate per K ---
    agg = df.groupby("K").agg(
        joint_conservative_rate=("joint_conservative", "mean"),
        median_ratio_L=("ratio_L", "median"),
        median_ratio_Rmax=("ratio_Rmax", "median"),
        median_ratio_L_capped=("ratio_L_capped", "median"),
        mean_m_F=("m_F", "mean"),
    ).reindex(k_grid)
    print("\nPer-K aggregate:")
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"))

    # --- PDF (left): joint conservative rate vs K ---
    fig, ax = new_fig()
    ax.plot(k_grid, agg["joint_conservative_rate"].values, "o-", color="steelblue",
            label="joint conservative rate")
    ax.axhline(1.0 - DELTA_G, color="k", ls="--", lw=1.0, label=f"1-δG = {1-DELTA_G:.2f} target")
    ax.set_xscale("log")
    ax.set_xlabel("rollout budget K")
    ax.set_ylabel("joint conservative rate")
    ax.set_ylim(0.80, 1.02)
    ax.set_title("Conservative rollout bridge: validity vs K")
    ax.legend(fontsize=9)
    save_pdf(fig, fig_path("exp_12_conservative_rate.pdf"))

    # --- PDF (right): tightness vs K ---
    fig, ax = new_fig()
    ax.plot(k_grid, agg["median_ratio_L"].values, "o-", color="darkorange", label=r"$\hat{B}_N/L$")
    ax.plot(k_grid, agg["median_ratio_Rmax"].values, "s--", color="seagreen", label=r"$\hat{B}_N/R_{max}$")
    ax.axhline(1.0, color="k", ls=":", lw=0.8, label="= 1 (trivial / tight)")
    ax.set_xscale("log")
    ax.set_xlabel("rollout budget K")
    ax.set_ylabel("median tightness ratio")
    ax.set_title("Conservative rollout bridge: tightness vs K")
    ax.legend(fontsize=9)
    save_pdf(fig, fig_path("exp_12_tightness.pdf"))

    # --- PASS/FAIL ---
    tol = 0.05  # statistical slack around the 1-delta_G target (finite repeats)
    validity_ok = bool((agg["joint_conservative_rate"] >= 1.0 - DELTA_G - tol).all())
    tight_first = agg["median_ratio_L"].iloc[0]
    tight_last = agg["median_ratio_L"].iloc[-1]
    monotone_ok = bool(tight_last <= tight_first)

    status = "PASS" if (validity_ok and monotone_ok) else "FAIL"
    rate_str = "  ".join(f"K={k}:rate={agg.loc[k,'joint_conservative_rate']:.3f}" for k in k_grid)
    tight_str = "  ".join(f"K={k}:B/L={agg.loc[k,'median_ratio_L']:.2f},B/Rmax={agg.loc[k,'median_ratio_Rmax']:.2f}" for k in k_grid)

    write_summary(
        f"exp_12 rollout_bridge [{status}] validity_ok={validity_ok} "
        f"monotone_tightening={monotone_ok}  {rate_str}  |  {tight_str}"
    )


if __name__ == "__main__":
    main()
