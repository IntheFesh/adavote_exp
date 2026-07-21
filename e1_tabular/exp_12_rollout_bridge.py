"""
exp_12_rollout_bridge.py — Theorem 4 conservative rollout value bound
(CORRECTED, conditional-mean / Jensen estimator) + Theorem 3 finite-sample
certificate, audited against exact-DP truth.

This reproduces the paper's Section 9.2 / Figure 6 "conservative rollout
bridge" result. Exact tabular DP is used only to AUDIT the rollout-based
bound against ground truth, never to compute the bound itself.

*** ESTIMATOR CORRECTION (supersedes the previous Hoeffding-radius version) ***
Old (discarded): W_fb^+(u) = min{B_Q, [Q_hat^ref - Q_hat^fb]_+ + 2*rad(K,delta')},
  delta' = delta_G/(2 m_F), validity argued via a per-unit high-probability
  "joint conservative event" (Assumption 2) at confidence delta_G.
New: W_tilde(u) = [Q_hat^ref(u,a^ref) - Q_hat^fb(u,a^fb)]_+, capped at B_Q
  for the clean range guarantee only (NOT a confidence radius -- there is no
  delta'/delta_G/m_F anywhere in this construction). Q_hat^ref, Q_hat^fb are
  empirical means of K independent resettable rollouts each (unbiased for
  every K >= 1). Since [.]_+ is convex, conditional Jensen gives
      E[W_tilde | u, F=1, a_fb] = E[[Y]_+] >= [E[Y]]_+ = Delta_+(u,a_fb)
  UNCONDITIONALLY (a deterministic inequality of expectations, verified by
  exact-Fraction toy enumeration including Delta<=0 boundary cases -- no
  probabilistic caveat, no per-unit union bound). Hence E[X_j] >= C2 >= L at
  the population level, and Theorem 3's empirical-Bernstein concentration
  over the m i.i.d. draws of X_j (delta_B only) is now the ONLY probabilistic
  layer: B_hat = mean(X) + EB_radius(X,delta_B) >= C2 >= L w.p. >= 1-delta_B.
  See common/certificates.py's "Conditional-mean conservative estimator"
  section for the full derivation and `wtilde_jensen_from_rollouts`.

Protocol per repeat (eta=0):
  1. Draw m i.i.d. certification episodes: each draws one unit U_j ~ mu
     (occupancy law) and one committee failure indicator F_j ~ Bernoulli(g(U_j)).
  2. For each LOGGED FAILED unit (F_j=1): draw the realized fallback action
     a_fb ~ fb(.|U_j); run K independent resettable Monte-Carlo rollouts of
     the reference tail and K of the fallback-then-reference tail (real
     stochastic simulation via e1_tabular.rollout_sim); form W_tilde(U_j) via
     the corrected Jensen estimator above (no rad, no delta').
  3. X_j = nH * W_tilde(U_j) * F_j; B_hat = empirical_bernstein(X, delta_B, b0);
     capped B_hat^ = min(H*Delta_r, B_hat).
  4. AUDIT ONLY (uses exact DP, never feeds the certificate): does
     B_hat_capped >= exact true loss L this repeat ("coverage")? Does the
     SAMPLE mean of X (Xbar) exceed the EXACT C2 for this game ("population
     conservatism", Xbar/C2 -- replaces the old "joint conservative event"
     rate, since that per-unit event no longer exists under this estimator)?

PASS <=> coverage rate >= 1-delta_B-tol for every K in the sweep
         AND median tightness at the largest K <= median tightness at the
         smallest K (conservatism does not increase with more rollout budget).

Outputs:
    results/data/exp_12_rollout_bridge.csv           (K-sweep, per game x K x repeat)
    results/data/exp_12_bound_comparison.csv         (4 bound types, m=5000, K=400 fixed)
    results/data/exp_12_m_sweep.csv                  (m in {500,1500,5000}, K=400 fixed)
    results/figs/consbound_rate_vs_K.pdf              (NEW: population conservatism Xbar/C2 vs K)
    results/figs/consbound_tightness_vs_K.pdf
    one PASS/FAIL line -> results/summary.txt (coverage audit reported explicitly)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from matplotlib.ticker import NullLocator
from tqdm import tqdm

from common import certificates as C
from common.baselines import hoeffding_bound, naive_plugin_mean, emp_bernstein_bound
from common.games import make_game
from common.io_utils import data_path, fig_path, write_summary
from common.plotting import new_fig, save_pdf
from e1_tabular._common import build, make_alpha_fn, const_fn
from e1_tabular.rollout_sim import mc_tail_returns_batch

DELTA_B = 0.05
K_GRID = [25, 50, 100, 200, 400]


def bootstrap_median_ci(values, n_boot=2000, alpha=0.05, rng=None):
    """Percentile-bootstrap 95% CI for the median of `values`. Used for the
    population-conservatism plot: if the CI at some K straddles 1, a
    median < 1 there is NOT a significant violation of the (population-level,
    exact) Jensen guarantee E[Xbar] >= C2 -- just sampling noise on top of an
    exact-in-expectation quantity."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(values)
    boot_medians = np.empty(n_boot)
    for b in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boot_medians[b] = np.median(sample)
    lo, hi = np.percentile(boot_medians, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.median(values)), float(lo), float(hi)


def clean_log_xaxis(ax, k_grid):
    """Explicit major ticks at exactly the K values used (25,50,100,200,400
    are not clean powers of 10), no minor ticks at all -- avoids matplotlib's
    default log-scale minor ticks (3x10^1, 4x10^1, 6x10^1, ...) cluttering
    and overlapping the axis."""
    ax.set_xticks(k_grid)
    ax.set_xticklabels([str(k) for k in k_grid])
    ax.xaxis.set_minor_locator(NullLocator())
BOUND_COMPARISON_K = 400   # fixed rollout budget for the m=5000 bound-type comparison
BOUND_COMPARISON_M = 5000
M_SWEEP = [500, 1500, 5000]
M_SWEEP_K = 400            # fixed rollout budget for the m-sweep


def draw_repeat_X(b, rng, m, K):
    """Draw one repeat's m certification episodes; return (X_jensen, X_exact,
    m_F, Xbar_jensen) where X_exact uses the EXACT Delta_+ (K->infinity
    oracle) instead of the rollout-based Jensen estimator, for the
    exact-value-baseline bound-type comparison."""
    game, table, rho = b.game, b.table, b.rho
    nH = game.n * game.H
    Rmax = game.H * game.delta_r
    B_Q = Rmax

    keys = list(table.keys())
    probs = np.array([rho[k] for k in keys])
    probs = probs / probs.sum()
    idxs = rng.choice(len(keys), size=m, p=probs)

    logged = []
    for j in range(m):
        u = table[keys[idxs[j]]]
        endorsers = rng.binomial(u.N, u.alpha)
        f = 1.0 if endorsers <= np.floor(u.N / 2.0) else 0.0
        if f == 1.0:
            a_fb = int(rng.choice(game.A, p=u.fb))
            logged.append((j, keys[idxs[j]], a_fb, float(u.delta_plus[a_fb])))
    m_F = len(logged)

    X = np.zeros(m)
    X_exact = np.zeros(m)
    for (j, key, a_fb, exact_dplus) in logged:
        t, s, i, prefix = key
        ref_samples = mc_tail_returns_batch(game, rng, t, s, i, prefix, table[key].ref_i, K)
        fb_samples = mc_tail_returns_batch(game, rng, t, s, i, prefix, a_fb, K)
        Qref_hat = float(ref_samples.mean())
        Qfb_hat = float(fb_samples.mean())
        w_tilde = C.wtilde_jensen_from_rollouts(Qref_hat, Qfb_hat, B_Q)
        X[j] = nH * w_tilde
        X_exact[j] = nH * exact_dplus

    return X, X_exact, m_F, Rmax, B_Q


def run_repeat(b, rng, m, K, delta_B):
    """One certification repeat under the corrected Jensen estimator."""
    nH = b.game.n * b.game.H
    C2_exact = b.chain["Ctilde"]
    X, X_exact, m_F, Rmax, B_Q = draw_repeat_X(b, rng, m, K)

    b0 = nH * B_Q
    Bhat = C.empirical_bernstein(X, delta_B, b0)
    Bhat_capped = min(Rmax, Bhat)
    Xbar = float(X.mean())

    return dict(
        m_F=m_F,
        Bhat=Bhat,
        Bhat_capped=Bhat_capped,
        true_loss=b.true_loss,
        Rmax=Rmax,
        C2_exact=C2_exact,
        covered=int(Bhat_capped >= b.true_loss - 1e-9),
        Xbar=Xbar,
        ratio_L=Bhat / b.true_loss if b.true_loss > 1e-9 else np.nan,
        ratio_Rmax=Bhat / Rmax,
        ratio_XC2=Xbar / C2_exact if C2_exact > 1e-9 else np.nan,
        jensen_pop_bound=C.jensen_population_bound_K(nH, B_Q, K),
    )


def run_ksweep(n_games, repeats, m, seed, S, A, H, N, k_grid):
    rows = []
    master_rng = np.random.default_rng(seed)
    builds = []
    for gidx in range(n_games):
        gseed = int(master_rng.integers(0, 2**31))
        game = make_game(S=S, A=A, H=H, n=2, seed=gseed, transition="dependent")
        alpha_fn = make_alpha_fn(gseed)
        b = build(game, alpha_fn, const_fn(N), eta=0.0, psi_rule="ref")
        builds.append(b)

    for gidx, b in enumerate(tqdm(builds, desc="exp_12 K-sweep games")):
        for K in k_grid:
            for rep in range(repeats):
                rep_rng = np.random.default_rng([seed, gidx, K, rep])
                res = run_repeat(b, rep_rng, m, K, DELTA_B)
                res.update(game=gidx, K=K, repeat=rep)
                rows.append(res)
    return pd.DataFrame(rows), builds


def run_bound_comparison(builds, repeats, seed, K, m):
    """Fig-5-style bound-type comparison: SAME m draws per repeat, 4 bound
    formulas evaluated on the same X (Jensen+EB, Jensen+Hoeffding,
    raw-plugin-mean) plus a 4th using the EXACT Delta_+ (X_exact) + EB, to
    isolate finite-K rollout noise from the m-sample concentration term."""
    rows = []
    for gidx, b in enumerate(tqdm(builds, desc=f"exp_12 bound comparison (K={K},m={m})")):
        for rep in range(repeats):
            rep_rng = np.random.default_rng([seed, gidx, 90001, rep])
            X, X_exact, m_F, Rmax, B_Q = draw_repeat_X(b, rep_rng, m, K)
            nH = b.game.n * b.game.H
            b0 = nH * B_Q
            tl = b.true_loss

            for bound_type, Xarr in [("jensen_eb", X), ("jensen_hoeffding", X),
                                      ("raw_plugin_mean", X), ("exact_value_eb", X_exact)]:
                if bound_type == "jensen_eb":
                    Bhat = emp_bernstein_bound(Xarr, DELTA_B, b0)
                elif bound_type == "jensen_hoeffding":
                    Bhat = hoeffding_bound(Xarr, DELTA_B, b0)
                elif bound_type == "raw_plugin_mean":
                    Bhat = naive_plugin_mean(Xarr, DELTA_B, b0)
                elif bound_type == "exact_value_eb":
                    Bhat = emp_bernstein_bound(Xarr, DELTA_B, b0)
                Bhat_capped = min(Rmax, Bhat)
                rows.append(dict(
                    game=gidx, repeat=rep, bound_type=bound_type,
                    Bhat=Bhat, Bhat_capped=Bhat_capped, true_loss=tl, Rmax=Rmax,
                    covered=int(Bhat_capped >= tl - 1e-9),
                    ratio_L=Bhat / tl if tl > 1e-9 else np.nan,
                    ratio_Rmax=Bhat / Rmax,
                ))
    return pd.DataFrame(rows)


def run_m_sweep(builds, repeats, seed, K, m_values):
    rows = []
    for m in tqdm(m_values, desc=f"exp_12 m-sweep (K={K})"):
        for gidx, b in enumerate(builds):
            for rep in range(repeats):
                rep_rng = np.random.default_rng([seed, gidx, 90002, m, rep])
                res = run_repeat(b, rep_rng, m, K, DELTA_B)
                res.update(game=gidx, m=m, repeat=rep)
                rows.append(res)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="exp_12: Theorem-4 conservative rollout bridge (corrected Jensen estimator), audited against exact-DP truth")
    ap.add_argument("--n_games", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--m", type=int, default=2000, help="certification episodes per repeat (K-sweep)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--S", type=int, default=4)
    ap.add_argument("--A", type=int, default=3)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--k_grid", default=",".join(str(k) for k in K_GRID))
    ap.add_argument("--skip_extra", action="store_true",
                     help="skip the bound-comparison and m-sweep passes (K-sweep only)")
    args = ap.parse_args()

    k_grid = [int(x) for x in args.k_grid.split(",")]

    df, builds = run_ksweep(args.n_games, args.repeats, args.m, args.seed,
                             args.S, args.A, args.H, args.N, k_grid)

    csv_path = data_path("exp_12_rollout_bridge.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

    # --- Aggregate per K ---
    agg = df.groupby("K").agg(
        coverage_rate=("covered", "mean"),
        median_ratio_L=("ratio_L", "median"),
        median_ratio_Rmax=("ratio_Rmax", "median"),
        median_ratio_XC2=("ratio_XC2", "median"),
        mean_m_F=("m_F", "mean"),
        jensen_pop_bound=("jensen_pop_bound", "mean"),
    ).reindex(k_grid)
    print("\nPer-K aggregate (coverage audit is the headline check):")
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"))

    # --- PDF (left, NEW): population conservatism Xbar/C2 vs K, with 95% bootstrap CI ---
    boot_rng = np.random.default_rng(args.seed + 1_000_003)
    xc2_med, xc2_lo, xc2_hi = [], [], []
    for K in k_grid:
        vals = df.loc[df.K == K, "ratio_XC2"].values
        med, lo, hi = bootstrap_median_ci(vals, rng=boot_rng)
        xc2_med.append(med); xc2_lo.append(lo); xc2_hi.append(hi)
    xc2_med, xc2_lo, xc2_hi = np.array(xc2_med), np.array(xc2_lo), np.array(xc2_hi)
    yerr = np.vstack([xc2_med - xc2_lo, xc2_hi - xc2_med])

    fig, ax = new_fig()
    ax.errorbar(k_grid, xc2_med, yerr=yerr, fmt="o-", color="steelblue", capsize=4, lw=1.5,
                label=r"median $\bar{X}/C_2$ (95% bootstrap CI)")
    ax.axhline(1.0, color="k", ls="--", lw=1.2, label="population target = 1 (Jensen exact)")
    ax.set_xscale("log")
    clean_log_xaxis(ax, k_grid)
    ax.set_xlabel("rollout budget K")
    ax.set_ylabel(r"median $\bar{X}/C_2$")
    ax.set_title("Conservative rollout bridge: population conservatism vs K\n"
                 "(CI crossing 1 = not significantly below the population target)")
    ax.legend(fontsize=8)
    save_pdf(fig, fig_path("consbound_rate_vs_K.pdf"))
    print("\nPopulation conservatism 95% CI per K:")
    for K, med, lo, hi in zip(k_grid, xc2_med, xc2_lo, xc2_hi):
        crosses = "crosses 1 (not significant)" if lo <= 1.0 <= hi else "does not cross 1"
        print(f"  K={K}: median={med:.4f}  CI=[{lo:.4f}, {hi:.4f}]  {crosses}")

    # --- PDF (right): tightness vs K ---
    fig, ax = new_fig()
    ax.plot(k_grid, agg["median_ratio_L"].values, "o-", color="darkorange", label=r"$\hat{B}_N/L$")
    ax.plot(k_grid, agg["median_ratio_Rmax"].values, "s--", color="seagreen", label=r"$\hat{B}_N/R_{max}$")
    ax.axhline(1.0, color="k", ls=":", lw=0.8, label="= 1 (trivial / tight)")
    ax.set_xscale("log")
    clean_log_xaxis(ax, k_grid)
    ax.set_xlabel("rollout budget K")
    ax.set_ylabel("median tightness ratio")
    ax.set_title("Conservative rollout bridge: tightness vs K")
    ax.legend(fontsize=9)
    save_pdf(fig, fig_path("consbound_tightness_vs_K.pdf"))

    # --- coverage audit (headline validity check) ---
    tol = 0.05
    coverage_ok = bool((agg["coverage_rate"] >= 1.0 - DELTA_B - tol).all())
    tight_first = agg["median_ratio_L"].iloc[0]
    tight_last = agg["median_ratio_L"].iloc[-1]
    monotone_ok = bool(tight_last <= tight_first)
    status = "PASS" if (coverage_ok and monotone_ok) else "FAIL"

    cov_str = "  ".join(f"K={k}:cov={agg.loc[k,'coverage_rate']:.4f}" for k in k_grid)
    tight_str = "  ".join(f"K={k}:B/L={agg.loc[k,'median_ratio_L']:.4f},B/Rmax={agg.loc[k,'median_ratio_Rmax']:.4f}" for k in k_grid)
    xc2_str = "  ".join(f"K={k}:Xbar/C2={agg.loc[k,'median_ratio_XC2']:.4f}" for k in k_grid)

    if not coverage_ok:
        print("\n*** COVERAGE AUDIT FAILED: at least one K has coverage_rate < 1-delta_B-tol ***")
        print(cov_str)

    write_summary(
        f"exp_12 rollout_bridge [{status}] estimator=jensen_conditional_mean "
        f"coverage_ok={coverage_ok} monotone_tightening={monotone_ok}  "
        f"COVERAGE: {cov_str}  |  TIGHTNESS: {tight_str}  |  POP_CONSERVATISM: {xc2_str}"
    )

    if args.skip_extra:
        return

    # --- bound-type comparison (Fig 5 style), m=5000, K=400 fixed ---
    df_bc = run_bound_comparison(builds, args.repeats, args.seed, BOUND_COMPARISON_K, BOUND_COMPARISON_M)
    df_bc.to_csv(data_path("exp_12_bound_comparison.csv"), index=False)
    agg_bc = df_bc.groupby("bound_type").agg(
        coverage=("covered", "mean"), median_ratio_L=("ratio_L", "median"),
    ).reindex(["jensen_eb", "jensen_hoeffding", "raw_plugin_mean", "exact_value_eb"])
    print(f"\nBound-type comparison (m={BOUND_COMPARISON_M}, K={BOUND_COMPARISON_K}):")
    print(agg_bc.to_string(float_format=lambda x: f"{x:.4f}"))
    bc_str = "  ".join(
        f"{bt}:cov={agg_bc.loc[bt,'coverage']:.4f},B/L={agg_bc.loc[bt,'median_ratio_L']:.4f}"
        for bt in agg_bc.index
    )
    write_summary(f"exp_12 bound_comparison [m={BOUND_COMPARISON_M},K={BOUND_COMPARISON_K}]  {bc_str}")

    # --- m-sweep, K=400 fixed ---
    df_ms = run_m_sweep(builds, args.repeats, args.seed, M_SWEEP_K, M_SWEEP)
    df_ms.to_csv(data_path("exp_12_m_sweep.csv"), index=False)
    agg_ms = df_ms.groupby("m").agg(
        coverage_rate=("covered", "mean"), median_ratio_L=("ratio_L", "median"),
    ).reindex(M_SWEEP)
    print(f"\nm-sweep (K={M_SWEEP_K}):")
    print(agg_ms.to_string(float_format=lambda x: f"{x:.4f}"))
    ms_str = "  ".join(
        f"m={m}:cov={agg_ms.loc[m,'coverage_rate']:.4f},B/L={agg_ms.loc[m,'median_ratio_L']:.4f}"
        for m in M_SWEEP
    )
    write_summary(f"exp_12 m_sweep [K={M_SWEEP_K}]  {ms_str}")


if __name__ == "__main__":
    main()
