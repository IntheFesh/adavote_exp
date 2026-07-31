"""
exp_14_correlated_committees.py — §9.3 correlated-committee certificate via
tabular Beta-Binomial over-dispersion.

REBUILT DRIVER. The original source file is not present in this repository
or any branch (git history has no record of it under any commit); only a
stray __pycache__/exp_14_correlated_committees.cpython-311.pyc bytecode
cache survived in the working tree, from which this file's original
docstring, the S_VALUES grid (500,300,200,150,100,75,50,30,20,10,5,2), and
these function names were recovered by direct code-object inspection
(marshal.load on the .pyc, NOT execution or decompilation):
  fail_prob_given_theta, oracle_g_betabinom, rebuild_with_oracle_g,
  cert_tight, run_one_game, run.
The recovered names list also confirmed the original imported scipy.stats
binom and beta (aliased beta_dist) and scipy.integrate.quad -- exactly the
tools this reconstruction uses, which is corroborating (not conclusive)
evidence this rebuild's approach matches the original's.

What is NOT recoverable from bytecode is what "s" ranges over and how
correlation was modeled. Reconstructed from the paper's own description
(Task 5 §3.2 in the governing instructions): a "binomial plug-in"
certificate whose under-coverage rate rises from ~0.00 at s=500 to 1.00 at
s<=30 as s shrinks -- interpreted here as s = number of PILOT committee
queries used to estimate the endorsement rate alpha before certifying,
since that is the natural quantity whose shrinkage degrades a plug-in
estimate monotonically to certain failure, matching the described curve
shape (not verified against the original; a plausible, self-consistent
reconstruction).

Model
-----
Standard certificate machinery (common.certificates.g_N) assumes N
committee members endorse i.i.d. Bernoulli(alpha). This experiment asks
what happens if members are correlated via a shared per-instance nuisance
theta (over-dispersion): the REALIZED endorsement probability for a given
committee query is theta ~ Beta(alpha*kappa, (1-alpha)*kappa) (mean alpha,
concentration kappa; kappa -> infinity recovers the i.i.d. case), and,
given theta, the N members' votes are Binomial(N, theta).

  fail_prob_given_theta(theta, N) = P(Binomial(N,theta) <= floor(N/2))
                                   = binom.cdf(floor(N/2), N, theta)

  oracle_g_betabinom(N, alpha, kappa)
      = E_theta[ fail_prob_given_theta(theta, N) ],  theta ~ Beta(alpha*kappa,(1-alpha)*kappa)
      computed by exact numerical integration (scipy.integrate.quad).

A practitioner unaware of the correlation estimates alpha from s pilot
committee queries (each an independent theta draw + one Binomial(N,theta)
trial), alpha_hat = mean fraction of endorsers over the s queries, then
certifies using the STANDARD independence-assuming g_N(N, alpha_hat)
(common.certificates.g_N -- reused, not reimplemented). As s shrinks,
alpha_hat is noisier and g_N never accounts for the correlation the oracle
does, so the naive plug-in increasingly UNDER-covers oracle_g_betabinom.

rebuild_with_oracle_g / cert_tight: for context (matching the paper's
"true loss 0.371 to 0.544" line), a small set of standard tabular games
(same construction as exp_01/03) are built and their EXACT true_loss is
reported alongside the abstract g-sweep above; the two are reported
side by side, not entangled (the true_loss range is deployment-loss
context, not a direct input to the naive-vs-oracle comparison).

HONEST CAVEAT (read before citing this experiment's numbers): this
reconstruction's under-coverage rate moves in the OPPOSITE direction from
the paper's description -- it is HIGH (~0.95) at s=500 and LOWER (~0.5-0.66)
at small s, not low-at-large-s / high-at-small-s as described. The reason is
mathematically clear and was verified directly (not guessed around): for
alpha > 0.5 (the eligible regime this codebase always uses), the naive
independence-assuming g_N(N,alpha) is a fixed, systematic UNDERESTIMATE of
the true correlated oracle_g_betabinom(N,alpha,kappa) -- a bias in the model
assumption itself (over-dispersion inflates the majority-failure
probability), which does NOT shrink as more pilot data (larger s) is used
to estimate alpha; more pilot data only removes the ESTIMATION noise that
occasionally, by chance, pushes alpha_hat low enough for g_naive to exceed
g_oracle. So under-coverage is *highest* exactly when the pilot estimate is
*most* accurate. A second candidate reconstruction was also tried (S_VALUES
as swept committee size N with alpha held fixed or swept over the codebase's
usual [0.55,0.9] eligible range): it gives an even worse mismatch --
under-coverage pinned near 1.0 for essentially the entire N range, with no
smooth 0-to-1 transition at all. Neither candidate reproduces the paper's
described curve; this is reported honestly as an unresolved reconstruction
gap (see Task 5 report) rather than forcing a match by tuning kappa/alpha
post hoc. The kept (first) design is retained because it at least produces
a genuine, non-degenerate rate (not pinned at 0 or 1) and is the more
directly bytecode-motivated of the two.

Outputs:
    results/data/exp_14_correlated_committees.csv
    results/data/exp_14_true_loss_context.csv
    one summary line -> results/summary.txt
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import beta as beta_dist
from scipy.stats import binom
from tqdm import tqdm

from common import certificates as C
from common.games import make_game
from common.io_utils import data_path, write_summary
from e1_tabular._common import build, make_alpha_fn, const_fn

N_COMMITTEE = 5
S_VALUES = (500, 300, 200, 150, 100, 75, 50, 30, 20, 10, 5, 2)
ALPHA_TRUE = 0.7
KAPPA = 20.0
N_TRIALS = 400


def fail_prob_given_theta(theta: float, N: int) -> float:
    return float(binom.cdf(np.floor(N / 2.0), N, theta))


def oracle_g_betabinom(N: int, alpha: float, kappa: float) -> float:
    a, b = alpha * kappa, (1.0 - alpha) * kappa
    val, _err = quad(lambda th: fail_prob_given_theta(th, N) * beta_dist.pdf(th, a, b),
                      0.0, 1.0, limit=200)
    return float(val)


def run_one_trial(N: int, alpha: float, kappa: float, s: int, rng: np.random.Generator):
    a, b = alpha * kappa, (1.0 - alpha) * kappa
    thetas = rng.beta(a, b, size=s)
    endorsers = rng.binomial(N, thetas)
    alpha_hat = float(endorsers.mean() / N)
    g_naive = C.g_N(N, alpha_hat)
    return g_naive, alpha_hat


def run(N: int, alpha: float, kappa: float, s_values, n_trials: int, seed: int):
    g_oracle = oracle_g_betabinom(N, alpha, kappa)
    rows = []
    for s in tqdm(s_values, desc="exp_14 s-sweep"):
        under = 0
        g_naives = []
        for trial in range(n_trials):
            rng = np.random.default_rng([seed, int(s), trial])
            g_naive, alpha_hat = run_one_trial(N, alpha, kappa, int(s), rng)
            g_naives.append(g_naive)
            if g_naive < g_oracle - 1e-9:
                under += 1
        rows.append(dict(
            s=int(s), N=N, alpha=alpha, kappa=kappa, g_oracle=g_oracle,
            mean_g_naive=float(np.mean(g_naives)),
            median_g_naive=float(np.median(g_naives)),
            under_coverage_rate=under / n_trials,
        ))
    return pd.DataFrame(rows), g_oracle


def true_loss_context(n_games: int, seed: int, S: int, A: int, H: int, N: int):
    """Exact deployment-loss range across standard tabular games, for
    context alongside the abstract naive-vs-oracle g sweep above (see
    docstring: reported side by side, not entangled)."""
    rows = []
    master_rng = np.random.default_rng(seed)
    for gidx in range(n_games):
        gseed = int(master_rng.integers(0, 2**31))
        game = make_game(S=S, A=A, H=H, n=2, seed=gseed, transition="dependent")
        alpha_fn = make_alpha_fn(gseed)
        b = build(game, alpha_fn, const_fn(N), eta=0.0, psi_rule="ref")
        rows.append(dict(game=gidx, true_loss=b.true_loss))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="exp_14: correlated-committee (Beta-Binomial over-dispersion) certificate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--N_committee", type=int, default=N_COMMITTEE)
    ap.add_argument("--alpha", type=float, default=ALPHA_TRUE)
    ap.add_argument("--kappa", type=float, default=KAPPA)
    ap.add_argument("--n_trials", type=int, default=N_TRIALS)
    ap.add_argument("--n_games", type=int, default=8)
    ap.add_argument("--S", type=int, default=4)
    ap.add_argument("--A", type=int, default=3)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--N", type=int, default=5)
    args = ap.parse_args()

    df, g_oracle = run(args.N_committee, args.alpha, args.kappa, S_VALUES,
                        args.n_trials, args.seed)
    csv_path = data_path("exp_14_correlated_committees.csv")
    df.to_csv(csv_path, index=False)

    df_tl = true_loss_context(args.n_games, args.seed, args.S, args.A, args.H, args.N)
    tl_path = data_path("exp_14_true_loss_context.csv")
    df_tl.to_csv(tl_path, index=False)

    # Self-assertion: recompute headline numbers from the just-written CSVs.
    df_check = pd.read_csv(csv_path)
    assert np.allclose(df_check["under_coverage_rate"].values, df["under_coverage_rate"].values)
    df_tl_check = pd.read_csv(tl_path)
    assert np.isclose(df_tl_check["true_loss"].min(), df_tl["true_loss"].min())
    assert np.isclose(df_tl_check["true_loss"].max(), df_tl["true_loss"].max())

    print(df.to_string(index=False))
    print(f"g_oracle={g_oracle:.4f}  true_loss range=[{df_tl['true_loss'].min():.3f},"
          f"{df_tl['true_loss'].max():.3f}]")

    s_list = ",".join(f"s={r.s}:undercov={r.under_coverage_rate:.2f}" for r in df.itertuples())
    write_summary(
        f"exp_14 correlated_committees [REBUILT,DIRECTION-MISMATCH-vs-paper,see docstring] "
        f"[N={args.N_committee},alpha={args.alpha},kappa={args.kappa}] "
        f"g_oracle={g_oracle:.4f}  {s_list}  "
        f"true_loss_range=[{df_tl['true_loss'].min():.3f},{df_tl['true_loss'].max():.3f}]")


if __name__ == "__main__":
    main()
