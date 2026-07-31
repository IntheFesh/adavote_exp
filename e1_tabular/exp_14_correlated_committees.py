"""
exp_14_correlated_committees.py — §9.3 correlated-committee certificate via
tabular Beta-Binomial over-dispersion (de Finetti mixture).

REBUILT DRIVER, v2 (Task 7 rework). v1 (see git history) modeled `s` as a
pilot-sample count used to noisily ESTIMATE alpha, holding the population
correlation strength fixed -- this gave an under-coverage curve running in
the OPPOSITE direction from the paper (checked directly, not assumed: v1's
rate ran high-at-large-s / low-at-small-s). Task 7's governing instructions
gave an exact mathematical diagnosis (their §2.1) and confirmed, with a
reference table of E_Theta[g_5(Theta)] at alpha_bar=0.7 for
s in {500,200,100,50,30,10,5,2}, that the correct model has NO estimation
step at all: `s` is the CONCENTRATION parameter of the underlying Beta
mixing distribution itself (Theta ~ Beta(alpha_bar*s, (1-alpha_bar)*s), so
Var(Theta) = alpha_bar(1-alpha_bar)/(s+1) -- shrinks as s GROWS). This was
verified independently before writing any code: recomputing that exact
reference table via scipy.integrate.quad reproduced all 8 values to full
displayed precision (0.164131, 0.165683, 0.168202, 0.173004, 0.178961,
0.202797, 0.226563, 0.261248).

v1's other bug (per Task 7 §2.2 point 1, "true loss 是否用了 de Finetti
混合"): true_loss was NOT recomputed under the mixture at all in v1 --
only abstract g values were compared in isolation, with no connection to
an actual deployment loss. v2 fixes this by actually rebuilding each
game's unit table with g(u) replaced by the mixture value (the recovered
bytecode's rebuild_with_oracle_g, now correctly understood: "oracle g" =
E_Theta[g_N(Theta)], not a second fixed-kappa Beta-Binomial as v1 assumed),
then recomputing true_loss and cert_Ctilde end-to-end via the existing
controller_value/true_loss/cert_Ctilde harness -- both the certificate
formula (Sum rho*g*W) and the controller's ACTUAL simulated behavior
(controller_state_dist, which uses u.g directly) are driven by the same
mixture g, so true_loss genuinely grows as s shrinks.

Model
-----
Standard unit table (common/certificates.py build_unit_table) gives every
unit u its own eligible alpha(u) drawn by make_alpha_fn (in [0.55,0.9], the
range this codebase always uses -- also exactly the range where g_5 is
strictly convex, g_5''(alpha) = -60 alpha(1-alpha)(1-2alpha) > 0 for
alpha>1/2, which is what makes the Jensen gap below positive).

For a given committee-size N and concentration s:
  g_plugin(u)  = g_N(N, alpha(u))                          [standard, s-free]
  g_mixture(u, s) = E_Theta[ g_N(N, Theta) ],  Theta ~ Beta(alpha(u)*s, (1-alpha(u))*s)
                 (exact numerical integration, scipy.integrate.quad)

Two tables per (game, s):
  - "mixture table": every unit's g field replaced (dataclasses.replace)
    by g_mixture(u,s). true_loss_mixture(game,s) and
    Ctilde_mixture(game,s) are both computed from THIS table -- by the
    same Theorem-1 chain argument used everywhere else in this codebase
    (which does not care what specific g values are used, only that they
    are used consistently), Ctilde_mixture must dominate true_loss_mixture
    with zero violations.
  - "plug-in table": the UNTOUCHED standard table (g(u) = g_N(N,alpha(u)),
    s-independent). Ctilde_plugin(game) is a SINGLE per-game number,
    computed once, that does NOT know about s or correlation.

Coverage / under-coverage, aggregated over n_games games:
  - mixture coverage      = frac[ Ctilde_mixture(game,s) >= true_loss_mixture(game,s) ]
  - plug-in under-coverage = frac[ Ctilde_plugin(game)    <  true_loss_mixture(game,s) ]

As s shrinks, true_loss_mixture grows (more correlation -> effectively
higher failure probability -> worse deployment loss) while Ctilde_plugin
stays fixed per game (it never accounted for correlation), so the naive
certificate should increasingly fail to cover -- this is exactly Task 7's
required direction, and follows from the same de Finetti-mixture
mechanism as the exact E_Theta[g_5(Theta)] - g_5(alpha_bar) gap verified
above, not from any per-s tuning.

Outputs:
    results/data/exp_14_correlated_committees.csv   (per-s intermediate quantities, Task 7 §2.3)
    results/data/exp_14_per_game.csv                (per-(game,s) raw detail)
    results/e2/exp_14_summary.json (not applicable; e2/ is E2/MPE only -- see results/summary.txt)
    one summary line -> results/summary.txt
"""

from __future__ import annotations

import argparse
import dataclasses

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
N_GAMES = 8
S = 4
A = 3
H = 4


def fail_prob_given_theta(theta: float, N: int) -> float:
    return float(binom.cdf(np.floor(N / 2.0), N, theta))


def g_mixture(N: int, alpha: float, s: float) -> float:
    """E_Theta[ g_N(N,Theta) ], Theta ~ Beta(alpha*s, (1-alpha)*s)."""
    a, b = alpha * s, (1.0 - alpha) * s
    val, _err = quad(lambda th: fail_prob_given_theta(th, N) * beta_dist.pdf(th, a, b),
                      0.0, 1.0, limit=200)
    return float(val)


def rebuild_with_oracle_g(table, g_fn):
    """Return a new unit table with every unit's g field replaced by
    g_fn(unit_alpha). Q/delta_plus/W/Wtilde/fb/psi are untouched (they do
    not depend on g)."""
    return {k: dataclasses.replace(u, g=float(g_fn(u.alpha))) for k, u in table.items()}


def run(n_games: int, s_values, N_committee: int, seed: int, S: int, A: int, H: int, N: int):
    master_rng = np.random.default_rng(seed)
    games = []
    for gidx in range(n_games):
        gseed = int(master_rng.integers(0, 2**31))
        game = make_game(S=S, A=A, H=H, n=2, seed=gseed, transition="dependent")
        alpha_fn = make_alpha_fn(gseed)
        b = build(game, alpha_fn, const_fn(N), eta=0.0, psi_rule="ref")
        # Plug-in certificate: the STANDARD, s-independent table.
        Ctilde_plugin = C.cert_Ctilde(game, b.table, b.rho)
        games.append((gidx, game, b, Ctilde_plugin))

    rows_summary = []
    rows_detail = []
    for s in tqdm(s_values, desc="exp_14 s-sweep"):
        true_losses, ctildes_mix, g_mix_means, g_plugin_means = [], [], [], []
        n_mix_violations = 0
        n_plugin_undercov = 0
        for gidx, game, b, Ctilde_plugin in games:
            g_fn = lambda alpha, s=s, N=N_committee: g_mixture(N, alpha, s)
            table_mix = rebuild_with_oracle_g(b.table, g_fn)
            V_ctrl_mix, d_ctrl_mix = C.controller_value(game, table_mix)
            rho_mix = C.unit_occupancy(game, table_mix, d_ctrl_mix)
            true_loss_mix = C.true_loss(game, b.V_ref, V_ctrl_mix)
            Ctilde_mix = C.cert_Ctilde(game, table_mix, rho_mix)

            g_plugin_mean = float(np.mean([u.g for u in b.table.values()]))
            g_mix_mean = float(np.mean([table_mix[k].g for k in table_mix]))

            mix_covered = Ctilde_mix >= true_loss_mix - 1e-9
            plugin_covered = Ctilde_plugin >= true_loss_mix - 1e-9
            if not mix_covered:
                n_mix_violations += 1
            if not plugin_covered:
                n_plugin_undercov += 1

            true_losses.append(true_loss_mix)
            ctildes_mix.append(Ctilde_mix)
            g_mix_means.append(g_mix_mean)
            g_plugin_means.append(g_plugin_mean)

            rows_detail.append(dict(
                s=int(s), game=gidx, true_loss_mixture=true_loss_mix,
                Ctilde_mixture=Ctilde_mix, Ctilde_plugin=Ctilde_plugin,
                mixture_covered=int(mix_covered), plugin_covered=int(plugin_covered),
                g_mixture_mean=g_mix_mean, g_plugin_mean=g_plugin_mean,
            ))

        n = len(games)
        rows_summary.append(dict(
            s=int(s),
            mean_g_mixture=float(np.mean(g_mix_means)),
            mean_g_plugin=float(np.mean(g_plugin_means)),
            gap_mixture_minus_plugin=float(np.mean(g_mix_means) - np.mean(g_plugin_means)),
            mean_true_loss=float(np.mean(true_losses)),
            mean_Ctilde_mixture=float(np.mean(ctildes_mix)),
            mixture_coverage=1.0 - n_mix_violations / n,
            mixture_violations=n_mix_violations,
            plugin_under_coverage_rate=n_plugin_undercov / n,
        ))
    return pd.DataFrame(rows_summary), pd.DataFrame(rows_detail)


def main():
    ap = argparse.ArgumentParser(
        description="exp_14 v2: correlated-committee certificate via de Finetti Beta mixture")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_games", type=int, default=N_GAMES)
    ap.add_argument("--N_committee", type=int, default=N_COMMITTEE)
    ap.add_argument("--S", type=int, default=S)
    ap.add_argument("--A", type=int, default=A)
    ap.add_argument("--H", type=int, default=H)
    ap.add_argument("--N", type=int, default=5)
    args = ap.parse_args()

    df, df_detail = run(args.n_games, S_VALUES, args.N_committee, args.seed,
                         args.S, args.A, args.H, args.N)
    csv_path = data_path("exp_14_correlated_committees.csv")
    df.to_csv(csv_path, index=False)
    detail_path = data_path("exp_14_per_game.csv")
    df_detail.to_csv(detail_path, index=False)

    # Self-assertion: headline numbers recomputed from the just-written CSVs.
    df_check = pd.read_csv(csv_path)
    assert np.allclose(df_check["plugin_under_coverage_rate"].values,
                        df["plugin_under_coverage_rate"].values)
    df_detail_check = pd.read_csv(detail_path)
    recomputed_mixture_cov = (
        df_detail_check.groupby("s")["mixture_covered"].mean().reindex(df["s"]).values)
    assert np.allclose(recomputed_mixture_cov, df["mixture_coverage"].values)

    print(df.to_string(index=False))

    # Acceptance checks (Task 7 §2.4) -- report, do not silently pass/fail.
    mixture_zero_viol = bool((df["mixture_violations"] == 0).all())
    # s in descending order in S_VALUES already (500 -> 2): check monotone non-decreasing
    # under-coverage / true_loss / gap as s DECREASES, i.e. non-increasing in s-order.
    def monotone_nondecreasing_as_s_shrinks(col):
        vals = df.sort_values("s", ascending=False)[col].values  # s: 500 -> 2
        return bool(np.all(np.diff(vals) >= -1e-9))

    undercov_ok = monotone_nondecreasing_as_s_shrinks("plugin_under_coverage_rate")
    trueloss_ok = monotone_nondecreasing_as_s_shrinks("mean_true_loss")
    gap_ok = monotone_nondecreasing_as_s_shrinks("gap_mixture_minus_plugin")

    print(f"\nAcceptance (Task 7 §2.4): mixture_zero_violations={mixture_zero_viol} "
          f"undercov_monotone={undercov_ok} true_loss_monotone={trueloss_ok} "
          f"gap_monotone={gap_ok}")

    verdict = "PASS" if (mixture_zero_viol and undercov_ok and trueloss_ok and gap_ok) else "FAIL"
    s_list = ",".join(f"s={r.s}:undercov={r.plugin_under_coverage_rate:.2f}"
                       for r in df.itertuples())
    write_summary(
        f"exp_14 correlated_committees [{verdict},v2-de-Finetti-mixture] "
        f"mixture_zero_viol={mixture_zero_viol} undercov_monotone={undercov_ok} "
        f"true_loss_monotone={trueloss_ok} gap_monotone={gap_ok}  {s_list}  "
        f"true_loss_range=[{df['mean_true_loss'].min():.3f},{df['mean_true_loss'].max():.3f}]")


if __name__ == "__main__":
    main()
