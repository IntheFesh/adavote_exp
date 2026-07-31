"""
tests/test_a3_a4_exp14_smoke.py — smoke tests for the Task-5 rebuilt drivers
(exp_a3_rare_unit_scaling, exp_a4_adjacent_baselines, exp_14_correlated_committees).

These are driver-level scripts (not shared common/ primitives), so the tests
here check basic self-consistency and validity properties rather than
re-deriving formulas already covered by tests/test_certificates.py.
"""

import numpy as np

from e1_tabular.exp_a3_rare_unit_scaling import (
    n_min_from_hoeffding, m_required_for_visits, fit_loglog_slope, run as run_a3,
)
from e1_tabular.exp_a4_adjacent_baselines import run as run_a4
from e1_tabular.exp_14_correlated_committees import (
    fail_prob_given_theta, g_mixture, run as run_exp14,
)


def test_a3_m_required_increases_as_p_shrinks():
    df, n_min = run_a3([0.1, 0.03, 0.01], eps_frac=0.1, delta_prime=0.05,
                        delta_visit=0.05, B_Q=1.0)
    m = df["m_required"].values
    assert np.all(np.diff(m) > 0), "m_required must strictly increase as p shrinks"
    assert n_min >= 2


def test_a3_m_required_at_least_naive_division():
    # Confidence-adjusted requirement must be >= the naive n_min/p (it accounts
    # for tail risk on top of the naive expectation-based count).
    df, n_min = run_a3([0.1, 0.01], eps_frac=0.1, delta_prime=0.05, delta_visit=0.05, B_Q=1.0)
    for row in df.itertuples():
        assert row.m_required >= row.naive_n_min_over_p


def test_a3_slope_near_one_for_tail_dominated_regime():
    df, _ = run_a3([0.1, 0.03, 0.01], eps_frac=0.1, delta_prime=0.05, delta_visit=0.05, B_Q=1.0)
    slope, r2 = fit_loglog_slope(df)
    assert 0.9 <= slope <= 1.3
    assert r2 > 0.9


def test_a4_I2_oracle_always_covers_exactly():
    df = run_a4(n_games=2, repeats=3, m=200, seed=0, S=4, A=3, H=4, N=5)
    i2o = df[df.method == "I2_O"]
    assert (i2o.covered == 1).all(), "I2_O (population C2/L) must always cover by Theorem 1"
    # I2_O is deterministic per game (no sampling): constant across repeats within a game.
    for _, sub in i2o.groupby("game"):
        assert sub["bound"].nunique() == 1


def test_a4_headline_methods_present_and_finite():
    df = run_a4(n_games=2, repeats=3, m=200, seed=0, S=4, A=3, H=4, N=5)
    for method in ["I2_O", "I2_B", "PDIS", "DR", "robust_sim_lemma"]:
        sub = df[df.method == method]
        assert len(sub) > 0
        assert np.isfinite(sub["bound"]).all()
        assert (sub["bound"] >= 0).all()


def test_exp14_g_mixture_matches_direct_binomial_at_large_s():
    # As s grows, Theta ~ Beta(alpha*s, (1-alpha)*s) concentrates at alpha, so
    # the mixture should approach the plain Binomial g_N(N, alpha). s is kept
    # moderate (not e.g. 1e6): at extreme concentration the Beta density
    # becomes a needle-thin spike that scipy.integrate.quad's default adaptive
    # sampling can miss entirely (verified directly: at s=1e6 quad returns
    # ~1e-172; a 20M-draw Monte Carlo cross-check at s=20 confirmed quad IS
    # accurate in the regime exp_14 actually uses -- a quad robustness limit
    # at extreme concentration, not a bug reachable by exp_14's own S_VALUES).
    from common.certificates import g_N
    N, alpha = 7, 0.7
    g_direct = g_N(N, alpha)
    g_mix_large_s = g_mixture(N, alpha, s=5000.0)
    assert abs(g_direct - g_mix_large_s) < 1e-3


def test_exp14_g_mixture_exceeds_plugin_for_alpha_above_half():
    # Jensen: g_N is strictly convex for alpha>1/2 (g_N''(alpha) =
    # -60 alpha(1-alpha)(1-2alpha) > 0 there for N=5), so the de Finetti
    # mixture must exceed the plug-in value, with the gap GROWING as s shrinks
    # (Var(Theta) = alpha(1-alpha)/(s+1) grows as s shrinks). This is the
    # exact mechanism Task 7 diagnosed as missing from exp_14 v1.
    from common.certificates import g_N
    N, alpha = 5, 0.7
    g_plugin = g_N(N, alpha)
    gaps = []
    for s in [500, 100, 30, 5, 2]:
        gap = g_mixture(N, alpha, s) - g_plugin
        assert gap > 0
        gaps.append(gap)
    # gaps computed at s descending (500->2): gap must be non-decreasing.
    assert np.all(np.diff(gaps) > 0)


def test_exp14_fail_prob_given_theta_matches_binom_cdf():
    from scipy.stats import binom
    N = 9
    for theta in [0.1, 0.4, 0.5, 0.6, 0.9]:
        assert np.isclose(fail_prob_given_theta(theta, N),
                          binom.cdf(np.floor(N / 2), N, theta))


def test_exp14_run_mixture_certificate_never_violates():
    df, df_detail = run_exp14(n_games=2, s_values=[100, 5], N_committee=5,
                               seed=0, S=4, A=3, H=4, N=5)
    assert (df["mixture_violations"] == 0).all()
    assert ((df["plugin_under_coverage_rate"] >= 0) &
            (df["plugin_under_coverage_rate"] <= 1)).all()


def test_exp14_run_true_loss_and_gap_increase_as_s_shrinks():
    df, _ = run_exp14(n_games=3, s_values=[200, 20, 2], N_committee=5,
                       seed=0, S=4, A=3, H=4, N=5)
    df = df.sort_values("s", ascending=False)  # s: 200 -> 20 -> 2
    assert np.all(np.diff(df["mean_true_loss"].values) >= -1e-9)
    assert np.all(np.diff(df["gap_mixture_minus_plugin"].values) >= -1e-9)
