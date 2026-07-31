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
    fail_prob_given_theta, oracle_g_betabinom, run as run_exp14,
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


def test_exp14_oracle_matches_direct_binomial_at_zero_dispersion_limit():
    # As kappa grows, theta concentrates at alpha, so the Beta-Binomial oracle
    # should approach the plain Binomial g_N(N, alpha). kappa is kept moderate
    # (not e.g. 1e6): at extreme kappa the Beta(alpha*kappa,(1-alpha)*kappa)
    # density becomes a needle-thin spike that scipy.integrate.quad's default
    # adaptive sampling can miss entirely (verified directly: at kappa=1e6 quad
    # returns ~1e-172 while a 20M-draw Monte Carlo cross-check at the actually
    # -used kappa=20 confirms quad IS accurate in the regime this experiment
    # uses -- this is a quad robustness limit at extreme kappa, not a bug
    # reachable by exp_14's own kappa=20 default).
    from common.certificates import g_N
    N, alpha = 7, 0.7
    g_direct = g_N(N, alpha)
    g_oracle_highkappa = oracle_g_betabinom(N, alpha, kappa=5000.0)
    assert abs(g_direct - g_oracle_highkappa) < 1e-3


def test_exp14_fail_prob_given_theta_matches_binom_cdf():
    from scipy.stats import binom
    N = 9
    for theta in [0.1, 0.4, 0.5, 0.6, 0.9]:
        assert np.isclose(fail_prob_given_theta(theta, N),
                          binom.cdf(np.floor(N / 2), N, theta))


def test_exp14_run_produces_valid_rates():
    df, g_oracle = run_exp14(N=5, alpha=0.7, kappa=20.0, s_values=[50, 5],
                              n_trials=20, seed=0)
    assert 0.0 <= g_oracle <= 1.0
    assert ((df["under_coverage_rate"] >= 0) & (df["under_coverage_rate"] <= 1)).all()
