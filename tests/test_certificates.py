"""
tests/test_certificates.py — unit tests for the certificate core.

Covered claims:
  * g_N(α) monotonicity in N (odd) for α>1/2 and anti-monotonicity for α<1/2
    (eligibility condition, Cor. 2.1 premise).
  * Non-negativity of the clip Δ_+ and of W, W̃.
  * Measure convention: Σ_u ρ(u) = nH exactly.
  * Certificate chain C_0 ≥ C* ≥ C̃* ≥ true_loss for η=0 (Theorem 1b).
  * Success-inclusive certificate ≥ true_loss for η>0.
  * binomial-cdf identity for g_N.
"""

import itertools

import numpy as np
import pytest
from scipy.stats import binom

from common import certificates as C
from common.games import make_game


# --------------------------------------------------------------------------- #
def _build(game, alpha=0.7, N=5, eta=0.0, psi_rule="ref"):
    V_ref = C.ref_value_iteration(game)
    table = C.build_unit_table(
        game, V_ref,
        alpha_fn=lambda *a: alpha,
        N_fn=lambda *a: N,
        eta=eta, psi_rule=psi_rule,
    )
    V_ctrl, d_ctrl = C.controller_value(game, table)
    rho = C.unit_occupancy(game, table, d_ctrl)
    tl = C.true_loss(game, V_ref, V_ctrl)
    return V_ref, table, rho, V_ctrl, tl


# --------------------------------------------------------------------------- #
def test_gN_matches_binom_cdf():
    for N in [1, 3, 5, 7, 11]:
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert np.isclose(C.g_N(N, alpha),
                              binom.cdf(np.floor(N / 2), N, alpha))


def test_gN_monotone_in_N_for_eligible_alpha():
    # α > 1/2 : g_N strictly decreasing along odd N.
    odds = [1, 3, 5, 7, 9, 11, 13]
    for alpha in [0.55, 0.7, 0.9]:
        vals = [C.g_N(N, alpha) for N in odds]
        assert all(vals[k] > vals[k + 1] - 1e-15 for k in range(len(vals) - 1))
        assert all(vals[k] > vals[k + 1] for k in range(len(vals) - 1))


def test_gN_increasing_in_N_for_ineligible_alpha():
    # α < 1/2 : adding members (odd +2) increases failure probability.
    odds = [1, 3, 5, 7, 9, 11, 13]
    for alpha in [0.1, 0.3, 0.45]:
        vals = [C.g_N(N, alpha) for N in odds]
        assert all(vals[k] < vals[k + 1] for k in range(len(vals) - 1))


def test_gN_half_is_constant_half():
    for N in [1, 3, 5, 7]:
        assert np.isclose(C.g_N(N, 0.5), 0.5)


def test_clip_nonneg_and_W_nonneg():
    game = make_game(S=4, A=3, H=4, n=2, seed=1)
    _, table, _, _, _ = _build(game)
    for u in table.values():
        assert np.all(u.delta_plus >= -1e-12)
        assert u.W >= -1e-12
        assert u.Wtilde >= -1e-12
        assert u.Wtilde <= u.W + 1e-9      # W̃ (mean) ≤ W (max)
        assert u.delta_plus[u.ref_i] <= 1e-12  # ref action has zero swing


def test_measure_convention_sums_to_nH():
    for seed in range(5):
        game = make_game(S=4, A=3, H=4, n=2, seed=seed)
        _, table, rho, _, _ = _build(game)
        total = sum(rho.values())
        assert np.isclose(total, game.n * game.H, atol=1e-9)


def test_occupancy_is_probability_per_coordinate():
    # For each (t, i), Σ_{s,prefix} ρ over states/prefixes should be the
    # state-marginal mass d_t summed = 1 per coordinate-step.
    game = make_game(S=3, A=3, H=3, n=2, seed=7)
    _, table, rho, _, _ = _build(game)
    for t in range(game.H):
        for i in range(game.n):
            mass = sum(rho[(t, s, i, pre)]
                       for s in range(game.S)
                       for pre in itertools.product(range(game.A), repeat=i))
            assert np.isclose(mass, 1.0, atol=1e-9)


def test_certificate_chain_eta0():
    # C_0 ≥ C* ≥ C̃* ≥ true_loss, with zero violations across many games.
    viol = 0
    for seed in range(60):
        game = make_game(S=4, A=3, H=4, n=2, seed=seed)
        V_ref, table, rho, V_ctrl, tl = _build(game, alpha=0.7, N=5)
        ch = C.cert_chain(game, table, rho)
        if not (ch["C0"] + 1e-9 >= ch["Cstar"] >= ch["Ctilde"] - 1e-12
                and ch["Ctilde"] + 1e-9 >= tl):
            viol += 1
    assert viol == 0, f"{viol} chain violations"


def test_success_inclusive_valid_for_eta_positive():
    # With η>0 and ψ = worst-in-G^η, the success-inclusive certificate must
    # still upper-bound the true loss.
    viol = 0
    for seed in range(40):
        game = make_game(S=4, A=3, H=4, n=2, seed=seed)
        V_ref, table, rho, V_ctrl, tl = _build(
            game, alpha=0.7, N=5, eta=0.2, psi_rule="worst_in_Geta")
        si = C.cert_success_inclusive(game, table, rho)
        if si + 1e-9 < tl:
            viol += 1
    assert viol == 0, f"{viol} success-inclusive violations"


def test_true_loss_nonnegative_for_greedy_reference():
    # Greedy reference is optimal => controller cannot beat it => loss ≥ 0.
    for seed in range(20):
        game = make_game(S=4, A=3, H=4, n=2, seed=seed, ref_mode="greedy")
        _, _, _, _, tl = _build(game)
        assert tl >= -1e-9


def test_index_roundtrip():
    A, n = 4, 3
    for idx in range(A ** n):
        assert C.joint_to_index(C.index_to_joint(idx, A, n), A) == idx


# --------------------------------------------------------------------------- #
# Theorem 4 (conservative rollout value bound) — shared rollout-bridge math.
# --------------------------------------------------------------------------- #
def test_rad_hoeffding_shrinks_with_K_and_grows_with_BQ():
    # rad(K, delta') = B_Q * sqrt(ln(2/delta') / (2K)): monotone decreasing in
    # K, monotone increasing in B_Q, and exactly zero only in the limit.
    r25 = C.rad_hoeffding(25, 0.05, 2.0)
    r400 = C.rad_hoeffding(400, 0.05, 2.0)
    assert r400 < r25
    assert np.isclose(r400, r25 * np.sqrt(25.0 / 400.0))  # rad ~ 1/sqrt(K)
    r_small_BQ = C.rad_hoeffding(100, 0.05, 1.0)
    r_big_BQ = C.rad_hoeffding(100, 0.05, 4.0)
    assert np.isclose(r_big_BQ, 4.0 * r_small_BQ)  # rad linear in B_Q


def test_wfb_plus_from_rollouts_nonneg_and_clips():
    # If Qref_hat <= Qfb_hat (no observed swing), the bound reduces to just
    # the radius term (clip_pos kills the negative part).
    B_Q = 3.0
    w = C.wfb_plus_from_rollouts(Qref_hat=1.0, Qfb_hat=1.5, K=100, delta_prime=0.05, B_Q=B_Q)
    rad = C.rad_hoeffding(100, 0.05, B_Q)
    assert np.isclose(w, 2.0 * rad)
    assert w >= 0.0
    # If Qref_hat > Qfb_hat, the swing adds on top of the radius term.
    w2 = C.wfb_plus_from_rollouts(Qref_hat=2.0, Qfb_hat=0.5, K=100, delta_prime=0.05, B_Q=B_Q)
    assert np.isclose(w2, 1.5 + 2.0 * rad)


def test_delta_prime_union_matches_definition_and_shrinks_with_more_failures():
    assert C.delta_prime_union(0.05, m_F=0) == 1.0  # no logged failures -> no radius needed
    d1 = C.delta_prime_union(0.05, m_F=1)
    d10 = C.delta_prime_union(0.05, m_F=10)
    assert np.isclose(d1, 0.05 / 2.0)
    assert d10 < d1  # more logged failures -> tighter per-unit budget (union bound)


def test_theorem4_coverage_on_toy_fraction_verified_instance():
    """Cross-check against the exact-Fraction-verified toy game used during
    development (n=1,A=2,S=2,H=2): Q^ref=2.0 exactly, Q^fb=0.5 exactly (see
    scratchpad/validate_theorem4.py), so Delta_+ = 1.5 exactly. Monte-Carlo
    rollout estimates of Q^ref/Q^fb, inflated by the Hoeffding radius, must
    dominate this EXACT value at rate >= 1-delta' (statistically, over many
    independent K-rollout trials)."""
    from e1_tabular.rollout_sim import mc_tail_returns_batch

    R = np.array([[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]])
    P = np.array([
        [[[0.5, 0.5], [0.25, 0.75]], [[0.75, 0.25], [0.5, 0.5]]],
        [[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]],
    ])
    d0 = np.array([1.0, 0.0])
    ref = np.zeros((2, 2, 1), dtype=int)
    delta_r = float(R.max() - R.min())
    game = C.Game(S=2, A=2, n=1, H=2, R=R, P=P, d0=d0, ref=ref, delta_r=delta_r)

    Q_ref_exact, Q_fb_exact, delta_plus_exact = 2.0, 0.5, 1.5
    B_Q = game.H * game.delta_r
    K, delta_prime, n_trials = 50, 0.10, 800
    violations = 0
    for trial in range(n_trials):
        rng = np.random.default_rng(trial)
        ref_s = mc_tail_returns_batch(game, rng, 0, 0, 0, (), 0, K)
        fb_s = mc_tail_returns_batch(game, rng, 0, 0, 0, (), 1, K)
        w = C.wfb_plus_from_rollouts(ref_s.mean(), fb_s.mean(), K, delta_prime, B_Q)
        if w < delta_plus_exact - 1e-9:
            violations += 1
    assert violations / n_trials <= delta_prime + 0.03  # statistical tolerance


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
