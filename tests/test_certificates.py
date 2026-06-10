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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
