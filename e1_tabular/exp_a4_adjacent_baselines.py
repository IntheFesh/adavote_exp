"""
exp_a4_adjacent_baselines.py — Appendix baselines table: I2 certificate vs.
adjacent OPE / robust-RL baselines (PDIS, doubly-robust, robust simulation
lemma), audited against exact-DP truth.

REBUILT DRIVER. The original A4 driver is not present in this repository or
any branch; only its output data survived (recovered from an old
supplementary bundle, data/exp_strawman.csv). That recovered data's R_max is
~4x smaller than what common/games.py's current CURRENT config produces for
the same gseed/config (see the optional Rmax diagnostic in the A3 driver /
Task 5 report), so it cannot be used to validate this rebuild's calibration.
This driver instead reuses the SAME game construction as exp_03/exp_01 (8
games via master_rng(0), S=4,A=3,H=4,n=2) and the shared certificate/
estimator machinery in common/certificates.py and common/baselines.py.

Four methods, one shared batch of m=2000 trajectories per (game, repeat):

  I2 certificate, TWO computational bases (dual output, per Task 5 §3.3):
    - 口径 O (oracle): population C2 = cert_Ctilde(...) / true_loss L.
      No sampling error -- deterministic given the game.
    - 口径 B (finite-sample bound): common.certificates.empirical_bernstein
      on X ~ estimator_samples(..., variant="Pi2") (the SAME one-sample-per-
      episode estimator exp_03 uses for its Pi2 variant), at m=2000.

  PDIS (Precup 2000) / DR (Jiang & Li 2016), via common/baselines.py's
  ope_pdis_upper / ope_dr_upper: m trajectories are rolled out under the
  ACTUAL controller policy (common.certificates.controller_state_dist),
  reweighted toward pi_ref (deterministic -> importance ratio is 1 if the
  logged joint action equals the reference joint action, else 0, standard
  for a deterministic target policy). J_ctrl_hat is supplied as the EXACT
  controller value (b.true_loss's V_ctrl @ d0), since this radius formula
  has exactly one statistical-noise term (attributed to J_ref_hat) and
  reusing the already-exact V_ctrl is the natural choice in this exactly
  solvable tabular setting -- not an approximation shortcut being hidden.
  DR's model term (V_hat_ref / Q_hat_ref) uses the EXACT V_ref / one-step
  Bellman-expanded Q_ref (also exact, via ref_value_iteration), which is the
  best-available "model" here; the importance-sampling correction is still
  computed from the real logged (state, action, reward) trajectories under
  pi_ctrl, so this remains a legitimate off-policy DR estimate, not an
  oracle shortcut of the OPE quantity itself.

  Robust simulation lemma (Kakade & Langford 2002): deterministic bound
  H * epsilon_pi_max * R_max, epsilon_pi_max = worst-state total-variation
  distance between pi_ref and pi_ctrl (exact, since pi_ctrl's full joint
  distribution is known via controller_state_dist).

R_max (per-step reward range) = game.delta_r, matching how
common/baselines.py's own simulation_lemma_upper multiplies by H
internally (R_max is a PER-STEP bound there, not H*delta_r).

Coverage is defined uniformly across all four methods+bases as the
empirical frequency, over (game, repeat) pairs, of bound >= true_loss.

Outputs:
    results/data/exp_a4_adjacent_baselines.csv
    one summary line -> results/summary.txt
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import certificates as C
from common.baselines import ope_pdis_upper, ope_dr_upper, simulation_lemma_upper, emp_bernstein_bound
from common.games import make_game
from common.io_utils import data_path, write_summary
from e1_tabular._common import build, make_alpha_fn, const_fn
from e1_tabular.rollout_sim import _categorical_batch

DELTA = 0.05


def exact_dr_ingredients(game: C.Game, V_ref: np.ndarray):
    """Exact model term for DR: V_hat_ref(t,s), Q_hat_ref(t,s,joint_a)."""
    H, S, Anj = game.H, game.S, game.A ** game.n
    V_hat_ref, Q_hat_ref = {}, {}
    for t in range(H):
        for s in range(S):
            V_hat_ref[(t, s)] = float(V_ref[t, s])
            cont = game.R[t, s] + game.P[t, s] @ V_ref[t + 1]
            for a in range(Anj):
                Q_hat_ref[(t, s, a)] = float(cont[a])
    return V_hat_ref, Q_hat_ref


def rollout_controller_trajectories(game: C.Game, table, m: int, rng: np.random.Generator):
    """m i.i.d. episodes under the controller policy (vectorized across
    episodes), returning per-episode (s,a_joint,r) lists plus per-step
    pi_ref_prob / pi_ctrl_prob for the ACTUAL logged joint action."""
    H, S, Anj = game.H, game.S, game.A ** game.n
    jdist = np.zeros((H, S, Anj))
    ref_idx = np.zeros((H, S), dtype=int)
    for t in range(H):
        for s in range(S):
            jdist[t, s], _ = C.controller_state_dist(game, table, t, s)
            ref_idx[t, s] = C.joint_to_index(game.ref_joint(t, s), game.A)

    cur_s = rng.choice(S, size=m, p=game.d0)
    trajs = [[] for _ in range(m)]
    pi_ref_probs = np.zeros((m, H))
    pi_ctrl_probs = np.zeros((m, H))
    for t in range(H):
        P_rows_a = jdist[t, cur_s]                      # (m, Anj)
        a_idx = _categorical_batch(rng, P_rows_a)        # (m,)
        r = game.R[t, cur_s, a_idx]
        pi_ctrl_probs[:, t] = P_rows_a[np.arange(m), a_idx]
        pi_ref_probs[:, t] = (a_idx == ref_idx[t, cur_s]).astype(float)
        for ep in range(m):
            trajs[ep].append((int(cur_s[ep]), int(a_idx[ep]), float(r[ep])))
        P_rows_s = game.P[t, cur_s, a_idx]                # (m, S)
        cur_s = _categorical_batch(rng, P_rows_s)
    return trajs, pi_ref_probs, pi_ctrl_probs


def worst_state_tv(game: C.Game, table) -> float:
    """max_{t,s} TV(pi_ref(.|t,s), pi_ctrl(.|t,s)) = max (1 - P_ctrl(ref_joint | t,s))."""
    worst = 0.0
    for t in range(game.H):
        for s in range(game.S):
            jdist, _ = C.controller_state_dist(game, table, t, s)
            ref_idx = C.joint_to_index(game.ref_joint(t, s), game.A)
            tv = 1.0 - float(jdist[ref_idx])
            worst = max(worst, tv)
    return worst


def run(n_games: int, repeats: int, m: int, seed: int, S: int, A: int, H: int, N: int):
    rows = []
    master_rng = np.random.default_rng(seed)
    R_max = None
    for gidx in tqdm(range(n_games), desc="exp_a4 games"):
        gseed = int(master_rng.integers(0, 2**31))
        game = make_game(S=S, A=A, H=H, n=2, seed=gseed, transition="dependent")
        alpha_fn = make_alpha_fn(gseed)
        b = build(game, alpha_fn, const_fn(N), eta=0.0, psi_rule="ref")
        tl = b.true_loss
        chain = C.cert_chain(game, b.table, b.rho)
        C2_pop = chain["Ctilde"]
        b_range = game.n * game.H * C.max_W(b.table)
        R_max = game.delta_r
        eps_pi_max = worst_state_tv(game, b.table)
        J_ctrl_exact = float(game.d0 @ b.V_ctrl[0])
        V_hat_ref, Q_hat_ref = exact_dr_ingredients(game, b.V_ref)
        sim_lemma = simulation_lemma_upper(eps_pi_max, R_max, game.H)

        for rep in range(repeats):
            rep_rng = np.random.default_rng([seed, gidx, rep])

            # I2, 口径 O (population, no sampling)
            i2_O = C2_pop

            # I2, 口径 B (finite-sample EB on the Pi2 estimator, m draws)
            X = C.estimator_samples(game, b.table, b.rho, m, rep_rng, variant="Pi2")
            i2_B = emp_bernstein_bound(X, DELTA, b_range)

            # PDIS / DR: shared batch of m trajectories under pi_ctrl
            trajs, pi_ref_probs, pi_ctrl_probs = rollout_controller_trajectories(
                game, b.table, m, rep_rng)
            pdis_upper, pdis_hat = ope_pdis_upper(
                trajs, pi_ref_probs, pi_ctrl_probs, J_ctrl_exact, DELTA, R_max, game.H)
            dr_upper, dr_hat = ope_dr_upper(
                trajs, pi_ref_probs, pi_ctrl_probs, V_hat_ref, Q_hat_ref,
                J_ctrl_exact, DELTA, R_max, game.H)

            for method, bound in [
                ("I2_O", i2_O), ("I2_B", i2_B), ("PDIS", pdis_upper),
                ("DR", dr_upper), ("robust_sim_lemma", sim_lemma),
            ]:
                rows.append(dict(
                    game=gidx, repeat=rep, method=method, bound=bound,
                    true_loss=tl, covered=int(bound >= tl - 1e-9),
                    ratio_L=bound / tl if tl > 1e-9 else np.nan,
                ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="exp_a4: I2 (dual basis) vs PDIS/DR/robust-simulation-lemma baselines")
    ap.add_argument("--n_games", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--m", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--S", type=int, default=4)
    ap.add_argument("--A", type=int, default=3)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--N", type=int, default=5)
    args = ap.parse_args()

    df = run(args.n_games, args.repeats, args.m, args.seed, args.S, args.A, args.H, args.N)
    csv_path = data_path("exp_a4_adjacent_baselines.csv")
    df.to_csv(csv_path, index=False)

    g = df.groupby("method").agg(coverage=("covered", "mean"),
                                  median_ratio_L=("ratio_L", "median"))
    print(g.round(4).to_string())

    # Self-assertion: headline numbers recomputed from the just-written CSV.
    df_check = pd.read_csv(csv_path)
    g_check = df_check.groupby("method").agg(
        coverage=("covered", "mean"), median_ratio_L=("ratio_L", "median"))
    assert np.allclose(g_check["coverage"].values, g["coverage"].values), \
        "headline coverage does not match recomputation from CSV"
    assert np.allclose(g_check["median_ratio_L"].values, g["median_ratio_L"].values,
                        equal_nan=True), \
        "headline median_ratio_L does not match recomputation from CSV"

    parts = "  ".join(f"{m}:cov={g.loc[m,'coverage']:.4f},B/L={g.loc[m,'median_ratio_L']:.4f}"
                       for m in g.index)
    write_summary(f"exp_a4 adjacent_baselines [n_games={args.n_games},m={args.m}]  {parts}")


if __name__ == "__main__":
    main()
