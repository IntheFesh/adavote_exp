"""
e1_tabular/rollout_sim.py — vectorized Monte-Carlo resettable-rollout tail
simulator, used to instantiate Theorem 4 (conservative rollout value bound)
in the exact-tabular setting.

This deliberately does NOT read Q^ref off the exact DP value V_ref. Instead
it simulates actual stochastic trajectories under the game's real transition
kernel P (reward realizations + sampled next states), exactly mirroring what
"K independent resettable rollouts of the reference tail / fallback-then-
reference tail" means when the exact-DP value is unavailable (Theorem 4's
learned-setting bridge). The exact V_ref is used ONLY as a separate ground
truth to AUDIT the resulting conservative bound, never to compute it.

All K rollouts for a unit are simulated in a single vectorized batch (an
array of K parallel particle states), so wall-clock is ~independent of K.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from common.certificates import Game, joint_to_index


def _joint_to_index_batch(actions_2d: np.ndarray, A: int) -> np.ndarray:
    """Vectorized mixed-radix encode: actions_2d shape (K, n) -> (K,) indices."""
    idx = np.zeros(actions_2d.shape[0], dtype=np.int64)
    for col in range(actions_2d.shape[1]):
        idx = idx * A + actions_2d[:, col]
    return idx


def _categorical_batch(rng: np.random.Generator, P_rows: np.ndarray) -> np.ndarray:
    """Vectorized categorical sample: P_rows shape (K, S) -> (K,) next-state indices."""
    cums = np.cumsum(P_rows, axis=1)
    u = rng.random(P_rows.shape[0])
    # guard against floating round-off leaving u slightly above the last cumsum
    cums[:, -1] = 1.0 + 1e-12
    return (u[:, None] < cums).argmax(axis=1)


def mc_tail_returns_batch(
    game: Game,
    rng: np.random.Generator,
    t: int,
    s: int,
    i: int,
    prefix: Tuple[int, ...],
    a_i: int,
    K: int,
) -> np.ndarray:
    """K i.i.d. Monte-Carlo realizations of the tail return for unit (t,s,i,prefix)
    playing coordinate action a_i, then following pi^ref for the remaining
    coordinates at t and all future timesteps t+1..H-1, under the game's real
    (stochastic) reward/transition kernel.

    Returns an array of shape (K,) of realized tail returns in
    [0, (H-t)*delta_r] (assuming non-negative rewards; see B_Q convention in
    the calling experiment).
    """
    ref_j0 = game.ref_joint(t, s)
    joint0 = tuple(prefix) + (int(a_i),) + tuple(ref_j0[i + 1:])
    j0 = joint_to_index(joint0, game.A)

    returns = np.full(K, game.R[t, s, j0], dtype=float)
    cur_s = _categorical_batch(rng, np.tile(game.P[t, s, j0], (K, 1)))

    for step_t in range(t + 1, game.H):
        ref_actions = game.ref[step_t, cur_s]  # (K, n)
        j_idx = _joint_to_index_batch(ref_actions, game.A)  # (K,)
        returns += game.R[step_t, cur_s, j_idx]
        P_rows = game.P[step_t, cur_s, j_idx]  # (K, S)
        cur_s = _categorical_batch(rng, P_rows)

    return returns
