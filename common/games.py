"""
common/games.py — Random Markov-game generators for E1 (CPU-only, numpy).

`make_game` produces a finite-horizon n-agent cooperative game with:
    reward  ~ U(0,1) over joint actions,
    transition ~ Dirichlet (action-DEPENDENT) or action-INDEPENDENT.

The action-independent class P_t(s'|s,a) = d_{t+1}(s') is used for the
fixed-occupancy subclass and the realizability witnesses (exp_07, exp_08b).

A reference policy is attached.  By default it is the per-agent *greedy*
joint action under the reference Q, but it MAY be non-greedy (the paper
allows arbitrary reference); pass `ref_mode='random'` for a random reference
or `ref_mode='given'` with `ref_actions`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .certificates import Game, joint_to_index, index_to_joint, ref_value_iteration


def make_game(
    S: int = 4,
    A: int = 3,
    H: int = 4,
    n: int = 2,
    seed: int = 0,
    transition: str = "dependent",   # 'dependent' | 'independent'
    eta: float = 0.0,                # kept for API symmetry (unused here)
    ref_mode: str = "greedy",        # 'greedy' | 'random' | 'given'
    ref_actions: Optional[np.ndarray] = None,
    reward_scale: float = 1.0,
    dirichlet_alpha: float = 1.0,
) -> Game:
    """Create a random tabular cooperative Markov game.

    Parameters
    ----------
    transition : 'dependent' -> P_t(s'|s,a) depends on the joint action;
                 'independent' -> P_t(s'|s,a) = d_{t+1}(s') (action-free),
                 giving a fixed state-occupancy subclass.
    ref_mode   : how the reference policy is chosen.
    """
    rng = np.random.default_rng(seed)
    Anj = A ** n

    R = reward_scale * rng.uniform(0.0, 1.0, size=(H, S, Anj))

    P = np.zeros((H, S, Anj, S))
    if transition == "dependent":
        for t in range(H):
            for s in range(S):
                for a in range(Anj):
                    P[t, s, a] = rng.dirichlet(np.full(S, dirichlet_alpha))
    elif transition == "independent":
        for t in range(H):
            # one next-state law per (t, s) shared across all joint actions
            for s in range(S):
                d_next = rng.dirichlet(np.full(S, dirichlet_alpha))
                P[t, s, :] = d_next[None, :]
    else:
        raise ValueError(f"unknown transition mode {transition}")

    d0 = rng.dirichlet(np.full(S, 1.0))
    delta_r = float(R.max() - R.min())

    game = Game(S=S, A=A, n=n, H=H, R=R, P=P, d0=d0,
                ref=np.zeros((H, S, n), dtype=int), delta_r=delta_r)

    if ref_mode == "given":
        assert ref_actions is not None
        game.ref = np.asarray(ref_actions, dtype=int)
    elif ref_mode == "random":
        game.ref = rng.integers(0, A, size=(H, S, n))
    elif ref_mode == "greedy":
        game.ref = _greedy_reference(game)
    else:
        raise ValueError(f"unknown ref_mode {ref_mode}")

    return game


def _greedy_reference(game: Game) -> np.ndarray:
    """Per-(t,s) greedy joint action maximizing one-step+go reward.

    Computed by backward induction over the *joint* action (this defines a
    well-defined reference policy; agents need not be individually greedy).
    """
    H, S, A, n = game.H, game.S, game.A, game.n
    V = np.zeros((H + 1, S))
    ref = np.zeros((H, S, n), dtype=int)
    for t in range(H - 1, -1, -1):
        for s in range(S):
            q = game.R[t, s] + game.P[t, s] @ V[t + 1]  # (A**n,)
            best = int(np.argmax(q))
            V[t, s] = q[best]
            ref[t, s] = index_to_joint(best, A, n)
    return ref


def make_independent_game_with_occupancy(
    S: int, A: int, H: int, n: int, seed: int,
    target_d: Optional[np.ndarray] = None,
) -> Game:
    """Action-independent game whose per-step next-state law is `target_d`
    (shape (H, S) -> next-state distribution, or random if None).

    Because transitions ignore actions, the state occupancy is fixed
    regardless of the controller — the 'fixed-occupancy' subclass used in
    exp_07 / exp_08.
    """
    rng = np.random.default_rng(seed)
    Anj = A ** n
    R = rng.uniform(0.0, 1.0, size=(H, S, Anj))
    P = np.zeros((H, S, Anj, S))
    for t in range(H):
        for s in range(S):
            if target_d is None:
                d_next = rng.dirichlet(np.full(S, 1.0))
            else:
                d_next = target_d[t]
            P[t, s, :] = d_next[None, :]
    d0 = rng.dirichlet(np.full(S, 1.0))
    delta_r = float(R.max() - R.min())
    game = Game(S=S, A=A, n=n, H=H, R=R, P=P, d0=d0,
                ref=np.zeros((H, S, n), dtype=int), delta_r=delta_r)
    game.ref = _greedy_reference(game)
    return game
