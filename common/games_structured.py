"""
common/games_structured.py — Structured, cooperation-semantic cooperative
Markov game family for exact-DP validity experiments (Route 1 / Pilot B).

The existing `common.games.make_game` generator uses i.i.d.-uniform-random
rewards and Dirichlet-random transitions: valid for stress-testing the
certificate machinery, but with no real cooperative structure (a reviewer's
"synthetic/toy" impression). This module replaces it with a small family that
has genuine coordination semantics while remaining exactly DP-solvable:

Resource-constrained coordination game.
  A shared depleting resource pool has `s in {0,...,S-1}` slots remaining.
  Each of `n` agents simultaneously chooses an action a_i in {0,...,A-2} to
  CLAIM resource type a_i, or a_i = A-1 to WAIT (defer, claim nothing).
  Reward at (t,s,joint a) is cooperative and requires genuine coordination:
    - Let c = number of agents claiming a resource this step (a_i != WAIT).
    - If c <= s (no over-claim / no collision on scarce slots): every
      claiming agent earns a per-claim reward `r_claim`, and the state
      depletes by c (unless resources_replenish=True, see below).
    - If c > s (collision / over-claim on a scarce resource): the round
      fails outright -- ALL claiming agents get zero reward this step (the
      shared resource is spoiled by the conflict), a strictly worse outcome
      than any single agent waiting. This is what makes joint (not just
      per-agent) coordination necessary: a myopically-greedy agent that
      ignores teammates' likely claims can trigger a collision that zeros
      out everyone's reward.
    - Waiting always earns 0 for that agent, deterministically, with no risk.
  Transition: s' = max(0, s - c) if no collision, else s' unchanged (a failed
  round wastes the round but does not destroy the resource). If
  `resources_replenish=True`, resources drift back up by +1 (capped at S-1)
  whenever no agent claims (encoding a renewable-resource variant).

This keeps the state space S small and INDEPENDENT of n (unlike encoding
each agent's position, which would blow up as L**n); the only place n enters
combinatorially is the joint action space A**n, exactly as in the general
theory (Definition 1) and matching the existing E1 infrastructure's Game
container (R, P indexed by flat joint-action index).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .certificates import Game, joint_to_index, index_to_joint


def make_resource_coordination_game(
    S: int = 4,
    A: int = 3,
    H: int = 4,
    n: int = 2,
    seed: int = 0,
    r_claim: float = 1.0,
    resources_replenish: bool = False,
    ref_mode: str = "greedy",
) -> Game:
    """Structured cooperative Markov game with genuine collision-avoidance
    coordination semantics (see module docstring). A = number of per-agent
    choices; action A-1 is always WAIT, actions 0..A-2 are distinct resource
    claims. S = number of shared-resource-pool levels (state space size).

    All quantities are DETERMINISTIC given (s, joint action) -- reward and
    next-state are both exact functions of (t, s, joint a), so the game is
    exactly DP-solvable at any n via the existing `Game` container (the joint
    action space A**n is the only combinatorial growth axis, as in the
    general theory).
    """
    assert A >= 2, "need at least one claim action + WAIT"
    rng = np.random.default_rng(seed)
    Anj = A ** n
    WAIT = A - 1

    R = np.zeros((H, S, Anj), dtype=float)
    P = np.zeros((H, S, Anj, S), dtype=float)

    for t in range(H):
        for s in range(S):
            for a_idx in range(Anj):
                joint = index_to_joint(a_idx, A, n)
                c = sum(1 for a in joint if a != WAIT)
                if c == 0:
                    R[t, s, a_idx] = 0.0
                    s_next = min(S - 1, s + 1) if resources_replenish else s
                elif c <= s:
                    R[t, s, a_idx] = r_claim * c
                    s_next = max(0, s - c)
                else:
                    R[t, s, a_idx] = 0.0  # collision: round spoiled, no reward
                    s_next = s  # resource untouched, but round wasted
                P[t, s, a_idx, s_next] = 1.0

    # Initial state: full resource pool.
    d0 = np.zeros(S)
    d0[S - 1] = 1.0

    delta_r = float(R.max() - R.min())
    game = Game(S=S, A=A, n=n, H=H, R=R, P=P, d0=d0,
                ref=np.zeros((H, S, n), dtype=int), delta_r=delta_r)

    if ref_mode == "greedy":
        game.ref = _greedy_reference(game)
    elif ref_mode == "random":
        game.ref = rng.integers(0, A, size=(H, S, n))
    else:
        raise ValueError(f"unknown ref_mode {ref_mode}")

    return game


def _greedy_reference(game: Game) -> np.ndarray:
    """Per-(t,s) joint-optimal reference action, by exact backward induction
    (identical convention to common.games._greedy_reference)."""
    H, S, A, n = game.H, game.S, game.A, game.n
    V = np.zeros((H + 1, S))
    ref = np.zeros((H, S, n), dtype=int)
    for t in range(H - 1, -1, -1):
        for s in range(S):
            q = game.R[t, s] + game.P[t, s] @ V[t + 1]
            best = int(np.argmax(q))
            V[t, s] = q[best]
            ref[t, s] = index_to_joint(best, A, n)
    return ref
