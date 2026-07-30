"""
common/certificates.py — Core mathematical objects for AdaVote (E1 + E2).

CPU-only (numpy + scipy). This module DOES NOT import any GPU/JAX library.

It implements the shared certificate machinery described in the paper:

  Setting: finite-horizon (H) n-agent cooperative Markov game, states s,
  joint action (a_1,...,a_n), shared bounded reward (global amplitude Δ_r),
  fixed reference policy π^ref (may be non-greedy).

  Unit u = (t, s, i, a_{<i})  — a *coordinate* decision point: at time t,
  state s, deciding agent i's action, conditioned on the already-executed
  prefix a_{<i} (agents 1..i-1).

  Non-negative coordinate amplitude (clipped throughout):
      Δ_+(u,a) = [ Q^{π^ref}(u, a_i^ref) - Q^{π^ref}(u, a) ]_+
      W(u)     = max_a Δ_+(u,a)
      W̃(u)     = E[ Δ_+(u, a^fb) | u, F=1 ]            (fallback-distribution mean)

  reference-relative endorsed set:
      G^η(u) = { a : Δ_+(u,a) ≤ η }.    Main experiments use η=0 (ψ=a_i^ref).

  Majority-failure probability (N odd):
      g_N(α) = Pr[ Bin(N,α) ≤ floor(N/2) ] = binom.cdf(floor(N/2), N, α).

  Measure convention: μ is a probability law over units; each certificate is
      C = nH · E_{U~μ}[ ω(U) g(U) ] = Σ_u ρ(u) ω(u) g(u),
  where ρ(u) is the controller's expected unit-occupancy (Σ_u ρ(u) = nH).

  Three-level certificate chain (η=0):
      C_0   = nH·E_μ[ g(U) (H - t_U) Δ_r ]
           ≥ C*   = nH·E_μ[ g(U) W(U) ]
           ≥ C̃*  = nH·E_μ[ g(U) W̃(U) ]
           ≥ true loss.

  The chain inequalities hold *by construction* of the occupancy weighting:
  see `cert_chain` and the performance-difference derivation in the README.

Author: AdaVote experiment suite.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import binom

# A unit key is (t, s, i, prefix) where prefix is a tuple of length i.
UnitKey = Tuple[int, int, int, Tuple[int, ...]]


# --------------------------------------------------------------------------- #
# Basic scalar objects
# --------------------------------------------------------------------------- #
def g_N(N: int, alpha: float) -> float:
    """Majority-failure probability g_N(α) = Pr[Bin(N,α) ≤ floor(N/2)].

    N should be odd (the paper uses odd committee sizes so that "majority"
    is unambiguous).  Returns a float in [0,1].
    """
    return float(binom.cdf(np.floor(N / 2.0), N, alpha))


def clip_pos(x):
    """Non-negative clip [x]_+ = max(x, 0)."""
    return np.maximum(x, 0.0)


# --------------------------------------------------------------------------- #
# Joint-action index helpers (n agents, A actions each -> flat index in A^n)
# --------------------------------------------------------------------------- #
def joint_to_index(actions: Tuple[int, ...], A: int) -> int:
    """Mixed-radix encode a joint action tuple into a flat index in [0, A^n)."""
    idx = 0
    for a in actions:
        idx = idx * A + int(a)
    return idx


def index_to_joint(idx: int, A: int, n: int) -> Tuple[int, ...]:
    """Decode a flat joint-action index into an n-tuple."""
    out = []
    for _ in range(n):
        out.append(idx % A)
        idx //= A
    return tuple(reversed(out))


# --------------------------------------------------------------------------- #
# Game container
# --------------------------------------------------------------------------- #
@dataclass
class Game:
    """A finite-horizon n-agent cooperative Markov game in tabular form.

    Attributes
    ----------
    S, A, n, H : ints
        #states, #actions per agent, #agents, horizon (t = 0..H-1).
    R : np.ndarray, shape (H, S, A**n)
        Shared reward R_t(s, joint_action).  Bounded; global amplitude Δ_r.
    P : np.ndarray, shape (H, S, A**n, S)
        Transition P_t(s' | s, joint_action). Rows sum to 1.
    d0 : np.ndarray, shape (S,)
        Initial state distribution.
    ref : np.ndarray, shape (H, S, n), int
        Deterministic reference action per agent at (t,s).
    delta_r : float
        Global reward amplitude Δ_r = max R - min R (used by C_0).
    """

    S: int
    A: int
    n: int
    H: int
    R: np.ndarray
    P: np.ndarray
    d0: np.ndarray
    ref: np.ndarray
    delta_r: float

    def ref_joint(self, t: int, s: int) -> Tuple[int, ...]:
        return tuple(int(x) for x in self.ref[t, s])


# --------------------------------------------------------------------------- #
# Reference value iteration (exact DP) and coordinate Q
# --------------------------------------------------------------------------- #
def ref_value_iteration(game: Game) -> np.ndarray:
    """Exact V^{π^ref} for the deterministic reference policy.

    Returns V_ref of shape (H+1, S) with V_ref[H] = 0.
    """
    V = np.zeros((game.H + 1, game.S), dtype=float)
    for t in range(game.H - 1, -1, -1):
        for s in range(game.S):
            j = joint_to_index(game.ref_joint(t, s), game.A)
            V[t, s] = game.R[t, s, j] + game.P[t, s, j] @ V[t + 1]
    return V


def coordinate_Q(
    game: Game, V_ref: np.ndarray, t: int, s: int, i: int,
    prefix: Tuple[int, ...], a_i: int,
) -> float:
    """Q^{π^ref}(u, a_i) for unit u=(t,s,i,prefix).

    The continuation joint action is:
        agents < i : prefix
        agent   i  : a_i
        agents > i : reference actions a^ref(t,s)
    followed by π^ref afterwards (value V_ref[t+1]).
    """
    ref_j = game.ref_joint(t, s)
    joint = tuple(prefix) + (int(a_i),) + tuple(ref_j[i + 1:])
    j = joint_to_index(joint, game.A)
    return float(game.R[t, s, j] + game.P[t, s, j] @ V_ref[t + 1])


def coordinate_Q_vec(
    game: Game, V_ref: np.ndarray, t: int, s: int, i: int, prefix: Tuple[int, ...]
) -> np.ndarray:
    """Vector of Q^{π^ref}(u, a) over all candidate actions a for agent i."""
    return np.array(
        [coordinate_Q(game, V_ref, t, s, i, prefix, a) for a in range(game.A)]
    )


# --------------------------------------------------------------------------- #
# Unit table: per-unit certificate ingredients
# --------------------------------------------------------------------------- #
@dataclass
class UnitInfo:
    """Per-unit certificate ingredients."""
    t: int
    s: int
    i: int
    prefix: Tuple[int, ...]
    ref_i: int
    Q: np.ndarray          # Q^{π^ref}(u, a) over a  (length A)
    delta_plus: np.ndarray  # Δ_+(u, a) over a       (length A)
    W: float               # max_a Δ_+
    Wtilde: float          # E[Δ_+(u,a^fb) | F=1]    (fallback mean)
    fb: np.ndarray         # fallback distribution over a (length A)
    psi: int               # success-executed action (η=0 -> ref_i)
    w_psi: float           # Δ_+(u, psi)            (0 for η=0)
    Geta: np.ndarray       # boolean endorsed set G^η(u)
    alpha: float           # per-member endorsement prob of ψ
    N: int                 # committee size (odd)
    g: float               # g_N(α) majority-failure prob


# Type aliases for per-unit configuration callbacks.
AlphaFn = Callable[[int, int, int, Tuple[int, ...]], float]
NFn = Callable[[int, int, int, Tuple[int, ...]], int]
FbFn = Callable[[int, int, int, Tuple[int, ...], "Game", np.ndarray], np.ndarray]


def uniform_fb(t, s, i, prefix, game: Game, delta_plus: np.ndarray) -> np.ndarray:
    """Default fallback: uniform over all A actions."""
    return np.ones(game.A) / game.A


def enumerate_units(game: Game) -> List[Tuple[int, int, int, Tuple[int, ...]]]:
    """All units (t,s,i,prefix).  prefix ranges over A^i (executed prefixes)."""
    units = []
    for t in range(game.H):
        for s in range(game.S):
            for i in range(game.n):
                for prefix in itertools.product(range(game.A), repeat=i):
                    units.append((t, s, i, tuple(prefix)))
    return units


def build_unit_table(
    game: Game,
    V_ref: np.ndarray,
    alpha_fn: AlphaFn,
    N_fn: NFn,
    fb_fn: FbFn = uniform_fb,
    eta: float = 0.0,
    psi_rule: str = "ref",  # "ref" or "worst_in_Geta"
) -> Dict[UnitKey, UnitInfo]:
    """Build the full per-unit table.

    psi_rule:
        "ref"           -> ψ = a_i^ref always (η=0 main setting; w_ψ=0).
        "worst_in_Geta" -> ψ = argmax_{a in G^η} Δ_+(u,a) (used by exp_02).
    """
    table: Dict[UnitKey, UnitInfo] = {}
    for (t, s, i, prefix) in enumerate_units(game):
        ref_i = int(game.ref[t, s, i])
        Q = coordinate_Q_vec(game, V_ref, t, s, i, prefix)
        delta_plus = clip_pos(Q[ref_i] - Q)
        Geta = delta_plus <= eta + 1e-12
        fb = np.asarray(fb_fn(t, s, i, prefix, game, delta_plus), dtype=float)
        fb = fb / fb.sum()
        W = float(delta_plus.max())
        Wtilde = float(fb @ delta_plus)

        if psi_rule == "ref":
            psi = ref_i
        elif psi_rule == "worst_in_Geta":
            # worst (largest Δ_+) action still inside the η-endorsed set;
            # ties broken toward the reference action so that η=0 yields ψ=ref_i
            # exactly (otherwise an equal-Q action could transition differently).
            cand = np.where(Geta)[0]
            worst_val = delta_plus[cand].max()
            tied = cand[np.isclose(delta_plus[cand], worst_val)]
            psi = ref_i if ref_i in tied else int(tied[0])
        else:
            raise ValueError(f"unknown psi_rule {psi_rule}")
        w_psi = float(delta_plus[psi])

        alpha = float(alpha_fn(t, s, i, prefix))
        N = int(N_fn(t, s, i, prefix))
        g = g_N(N, alpha)

        table[(t, s, i, prefix)] = UnitInfo(
            t=t, s=s, i=i, prefix=tuple(prefix), ref_i=ref_i, Q=Q,
            delta_plus=delta_plus, W=W, Wtilde=Wtilde, fb=fb, psi=psi,
            w_psi=w_psi, Geta=Geta, alpha=alpha, N=N, g=g,
        )
    return table


# --------------------------------------------------------------------------- #
# Controller: agreement-gated, prefix-drift joint-action distribution
# --------------------------------------------------------------------------- #
def controller_state_dist(
    game: Game, table: Dict[UnitKey, UnitInfo], t: int, s: int
) -> Tuple[np.ndarray, Dict[Tuple[int, ...], float]]:
    """Controller joint-action distribution at (t,s), built coordinate-by-
    coordinate with prefix conditioning (prefix drift).

    Per coordinate i with unit u=(t,s,i,prefix):
        p_exec(a) = (1 - g(u)) · [a == ψ(u)]  +  g(u) · fb(a | u).

    Returns
    -------
    joint_dist : np.ndarray, shape (A**n,)
        P_ctrl(joint_action | t, s).
    prefix_probs : dict prefix_tuple -> prob
        Executed-prefix probabilities for ALL prefix lengths 0..n
        (prefix_probs[()] = 1.0).  Used to assemble unit occupancy ρ(u).
    """
    A, n = game.A, game.n
    prefix_probs: Dict[Tuple[int, ...], float] = {(): 1.0}
    # frontier: list of (prefix_tuple, prob)
    frontier = [((), 1.0)]
    for i in range(n):
        new_frontier = []
        for prefix, prob in frontier:
            u = table[(t, s, i, prefix)]
            p_exec = np.zeros(A)
            p_exec += u.g * u.fb
            p_exec[u.psi] += (1.0 - u.g)
            for a in range(A):
                pa = p_exec[a]
                if pa <= 0.0:
                    continue
                new_prefix = prefix + (a,)
                new_prob = prob * pa
                new_frontier.append((new_prefix, new_prob))
                prefix_probs[new_prefix] = prefix_probs.get(new_prefix, 0.0) + new_prob
        frontier = new_frontier

    joint_dist = np.zeros(A ** n)
    for prefix, prob in frontier:  # prefix length == n now
        joint_dist[joint_to_index(prefix, A)] += prob
    return joint_dist, prefix_probs


def controller_value(
    game: Game, table: Dict[UnitKey, UnitInfo]
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact controller value V^ctrl and occupancy d^ctrl via DP.

    Returns
    -------
    V_ctrl : np.ndarray, shape (H+1, S)
    d_ctrl : np.ndarray, shape (H, S)   state-visitation under controller
    """
    H, S = game.H, game.S
    # Cache per-(t,s) joint dist.
    jdist = np.zeros((H, S, game.A ** game.n))
    for t in range(H):
        for s in range(S):
            jdist[t, s], _ = controller_state_dist(game, table, t, s)

    V = np.zeros((H + 1, S))
    for t in range(H - 1, -1, -1):
        for s in range(S):
            cont = game.R[t, s] + game.P[t, s] @ V[t + 1]  # (A**n,)
            V[t, s] = jdist[t, s] @ cont

    d = np.zeros((H, S))
    d[0] = game.d0.copy()
    for t in range(H - 1):
        nxt = np.zeros(S)
        for s in range(S):
            # P_ctrl-averaged transition out of (t,s)
            trans = jdist[t, s] @ game.P[t, s]  # (S,)
            nxt += d[t, s] * trans
        d[t + 1] = nxt
    return V, d


def unit_occupancy(
    game: Game, table: Dict[UnitKey, UnitInfo], d_ctrl: np.ndarray
) -> Dict[UnitKey, float]:
    """Controller unit-occupancy ρ(u) = d^ctrl_t(s) · P(prefix a_{<i} | t,s).

    Satisfies Σ_u ρ(u) = nH.
    """
    rho: Dict[UnitKey, float] = {}
    for t in range(game.H):
        for s in range(game.S):
            _, prefix_probs = controller_state_dist(game, table, t, s)
            base = d_ctrl[t, s]
            for i in range(game.n):
                for prefix in itertools.product(range(game.A), repeat=i):
                    p_prefix = prefix_probs.get(tuple(prefix), 0.0)
                    rho[(t, s, i, tuple(prefix))] = base * p_prefix
    return rho


def true_loss(game: Game, V_ref: np.ndarray, V_ctrl: np.ndarray) -> float:
    """True deployment loss  d_0^T (V^ref - V^ctrl)  (≥ 0 expected)."""
    return float(game.d0 @ (V_ref[0] - V_ctrl[0]))


# --------------------------------------------------------------------------- #
# Certificates (three-level chain + success-inclusive variant)
# --------------------------------------------------------------------------- #
def cert_C0(game: Game, table: Dict[UnitKey, UnitInfo], rho: Dict[UnitKey, float]) -> float:
    """C_0 = Σ_u ρ(u) g(u) (H - t_u) Δ_r."""
    total = 0.0
    for key, u in table.items():
        total += rho[key] * u.g * (game.H - u.t) * game.delta_r
    return total


def cert_Cstar(game: Game, table: Dict[UnitKey, UnitInfo], rho: Dict[UnitKey, float]) -> float:
    """C* = Σ_u ρ(u) g(u) W(u)."""
    total = 0.0
    for key, u in table.items():
        total += rho[key] * u.g * u.W
    return total


def cert_Ctilde(game: Game, table: Dict[UnitKey, UnitInfo], rho: Dict[UnitKey, float]) -> float:
    """C̃* = Σ_u ρ(u) g(u) W̃(u)."""
    total = 0.0
    for key, u in table.items():
        total += rho[key] * u.g * u.Wtilde
    return total


def cert_failure_only(game: Game, table, rho) -> float:
    """Pure-failure certificate Σ_u ρ(u) g(u) W̃(u).

    Identical to C̃* but named to emphasize it OMITS the success term
    (1-g) w_ψ.  For η=0 this is valid (w_ψ=0); for η>0 it can UNDERBOUND
    the true loss (exp_02).
    """
    return cert_Ctilde(game, table, rho)


def cert_success_inclusive(game: Game, table, rho) -> float:
    """Success-inclusive certificate (valid for general η):

        Σ_u ρ(u) [ (1 - g(u)) w_ψ(u) + g(u) W̃(u) ].
    """
    total = 0.0
    for key, u in table.items():
        total += rho[key] * ((1.0 - u.g) * u.w_psi + u.g * u.Wtilde)
    return total


def cert_chain(game: Game, table, rho) -> Dict[str, float]:
    """Convenience: returns all certificate levels + true-loss surrogate parts."""
    return {
        "C0": cert_C0(game, table, rho),
        "Cstar": cert_Cstar(game, table, rho),
        "Ctilde": cert_Ctilde(game, table, rho),
        "success_inclusive": cert_success_inclusive(game, table, rho),
    }


def max_W(table: Dict[UnitKey, UnitInfo]) -> float:
    """max_u W(u) over the unit table (used for estimator range b)."""
    return max(u.W for u in table.values())


# --------------------------------------------------------------------------- #
# Finite-sample estimator (Theorem 3): empirical-Bernstein
# --------------------------------------------------------------------------- #
def empirical_bernstein(X: np.ndarray, delta: float, b: float) -> float:
    """Empirical-Bernstein upper bound B̂ for E[X].

        B̂ = X̄ + sqrt(2 σ̂² ln(2/δ) / m) + 7 b ln(2/δ) / (3 (m-1))

    where σ̂² is the UNBIASED sample variance (ddof=1),
        σ̂² = (1/(m-1)) Σ_j (X_j - X̄)²
            = (1/(m(m-1))) Σ_{i<j} (X_i - X_j)²   (equivalent pairwise form),
    m=len(X), b is the range bound (X ∈ [0, b]). Maurer & Pontil (2009) state
    the inequality in the pairwise form; using the biased (ddof=0) variance
    here would understate σ̂² by a factor of (m-1)/m and understate the
    resulting radius, which is NOT covered by their guarantee.
    """
    X = np.asarray(X, dtype=float)
    m = len(X)
    assert m >= 2, "empirical-Bernstein needs m >= 2"
    xbar = X.mean()
    var = X.var(ddof=1)
    L = np.log(2.0 / delta)
    return float(xbar + np.sqrt(2.0 * var * L / m) + 7.0 * b * L / (3.0 * (m - 1)))


def estimator_samples(
    game: Game,
    table: Dict[UnitKey, UnitInfo],
    rho: Dict[UnitKey, float],
    m: int,
    rng: np.random.Generator,
    variant: str = "Pi2",
) -> np.ndarray:
    """Draw m i.i.d. estimator samples X_j for the Theorem-3 estimator.

    Each sample:
      * draw a unit U ~ μ = ρ / (nH)  (occupancy-weighted);
      * simulate committee: N members each endorse ψ w.p. α; failure
        F=1 iff #endorsers ≤ floor(N/2);
      * on failure, executed action a^fb ~ fb(·|U);
      * X_j = nH · w_j · F_j, where
            w_j = Δ_+(U, a^fb)   (Π_2, realized fallback advantage)
            w_j = W(U)           (Π_1, worst-case swing, ignores logging).

    E[X] = C̃* (Π_2) or C* (Π_1), both ≥ true loss; empirical-Bernstein on
    these samples yields the high-probability certificate.
    """
    keys = list(table.keys())
    probs = np.array([rho[k] for k in keys])
    probs = probs / probs.sum()  # = μ
    nH = game.n * game.H

    idxs = rng.choice(len(keys), size=m, p=probs)
    X = np.zeros(m)
    for j in range(m):
        u = table[keys[idxs[j]]]
        endorsers = rng.binomial(u.N, u.alpha)
        F = 1.0 if endorsers <= np.floor(u.N / 2.0) else 0.0
        if F == 0.0:
            X[j] = 0.0
            continue
        if variant == "Pi2":
            a_fb = rng.choice(game.A, p=u.fb)
            w = u.delta_plus[a_fb]
        elif variant == "Pi1":
            w = u.W
        else:
            raise ValueError(variant)
        X[j] = nH * w * F
    return X


# --------------------------------------------------------------------------- #
# Conservative rollout value bound (Theorem 4) — shared math, reused by both
# E1 (tabular Monte-Carlo tail simulation) and E2 (real-environment rollouts).
# --------------------------------------------------------------------------- #
def rad_hoeffding(K: int, delta_prime: float, B_Q: float) -> float:
    """Hoeffding confidence radius  rad(K, δ') = B_Q · sqrt( ln(2/δ') / (2K) )."""
    return float(B_Q * np.sqrt(np.log(2.0 / delta_prime) / (2.0 * K)))


def wfb_plus_from_rollouts(
    Qref_hat: float, Qfb_hat: float, K: int, delta_prime: float, B_Q: float
) -> float:
    """Theorem 4 conservative fallback-swing bound from K-rollout Q estimates.

        W_fb^+(u) = [ Q̂^ref(u,a^ref) − Q̂^fb(u,a^fb) ]_+ + 2·rad(K, δ').

    Q̂^ref, Q̂^fb are empirical means of K independent resettable rollouts each
    (reference tail / fallback-then-reference tail).  Under Hoeffding's
    inequality per tail (union bound over the two tails, and over all logged
    failed units via δ' = δ_G / (2 m_F)), this dominates the true coordinate
    swing Δ_+(u, a^fb) jointly with probability ≥ 1 − δ_G (Theorem 4).
    """
    rad = rad_hoeffding(K, delta_prime, B_Q)
    return float(clip_pos(np.asarray(Qref_hat - Qfb_hat)) + 2.0 * rad)


def delta_prime_union(delta_G: float, m_F: int) -> float:
    """Per-tail-event confidence budget δ' = δ_G / (2 m_F) (Theorem 4).

    m_F is the number of logged failed units in the certification batch; the
    union bound is over the 2 tails (reference, fallback) of each such unit.
    m_F=0 (no failures logged) returns 1.0 (no radius needed / unused).
    """
    if m_F <= 0:
        return 1.0
    return float(delta_G / (2.0 * m_F))


def rad_empirical_bernstein(K: int, delta_prime: float, B_Q: float, sigma2_hat: float) -> float:
    """Empirical-Bernstein one-sided confidence radius for a K-sample mean of
    a random variable bounded within a range of width B_Q (Maurer & Pontil,
    2009 form -- the same functional form as `empirical_bernstein` above,
    applied per-tail with K resettable rollouts):

        rad_EB(K, δ') = sqrt(2 σ̂² ln(2/δ') / K) + 7 B_Q ln(2/δ') / (3(K-1)).

    Remark 6 in the paper notes empirical Bernstein may replace Hoeffding to
    sharpen rad(K,δ') at variance-dependent rates: when the true tail-return
    variance sigma2_hat is small relative to B_Q^2 (e.g. near-deterministic
    dynamics under a greedy policy), this radius is much tighter than
    `rad_hoeffding`, and its dominant (bias) term decays as 1/K rather than
    1/sqrt(K).
    """
    assert K >= 2, "empirical-Bernstein needs K >= 2"
    L = np.log(2.0 / delta_prime)
    return float(np.sqrt(2.0 * sigma2_hat * L / K) + 7.0 * B_Q * L / (3.0 * (K - 1)))


def wfb_plus_from_rollouts_eb(
    Qref_hat: float, Qfb_hat: float, K: int, delta_prime: float, B_Q: float,
    sigma2_ref: float, sigma2_fb: float,
) -> float:
    """Empirical-Bernstein variant of `wfb_plus_from_rollouts`: replaces the
    shared Hoeffding radius with two (potentially different) per-tail
    empirical-Bernstein radii, one for each tail's own measured variance:

        W_fb^+_EB(u) = [ Q̂^ref − Q̂^fb ]_+ + rad_EB(K,δ',B_Q,σ̂²_ref) + rad_EB(K,δ',B_Q,σ̂²_fb).

    Each one-sided EB event (Q̂^ref ≥ Q^ref − rad_EB_ref, Q̂^fb ≤ Q^fb + rad_EB_fb)
    holds with probability ≥ 1−δ' individually, matching Theorem 4's union
    bound over 2 tails × m_F logged units at δ' = δ_G/(2 m_F).
    """
    rad_ref = rad_empirical_bernstein(K, delta_prime, B_Q, sigma2_ref)
    rad_fb = rad_empirical_bernstein(K, delta_prime, B_Q, sigma2_fb)
    return float(clip_pos(np.asarray(Qref_hat - Qfb_hat)) + rad_ref + rad_fb)


# --------------------------------------------------------------------------- #
# Conditional-mean conservative estimator (Theorem 4, corrected) — replaces
# the per-unit high-probability radius bump above with a Jensen-inequality
# argument that holds unconditionally (no delta_G, no per-unit union bound,
# no joint-coverage event).
#
# Q_hat^ref, Q_hat^fb are each the empirical mean of K i.i.d. resettable
# rollouts (independent draws for the two tails), so E[Q_hat^ref] = Q^ref and
# E[Q_hat^fb] = Q^fb exactly (unbiased), for every K >= 1. Let
#     Y = Q_hat^ref - Q_hat^fb            (E[Y] = Delta = Q^ref - Q^fb)
#     W_tilde = [Y]_+ = max(Y, 0).
# [.]_+ is convex, so by (conditional) Jensen's inequality:
#     E[W_tilde | U, F=1, a_fb] = E[[Y]_+] >= [E[Y]]_+ = [Delta]_+ = Delta_+.
# This holds EXACTLY, for every K (verified by exact-Fraction enumeration on
# toy discrete tail-return distributions, including Delta<=0 boundary cases),
# with no probabilistic caveat -- unlike the old Hoeffding-radius construction,
# domination here is a deterministic inequality of expectations, not a
# high-probability event. The finite-K excess (Jensen gap) shrinks as K grows
# (empirically ~ K^{-1/2} on the toy checks), with a population-level upper
# bound nH*B_Q/sqrt(2K) (Popoviciu-type variance bound on a range-[0,B_Q]
# difference).
#
# Since E[X_j] = nH*E[F_j*W_tilde_j] >= nH*E_mu[g*Delta_+] = C2 >= L holds at
# the POPULATION level (in expectation over the K-rollout draws AND the unit
# sampling), Theorem 3's empirical-Bernstein concentration over the m i.i.d.
# draws of X_j (delta_B only -- no delta_G, no delta', no m_F) is now the
# ONLY probabilistic layer: B_hat = mean(X) + EB_radius(X, delta_B) >= E[X]
# >= C2 >= L with probability >= 1 - delta_B.
# --------------------------------------------------------------------------- #
def wtilde_jensen_from_rollouts(Qref_hat: float, Qfb_hat: float, B_Q: float) -> float:
    """Corrected Theorem-4 per-unit estimator: W_tilde(u) = [Q_hat^ref - Q_hat^fb]_+,
    capped at B_Q only for the clean range guarantee 0 <= W_tilde <= B_Q (NOT
    a confidence-radius bump -- there is no radius/delta' in this construction
    at all). Q_hat^ref, Q_hat^fb are empirical means over K resettable
    rollouts each, with returns in [0, B_Q].
    """
    w = float(clip_pos(np.asarray(Qref_hat - Qfb_hat)))
    return min(B_Q, w)


def jensen_population_bound_K(nH: float, B_Q: float, K: int) -> float:
    """Population-level upper bound on the Jensen gap nH*(E[W_tilde]-Delta_+),
    O(K^{-1/2}): nH*B_Q / sqrt(2K). Reported alongside the empirical Xbar/C2
    ratio as a theoretical cross-check, not used in the certificate itself.
    """
    return float(nH * B_Q / np.sqrt(2.0 * K))
