"""
e1_tabular/_common.py — shared scaffolding for the E1 tabular experiments.

Provides a standard "build a controller + certificates for a game" pipeline
with configurable, reproducible per-unit endorsement probabilities α(u) and
committee sizes N(u).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from common import certificates as C
from common.certificates import Game, UnitKey, UnitInfo


_ALPHA_TAG = 1  # integer stream tag, disambiguates this seed use from others

def make_alpha_fn(seed: int, lo: float = 0.55, hi: float = 0.9):
    """Per-unit α drawn deterministically from U(lo, hi) keyed on the unit.

    Uses np.random.default_rng with an all-integer entropy tuple (unit key +
    an integer stream tag), NOT Python's built-in hash(). hash() of a tuple
    containing a str is randomized per-process (PEP 456 SipHash) unless
    PYTHONHASHSEED is fixed, which silently made this "reproducible" claim
    false: two runs with an identical `seed` produced different alpha(u), and
    hence different C0/C1/C2/true_loss, every fresh process invocation.
    default_rng's SeedSequence is deterministic across processes/machines for
    a given integer entropy list, so this is genuinely reproducible.
    len(prefix) is included ahead of the prefix digits to disambiguate
    prefixes of different lengths (e.g. (1,2) vs (12,)).
    """
    def alpha_fn(t, s, i, prefix):
        entropy = [seed, _ALPHA_TAG, t, s, i, len(prefix)] + [int(p) for p in prefix]
        r = np.random.default_rng(entropy).random()
        return lo + (hi - lo) * r
    return alpha_fn


def const_fn(value):
    def f(*args):
        return value
    return f


@dataclass
class Built:
    game: Game
    V_ref: np.ndarray
    table: Dict[UnitKey, UnitInfo]
    V_ctrl: np.ndarray
    d_ctrl: np.ndarray
    rho: Dict[UnitKey, float]
    true_loss: float
    chain: Dict[str, float]


def build(game: Game, alpha_fn, N_fn, fb_fn=C.uniform_fb,
          eta: float = 0.0, psi_rule: str = "ref") -> Built:
    V_ref = C.ref_value_iteration(game)
    table = C.build_unit_table(game, V_ref, alpha_fn, N_fn, fb_fn,
                               eta=eta, psi_rule=psi_rule)
    V_ctrl, d_ctrl = C.controller_value(game, table)
    rho = C.unit_occupancy(game, table, d_ctrl)
    tl = C.true_loss(game, V_ref, V_ctrl)
    chain = C.cert_chain(game, table, rho)
    return Built(game, V_ref, table, V_ctrl, d_ctrl, rho, tl, chain)
