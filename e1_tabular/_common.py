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


def make_alpha_fn(seed: int, lo: float = 0.55, hi: float = 0.9):
    """Per-unit α drawn deterministically from U(lo, hi) keyed on the unit.

    Using a hash of the unit key keeps α reproducible and independent of
    enumeration order.
    """
    def alpha_fn(t, s, i, prefix):
        h = hash((seed, "alpha", t, s, i, tuple(prefix))) & 0xFFFFFFFF
        r = (h / 0xFFFFFFFF)
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
