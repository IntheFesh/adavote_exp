"""
exp_10b_anchored_eligibility.py — Anchored-committee eligibility threshold.

Companion to exp_10 (unanchored, threshold 1/2). Verifies the EXACT threshold
for the ANCHORED committee protocol (reference is a seated member that always
self-endorses; the remaining N-1 advisors are conditional-i.i.d. Bernoulli(p)).

Anchored majority-failure (N odd):
    h_N(p) = Pr[ 1 + Bin(N-1, p) <= floor(N/2) ] = Pr[ Bin(N-1, p) <= (N-3)/2 ].

CLAIM (anchored eligibility iff):
    h_{N+2}(p) < h_N(p)   <=>   p > (N+1)/(2N),  equality exactly at p*=(N+1)/(2N).
Same stepping identity gives threshold 1/2 for the unanchored protocol
(recovering exp_10): a PARALLEL protocol, not a strengthening.

ADVERSARIAL: written to BREAK the claim. Exact Fraction arithmetic on a grid
straddling each threshold; asserts sign (+) below p*, (0) at p*, (-) above p*,
FAILS LOUDLY on violation. Cross-checks h_N against independent Monte-Carlo.

PASS <=> exact threshold verified at every N with 0 sign violations AND
         unanchored 1/2 recovered AND MC matches closed form.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
from math import comb, floor
import numpy as np
import pandas as pd
from common.io_utils import data_path, write_summary


def cdf_bin_exact(n: int, k: int, p: F) -> F:
    if k < 0:
        return F(0)
    if k >= n:
        return F(1)
    q = 1 - p
    return sum(comb(n, j) * p**j * q**(n - j) for j in range(k + 1))


def h_anchored_exact(N: int, p: F) -> F:
    return cdf_bin_exact(N - 1, (N - 3) // 2, p)


def g_unanchored_exact(N: int, p: F) -> F:
    return cdf_bin_exact(N, floor(N / 2), p)


def h_anchored_mc(N: int, p: float, trials: int, rng: np.random.Generator) -> float:
    adv = rng.random((trials, N - 1)) < p
    endorse = 1 + adv.sum(axis=1)
    return float((endorse <= floor(N / 2)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=int, nargs="+",
                    default=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
    ap.add_argument("--eps_den", type=int, default=1000)
    ap.add_argument("--mc_trials", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    eps = F(1, args.eps_den)
    sign_violations = 0
    mc_max_err = 0.0
    rows = []

    print(f"{'N':>3} {'p*':>14} {'diff@p*':>10} {'below':>6} {'at':>4} {'above':>6} {'MC_err':>9}")
    print("-" * 60)
    for N in args.Ns:
        pstar = F(N + 1, 2 * N)
        d_at = h_anchored_exact(N + 2, pstar) - h_anchored_exact(N, pstar)
        d_below = h_anchored_exact(N + 2, pstar - eps) - h_anchored_exact(N, pstar - eps)
        d_above = h_anchored_exact(N + 2, pstar + eps) - h_anchored_exact(N, pstar + eps)
        below_ok = d_below > 0
        at_ok = d_at == 0
        above_ok = d_above < 0
        if not (below_ok and at_ok and above_ok):
            sign_violations += 1
        p_test = float(pstar) - 0.05
        if 0.0 < p_test < 1.0:
            mc = h_anchored_mc(N, p_test, args.mc_trials, rng)
            formula = float(h_anchored_exact(N, F(p_test).limit_denominator(10**6)))
            err = abs(mc - formula)
            mc_max_err = max(mc_max_err, err)
        else:
            err = 0.0
        print(f"{N:>3} {str(pstar):>14} {str(d_at):>10} "
              f"{'+' if below_ok else 'X':>6} {'0' if at_ok else 'X':>4} "
              f"{'-' if above_ok else 'X':>6} {err:>9.5f}")
        rows.append(dict(N=N, p_star=float(pstar), p_star_exact=str(pstar),
                         below_increasing=int(below_ok), flat_at_threshold=int(at_ok),
                         above_decreasing=int(above_ok), mc_err=err))

    unanch_violations = 0
    for N in args.Ns:
        half = F(1, 2)
        d_at = g_unanchored_exact(N + 2, half) - g_unanchored_exact(N, half)
        d_below = g_unanchored_exact(N + 2, half - eps) - g_unanchored_exact(N, half - eps)
        d_above = g_unanchored_exact(N + 2, half + eps) - g_unanchored_exact(N, half + eps)
        if not ((d_below > 0) and (d_at == 0) and (d_above < 0)):
            unanch_violations += 1

    df = pd.DataFrame(rows)
    df.to_csv(data_path("exp_10b_anchored_eligibility.csv"), index=False)

    status = ("PASS" if (sign_violations == 0 and unanch_violations == 0
                         and mc_max_err < 0.01) else "FAIL")
    write_summary(
        f"exp_10b anchored_eligibility [{status}] "
        f"anchored_sign_violations={sign_violations} "
        f"unanchored_violations={unanch_violations} "
        f"mc_max_err={mc_max_err:.5f} n_N={len(args.Ns)}"
    )


if __name__ == "__main__":
    main()
