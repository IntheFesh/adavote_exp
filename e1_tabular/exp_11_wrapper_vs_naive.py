"""
exp_11_wrapper_vs_naive.py  —  Proposition 0.

Claim: under the η=0 condition, the agreement-gated AdaVote wrapper executes the
SAME per-unit action distribution as plain naive committee plurality voting
(when the wrapper's fallback IS the committee plurality).  For η>0 the wrapper
executes ψ = worst-in-G^η on the success side (≠ reference), so it diverges from
naive voting; the value gap grows with η.

Voting model (per unit u, reference action a_i^ref):
    each of N members endorses a_i^ref w.p. α (votes reference); otherwise votes
    a non-reference action ~ Uniform(non-ref).  #endorsers ~ Bin(N, α), so the
    majority-FAILURE probability (reference not a strict majority) is exactly
    g_N(α).  Naive executed action = plurality of the N votes (ties -> reference).

Decomposition of naive plurality:
    success (endorsers > ⌊N/2⌋) -> reference wins plurality, prob 1-g_N(α);
    failure -> plurality among the votes, prob g_N(α).
=>  naive_exec(a) = (1-g)·[a=ref] + g·P(plurality=a | failure).
The wrapper with fallback fb(a)=P(plurality=a|failure) reproduces this EXACTLY
(Prop 0).  We verify the per-unit distributions coincide on real games.

Outputs:
    results/data/exp_11_wrapper_vs_naive.csv
    results/figs/exp_11_value_gap_vs_eta.pdf
    one PASS/FAIL line -> results/summary.txt

PASS <=> zero per-unit mismatches at η=0; relative value gap ≈0 at η=0 and
         increasing with η.
"""

from __future__ import annotations

import argparse
import itertools
from functools import lru_cache

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import certificates as C
from common.games import make_game
from common.io_utils import data_path, fig_path, write_summary
from common.plotting import new_fig, save_pdf
from e1_tabular._common import build, make_alpha_fn, const_fn

# Tolerance for per-unit distribution equality.  The naive distribution is
# computed by exact enumeration over A**N vote profiles, so accumulated
# floating-point summation error is ~1e-9; 1e-7 is a safe equality threshold.
TOL = 1e-7


@lru_cache(maxsize=None)
def _naive_canonical(A: int, N: int, alpha_key: int):
    """Naive plurality distributions in a CANONICAL frame (reference index 0).

    alpha_key encodes round(alpha, 9) to make the cache hashable.
    Returns (exec_dist, fail_prob, cond_fail_dist) as tuples, length A.
    Enumerates all A**N vote profiles exactly.
    """
    alpha = alpha_key / 1e9
    vote_prob = np.full(A, (1.0 - alpha) / (A - 1))
    vote_prob[0] = alpha  # reference = index 0
    half = np.floor(N / 2.0)

    exec_dist = np.zeros(A)
    cond_fail = np.zeros(A)
    fail_prob = 0.0
    for profile in itertools.product(range(A), repeat=N):
        p = 1.0
        for v in profile:
            p *= vote_prob[v]
        if p == 0.0:
            continue
        counts = np.bincount(profile, minlength=A)
        endorsers = counts[0]
        # plurality, ties broken toward reference (index 0) then lowest index
        mx = counts.max()
        winner = 0 if counts[0] == mx else int(np.argmax(counts == mx))
        exec_dist[winner] += p
        if endorsers <= half:  # majority failure
            fail_prob += p
            cond_fail[winner] += p
    if fail_prob > 0:
        cond_fail = cond_fail / fail_prob
    return tuple(exec_dist), float(fail_prob), tuple(cond_fail)


def naive_for_unit(A, N, alpha, ref_i):
    """Naive (exec_dist, fail_prob, cond_fail) remapped so reference = ref_i."""
    exec_c, fail_p, cond_c = _naive_canonical(A, N, int(round(alpha * 1e9)))
    exec_c = np.array(exec_c)
    cond_c = np.array(cond_c)
    # canonical uses index 0 for reference and 1..A-1 for the non-ref actions
    # in their natural order; remap index 0 -> ref_i and shift the rest into the
    # remaining slots in increasing action order.
    nonref = [a for a in range(A) if a != ref_i]
    perm = np.empty(A, dtype=int)
    perm[0] = ref_i
    for k, a in enumerate(nonref):
        perm[k + 1] = a
    exec_d = np.zeros(A)
    cond_d = np.zeros(A)
    for c in range(A):
        exec_d[perm[c]] = exec_c[c]
        cond_d[perm[c]] = cond_c[c]
    return exec_d, fail_p, cond_d


def run_eta0(n_games, seed, S, A, H, N):
    """Verify wrapper == naive plurality per unit at η=0."""
    mismatches = 0
    total_units = 0
    max_diff = 0.0
    max_g_diff = 0.0
    rows = []
    for g in tqdm(range(n_games), desc="exp_11 eta=0"):
        gseed = seed * 100003 + g
        game = make_game(S=S, A=A, H=H, n=2, seed=gseed, transition="dependent")
        alpha_fn = make_alpha_fn(gseed)

        # wrapper fallback = committee plurality conditioned on failure
        def fb_fn(t, s, i, prefix, game, delta_plus, _af=alpha_fn):
            ref_i = int(game.ref[t, s, i])
            alpha = _af(t, s, i, prefix)
            _, _, cond = naive_for_unit(game.A, N, alpha, ref_i)
            if cond.sum() <= 0:  # g==0 edge: fallback never used; any valid dist
                cond = np.ones(game.A) / game.A
            return cond

        b = build(game, alpha_fn, const_fn(N), fb_fn=fb_fn, eta=0.0, psi_rule="ref")

        for key, u in b.table.items():
            t, s, i, prefix = key
            ref_i = u.ref_i
            alpha = alpha_fn(t, s, i, prefix)
            naive_exec, fail_p, cond = naive_for_unit(A, N, alpha, ref_i)
            # wrapper executed dist
            wrap = u.g * u.fb.copy()
            wrap[u.psi] += (1.0 - u.g)
            diff = float(np.max(np.abs(wrap - naive_exec)))
            gdiff = abs(fail_p - u.g)
            max_diff = max(max_diff, diff)
            max_g_diff = max(max_g_diff, gdiff)
            total_units += 1
            if diff > TOL:
                mismatches += 1
        rows.append(dict(game=g, max_exec_diff=max_diff, max_g_diff=max_g_diff))
    return mismatches, total_units, max_diff, max_g_diff, pd.DataFrame(rows)


def run_eta_positive(n_games, seed, S, A, H, N, etas):
    """For η>0 the wrapper executes ψ=worst-in-G^η on success (≠ ref) while
    naive voting still executes ref.  Report the relative value gap vs η."""
    rows = []
    for g in range(n_games):
        gseed = seed * 100003 + g
        game = make_game(S=S, A=A, H=H, n=2, seed=gseed, transition="dependent")
        alpha_fn = make_alpha_fn(gseed)
        # naive (== wrapper at η=0) reference value
        b_naive = build(game, alpha_fn, const_fn(N), eta=0.0, psi_rule="ref")
        V_naive = float(game.d0 @ b_naive.V_ctrl[0])
        scale = abs(V_naive) + 1e-9
        for eta in etas:
            b_w = build(game, alpha_fn, const_fn(N), eta=eta,
                        psi_rule="worst_in_Geta")
            V_w = float(game.d0 @ b_w.V_ctrl[0])
            rows.append(dict(game=g, eta=eta,
                             V_naive=V_naive, V_wrapper=V_w,
                             rel_gap=abs(V_w - V_naive) / scale))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--S", type=int, default=4)
    ap.add_argument("--A", type=int, default=3)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--N", type=int, default=5)
    args = ap.parse_args()

    # (a) η=0 equivalence
    mism, total, max_diff, max_g_diff, df0 = run_eta0(
        args.n_games, args.seed, args.S, args.A, args.H, args.N)

    # (b) η>0 divergence
    etas = [0.0, 0.1, 0.2, 0.3]
    dfp = run_eta_positive(min(args.n_games, 60), args.seed,
                           args.S, args.A, args.H, args.N, etas)

    # CSV
    dfp.to_csv(data_path("exp_11_wrapper_vs_naive.csv"), index=False)

    # Figure: relative value gap vs η
    gap_by_eta = dfp.groupby("eta")["rel_gap"].median()
    fig, ax = new_fig()
    ax.plot(gap_by_eta.index, gap_by_eta.values, "o-", color="darkred")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("median |V_wrapper - V_naive| / |V_naive|")
    ax.set_title("exp_11  wrapper vs naive: value gap grows with η\n"
                 "(η=0 => identical, Prop. 0)")
    save_pdf(fig, fig_path("exp_11_value_gap_vs_eta.pdf"))

    gap0 = float(gap_by_eta.get(0.0, 0.0))
    gap_max = float(gap_by_eta.loc[max(etas)])
    increasing = all(
        gap_by_eta.loc[etas[k]] <= gap_by_eta.loc[etas[k + 1]] + 1e-9
        for k in range(len(etas) - 1)
    )
    pass_eq = (mism == 0) and (max_g_diff <= TOL)
    pass_div = (gap0 <= TOL) and (gap_max > gap0) and increasing
    status = "PASS" if (pass_eq and pass_div) else "FAIL"

    write_summary(
        f"exp_11 wrapper_vs_naive [{status}] "
        f"eta0_unit_mismatches={mism}/{total} "
        f"max_exec_diff={max_diff:.2e} max_g_diff={max_g_diff:.2e} "
        f"rel_gap(eta=0)={gap0:.2e} rel_gap(eta={max(etas)})={gap_max:.3f} "
        f"monotone_in_eta={increasing}"
    )
    if not pass_eq:
        print(f"[FAIL] η=0 equivalence broken: {mism} mismatches, "
              f"max_g_diff={max_g_diff:.2e}")
    if not pass_div:
        print(f"[FAIL] η>0 divergence not as expected: gap0={gap0:.2e} "
              f"gap_max={gap_max:.3f} monotone={increasing}")


if __name__ == "__main__":
    main()
