"""
exp_02_eta_positive_success_term.py  —  Paper §5 Remark: η>0 and the
success-term gap.

Claim verified:
  * pure-failure certificate  Σ_u ρ(u) g(u) W̃(u)  (cert_failure_only)
    can UNDERBOUND the true loss when η>0 and psi_rule="worst_in_Geta"
    (violation rate 0 at η=0, rises substantially for η≥0.2).
  * success-inclusive certificate  Σ_u ρ(u)[(1-g)w_ψ + g W̃]
    (cert_success_inclusive) stays valid (0 violations) at ALL η.

PASS <=> success_inclusive has 0 violations at every η AND
         pure-failure violation rate is 0 at η=0 and is substantial
         (≥ 5 %) for at least one η > 0.

Outputs:
    results/data/exp_02_eta_positive_success_term.csv
    results/figs/exp_02_violation_rate.pdf
    one PASS/FAIL line  ->  results/summary.txt
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.games import make_game
from common import certificates as C
from common.io_utils import data_path, fig_path, write_summary
from common.plotting import new_fig, save_pdf
from e1_tabular._common import build, make_alpha_fn, const_fn

TOL = 1e-9
ETAS = [0.0, 0.1, 0.2, 0.3]


def run(n_games: int, seed: int, S: int, A: int, H: int, N: int):
    rows = []

    for eta in ETAS:
        desc = f"exp_02 eta={eta:.1f}"
        for g in tqdm(range(n_games), desc=desc):
            gseed = seed * 100000 + g
            game = make_game(S=S, A=A, H=H, n=2, seed=gseed,
                             transition="dependent")
            alpha_fn = make_alpha_fn(gseed)
            N_fn = const_fn(N)

            # Build with worst_in_Geta to expose the success-term gap
            b = build(game, alpha_fn, N_fn,
                      eta=eta, psi_rule="worst_in_Geta")

            tl = b.true_loss
            fail_only = C.cert_failure_only(game, b.table, b.rho)
            succ_incl = C.cert_success_inclusive(game, b.table, b.rho)

            viol_fail = int(fail_only + TOL < tl)
            viol_succ = int(succ_incl + TOL < tl)

            if viol_succ:
                print(f"[BUG] success_inclusive VIOLATED: eta={eta} game={g} "
                      f"succ_incl={succ_incl:.6f} true_loss={tl:.6f}")
                assert False, "success_inclusive must never violate"

            rows.append(dict(
                eta=eta, game=g,
                true_loss=tl,
                cert_failure_only=fail_only,
                cert_success_inclusive=succ_incl,
                viol_failure_only=viol_fail,
                viol_success_inclusive=viol_succ,
            ))

    df = pd.DataFrame(rows)
    return df


def main():
    ap = argparse.ArgumentParser(
        description="exp_02: eta>0 success-term gap counterexample")
    ap.add_argument("--n_games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--S", type=int, default=4)
    ap.add_argument("--A", type=int, default=4)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--N", type=int, default=5)
    args = ap.parse_args()

    df = run(args.n_games, args.seed, args.S, args.A, args.H, args.N)

    # --- CSV ---
    csv_file = data_path("exp_02_eta_positive_success_term.csv")
    df.to_csv(csv_file, index=False)

    # --- Summary stats per eta ---
    rate_fail = {}
    rate_succ = {}
    for eta in ETAS:
        sub = df[df.eta == eta]
        rate_fail[eta] = sub.viol_failure_only.mean()
        rate_succ[eta] = sub.viol_success_inclusive.mean()

    # --- PDF: violation rate vs eta ---
    fig, ax = new_fig()
    etas_x = ETAS
    ax.plot(etas_x, [rate_fail[e] for e in etas_x],
            marker="o", label=r"$nH\,\mathbb{E}[g \cdot W_{fb}]$  (failure-only)")
    ax.plot(etas_x, [rate_succ[e] for e in etas_x],
            marker="s", linestyle="--",
            label=r"$nH\,\mathbb{E}[(1-g)w_\psi + g \cdot W_{fb}]$  (success-inclusive)")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("violation rate  (fraction of games)")
    ax.set_title(r"$\eta = 0$ boundary: pure-failure term validity")
    ax.legend()
    fig.savefig(fig_path("eta_success_term_violation.pdf"), format="pdf", bbox_inches="tight")
    save_pdf(fig, fig_path("exp_02_violation_rate.pdf"))

    # --- PASS/FAIL ---
    succ_all_zero = all(rate_succ[e] == 0.0 for e in ETAS)
    fail_zero_at_zero = (rate_fail[0.0] == 0.0)
    fail_rises = any(rate_fail[e] >= 0.05 for e in ETAS if e > 0.0)
    status = "PASS" if (succ_all_zero and fail_zero_at_zero and fail_rises) else "FAIL"

    rate_str = "  ".join(
        f"eta={e:.1f}: fail_rate={rate_fail[e]:.3f} succ_rate={rate_succ[e]:.3f}"
        for e in ETAS
    )
    write_summary(
        f"exp_02 eta_positive_success_term [{status}] games={args.n_games}  "
        + rate_str
    )


if __name__ == "__main__":
    main()
