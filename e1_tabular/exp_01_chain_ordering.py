"""
exp_01_chain_ordering.py  —  Theorem 1b (three-level certificate chain).

Claim verified:  for η=0, on every random game,
        C_0  ≥  C*  ≥  C̃*  ≥  true loss,
with ZERO instance-level violations.  We also report the distribution of the
ratios C_0/true, C*/true, C̃*/true and the coordinate-decomposition slack
(C̃* - true)/true.

Outputs:
    results/data/exp_01_chain_ordering.csv
    results/figs/exp_01_ratios_box.pdf
    results/figs/exp_01_slack_hist.pdf
    one PASS/FAIL line -> results/summary.txt

PASS  <=>  zero chain violations across all games.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.games import make_game
from common.io_utils import data_path, fig_path, write_summary
from common.plotting import new_fig, save_pdf
from e1_tabular._common import build, make_alpha_fn, const_fn

TOL = 1e-9


def run(n_games: int, seed: int, S: int, A: int, H: int, N: int):
    rows = []
    viol = 0
    for g in tqdm(range(n_games), desc="exp_01"):
        gseed = seed * 100000 + g
        game = make_game(S=S, A=A, H=H, n=2, seed=gseed, transition="dependent")
        b = build(game, make_alpha_fn(gseed), const_fn(N))
        tl = b.true_loss
        ch = b.chain
        # instance-level chain assertions
        ok = (ch["C0"] + TOL >= ch["Cstar"]
              and ch["Cstar"] + TOL >= ch["Ctilde"]
              and ch["Ctilde"] + TOL >= tl)
        if not ok:
            viol += 1
            print(f"[VIOLATION] game {g}: C0={ch['C0']:.5f} C*={ch['Cstar']:.5f} "
                  f"C~={ch['Ctilde']:.5f} true={tl:.5f}")
        # ratios only meaningful when there is genuine loss
        safe = max(tl, 1e-9)
        rows.append(dict(
            game=g, true_loss=tl, C0=ch["C0"], Cstar=ch["Cstar"],
            Ctilde=ch["Ctilde"], delta_r=game.delta_r,
            r_C0=ch["C0"] / safe, r_Cstar=ch["Cstar"] / safe,
            r_Ctilde=ch["Ctilde"] / safe,
            slack_Ctilde=(ch["Ctilde"] - tl) / safe,
            true_over_range=tl / max(game.delta_r * game.H, 1e-9),
            usable=tl > 1e-6,
        ))
    df = pd.DataFrame(rows)
    return df, viol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--S", type=int, default=4)
    ap.add_argument("--A", type=int, default=3)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--N", type=int, default=5)
    args = ap.parse_args()

    df, viol = run(args.n_games, args.seed, args.S, args.A, args.H, args.N)
    csv = data_path("exp_01_chain_ordering.csv")
    df.to_csv(csv, index=False)

    use = df[df.usable]
    # ----- ratios boxplot -----
    fig, ax = new_fig()
    data = [use.r_C0, use.r_Cstar, use.r_Ctilde]
    ax.boxplot(data, tick_labels=["C0/true", "C*/true", "C~*/true"], showfliers=False)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_ylabel("certificate / true loss")
    ax.set_title("exp_01  three-level certificate ratios")
    save_pdf(fig, fig_path("exp_01_ratios_box.pdf"))

    # ----- slack histogram -----
    fig, ax = new_fig()
    ax.hist(use.slack_Ctilde, bins=30, color="steelblue", edgecolor="k", lw=0.3)
    ax.set_xlabel(r"(C$\tilde{}^*$ - true)/true  (coordinate-decomposition slack)")
    ax.set_ylabel("# games")
    ax.set_title("exp_01  tightest-certificate slack")
    save_pdf(fig, fig_path("exp_01_slack_hist.pdf"))

    med = use[["r_C0", "r_Cstar", "r_Ctilde"]].median()
    status = "PASS" if viol == 0 else "FAIL"
    write_summary(
        f"exp_01 chain_ordering [{status}] games={len(df)} violations={viol} "
        f"median C0/true={med.r_C0:.2f} C*/true={med.r_Cstar:.2f} "
        f"C~*/true={med.r_Ctilde:.2f}"
    )


if __name__ == "__main__":
    main()
