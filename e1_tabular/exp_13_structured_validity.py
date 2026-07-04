"""
exp_13_structured_validity.py — Route 1: validity chain + tightness +
prefix-drift necessity on a STRUCTURED, cooperation-semantic game family,
replacing the i.i.d.-uniform-random-reward games used by exp_01/exp_06 (which
are always instantiated at a fixed n=2 agents there).

Motivation (reviewer: "validity checks are synthetic/tabular"): the existing
exact-DP suite is valid but has no real cooperative structure and never
varies past 2 agents. This experiment reuses the SAME certificate machinery
but on common.games_structured.make_resource_coordination_game -- a shared
depleting-resource collision-avoidance game where reward is genuinely
COOPERATIVE (joint over-claiming zeroes out everyone's reward that round) --
swept across n = 2..7 agents (the exact-DP-feasible envelope established by
the Pilot-B feasibility sweep: joint action space A^n is the exponential
bottleneck; S, H scale ~linearly and are not limiting).

Reports, per instance across VARYING n (not fixed at 2):
  (a) Chain ordering C0 >= C1 >= C2 >= true_loss (zero violations expected).
  (b) Tightness ratios C0/L, C1/L, C2/L (does tightness hold up as n grows?).
  (c) Prefix-drift necessity: the reference-prefix naive bound (as in exp_06)
      under-bounds true loss under this game's action-dependent (resource-
      depleting) transitions, while the joint C2 remains valid.

PASS <=> zero chain violations at every n AND naive bound shows >=1 violation
         (prefix drift is real here too) AND joint C2 has zero violations.

Outputs:
    results/data/exp_13_structured_validity.csv
    results/figs/exp_13_ratios_by_n.pdf
    results/figs/exp_13_naive_viol_by_n.pdf
    one PASS/FAIL line -> results/summary.txt
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import certificates as C
from common.games_structured import make_resource_coordination_game
from common.io_utils import data_path, fig_path, write_summary
from common.plotting import new_fig, save_pdf
from e1_tabular._common import build, make_alpha_fn, const_fn
from e1_tabular.exp_06_prefix_drift import naive_bound, compute_drift_rate

TOL = 1e-9


def run(games_schedule, n_grid, seed, S, A, H, N):
    rows = []
    master_rng = np.random.default_rng(seed)

    for n, n_games in zip(n_grid, games_schedule):
        for g in tqdm(range(n_games), desc=f"exp_13 n={n}"):
            gseed = int(master_rng.integers(0, 2**31))
            replenish = bool(master_rng.integers(0, 2))
            game = make_resource_coordination_game(
                S=S, A=A, H=H, n=n, seed=gseed, resources_replenish=replenish)
            alpha_fn = make_alpha_fn(gseed)
            b = build(game, alpha_fn, const_fn(N), eta=0.0, psi_rule="ref")

            tl = b.true_loss
            ch = b.chain
            chain_ok = (ch["C0"] + TOL >= ch["Cstar"]
                        and ch["Cstar"] + TOL >= ch["Ctilde"]
                        and ch["Ctilde"] + TOL >= tl)

            naive = naive_bound(game, b.table, b.d_ctrl)
            drift = compute_drift_rate(game, b.table)
            naive_viol = int(naive + TOL < tl)
            joint_viol = int(ch["Ctilde"] + TOL < tl)

            safe = max(tl, 1e-9)
            rows.append(dict(
                n=n, game=g, seed=gseed, replenish=replenish, true_loss=tl,
                C0=ch["C0"], Cstar=ch["Cstar"], Ctilde=ch["Ctilde"],
                r_C0=ch["C0"] / safe, r_Cstar=ch["Cstar"] / safe, r_Ctilde=ch["Ctilde"] / safe,
                chain_violation=int(not chain_ok),
                naive=naive, drift_rate=drift,
                naive_violation=naive_viol, joint_violation=joint_viol,
                usable=tl > 1e-6,
            ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="exp_13: structured cooperation-semantic validity + prefix-drift, swept over n agents")
    ap.add_argument("--n_games_per_n", type=int, default=40)
    ap.add_argument("--n_grid", default="2,3,4,5,6,7")
    ap.add_argument("--games_schedule", default="",
                     help="optional comma list of per-n game counts (same length as --n_grid); "
                          "overrides --n_games_per_n. Use to shrink counts for large (slow) n.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--S", type=int, default=4)
    ap.add_argument("--A", type=int, default=3)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--N", type=int, default=5)
    args = ap.parse_args()

    n_grid = [int(x) for x in args.n_grid.split(",")]
    if args.games_schedule:
        games_schedule = [int(x) for x in args.games_schedule.split(",")]
        assert len(games_schedule) == len(n_grid)
    else:
        games_schedule = [args.n_games_per_n] * len(n_grid)
    df = run(games_schedule, n_grid, args.seed, args.S, args.A, args.H, args.N)
    csv_path = data_path("exp_13_structured_validity.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

    # --- Figure 1: tightness ratios by n ---
    use = df[df.usable]
    fig, ax = new_fig(figsize=(7, 4))
    positions = []
    data = []
    labels = []
    width = 0.25
    for idx, n in enumerate(n_grid):
        sub = use[use.n == n]
        if len(sub) == 0:
            continue
        for j, col in enumerate(["r_C0", "r_Cstar", "r_Ctilde"]):
            positions.append(idx + (j - 1) * width)
            data.append(sub[col].values)
    bp = ax.boxplot(data, positions=positions, widths=width * 0.9, showfliers=False,
                     patch_artist=True)
    colors = ["lightcoral", "khaki", "lightgreen"] * len(n_grid)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(len(n_grid)))
    ax.set_xticklabels([str(n) for n in n_grid])
    ax.set_xlabel("number of agents n (structured resource-coordination game)")
    ax.set_ylabel("certificate / true loss")
    ax.set_title("Structured game: chain tightness vs. agent count n\n(red=C0/L, yellow=C1/L, green=C2/L)")
    save_pdf(fig, fig_path("exp_13_ratios_by_n.pdf"))

    # --- Figure 2: naive violation rate by n ---
    agg = df.groupby("n").agg(
        naive_viol_rate=("naive_violation", "mean"),
        joint_viol_rate=("joint_violation", "mean"),
        chain_viol_rate=("chain_violation", "mean"),
        drift_rate=("drift_rate", "mean"),
        median_r_Ctilde=("r_Ctilde", "median"),
    ).reindex(n_grid)
    print("\nPer-n aggregate:")
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"))

    fig, ax = new_fig()
    ax.bar(np.arange(len(n_grid)), agg["naive_viol_rate"].values * 100,
           color="tomato", edgecolor="k", lw=0.5)
    ax.set_xticks(range(len(n_grid)))
    ax.set_xticklabels([str(n) for n in n_grid])
    ax.set_xlabel("number of agents n")
    ax.set_ylabel("naive (reference-prefix) violation rate (%)")
    ax.set_title("Structured game: prefix-drift necessity vs. agent count n")
    save_pdf(fig, fig_path("exp_13_naive_viol_by_n.pdf"))

    # --- PASS/FAIL ---
    total_chain_viol = int(df["chain_violation"].sum())
    total_joint_viol = int(df["joint_violation"].sum())
    total_naive_viol = int(df["naive_violation"].sum())
    status = "PASS" if (total_chain_viol == 0 and total_joint_viol == 0 and total_naive_viol > 0) else "FAIL"

    med = use[["r_C0", "r_Cstar", "r_Ctilde"]].median()
    write_summary(
        f"exp_13 structured_validity [{status}] "
        f"n_grid={n_grid} total_games={len(df)} "
        f"chain_violations={total_chain_viol} joint_violations={total_joint_viol} "
        f"naive_violations={total_naive_viol} "
        f"median C0/L={med.r_C0:.2f} C1/L={med.r_Cstar:.2f} C2/L={med.r_Ctilde:.2f}"
    )


if __name__ == "__main__":
    main()
