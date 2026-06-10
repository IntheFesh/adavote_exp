"""
exp_06_prefix_drift.py  —  Prefix drift: why naive single-agent bound fails.

Action-DEPENDENT transitions cause PREFIX DRIFT: the controller's executed
prefix a_{<i} differs from the reference prefix, and coordinate i's swing
W̃ depends on that prefix.

(a) Prefix-drift occurrence rate: fraction of (game, unit with g>0) where
    the controller assigns positive probability to a NON-reference prefix.

(b) Over many random games, compute true_loss, naive bound, and joint C̃*.
    - naive violation rate (fraction with naive + 1e-9 < true_loss)
    - joint C̃* violation rate (expect exactly 0)
    Sweep: N in {3,5,7}, H in {3,4,5}.

(c) Save a concrete instance where naive STRICTLY underbounds true loss
    with margin > 1e-3, to results/data/exp_06_counterexample.txt.

Outputs:
    results/data/exp_06_prefix_drift.csv
    results/data/exp_06_counterexample.txt
    results/figs/exp_06_naive_viol_rate.pdf
    results/figs/exp_06_naive_vs_true_scatter.pdf
    one PASS/FAIL line -> results/summary.txt

PASS <=> drift_rate ~100% when g>0; joint C̃* 0 violations; naive shows
         persistent nonzero violations; a margin>1e-3 counterexample saved.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import certificates as C
from common.games import make_game
from common.io_utils import data_path, fig_path, write_summary
from common.plotting import new_fig, save_pdf
from e1_tabular._common import build, make_alpha_fn, const_fn

TOL = 1e-9
COUNTEREXAMPLE_MARGIN = 1e-3


def naive_bound(game: C.Game, table: dict, d_ctrl: np.ndarray) -> float:
    """Naive single-agent bound: evaluates coordinate i's swing ONLY at the
    REFERENCE prefix (a_{<i} = ref actions of agents <i), weighted by
    state occupancy d^ctrl_t(s).

    naive = Σ_{t,s,i} d^ctrl_t(s) · g(t,s,i,ref_prefix) · W̃(t,s,i,ref_prefix)

    where ref_prefix = tuple of reference actions for agents < i at (t,s).
    """
    total = 0.0
    for t in range(game.H):
        for s in range(game.S):
            ref_j = game.ref_joint(t, s)  # tuple of all n ref actions
            for i in range(game.n):
                ref_prefix = tuple(ref_j[:i])
                key = (t, s, i, ref_prefix)
                if key not in table:
                    continue
                u = table[key]
                total += d_ctrl[t, s] * u.g * u.Wtilde
    return total


def compute_drift_rate(game: C.Game, table: dict) -> float:
    """Fraction of (t,s,i) with g>0 where the controller puts positive mass
    on a NON-reference prefix (prefix_probs has support off the ref prefix).

    Only coordinates i>=1 are counted: coordinate i=0 has an empty prefix and
    therefore CANNOT drift by construction, so including it would mechanically
    cap the rate at (n-1)/n.  Among coordinates where a prefix exists, drift
    occurs essentially whenever g>0.
    """
    drifted = 0
    total_eligible = 0
    for t in range(game.H):
        for s in range(game.S):
            _, prefix_probs = C.controller_state_dist(game, table, t, s)
            ref_j = game.ref_joint(t, s)
            for i in range(1, game.n):  # i=0 cannot drift (empty prefix)
                ref_prefix = tuple(ref_j[:i])
                key = (t, s, i, ref_prefix)
                if key not in table:
                    continue
                u = table[key]
                if u.g <= 0.0:
                    continue
                total_eligible += 1
                for prefix, prob in prefix_probs.items():
                    if len(prefix) == i and prefix != ref_prefix and prob > TOL:
                        drifted += 1
                        break
    if total_eligible == 0:
        return 0.0
    return drifted / total_eligible


def run_setting(n_games: int, seed: int, S: int, A: int, H: int, N: int,
                setting_label: str, counterexample_store: list):
    """Run one (N, H) setting and return per-game rows."""
    rows = []
    for g in range(n_games):
        gseed = seed * 10_000_000 + N * 1_000_000 + H * 100_000 + g
        game = make_game(S=S, A=A, H=H, n=2, seed=gseed, transition="dependent")
        alpha_fn = make_alpha_fn(gseed)
        b = build(game, alpha_fn, const_fn(N))
        tl = b.true_loss
        ctilde = b.chain["Ctilde"]
        naive = naive_bound(game, b.table, b.d_ctrl)
        drift = compute_drift_rate(game, b.table)

        naive_viol = int(naive + TOL < tl)
        joint_viol = int(ctilde + TOL < tl)

        margin = tl - naive
        # Track the LARGEST-margin underbound instance seen so far.
        best = counterexample_store[0] if counterexample_store else None
        if margin > TOL and (best is None or margin > best["margin"]):
            ce = {
                "setting_label": setting_label,
                "game_seed": gseed,
                "S": S, "A": A, "H": H, "N": N,
                "true_loss": tl,
                "naive_bound": naive,
                "ctilde": ctilde,
                "margin": margin,
                "game": game,
                "table": b.table,
                "d_ctrl": b.d_ctrl,
                "V_ref": b.V_ref,
            }
            counterexample_store.clear()
            counterexample_store.append(ce)

        rows.append(dict(
            setting=setting_label, game_seed=gseed, H=H, N=N,
            drift_rate=drift, naive=naive, true_loss=tl, ctilde=ctilde,
            naive_violation=naive_viol, joint_violation=joint_viol,
        ))
    return rows


def save_counterexample(ce: dict) -> str:
    """Write mechanistic counterexample dump to file."""
    game = ce["game"]
    table = ce["table"]
    d_ctrl = ce["d_ctrl"]

    lines = []
    lines.append("=" * 70)
    lines.append("exp_06 — PREFIX DRIFT COUNTEREXAMPLE")
    lines.append("=" * 70)
    lines.append(f"Setting:    {ce['setting_label']}")
    lines.append(f"Game seed:  {ce['game_seed']}")
    lines.append(f"S={ce['S']}, A={ce['A']}, H={ce['H']}, N={ce['N']}, n={game.n}")
    lines.append(f"true_loss:   {ce['true_loss']:.8f}")
    lines.append(f"naive_bound: {ce['naive_bound']:.8f}")
    lines.append(f"C~*:         {ce['ctilde']:.8f}")
    tag = "[> 1e-3  CONFIRMED]" if ce["margin"] > 1e-3 else "[largest found]"
    lines.append(f"Margin (true_loss - naive): {ce['margin']:.8f}  {tag}")
    lines.append("")
    lines.append("Mechanistic note:")
    lines.append(
        "  The naive bound computes W~(t,s,i,ref_prefix) at the REFERENCE prefix only,\n"
        "  then multiplies by d^ctrl_t(s), ignoring that the controller's actual\n"
        "  prefix distribution at (t,s) places positive mass on NON-reference prefixes.\n"
        "  Because transitions are action-DEPENDENT, different prefixes lead to\n"
        "  different next-state distributions and hence different Q-values and W~.\n"
        "  The joint C~* correctly integrates over all prefixes via\n"
        "      rho(u) = d_ctrl[t,s] * P(prefix a_{<i} | t,s),\n"
        "  capturing the full drift. The naive bound misses drifted-prefix mass,\n"
        "  underbounding true loss by the margin shown above."
    )
    lines.append("")

    # Find the most impactful drifted units
    drift_units = []
    for t in range(game.H):
        for s in range(game.S):
            _, prefix_probs = C.controller_state_dist(game, table, t, s)
            ref_j = game.ref_joint(t, s)
            for i in range(game.n):
                ref_prefix = tuple(ref_j[:i])
                for prefix_tuple in itertools.product(range(game.A), repeat=i):
                    prefix = tuple(prefix_tuple)
                    if prefix == ref_prefix:
                        continue
                    prob = prefix_probs.get(prefix, 0.0)
                    if prob <= TOL:
                        continue
                    key = (t, s, i, prefix)
                    if key not in table:
                        continue
                    u = table[key]
                    if u.g <= 0.0:
                        continue
                    contrib = d_ctrl[t, s] * prob * u.g * u.Wtilde
                    if contrib > 1e-8:
                        drift_units.append((contrib, t, s, i, prefix, ref_prefix,
                                            prob, u.g, u.Wtilde))
    drift_units.sort(reverse=True)
    lines.append("Top drifted units (non-ref prefix, g>0, rho*g*W~ contribution):")
    for contrib, t, s, i, prefix, ref_prefix, prob, g, Wtilde in drift_units[:10]:
        lines.append(
            f"  (t={t},s={s},i={i}) prefix={prefix} ref_prefix={ref_prefix}: "
            f"P(prefix|t,s)={prob:.4f}, g={g:.4f}, W~={Wtilde:.4f}, "
            f"rho*g*W~={contrib:.6f}"
        )
    lines.append("")

    # Show Q-tables for key units
    lines.append("Q-tables at affected (t,s) positions:")
    shown = set()
    for contrib, t, s, i, prefix, ref_prefix, prob, g, Wtilde in drift_units[:5]:
        if (t, s) in shown:
            continue
        shown.add((t, s))
        lines.append(f"  (t={t}, s={s})  ref_joint={game.ref_joint(t,s)}")
        for ii in range(game.n):
            ref_p = tuple(game.ref_joint(t, s)[:ii])
            key_ref = (t, s, ii, ref_p)
            if key_ref in table:
                u_ref = table[key_ref]
                lines.append(
                    f"    Agent {ii} (ref_prefix={ref_p}): "
                    f"Q={u_ref.Q.round(4)}, delta_+={u_ref.delta_plus.round(4)}, "
                    f"W~={u_ref.Wtilde:.4f}, g={u_ref.g:.4f}"
                )

    path = data_path("exp_06_counterexample.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[exp_06] Counterexample saved to {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_games", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--S", type=int, default=4)
    ap.add_argument("--A", type=int, default=3)
    args = ap.parse_args()

    S, A = args.S, args.A
    N_grid = [3, 5, 7]
    H_grid = [3, 4, 5]

    all_rows = []
    counterexample_store = []

    settings = [(N, H) for N in N_grid for H in H_grid]
    for N, H in tqdm(settings, desc="exp_06 settings"):
        label = f"N={N},H={H}"
        rows = run_setting(
            args.n_games, args.seed, S, A, H, N, label, counterexample_store
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    csv_path = data_path("exp_06_prefix_drift.csv")
    df.to_csv(csv_path, index=False)
    print(f"[exp_06] CSV saved: {csv_path}")

    # --- Dedicated counterexample search ---
    # The grid may not surface a margin>1e-3 instance at small --n_games, so
    # scan a large dedicated pool until one is found (or budget exhausted),
    # always keeping the largest-margin instance.
    def best_margin():
        return counterexample_store[0]["margin"] if counterexample_store else 0.0

    if best_margin() < COUNTEREXAMPLE_MARGIN:
        print("[exp_06] grid margin below threshold; running dedicated search...")
        for cg in tqdm(range(4000), desc="exp_06 ce-search"):
            gseed = 777_000_000 + cg
            game = make_game(S=S, A=A, H=4, n=2, seed=gseed, transition="dependent")
            alpha_fn = make_alpha_fn(gseed)
            b = build(game, alpha_fn, const_fn(5))
            tl = b.true_loss
            naive = naive_bound(game, b.table, b.d_ctrl)
            margin = tl - naive
            if margin > best_margin():
                counterexample_store.clear()
                counterexample_store.append({
                    "setting_label": "ce-search N=5,H=4", "game_seed": gseed,
                    "S": S, "A": A, "H": 4, "N": 5, "true_loss": tl,
                    "naive_bound": naive, "ctilde": b.chain["Ctilde"],
                    "margin": margin, "game": game, "table": b.table,
                    "d_ctrl": b.d_ctrl, "V_ref": b.V_ref,
                })
            if best_margin() > COUNTEREXAMPLE_MARGIN:
                break

    ce_margin = best_margin()
    pass_ce = ce_margin > COUNTEREXAMPLE_MARGIN
    if counterexample_store:
        save_counterexample(counterexample_store[0])
    if not pass_ce:
        print(f"[exp_06] WARNING: best margin {ce_margin:.2e} <= "
              f"{COUNTEREXAMPLE_MARGIN:.0e}")

    # --- Per-setting aggregation ---
    agg = (df.groupby("setting")
           .agg(
               drift_rate=("drift_rate", "mean"),
               naive_viol_rate=("naive_violation", "mean"),
               joint_viol_rate=("joint_violation", "mean"),
               n_games=("game_seed", "count"),
           )
           .reset_index())

    # --- Figure 1: naive violation rate bar per (N, H) ---
    fig, ax = new_fig(figsize=(8, 4))
    settings_labels = agg["setting"].tolist()
    viol_rates = agg["naive_viol_rate"].tolist()
    x = np.arange(len(settings_labels))
    bars = ax.bar(x, [v * 100 for v in viol_rates], color="tomato",
                  edgecolor="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(settings_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Naive violation rate (%)")
    ax.set_title(
        "exp_06  Naive bound underbound rate per (N, H) setting\n"
        "(nonzero = prefix drift causes missed W̃ mass)"
    )
    ax.axhline(0, color="k", lw=0.7)
    for bar, v in zip(bars, viol_rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{v*100:.1f}%", ha="center", va="bottom", fontsize=7)
    save_pdf(fig, fig_path("exp_06_naive_viol_rate.pdf"))

    # --- Figure 2: naive vs true scatter (points below diagonal = underbound) ---
    fig, ax = new_fig(figsize=(5, 5))
    mask_viol = df["naive_violation"] == 1
    ax.scatter(df.loc[~mask_viol, "true_loss"], df.loc[~mask_viol, "naive"],
               s=5, alpha=0.3, c="steelblue", label="OK")
    if mask_viol.any():
        ax.scatter(df.loc[mask_viol, "true_loss"], df.loc[mask_viol, "naive"],
                   s=20, alpha=0.9, c="tomato", marker="x",
                   label="naive underbounds (VIOLATION)")
    mx = max(df["true_loss"].max(), df["naive"].max()) * 1.05
    ax.plot([0, mx], [0, mx], "k--", lw=0.8, label="diagonal (naive = true)")
    ax.set_xlabel("true loss")
    ax.set_ylabel("naive bound")
    ax.set_title(
        "exp_06  Naive bound vs true loss\n"
        "(red x below diagonal = naive underbound)"
    )
    ax.legend(fontsize=8)
    save_pdf(fig, fig_path("exp_06_naive_vs_true_scatter.pdf"))

    # --- Summary stats ---
    overall_drift = df["drift_rate"].mean()
    overall_naive_viol = df["naive_violation"].mean()
    overall_joint_viol = df["joint_violation"].mean()
    n_total = len(df)

    pass_drift = overall_drift > 0.9   # expect ~100% among coords i>=1
    pass_joint = overall_joint_viol == 0.0
    pass_naive = overall_naive_viol > 0.0
    status = "PASS" if (pass_drift and pass_joint and pass_naive and pass_ce) else "FAIL"

    write_summary(
        f"exp_06 prefix_drift [{status}] "
        f"total_games={n_total} "
        f"drift_rate={overall_drift:.3f} "
        f"naive_viol_rate={overall_naive_viol:.4f} "
        f"joint_viol_rate={overall_joint_viol:.6f} "
        f"counterexample_margin={ce_margin:.6f}"
    )

    if not pass_drift:
        print(f"[FAIL] drift_rate={overall_drift:.3f} (expected ~1.0)")
    if not pass_joint:
        print(f"[FAIL] joint C~* violations={int(overall_joint_viol*n_total)}")
    if not pass_naive:
        print("[FAIL] no naive violations found (expected persistent nonzero)")
    if not pass_ce:
        print("[FAIL] no margin>1e-3 counterexample found")


if __name__ == "__main__":
    main()
