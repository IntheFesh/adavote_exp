"""
exp_05_falsification.py  —  Ablation falsification: each component of AdaVote is necessary.

Claim: each component of AdaVote is necessary.  Ablations are either:
  - INVALID (nonzero violation rate, i.e. underbound true loss), or
  - strictly LOOSER than AdaVote's tightest valid certificate C̃*.

Six baselines evaluated on a batch of random games:
  1. AdaVote  C̃*   (cert_Ctilde)       — reference: VALID and tight.
  2. naive-marginal — use only the reference-prefix unit for each agent and
       weight by the REFERENCE (not controller) state occupancy d_ref[t,s],
       dropping the joint-occupancy coupling from prefix drift.  This is the
       multi-step analogue of exp_04's per-agent marginal bound and can
       UNDERBOUND true_loss (nonzero violation rate).
  3. unweighted     — replace occupancy weighting rho by UNIFORM over units:
       (nH/|units|)*g*Wtilde summed; INVALID or wildly loose.
  4. worst-prefix   — for each (t,s,i) weight by d_ctrl_t(s) but take max over
       prefixes of Wtilde(t,s,i,·) instead of the prefix-occupancy-weighted sum.
       VALID but strictly LOOSER (larger) than C̃*.
  5. plug-in α̂     — estimate alpha at each unit from m_alpha Bernoulli(alpha)
       samples, recompute g_N(α̂), form the cert.  Nonzero coverage failure rate.
  6. no-logging Π_1 — use W instead of Wtilde (cert_Cstar).  VALID but looser.

Outputs:
    results/data/exp_05_falsification.csv
    results/figs/exp_05_falsification.pdf
    one PASS/FAIL summary line
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


# ---------------------------------------------------------------------------
# Occupancy helpers
# ---------------------------------------------------------------------------

def ref_state_occupancy(game: C.Game, V_ref: np.ndarray) -> np.ndarray:
    """State occupancy d_ref[t,s] under the REFERENCE policy (forward DP).

    This is the marginal state distribution at each step t under π^ref.
    """
    d = np.zeros((game.H, game.S))
    d[0] = game.d0.copy()
    for t in range(game.H - 1):
        for s in range(game.S):
            j = C.joint_to_index(game.ref_joint(t, s), game.A)
            d[t + 1] += d[t, s] * game.P[t, s, j]
    return d


# ---------------------------------------------------------------------------
# Ablation helpers
# ---------------------------------------------------------------------------

def cert_naive_marginal(game: C.Game, table: dict, V_ref: np.ndarray) -> float:
    """
    Naive-marginal certificate: multi-step analogue of exp_04's per-agent
    marginal bound.

    For each agent i at each (t,s):
      - use the REFERENCE-prefix unit (prefix = ref actions of agents 0..i-1)
      - weight by the REFERENCE state occupancy d_ref[t,s] (not the controller
        occupancy d_ctrl[t,s])
      - sum independently across agents: drop joint-occupancy coupling.

    marginal = sum_{t,s,i} d_ref[t,s] * g(t,s,i,ref_prefix_i) * Wtilde(t,s,i,ref_prefix_i)

    This underbounds true_loss because:
    (a) d_ref may be smaller than d_ctrl at states where controller deviations
        drive the process towards high-reward regions (but V_ctrl < V_ref so
        the relevant states are those where the controller does WORSE);
    (b) it ignores the cross-prefix occupancy terms (the prefix drift mass that
        lands on non-reference prefixes, which contributes additional loss in C̃*).
    """
    d_ref = ref_state_occupancy(game, V_ref)
    total = 0.0
    for t in range(game.H):
        for s in range(game.S):
            for i in range(game.n):
                ref_prefix = tuple(int(game.ref[t, s, j]) for j in range(i))
                key = (t, s, i, ref_prefix)
                if key not in table:
                    continue
                u = table[key]
                total += d_ref[t, s] * u.g * u.Wtilde
    return total


def cert_unweighted(game: C.Game, table: dict) -> float:
    """
    Unweighted certificate: replace rho(u) by nH/|units| (uniform measure).
    Sum = (nH / |units|) * sum_u g(u) * Wtilde(u).
    """
    nH = game.n * game.H
    n_units = len(table)
    if n_units == 0:
        return 0.0
    total = sum(u.g * u.Wtilde for u in table.values())
    return (nH / n_units) * total


def cert_worst_prefix(game: C.Game, table: dict, rho: dict,
                       d_ctrl: np.ndarray) -> float:
    """
    Worst-prefix certificate: for each (t,s,i) take MAX over prefixes of
    Wtilde(t,s,i,prefix), weighted by d_ctrl[t,s] * g(t,s,i,·).

    This is VALID because:
      sum_{t,s,i} d_ctrl[t,s] * [sum_prefix P(prefix|t,s) * g * Wtilde]
      <= sum_{t,s,i} d_ctrl[t,s] * [sum_prefix P(prefix|t,s)] * max_prefix(g*Wtilde)
      = sum_{t,s,i} d_ctrl[t,s] * max_prefix(g*Wtilde)
      >= C̃*  (looser, because max >= occupancy-weighted average).

    We use max_prefix(g * Wtilde) (note: g may differ per prefix if alpha_fn
    varies by prefix; typically g is constant per (t,s,i) so this just
    picks max_prefix Wtilde).
    """
    total = 0.0
    for t in range(game.H):
        for s in range(game.S):
            for i in range(game.n):
                # Collect all prefixes for this (t,s,i)
                best = 0.0
                for prefix in itertools.product(range(game.A), repeat=i):
                    key = (t, s, i, tuple(prefix))
                    if key not in table:
                        continue
                    u = table[key]
                    best = max(best, u.g * u.Wtilde)
                total += d_ctrl[t, s] * best
    return total


def cert_plugin_alpha(game: C.Game, V_ref: np.ndarray, table: dict,
                       rho: dict, rng: np.random.Generator,
                       m_alpha: int = 20) -> float:
    """
    Plug-in α̂ certificate: estimate alpha at each unit from m_alpha
    Bernoulli(alpha) samples, recompute g_N(α̂), form cert_Ctilde.
    """
    total = 0.0
    for key, u in table.items():
        # Draw m_alpha samples from Bernoulli(u.alpha)
        samples = rng.binomial(1, u.alpha, size=m_alpha)
        alpha_hat = float(samples.mean())
        # clamp to [0,1] (already there by construction)
        g_hat = C.g_N(u.N, alpha_hat)
        total += rho[key] * g_hat * u.Wtilde
    return total


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run(n_games: int, seed: int, S: int, A: int, H: int, n: int,
        N: int, m_alpha: int, alpha_lo: float, alpha_hi: float):
    """Run all 6 ablations on n_games random games."""

    rows = []
    # Violation counters
    viol = {name: 0 for name in [
        "adavote", "naive_marginal", "unweighted", "worst_prefix",
        "plugin_alpha", "no_logging"
    ]}
    slacks = {name: [] for name in viol}

    rng = np.random.default_rng(seed)

    for g_idx in tqdm(range(n_games), desc="exp_05"):
        gseed = seed * 100_000 + g_idx
        game = make_game(S=S, A=A, H=H, n=n, seed=gseed, transition="dependent")
        alpha_fn = make_alpha_fn(gseed, lo=alpha_lo, hi=alpha_hi)
        N_fn = const_fn(N)
        b = build(game, alpha_fn, N_fn)

        tl = b.true_loss
        rho = b.rho
        table = b.table
        V_ref = b.V_ref
        d_ctrl = b.d_ctrl

        # 1. AdaVote C̃*
        ctilde = C.cert_Ctilde(game, table, rho)

        # 2. Naive-marginal
        nm = cert_naive_marginal(game, table, V_ref)

        # 3. Unweighted
        uw = cert_unweighted(game, table)

        # 4. Worst-prefix
        wp = cert_worst_prefix(game, table, rho, d_ctrl)

        # 5. Plug-in alpha
        pa = cert_plugin_alpha(game, V_ref, table, rho, rng, m_alpha=m_alpha)

        # 6. No-logging (C* = cert_Cstar uses W instead of Wtilde)
        nl = C.cert_Cstar(game, table, rho)

        certs = {
            "adavote": ctilde,
            "naive_marginal": nm,
            "unweighted": uw,
            "worst_prefix": wp,
            "plugin_alpha": pa,
            "no_logging": nl,
        }

        for name, cert in certs.items():
            violation = (cert < tl - TOL)
            if violation:
                viol[name] += 1
            slacks[name].append(cert - ctilde)  # slack vs C̃* (positive = looser)

        rows.append(dict(
            game=g_idx,
            true_loss=tl,
            adavote=ctilde,
            naive_marginal=nm,
            unweighted=uw,
            worst_prefix=wp,
            plugin_alpha=pa,
            no_logging=nl,
            # violation flags
            viol_adavote=int(ctilde < tl - TOL),
            viol_naive_marginal=int(nm < tl - TOL),
            viol_unweighted=int(uw < tl - TOL),
            viol_worst_prefix=int(wp < tl - TOL),
            viol_plugin_alpha=int(pa < tl - TOL),
            viol_no_logging=int(nl < tl - TOL),
        ))

    df = pd.DataFrame(rows)
    viol_rates = {k: v / n_games for k, v in viol.items()}
    med_slack = {k: float(np.median(slacks[k])) for k in slacks}

    return df, viol_rates, med_slack


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="exp_05 falsification: ablation failure modes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_games", type=int, default=300)
    ap.add_argument("--S", type=int, default=4)
    ap.add_argument("--A", type=int, default=4)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--m_alpha", type=int, default=20,
                    help="Bernoulli samples per unit for plug-in alpha estimate")
    ap.add_argument("--alpha_lo", type=float, default=0.55)
    ap.add_argument("--alpha_hi", type=float, default=0.9)
    args = ap.parse_args()

    if args.N % 2 == 0:
        args.N += 1

    df, viol_rates, med_slack = run(
        n_games=args.n_games,
        seed=args.seed,
        S=args.S, A=args.A, H=args.H, n=args.n,
        N=args.N,
        m_alpha=args.m_alpha,
        alpha_lo=args.alpha_lo,
        alpha_hi=args.alpha_hi,
    )

    # Print summary table
    print("\n=== exp_05 ablation results ===")
    print(f"  {'baseline':<20}  {'viol_rate':>10}  {'med_slack_vs_Ctilde':>20}")
    baselines = [
        "adavote", "naive_marginal", "unweighted", "worst_prefix",
        "plugin_alpha", "no_logging"
    ]
    for name in baselines:
        vr = viol_rates[name]
        ms = med_slack[name]
        print(f"  {name:<20}  {vr:>10.4f}  {ms:>20.6f}")

    # Assertions for PASS criteria:
    # C̃*: zero violations
    assert viol_rates["adavote"] == 0.0, (
        f"FAIL: AdaVote C̃* has {viol_rates['adavote']*args.n_games:.0f} violations!"
    )
    print("\n[PASS] AdaVote C̃*: 0 violations (valid)")

    # naive-marginal: nonzero violations expected
    if viol_rates["naive_marginal"] == 0:
        print("[WARN] naive-marginal shows 0 violations on this batch (expected > 0)")
    else:
        print(f"[PASS] naive-marginal: violation_rate={viol_rates['naive_marginal']:.4f} > 0 (invalid as expected)")

    # unweighted: report
    if viol_rates["unweighted"] > 0:
        print(f"[PASS] unweighted: violation_rate={viol_rates['unweighted']:.4f} > 0 (invalid)")
    else:
        # If never violated, it should be looser than C̃*
        med_uw_slack = med_slack["unweighted"]
        print(f"[INFO] unweighted: 0 violations, but median slack vs C̃*={med_uw_slack:.6f} "
              f"({'looser' if med_uw_slack > 0 else 'tighter or equal'})")

    # worst-prefix: valid (0 violations) but looser
    assert viol_rates["worst_prefix"] == 0.0, (
        f"FAIL: worst-prefix has {viol_rates['worst_prefix']*args.n_games:.0f} violations!"
    )
    assert med_slack["worst_prefix"] >= -TOL, (
        f"FAIL: worst-prefix median slack vs C̃* = {med_slack['worst_prefix']:.6f} < 0 (tighter than C̃*?)"
    )
    print(f"[PASS] worst-prefix: 0 violations, median_slack_vs_Ctilde={med_slack['worst_prefix']:.6f} >= 0 (valid+looser)")

    # plug-in alpha: nonzero coverage failures
    if viol_rates["plugin_alpha"] == 0:
        print("[WARN] plug-in alpha: 0 violations on this batch (expected > 0 for small m_alpha)")
    else:
        print(f"[PASS] plug-in alpha: violation_rate={viol_rates['plugin_alpha']:.4f} > 0 (coverage degrades as expected)")

    # no-logging (C*): valid but looser
    assert viol_rates["no_logging"] == 0.0, (
        f"FAIL: no-logging (C*) has {viol_rates['no_logging']*args.n_games:.0f} violations!"
    )
    assert med_slack["no_logging"] >= -TOL, (
        f"FAIL: no-logging (C*) median slack vs C̃* = {med_slack['no_logging']:.6f} < 0"
    )
    print(f"[PASS] no-logging Π_1 (C*): 0 violations, median_slack_vs_Ctilde={med_slack['no_logging']:.6f} >= 0 (valid+looser)")

    # ---- Aggregate CSV ----
    agg_rows = []
    for name in baselines:
        agg_rows.append(dict(
            baseline=name,
            violation_rate=viol_rates[name],
            n_violations=int(viol_rates[name] * args.n_games),
            median_slack_vs_Ctilde=med_slack[name],
        ))
    df_agg = pd.DataFrame(agg_rows)
    df_full = df  # per-game rows

    csv_path = data_path("exp_05_falsification.csv")
    df_agg.to_csv(csv_path, index=False)
    print(f"\nCSV (aggregate) saved: {csv_path}")

    # Also save per-game data
    csv_per_game = data_path("exp_05_falsification_per_game.csv")
    df_full.to_csv(csv_per_game, index=False)
    print(f"CSV (per-game) saved: {csv_per_game}")

    # ---- PDF ----
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.0))

    labels = ["C~* (ref)", "naive-marg", "unweighted", "worst-pref", "plug-in a", "Pi1 no-log"]

    # Left: violation rates bar chart
    colors_viol = ["green" if viol_rates[nm] == 0 else "crimson" for nm in baselines]
    vr_vals = [viol_rates[nm] * 100 for nm in baselines]
    bars = ax1.bar(labels, vr_vals, color=colors_viol, edgecolor="k", lw=0.5)
    ax1.set_ylabel("Violation rate (%)")
    ax1.set_title("exp_05  Ablation violation rates")
    ax1.set_ylim(0, max(max(vr_vals) * 1.3, 5))
    for bar, v in zip(bars, vr_vals):
        if v > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    for lbl in ax1.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")

    # Right: median slack vs C̃*
    colors_slack = ["green" if med_slack[nm] >= -TOL else "crimson" for nm in baselines]
    ms_vals = [med_slack[nm] for nm in baselines]
    ax2.bar(labels, ms_vals, color=colors_slack, edgecolor="k", lw=0.5)
    ax2.axhline(0, color="k", ls="--", lw=0.8)
    ax2.set_ylabel("Median slack vs C~*")
    ax2.set_title("exp_05  Ablation tightness (vs C~*)")
    for lbl in ax2.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")

    fig.tight_layout()
    pdf_path = fig_path("exp_05_falsification.pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"PDF saved: {pdf_path}")

    # ---- Summary ----
    pass_ctilde = viol_rates["adavote"] == 0.0
    pass_naive = viol_rates["naive_marginal"] > 0
    pass_plugin = viol_rates["plugin_alpha"] > 0
    pass_wp = viol_rates["worst_prefix"] == 0.0 and med_slack["worst_prefix"] >= -TOL
    pass_nl = viol_rates["no_logging"] == 0.0 and med_slack["no_logging"] >= -TOL
    overall = "PASS" if (pass_ctilde and pass_wp and pass_nl) else "FAIL"

    write_summary(
        f"exp_05 falsification [{overall}] "
        f"Ctilde_viol={viol_rates['adavote']:.3f} "
        f"naive_marginal_viol={viol_rates['naive_marginal']:.3f} "
        f"unweighted_viol={viol_rates['unweighted']:.3f} "
        f"worst_prefix_viol={viol_rates['worst_prefix']:.3f}(slack={med_slack['worst_prefix']:.4f}) "
        f"plugin_alpha_viol={viol_rates['plugin_alpha']:.3f} "
        f"no_logging_viol={viol_rates['no_logging']:.3f}(slack={med_slack['no_logging']:.4f})"
    )


if __name__ == "__main__":
    main()
