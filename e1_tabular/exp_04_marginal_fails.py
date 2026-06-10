"""
exp_04_marginal_fails.py  —  Prop. 1 / 1b: marginal certificate underbounds joint true loss.

Claim verified:
  (a) Explicit H=1, n=2 counterexample: marginal cert < true loss (margin > 1e-3).
  (b) Canonical n-agent amplification: ratio true/marginal = (1-g) + n*g,
      linear in n, equals 1 at n=1.  Closed form validated against enumeration
      for n<=14 (tolerance ~1e-9).

Reward R(a) = 1 - (k/n)^2  where k = # agents deviating from reference action 0.
Reference action for each agent: action 0.
true expected loss = E[(k/n)^2], k~Bin(n,g) = g(1-g)/n + g^2.
marginal cert      = n * g * (1/n)^2 = g/n.
ratio              = true/marginal = (1-g) + n*g.

Outputs:
    results/data/exp_04_marginal_fails.csv
    results/figs/exp_04_marginal_fails.pdf
    one PASS/FAIL summary line
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import certificates as C
from common.io_utils import data_path, fig_path, write_summary
from common.plotting import new_fig, save_pdf


# --------------------------------------------------------------------------- #
# Part (a): explicit H=1, n=2 two-agent counterexample
# --------------------------------------------------------------------------- #
def part_a_counterexample(alpha: float, N: int) -> dict:
    """
    Minimal H=1, n=2, S=1, A=2 game.
    Reference action for each agent: 0 (both play 0).
    Reward uses canonical form: R(k) = 1 - (k/n)^2, k=#{agents deviating from 0}.
      R(a1=0,a2=0) = 1 - 0    = 1.0   (k=0)
      R(a1=1,a2=0) = 1 - 1/4  = 0.75  (k=1)
      R(a1=0,a2=1) = 1 - 1/4  = 0.75  (k=1)
      R(a1=1,a2=1) = 1 - 4/4  = 0.0   (k=2)
    There is only one state (S=1), one step (H=1), A=2 actions per agent.

    Reference V^ref at t=0 is the reward under (a1=0, a2=0) = 1.0  (no continuation).
    Q^ref(t=0,s=0, i=0, prefix=(), a=0) = R(0,0) = 1.0  (followed by ref a2=0)
    Q^ref(t=0,s=0, i=0, prefix=(), a=1) = R(1,0) = 0.75
    So Delta_+(i=0, a=0)=0, Delta_+(i=0,a=1)=0.25   W(i=0)=0.25, Wtilde(i=0)=0.5*0+0.5*0.25=0.125.

    Similarly for i=1 with prefix conditioned on a0:
      prefix=(0,): Q(a=0)=R(0,0)=1.0, Q(a=1)=R(0,1)=0.75 => Delta_+(a=1)=0.25.
      prefix=(1,): Q(a=0)=R(1,0)=0.75, Q(a=1)=R(1,1)=0.0  => Delta_+(a=1)=0.75.

    g = g_N(N, alpha).
    true_loss = d0 (V^ref - V^ctrl).
    marginal cert = sum_u rho(u) g(u) Wtilde(u)  ...but here we compute
      the *marginal* bound (per-agent deviation, others at reference):
      marginal = sum_i g * Wtilde_ref_prefix(i)
               = g * Wtilde(i=0, prefix=()) + g * Wtilde(i=1, prefix=(ref_0=0,))
               = g * 0.125 + g * 0.125 = 0.25 * g.

    true loss under controller:
      controller plays (fb if failure): uniform fallback over {0,1} for each agent.
      For n=2 with uniform fallback:
        P(k=0) = (1-g)^2  (both succeed, play ref=0)
        P(k=1) = 2*(1-g)*g*0.5 + ... need explicit joint distribution.
    """
    n, A, H, S = 2, 2, 1, 1
    g = C.g_N(N, alpha)

    # Build reward tensor shape (H=1, S=1, A^2=4)
    R = np.zeros((H, S, A**n))
    for a0 in range(A):
        for a1 in range(A):
            k = (a0 != 0) + (a1 != 0)
            R[0, 0, C.joint_to_index((a0, a1), A)] = 1.0 - (k / n) ** 2

    # Transitions: trivial (H=1, no next state matters)
    P = np.ones((H, S, A**n, S))  # always stay in state 0
    d0 = np.array([1.0])
    ref = np.zeros((H, S, n), dtype=int)  # reference = action 0 for all
    delta_r = float(R.max() - R.min())

    game = C.Game(S=S, A=A, n=n, H=H, R=R, P=P, d0=d0, ref=ref, delta_r=delta_r)

    # Build with constant alpha/N and uniform fallback
    V_ref = C.ref_value_iteration(game)
    alpha_fn = lambda t, s, i, prefix: alpha
    N_fn = lambda t, s, i, prefix: N
    table = C.build_unit_table(game, V_ref, alpha_fn, N_fn, C.uniform_fb, eta=0.0)

    V_ctrl, d_ctrl = C.controller_value(game, table)
    rho = C.unit_occupancy(game, table, d_ctrl)

    tl = C.true_loss(game, V_ref, V_ctrl)
    ctilde = C.cert_Ctilde(game, table, rho)

    # Marginal cert: per-unit rho_marginal uses REFERENCE prefix only
    # For unit (t=0, s=0, i=0, prefix=()): rho_marginal = d_ctrl[0,0] * 1.0
    # For unit (t=0, s=0, i=1, prefix=(ref_0,)): rho_marginal = d_ctrl[0,0] * 1.0
    # Others (prefix != ref prefix) get 0 weight in the marginal cert.
    marginal_cert = 0.0
    for i in range(n):
        ref_prefix = tuple(int(game.ref[0, 0, j]) for j in range(i))
        key = (0, 0, i, ref_prefix)
        u = table[key]
        marginal_cert += d_ctrl[0, 0] * u.g * u.Wtilde

    # Print explicit table
    print("\n[Part (a)] Two-agent H=1 counterexample")
    print(f"  alpha={alpha}, N={N}, g={g:.6f}")
    print(f"  Reward table (k=# deviators, R=1-(k/n)^2):")
    for a0 in range(A):
        for a1 in range(A):
            k = (a0 != 0) + (a1 != 0)
            r = 1.0 - (k / n) ** 2
            j = C.joint_to_index((a0, a1), A)
            print(f"    a=({a0},{a1})  k={k}  R={r:.4f}  [idx={j}]")
    print(f"  V_ref[0,0] = {V_ref[0,0]:.6f}")
    for key, u in sorted(table.items()):
        print(f"  unit {key}: W={u.W:.4f}  Wtilde={u.Wtilde:.4f}  g={u.g:.6f}  rho={rho[key]:.6f}")
    print(f"  true_loss      = {tl:.6f}")
    print(f"  C~* (joint)    = {ctilde:.6f}")
    print(f"  marginal_cert  = {marginal_cert:.6f}")
    print(f"  gap (true - marginal) = {tl - marginal_cert:.6f}")

    return dict(alpha=alpha, N=N, g=g, true_loss=tl, ctilde=ctilde,
                marginal_cert=marginal_cert, gap=tl - marginal_cert)


# --------------------------------------------------------------------------- #
# Part (b): canonical n-agent amplification
# --------------------------------------------------------------------------- #
def closed_form_true(n: int, g: float) -> float:
    """E[(k/n)^2], k~Bin(n,g).  E[k^2] = n*g*(1-g) + n^2*g^2."""
    ek2 = n * g * (1 - g) + n ** 2 * g ** 2
    return ek2 / n ** 2


def closed_form_marginal(n: int, g: float) -> float:
    """Sum of per-agent single-deviation swings * g = n * g * (1/n)^2 = g/n."""
    return g / n


def enumerated_true(n: int, g: float, max_n_enum: int = 14) -> float | None:
    """Direct enumeration over k in 0..n (using Bin pmf)."""
    if n > max_n_enum:
        return None
    from scipy.stats import binom as sp_binom
    total = 0.0
    for k in range(n + 1):
        pmf = sp_binom.pmf(k, n, g)
        total += pmf * (k / n) ** 2
    return total


def part_b_amplification(alpha: float, N: int, n_max: int = 16) -> pd.DataFrame:
    """Compute amplification table for n in 1..n_max."""
    g = C.g_N(N, alpha)
    print(f"\n[Part (b)] Amplification  alpha={alpha}, N={N}, g={g:.6f}")

    rows = []
    for n in range(1, n_max + 1):
        true_cf = closed_form_true(n, g)
        marginal = closed_form_marginal(n, g)
        ratio_cf = true_cf / marginal if marginal > 1e-15 else float("nan")
        ratio_theory = (1 - g) + n * g

        true_enum = enumerated_true(n, g)
        ratio_enum = (true_enum / marginal) if (true_enum is not None and marginal > 1e-15) else None

        # Validate closed form vs enumeration
        if true_enum is not None:
            err = abs(true_cf - true_enum)
            assert err < 1e-9, (
                f"n={n}: closed_form_true={true_cf} vs enum={true_enum}  err={err}"
            )

        # Check ratio matches theory
        err_ratio = abs(ratio_cf - ratio_theory)
        assert err_ratio < 1e-9, (
            f"n={n}: ratio_cf={ratio_cf} vs theory={ratio_theory}  err={err_ratio}"
        )

        rows.append(dict(
            n=n, g=g, true_closed_form=true_cf, marginal=marginal,
            ratio_closed_form=ratio_cf,
            ratio_theory=(1 - g) + n * g,
            ratio_enumerated=ratio_enum if ratio_enum is not None else float("nan"),
            enum_feasible=(true_enum is not None),
        ))
        print(f"  n={n:2d}  true={true_cf:.6f}  marginal={marginal:.6f}  "
              f"ratio={ratio_cf:.4f}  theory={(1-g)+n*g:.4f}"
              + (f"  enum_ratio={ratio_enum:.4f}" if ratio_enum is not None else ""))

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="exp_04 marginal certificate failure")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="Per-member endorsement probability (>0.5 eligible)")
    ap.add_argument("--N", type=int, default=5,
                    help="Committee size (odd)")
    ap.add_argument("--n_max", type=int, default=16,
                    help="Maximum number of agents for amplification sweep")
    args = ap.parse_args()

    if args.N % 2 == 0:
        args.N += 1  # ensure odd

    # ---- Part (a) ----
    res_a = part_a_counterexample(alpha=args.alpha, N=args.N)
    assert res_a["gap"] > 1e-3, (
        f"Part (a) FAIL: gap={res_a['gap']:.6f} not > 1e-3  "
        f"(true={res_a['true_loss']:.6f}, marginal={res_a['marginal_cert']:.6f})"
    )
    print(f"\n[Part (a)] PASS: marginal={res_a['marginal_cert']:.6f} < "
          f"true={res_a['true_loss']:.6f}  gap={res_a['gap']:.6f} > 1e-3")

    # ---- Part (b) ----
    df = part_b_amplification(alpha=args.alpha, N=args.N, n_max=args.n_max)

    # Check n=1 ratio exactly 1
    r1 = df.loc[df.n == 1, "ratio_closed_form"].iloc[0]
    assert abs(r1 - 1.0) < 1e-12, f"ratio at n=1 = {r1} != 1.0"
    print(f"\n[Part (b)] ratio at n=1 = {r1:.12f}  (should be 1.0) PASS")

    # Check linearity: ratio = (1-g)+n*g for all n
    g_val = df["g"].iloc[0]
    r_last = df.loc[df.n == args.n_max, "ratio_closed_form"].iloc[0]
    theory_last = (1 - g_val) + args.n_max * g_val
    print(f"[Part (b)] n={args.n_max}: ratio={r_last:.8f}  theory={theory_last:.8f}")

    # Verify all closed form == enumeration rows match
    enum_rows = df[df.enum_feasible]
    print(f"[Part (b)] Enumeration validation: {len(enum_rows)} rows checked to 1e-9")

    # ---- CSV ----
    # Add part_a row indicator
    df_a = pd.DataFrame([{
        "n": "part_a_n2", "g": res_a["g"],
        "true_closed_form": res_a["true_loss"],
        "marginal": res_a["marginal_cert"],
        "ratio_closed_form": res_a["true_loss"] / res_a["marginal_cert"],
        "ratio_theory": (1 - res_a["g"]) + 2 * res_a["g"],
        "ratio_enumerated": float("nan"),
        "enum_feasible": True,
    }])
    df_out = pd.concat([df, df_a], ignore_index=True)
    csv_path = data_path("exp_04_marginal_fails.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path}")

    # ---- PDF ----
    fig, ax = new_fig()
    n_vals = df["n"].values
    ratio_vals = df["ratio_closed_form"].values
    theory_vals = df["ratio_theory"].values
    ax.plot(n_vals, ratio_vals, "o-", color="steelblue", label="closed-form ratio")
    ax.plot(n_vals, theory_vals, "r--", lw=1.5, label=r"theory $(1-g)+ng$")
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("number of agents $n$")
    ax.set_ylabel(r"true / marginal cert ratio")
    ax.set_title(r"exp_04  marginal-cert amplification: true/marginal $= (1-g)+ng$")
    ax.legend(fontsize=9)
    pdf_path = fig_path("exp_04_marginal_fails.pdf")
    save_pdf(fig, pdf_path)
    print(f"PDF saved: {pdf_path}")

    # ---- Summary ----
    pass_a = res_a["gap"] > 1e-3
    pass_b_ratio = abs(r1 - 1.0) < 1e-12
    pass_b_enum = all(
        abs(r - t) < 1e-9
        for r, t, fe in zip(
            df["ratio_closed_form"].values,
            df["ratio_theory"].values,
            df["enum_feasible"].values,
        )
    )
    status = "PASS" if (pass_a and pass_b_ratio and pass_b_enum) else "FAIL"
    write_summary(
        f"exp_04 marginal_fails [{status}] "
        f"part_a_gap={res_a['gap']:.4f} "
        f"n16_ratio={r_last:.4f} "
        f"theory_n16={theory_last:.4f} "
        f"closed_form==enum(n<=14)=True "
        f"ratio_n1=1.0"
    )


if __name__ == "__main__":
    main()
