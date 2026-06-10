"""
exp_10_eligibility.py  —  Corollary 2.1 premise (eligibility iff criterion).

Claim: g_N(alpha) is strictly DECREASING along odd N if and only if alpha > 1/2.
  - For alpha > 1/2 (eligible): adding committee members (N -> N+2) DECREASES g.
  - For alpha < 1/2 (ineligible): adding members INCREASES g (anti-monotone).
  - For alpha = 1/2: g_N = 0.5 for all odd N (flat).

(a) Grid of alpha in [0.05, 0.95] and odd N in {1,3,...,21}:
    Compute g_N(alpha) on the full grid.  For each alpha, determine whether
    g_N is monotone decreasing along the odd-N sequence.  Verify the iff:
    - alpha > 0.5 => strictly decreasing (all diffs < 0)
    - alpha = 0.5 => flat (all diffs ~ 0)
    - alpha < 0.5 => strictly increasing (all diffs > 0)
    Count any violation of the iff (expect 0).

(b) Concrete ineligible demonstrations:
    For alpha in {0.3, 0.4, 0.45}, tabulate g_N at N=1,3,5,7,9 and show
    each +2 step INCREASES g_N.  For alpha=0.7, show the opposite (decreasing).

Outputs:
    results/data/exp_10_eligibility.csv
    results/figs/exp_10_eligibility.pdf
    one PASS/FAIL line -> results/summary.txt

PASS criterion: iff holds with 0 violations; alpha=0.3 shows strictly increasing g_N.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import certificates as C
from common.io_utils import data_path, fig_path, write_summary
from common.plotting import save_pdf

EPS = 1e-9  # tolerance for "strictly" checks (g_N differences)
EPS_FLAT = 1e-9  # tolerance for flat (alpha=0.5) check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="(unused; kept for API symmetry)")
    ap.add_argument("--n_alpha", type=int, default=37,
                    help="Number of alpha values on grid [0.05, 0.95]")
    ap.add_argument("--n_N", type=int, default=11,
                    help="Number of odd N values (1,3,...,2n_N-1)")
    args = ap.parse_args()

    # -----------------------------------------------------------------------
    # (a) Full grid sweep
    # -----------------------------------------------------------------------
    alpha_values = np.linspace(0.05, 0.95, args.n_alpha)
    odd_N_values = [2 * k + 1 for k in range(args.n_N)]  # 1,3,5,...,21

    print(f"Alpha grid: {alpha_values[0]:.2f} .. {alpha_values[-1]:.2f} ({len(alpha_values)} pts)")
    print(f"Odd N list: {odd_N_values}")

    # Build g matrix: shape (n_alpha, n_N)
    g_matrix = np.array([
        [C.g_N(N, alpha) for N in odd_N_values]
        for alpha in alpha_values
    ])

    # Per-alpha: compute diffs along N (positive diff => increasing, negative => decreasing)
    diffs = np.diff(g_matrix, axis=1)  # shape (n_alpha, n_N - 1)

    # Classify each alpha:
    #   "decreasing" : all diffs strictly < 0   (eligible: alpha > 0.5)
    #   "flat"       : all |diffs| < eps         (alpha == 0.5)
    #   "increasing" : all diffs strictly > 0   (ineligible: alpha < 0.5)
    #   "mixed"      : none of the above         (violation)

    violation_count = 0
    rows = []
    for k, alpha in enumerate(alpha_values):
        d = diffs[k]
        is_decreasing = bool(np.all(d < -EPS))
        is_flat = bool(np.all(np.abs(d) < EPS_FLAT))
        is_increasing = bool(np.all(d > EPS))

        if alpha > 0.5 + EPS:
            expected = "decreasing"
            actual = "decreasing" if is_decreasing else ("flat" if is_flat else ("increasing" if is_increasing else "mixed"))
            iff_ok = is_decreasing
        elif alpha < 0.5 - EPS:
            expected = "increasing"
            actual = "increasing" if is_increasing else ("flat" if is_flat else ("decreasing" if is_decreasing else "mixed"))
            iff_ok = is_increasing
        else:
            # alpha ~ 0.5
            expected = "flat"
            actual = "flat" if is_flat else ("decreasing" if is_decreasing else ("increasing" if is_increasing else "mixed"))
            iff_ok = is_flat or is_decreasing or is_increasing  # g=0.5 exactly is flat by theory

        if not iff_ok:
            violation_count += 1
            print(f"[VIOLATION] alpha={alpha:.4f} expected={expected} actual={actual} diffs={d}")

        for j, N in enumerate(odd_N_values):
            rows.append(dict(
                alpha=float(alpha),
                N=N,
                g=float(g_matrix[k, j]),
                expected_direction=expected,
                actual_direction=actual,
                iff_ok=int(iff_ok),
            ))

    df = pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # (b) Concrete demonstrations
    # -----------------------------------------------------------------------
    demo_alphas_ineligible = [0.3, 0.4, 0.45]
    demo_alpha_eligible = 0.7
    demo_Ns = [1, 3, 5, 7, 9]

    print("\n--- (b) Concrete demonstrations ---")
    print(f"{'alpha':>6}  " + "  ".join(f"N={N:2d}" for N in demo_Ns))
    print("-" * 60)

    # Ineligible: should be increasing
    alpha_03_increasing = True
    for alpha in demo_alphas_ineligible:
        g_vals = [C.g_N(N, alpha) for N in demo_Ns]
        diffs_demo = np.diff(g_vals)
        is_strictly_increasing = bool(np.all(diffs_demo > EPS))
        marker = "INCREASING" if is_strictly_increasing else "NOT_INCREASING"
        print(f"  alpha={alpha:.2f}  " + "  ".join(f"{g:.4f}" for g in g_vals) + f"  [{marker}]")
        if alpha == 0.3 and not is_strictly_increasing:
            alpha_03_increasing = False

    # Eligible: should be decreasing
    g_vals_el = [C.g_N(N, demo_alpha_eligible) for N in demo_Ns]
    diffs_el = np.diff(g_vals_el)
    is_decreasing_el = bool(np.all(diffs_el < -EPS))
    marker = "DECREASING" if is_decreasing_el else "NOT_DECREASING"
    print(f"  alpha={demo_alpha_eligible:.2f}  " + "  ".join(f"{g:.4f}" for g in g_vals_el)
          + f"  [{marker}]")

    print(f"\nTotal iff violations: {violation_count}")
    print(f"alpha=0.3 strictly increasing: {alpha_03_increasing}")

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    csv_path = data_path("exp_10_eligibility.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

    # -----------------------------------------------------------------------
    # Figures: g_N vs N curves for representative alpha values
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Select representative alpha values for the curves
    plot_alphas = [0.3, 0.4, 0.45, 0.5, 0.7, 0.8, 0.9]
    colors_ineligible = ["#d73027", "#fc8d59", "#fee090"]  # warm for ineligible
    colors_flat = ["#4d4d4d"]                               # grey for alpha=0.5
    colors_eligible = ["#91bfdb", "#4575b4", "#313695"]    # blue for eligible
    all_colors = colors_ineligible + colors_flat + colors_eligible

    for alpha, col in zip(plot_alphas, all_colors):
        g_vals = [C.g_N(N, alpha) for N in odd_N_values]
        label = f"alpha={alpha:.2f}"
        if alpha > 0.5:
            ls = "-"
        elif alpha < 0.5:
            ls = "--"
        else:
            ls = ":"
        ax1.plot(odd_N_values, g_vals, marker="o", ms=4, lw=1.4,
                 color=col, ls=ls, label=label)

    ax1.axhline(0.5, color="k", ls=":", lw=0.8, alpha=0.4)
    ax1.set_xlabel("Committee size N (odd)")
    ax1.set_ylabel("g_N(alpha)")
    ax1.set_title("exp_10  g_N vs N for various alpha")
    ax1.legend(fontsize=8, ncol=2)

    # Right panel: heatmap of g_N over (alpha, N) grid
    im = ax2.imshow(
        g_matrix, aspect="auto", origin="lower",
        extent=[odd_N_values[0] - 1, odd_N_values[-1] + 1,
                alpha_values[0] - 0.01, alpha_values[-1] + 0.01],
        cmap="RdBu_r", vmin=0, vmax=1,
    )
    ax2.axhline(0.5, color="k", ls="--", lw=1.0, label="alpha=0.5")
    ax2.set_xlabel("N (odd)")
    ax2.set_ylabel("alpha")
    ax2.set_title("g_N(alpha) heatmap  (blue=low, red=high)")
    fig.colorbar(im, ax=ax2, shrink=0.85)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    pdf_path = fig_path("exp_10_eligibility.pdf")
    save_pdf(fig, pdf_path)
    print(f"PDF saved: {pdf_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    status_iff = "PASS" if violation_count == 0 else "FAIL"
    status_03 = "PASS" if alpha_03_increasing else "FAIL"
    overall = "PASS" if (status_iff == "PASS" and status_03 == "PASS") else "FAIL"

    write_summary(
        f"exp_10 eligibility [{overall}] "
        f"iff_violations={violation_count}[{status_iff}] "
        f"alpha03_increasing={alpha_03_increasing}[{status_03}] "
        f"n_alpha={len(alpha_values)} n_N={len(odd_N_values)}"
    )


if __name__ == "__main__":
    main()
