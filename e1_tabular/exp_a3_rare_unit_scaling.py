"""
exp_a3_rare_unit_scaling.py — sample complexity for certifying a RARE unit
(one with small occupancy rho(u)=p under the controller) as p shrinks.

REBUILT DRIVER. The original A3 driver is not present in this repository or
any branch (confirmed via `git log --all`); nothing of it survived, not
even a bytecode cache. This is a from-scratch reconstruction of the
"rare-unit sample-complexity scaling" phenomenon the paper describes
(required episode budget m growing as occupancy p shrinks, roughly linear
in 1/p on a log-log plot). It is NOT guaranteed to reproduce the original
numbers, and per Task 5 (E-4 follow-up) instructions this is expected and
accepted: report the actually-measured slope/R^2, do not tune anything to
match the paper's stated 1.082 / 0.966.

Definition used here (exact, closed-form + numeric root-find, no
simulation noise in "m_required" itself):

  A unit u with controller-occupancy p is visited, in an m-episode
  deployment batch, Binomial(m, p) times. The finite-sample certificate for
  that unit's local contribution (empirical-Bernstein on its own realized
  W-samples, common.certificates.empirical_bernstein) only starts to be
  informative once it has accumulated at least n_min local visits -- below
  that it is either undefined (m<2) or dominated by the O(1/n) range term.
  We fix n_min via the existing Hoeffding radius primitive
  (common.certificates.rad_hoeffding), inverted for a target absolute
  radius eps (as a fraction of B_Q) at confidence delta':

      n_min = ceil( B_Q^2 * ln(2/delta') / (2 * eps^2) )     [rad_hoeffding(n,delta',B_Q) = eps]

  We then define the required total episode budget as the smallest m such
  that a Binomial(m, p) visit count reaches n_min with probability >=
  1-delta_visit:

      m_required(p) = min { m : P(Binomial(m,p) >= n_min) >= 1-delta_visit }

  found by exact evaluation of scipy.stats.binom.sf (monotone in m, so a
  simple increasing search / bisection is exact and well-defined). This is
  the standard "how many trials until you've observed a rare event enough
  times, with confidence" sample-complexity question, applied to the
  per-unit visit count that gates a non-vacuous local empirical-Bernstein
  certificate. Because hitting n_min is a *tail* event when p is small, the
  m required to do so with high confidence exceeds the naive n_min/p by an
  amount that shrinks as n_min grows -- this is what gives the log-log
  slope observed below a value slightly above 1 rather than exactly 1
  (which is what a deterministic m=n_min/p division would give by
  construction, with R^2 exactly 1 and hence a less informative check).

Outputs:
    results/data/exp_a3_rare_unit_scaling.csv
    one summary line -> results/summary.txt
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import binom

from common.io_utils import data_path, write_summary

DELTA_PRIME = 0.05     # confidence for the local Hoeffding radius target
DELTA_VISIT = 0.05     # confidence for "enough visits accumulated"
B_Q = 1.0               # normalized local range bound
EPS_FRAC = 0.1          # target radius, as a fraction of B_Q


def n_min_from_hoeffding(eps: float, delta_prime: float, B_Q: float) -> int:
    """Invert rad_hoeffding(n, delta', B_Q) = eps for n (exact algebra)."""
    n = (B_Q ** 2) * np.log(2.0 / delta_prime) / (2.0 * eps ** 2)
    return int(np.ceil(n))


def m_required_for_visits(p: float, n_min: int, delta_visit: float,
                           m_max: int = 5_000_000) -> int:
    """Smallest m with P(Binomial(m,p) >= n_min) >= 1 - delta_visit, found by
    exact evaluation of the Binomial survival function (monotone in m)."""
    # binom.sf(k, m, p) = P(X > k) = P(X >= k+1); we want P(X >= n_min) = sf(n_min-1, m, p).
    lo, hi = n_min, m_max
    # geometric expansion to find a valid upper bracket where the condition holds
    while binom.sf(n_min - 1, hi, p) < 1 - delta_visit:
        hi *= 2
        if hi > m_max:
            raise RuntimeError(f"m_required exceeds m_max search bound for p={p}")
    while lo < hi:
        mid = (lo + hi) // 2
        if binom.sf(n_min - 1, mid, p) >= 1 - delta_visit:
            hi = mid
        else:
            lo = mid + 1
    return lo


def run(p_grid, eps_frac: float, delta_prime: float, delta_visit: float, B_Q: float):
    eps = eps_frac * B_Q
    n_min = n_min_from_hoeffding(eps, delta_prime, B_Q)
    rows = []
    for p in p_grid:
        m_req = m_required_for_visits(p, n_min, delta_visit)
        rows.append(dict(p=p, inv_p=1.0 / p, n_min=n_min, m_required=m_req,
                          naive_n_min_over_p=n_min / p))
    return pd.DataFrame(rows), n_min


def fit_loglog_slope(df: pd.DataFrame):
    x = np.log(df["inv_p"].values)
    y = np.log(df["m_required"].values)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return float(slope), float(r2)


def main():
    ap = argparse.ArgumentParser(
        description="exp_a3: rare-unit sample-complexity scaling (required m vs occupancy p)")
    ap.add_argument("--p_grid", default="0.1,0.03,0.01",
                     help="comma-separated occupancy levels p (paper's grid by default)")
    ap.add_argument("--eps_frac", type=float, default=EPS_FRAC)
    ap.add_argument("--delta_prime", type=float, default=DELTA_PRIME)
    ap.add_argument("--delta_visit", type=float, default=DELTA_VISIT)
    ap.add_argument("--B_Q", type=float, default=B_Q)
    args = ap.parse_args()

    p_grid = [float(x) for x in args.p_grid.split(",")]
    df, n_min = run(p_grid, args.eps_frac, args.delta_prime, args.delta_visit, args.B_Q)
    csv_path = data_path("exp_a3_rare_unit_scaling.csv")
    df.to_csv(csv_path, index=False)

    slope, r2 = fit_loglog_slope(df)

    # Self-assertion: recompute slope/R^2 from the just-written CSV.
    df_check = pd.read_csv(csv_path)
    slope_check, r2_check = fit_loglog_slope(df_check)
    assert np.isclose(slope, slope_check) and np.isclose(r2, r2_check), \
        "slope/R^2 do not match recomputation from CSV"

    print(df.to_string(index=False))
    print(f"n_min={n_min}  log-log slope={slope:.4f}  R^2={r2:.4f}")

    flag = "" if 0.8 <= slope <= 1.3 else "  [OUT OF [0.8,1.3] -- reported as-is, not tuned]"
    m_list = ",".join(f"p={r.p}:m={int(r.m_required)}" for r in df.itertuples())
    write_summary(f"exp_a3 rare_unit_scaling [n_min={n_min}] {m_list}  "
                  f"slope={slope:.4f} R2={r2:.4f}{flag}")


if __name__ == "__main__":
    main()
