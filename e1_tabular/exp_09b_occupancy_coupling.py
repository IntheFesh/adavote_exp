"""
exp_09b_occupancy_coupling.py — Satisfiability of the occupancy-coupled
eligibility sufficient condition (Refinement target (b) for §8.2 / Cor. 2.1).

Companion to exp_09 group 3. Group 3 reported only HOW MANY allocation events
raise the true loss under occupancy drift. This experiment performs, for each
drift event, the EXACT first-order decomposition of the certificate change into
a failure-reduction term and an occupancy-redistribution term, records the
eligibility margin, and DECIDES whether the sufficient condition

    |T_fail| >= T_occ      (equivalently  Delta C_tilde <= 0)

becomes satisfiable above some eligibility-margin threshold, or is vacuous.

First-order EXACT decomposition (identity; self-checked to machine precision):
    Delta C_tilde = C_tilde_1 - C_tilde_0
      = sum_u rho0(u) Wt(u) [ g_{N1(u)}(a_u) - g_{N0(u)}(a_u) ]   (= T_fail, <= 0)
      + sum_u (rho1(u) - rho0(u)) g_{N1(u)}(a_u) Wt(u)            (= T_occ)

Because g is monotone in N (odd N) and only the bumped unit changes N, T_fail is
the (non-positive) failure-reduction; T_occ is the occupancy-redistribution term
whose sign is free. Hence  Delta C_tilde > 0  <=>  T_occ > |T_fail|, which the
script checks as an internal consistency identity (occ_dominates == cert_rise).

ADVERSARIAL DESIGN: written to show (b) is VACUOUS. It actively searches for
high-margin events that still raise the certificate, and reports the full
margin-binned rise-rate curve so the verdict is auditable rather than a single
fragile threshold. (b) is declared SATISFIABLE only if some margin threshold has
>= MIN_PER_BIN events AND >= MIN_FRAC_ABOVE of all events above it AND zero
certificate-rise events above it.

Interfaces used (all confirmed; committee size tracked via explicit N_dict,
never via a u.N field):
    make_game(S,A,H,n,seed,transition="dependent")        [exp_09 group3]
    build(game, alpha_fn, N_fn) -> Built                  [_common.py]
    Built.rho : Dict[UnitKey,float]; Built.table; Built.chain  [_common.py]
    u.alpha, u.Wtilde on table units                      [cert_Ctilde impl]
    g_N(N, alpha)                                          [common.certificates]
    make_alpha_fn(seed,lo,hi), const_fn(N0)               [_common.py]
    data_path, write_summary                              [common.io_utils]
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from common.certificates import g_N
from common.games import make_game
from common.io_utils import data_path, write_summary
from e1_tabular._common import build, make_alpha_fn, const_fn

TOL = 1e-12
MIN_PER_BIN = 30        # min events above a candidate threshold to trust it
MIN_FRAC_ABOVE = 0.10   # threshold must retain >=10% of all events above it
N_BINS = 10


def build_with_N_dict(game, alpha_fn, N_dict, default_N):
    def N_fn(t, s, i, prefix):
        return N_dict.get((t, s, i, tuple(prefix)), default_N)
    return build(game, alpha_fn, N_fn)


def decompose(b0, b1, N0_dict, N1_dict, default_N):
    """Exact first-order decomposition Delta C_tilde = T_fail + T_occ.

    T_fail: occupancy frozen at rho0, g moves N0 -> N1 (only bumped unit differs).
    T_occ:  g evaluated at N1, occupancy moves rho0 -> rho1.
    alpha and Wtilde come from b0.table (identical in b1; only N changes).
    """
    rho0, rho1 = b0.rho, b1.rho
    tab = b0.table
    keys = set(rho0) | set(rho1)
    T_fail = 0.0
    T_occ = 0.0
    for k in keys:
        u = tab.get(k)
        if u is None:
            continue
        alpha = u.alpha
        Wt = u.Wtilde
        N0 = N0_dict.get(k, default_N)
        N1 = N1_dict.get(k, default_N)
        gN0 = g_N(N0, alpha)
        gN1 = g_N(N1, alpha)
        r0 = rho0.get(k, 0.0)
        r1 = rho1.get(k, 0.0)
        T_fail += r0 * Wt * (gN1 - gN0)
        T_occ += (r1 - r0) * gN1 * Wt
    return T_fail, T_occ


def run(n_events, seed, S, A, H, N0, max_units=12):
    rows = []
    total = 0
    g = 0
    pbar = tqdm(total=n_events, desc="exp_09b events")
    while total < n_events:
        gseed = seed * 700000 + g
        g += 1
        game = make_game(S=S, A=A, H=H, n=2, seed=gseed, transition="dependent")
        alpha_fn = make_alpha_fn(gseed, lo=0.55, hi=0.9)
        rng = np.random.default_rng(gseed)

        b0 = build(game, alpha_fn, const_fn(N0))
        N0_dict = {k: N0 for k in b0.table.keys()}
        eligible = [k for k, u in b0.table.items() if u.alpha > 0.5]
        if not eligible:
            continue
        if len(eligible) > max_units:
            idx = rng.choice(len(eligible), size=max_units, replace=False)
            eligible = [eligible[i] for i in idx]

        c0 = b0.chain["Ctilde"]
        for key in eligible:
            if total >= n_events:
                break
            N1_dict = dict(N0_dict)
            N1_dict[key] = N0 + 2
            b1 = build_with_N_dict(game, alpha_fn, N1_dict, N0)
            c1 = b1.chain["Ctilde"]
            T_fail, T_occ = decompose(b0, b1, N0_dict, N1_dict, N0)
            resid = abs((T_fail + T_occ) - (c1 - c0))
            alpha_star = b0.table[key].alpha
            margin = alpha_star - (N0 + 1) / (2 * N0)
            dcert = c1 - c0
            total += 1
            pbar.update(1)
            rows.append(dict(
                game=g - 1, unit_t=key[0], unit_s=key[1], unit_i=key[2],
                alpha_star=float(alpha_star), margin=float(margin),
                delta_ctilde=float(dcert),
                T_fail=float(T_fail), T_occ=float(T_occ),
                identity_resid=float(resid),
                cert_rise=int(dcert > TOL),
                occ_dominates=int(T_occ > -T_fail + TOL),
            ))
    pbar.close()
    return pd.DataFrame(rows)


def binned_curve(df):
    margins = df["margin"].to_numpy()
    dcert = df["delta_ctilde"].to_numpy()
    occd = df["occ_dominates"].to_numpy()
    edges = np.linspace(margins.min(), margins.max(), N_BINS + 1)
    print(f"\n{'margin-bin':>20}  {'n':>5}  {'cert_rise':>9}  {'occ_dom':>8}")
    print("-" * 50)
    curve_rows = []
    for i in range(N_BINS):
        lo, hi = edges[i], edges[i + 1]
        if i < N_BINS - 1:
            m = (margins >= lo) & (margins < hi)
        else:
            m = (margins >= lo) & (margins <= hi)
        n = int(m.sum())
        if n == 0:
            print(f"  [{lo:+.3f},{hi:+.3f})  {0:5d}  {'--':>9}  {'--':>8}")
            curve_rows.append(dict(bin_lo=lo, bin_hi=hi, n=0,
                                   cert_rise_rate=np.nan, occ_dom_rate=np.nan))
            continue
        rr = float((dcert[m] > TOL).mean())
        od = float(occd[m].mean())
        print(f"  [{lo:+.3f},{hi:+.3f})  {n:5d}  {rr:9.3f}  {od:8.3f}")
        curve_rows.append(dict(bin_lo=float(lo), bin_hi=float(hi), n=n,
                               cert_rise_rate=rr, occ_dom_rate=od))
    return pd.DataFrame(curve_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_events", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--S", type=int, default=4)
    ap.add_argument("--A", type=int, default=3)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--N0", type=int, default=3)
    args = ap.parse_args()

    df = run(args.n_events, args.seed, args.S, args.A, args.H, args.N0)
    df.to_csv(data_path("exp_09b_occupancy_coupling.csv"), index=False)

    # ---- self-checks ----
    max_resid = float(df["identity_resid"].max())
    tfail_nonpos = bool((df["T_fail"] <= TOL).all())
    # consistency: cert_rise must equal occ_dominates everywhere
    consistency_ok = bool((df["cert_rise"] == df["occ_dominates"]).all())

    # ---- auditable binned curve ----
    curve = binned_curve(df)
    curve.to_csv(data_path("exp_09b_margin_curve.csv"), index=False)

    # ---- adversarial satisfiability scan (strict) ----
    margins = df["margin"].to_numpy()
    dcert = df["delta_ctilde"].to_numpy()
    N = len(margins)
    overall_rise = float((dcert > TOL).mean())
    thr_grid = np.linspace(margins.min(), margins.max(), 100)
    sat_thr = None
    for thr in thr_grid:
        mask = margins >= thr
        if (mask.sum() >= MIN_PER_BIN and mask.sum() >= MIN_FRAC_ABOVE * N
                and (dcert[mask] <= TOL).all()):
            sat_thr = float(thr)
            break
    n_above = int((margins >= sat_thr).sum()) if sat_thr is not None else 0
    verdict = "SATISFIABLE" if sat_thr is not None else "VACUOUS"

    status = "PASS" if (max_resid < 1e-9 and tfail_nonpos and consistency_ok) else "FAIL"
    print(f"\n[exp_09b] occupancy_coupling [{status}]")
    print(f"  identity_max_resid = {max_resid:.2e}  (expect ~1e-16)")
    print(f"  T_fail_all_nonpos  = {tfail_nonpos}")
    print(f"  cert_rise==occ_dom = {consistency_ok}")
    print(f"  overall_cert_rise_rate = {overall_rise:.3f}")
    print(f"  (b) VERDICT = {verdict}"
          + (f"  (margin>={sat_thr:.3f}, n_above={n_above})" if sat_thr else ""))

    write_summary(
        f"exp_09b occupancy_coupling [{status}] "
        f"identity_max_resid={max_resid:.2e} T_fail_all_nonpos={tfail_nonpos} "
        f"cert_rise_eq_occ_dom={consistency_ok} "
        f"overall_cert_rise_rate={overall_rise:.3f} "
        f"(b)_verdict={verdict} "
        f"sat_margin_threshold={sat_thr if sat_thr is not None else 'none'} "
        f"events_above_threshold={n_above} n_events={len(df)}"
    )


if __name__ == "__main__":
    main()
