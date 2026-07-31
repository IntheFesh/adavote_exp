#!/usr/bin/env python3
"""
verify_paper_numbers.py — Task 11: recompute every headline number and every
figure's underlying data directly from results/ (CSV/JSON), compare against
the paper's final values, and print a PASS/FAIL table.

This is the concrete implementation of the paper's Reproducibility Statement
claim that all headline numbers are recomputed from the committed logs.
Nothing here reruns an experiment; everything reads an already-committed
results/data/*.csv or results/e2/*.json file. Run from the repository root:

    python verify_paper_numbers.py

Exit code 0 iff every check PASSes. On any FAIL, the script does NOT modify
its own targets -- it reports the check name, recomputed value, paper value,
and tolerance, and exits 1.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

RESULTS = "results"
CHECKS = []  # list of dicts: section, name, actual, expected, tol, passed


def check(section, name, actual, expected, tol=1e-3, cmp="abs"):
    if isinstance(expected, bool):
        passed = bool(actual) == expected
    elif cmp == "abs":
        passed = abs(float(actual) - float(expected)) <= tol
    elif cmp == "rel":
        passed = abs(float(actual) - float(expected)) <= tol * abs(float(expected))
    elif cmp == "eq":
        passed = actual == expected
    else:
        raise ValueError(cmp)
    CHECKS.append(dict(section=section, name=name, actual=actual, expected=expected,
                        tol=tol, passed=passed))
    return passed


def data(name):
    return f"{RESULTS}/data/{name}"


def e2(name):
    return f"{RESULTS}/e2/{name}"


# --------------------------------------------------------------------------- #
def verify_exp01():
    s = "9.1 exp_01"
    df = pd.read_csv(data("exp_01_chain_ordering.csv"))
    check(s, "n_games", len(df), 300, tol=0, cmp="eq")
    viol = ((df["r_C0"] + 1e-9 < df["r_Cstar"]) | (df["r_Cstar"] + 1e-9 < df["r_Ctilde"]) |
            (df["r_Ctilde"] < 1.0 - 1e-9)).sum()
    check(s, "chain violations", int(viol), 0, tol=0, cmp="eq")
    check(s, "median C0/L", df["r_C0"].median(), 8.53, tol=0.005)
    check(s, "median C1/L", df["r_Cstar"].median(), 2.04, tol=0.005)
    check(s, "median C2/L", df["r_Ctilde"].median(), 1.02, tol=0.005)
    slack = df["slack_Ctilde"]
    check(s, "slack median", slack.median(), 0.017, tol=0.0005)
    check(s, "slack p95", slack.quantile(0.95), 0.030, tol=0.0005)
    check(s, "slack frac<=0.04", (slack <= 0.04).mean(), 0.993, tol=0.001)
    check(s, "slack max", slack.max(), 0.042, tol=0.0005)


def verify_exp06():
    s = "9.1 exp_06"
    df = pd.read_csv(data("exp_06_prefix_drift.csv"))
    check(s, "instances", len(df), 3600, tol=0, cmp="eq")


def verify_exp13():
    s = "9.1 exp_13"
    df = pd.read_csv(data("exp_13_structured_validity.csv"))
    check(s, "total games", len(df), 315, tol=0, cmp="eq")
    g = df.groupby("n").agg(games=("game", "count"), chain_viol=("chain_violation", "sum"),
                             C2_viol=("joint_violation", "sum"), C0_L=("r_C0", "median"),
                             C1_L=("r_Cstar", "median"), C2_L=("r_Ctilde", "median"),
                             mean_L=("true_loss", "mean"))
    target = {
        2: (80, 0, 0, 31.12, 1.89, 1.003, 0.19),
        3: (80, 0, 0, 21.50, 1.70, 1.012, 0.45),
        4: (60, 0, 0, 21.41, 1.67, 1.019, 0.70),
        5: (50, 0, 0, 25.39, 1.65, 1.020, 0.93),
        6: (30, 0, 0, 21.50, 1.59, 1.015, 1.28),
        7: (15, 0, 0, 20.18, 1.58, 1.014, 1.50),
    }
    for n, (games, cv, c2v, c0l, c1l, c2l, meanl) in target.items():
        r = g.loc[n]
        check(s, f"n={n} games", int(r["games"]), games, tol=0, cmp="eq")
        check(s, f"n={n} chain_viol", int(r["chain_viol"]), cv, tol=0, cmp="eq")
        check(s, f"n={n} C2_viol", int(r["C2_viol"]), c2v, tol=0, cmp="eq")
        check(s, f"n={n} C0/L", r["C0_L"], c0l, tol=0.01)
        check(s, f"n={n} C1/L", r["C1_L"], c1l, tol=0.01)
        check(s, f"n={n} C2/L", r["C2_L"], c2l, tol=0.005)
        check(s, f"n={n} mean L", r["mean_L"], meanl, tol=0.01)
    check(s, "C2/L all in [1.003,1.020]", bool(g["C2_L"].between(1.0025, 1.0205).all()), True)
    check(s, "mean L range 0.19->1.50", f"{g['mean_L'].min():.2f}->{g['mean_L'].max():.2f}",
          "0.19->1.50", tol=0, cmp="eq")


# --------------------------------------------------------------------------- #
def verify_exp03():
    s = "9.2 exp_03"
    df = pd.read_csv(data("exp_03_estimator.csv"))
    med = df[df.m == 5000].groupby("variant")["tightness"].median()
    check(s, "median I2 tightness (m=5000)", med["Pi2"], 1.194, tol=0.005)
    check(s, "median I1 tightness (m=5000)", med["Pi1"], 2.307, tol=0.005)

    cov = df.groupby(["variant", "m"])["covered"].mean()
    cov_target = {("Pi1", 500): 1.000, ("Pi1", 1500): 1.000, ("Pi1", 5000): 1.000,
                  ("Pi2", 500): 1.000, ("Pi2", 1500): 1.000, ("Pi2", 5000): 0.9994}
    for (variant, m), val in cov_target.items():
        check(s, f"coverage {variant} m={m}", cov[(variant, m)], val, tol=0.0005)

    dfb = pd.read_csv(data("exp_03_bound_types.csv"))
    gb = dfb.groupby("bound_type").agg(coverage=("covered", "mean"), tightness=("ratio_L", "median"))
    bt_target = {"empirical_bernstein": (1.000, 1.201), "hoeffding": (1.000, 1.396),
                 "clipped_empirical_bernstein": (1.000, 1.180), "naive_plugin": (0.650, 1.022)}
    for bt, (cov_v, tight_v) in bt_target.items():
        check(s, f"bound_type={bt} coverage", gb.loc[bt, "coverage"], cov_v, tol=0.005)
        check(s, f"bound_type={bt} tightness", gb.loc[bt, "tightness"], tight_v, tol=0.005)


# --------------------------------------------------------------------------- #
def verify_exp12():
    s = "9.3 exp_12"
    df = pd.read_csv(data("exp_12_rollout_bridge.csv"))
    Ks = [25, 50, 100, 200, 400]
    target_BL = dict(zip(Ks, [1.624, 1.637, 1.646, 1.614, 1.608]))
    target_BR = dict(zip(Ks, [0.148, 0.149, 0.147, 0.145, 0.145]))
    max_dev, k25_dev = 0.0, None
    for K in Ks:
        sub = df[df.K == K]
        med_BL = sub["ratio_L"].median()
        med_BR = sub["ratio_Rmax"].median()
        cov = sub["covered"].mean()
        med_xc2 = sub["ratio_XC2"].median()
        check(s, f"K={K} median B/L", med_BL, target_BL[K], tol=0.005)
        check(s, f"K={K} median B/Rmax", med_BR, target_BR[K], tol=0.005)
        check(s, f"K={K} coverage", cov, 1.000, tol=0.001)
        dev = abs(med_xc2 - 1.0)
        max_dev = max(max_dev, dev)
        if K == 25:
            k25_dev = dev
    CHECKS.append(dict(section=s, name="max |Xbar/C2-1| <= 1.3%",
                        actual=round(max_dev, 4), expected="<=0.013",
                        tol="n/a", passed=max_dev <= 0.013))
    CHECKS.append(dict(section=s, name="K=25 |Xbar/C2-1| <= 0.2%",
                        actual=round(k25_dev, 4), expected="<=0.002",
                        tol="n/a", passed=k25_dev <= 0.002))

    dfm = pd.read_csv(data("exp_12_m_sweep.csv"))
    gm = dfm.groupby("m")["ratio_L"].median()
    check(s, "m-sweep median B/L m=500", gm[500], 2.97, tol=0.01)
    check(s, "m-sweep median B/L m=1500", gm[1500], 1.79, tol=0.01)
    check(s, "m-sweep median B/L m=5000", gm[5000], 1.32, tol=0.01)

    dfc = pd.read_csv(data("exp_12_bound_comparison.csv"))
    gc = dfc.groupby("bound_type").agg(coverage=("covered", "mean"), tightness=("ratio_L", "median"))
    check(s, "bound_comparison exact_value_eb", gc.loc["exact_value_eb", "tightness"], 1.310, tol=0.005)
    check(s, "bound_comparison jensen_eb", gc.loc["jensen_eb", "tightness"], 1.313, tol=0.005)
    check(s, "bound_comparison jensen_hoeffding", gc.loc["jensen_hoeffding", "tightness"], 2.494, tol=0.005)
    check(s, "bound_comparison raw_plugin_mean tightness", gc.loc["raw_plugin_mean", "tightness"], 1.009, tol=0.005)
    check(s, "bound_comparison raw_plugin_mean coverage", gc.loc["raw_plugin_mean", "coverage"], 0.531, tol=0.005)


def verify_exp14():
    s = "9.3 exp_14"
    df = pd.read_csv(data("exp_14_correlated_committees.csv")).sort_values("s", ascending=False)
    check(s, "n_levels", len(df), 12, tol=0, cmp="eq")
    check(s, "true_loss min", df["mean_true_loss"].min(), 0.351, tol=0.002)
    check(s, "true_loss max", df["mean_true_loss"].max(), 0.523, tol=0.002)

    undercov = df["plugin_under_coverage_rate"].values
    target_seq = [0.00, 0.12, 0.50, 0.62, 0.62, 0.88]
    for i, v in enumerate(target_seq):
        check(s, f"plugin undercov #{i} (s={df['s'].values[i]})", undercov[i], v, tol=0.01)
    check(s, "plugin undercov ==1.00 for s<=50", bool(np.all(undercov[6:] >= 0.999)), True)

    gap = df["gap_mixture_minus_plugin"].values
    check(s, "gap min", gap[0], 8.2e-4, tol=2e-4)
    check(s, "gap max", gap[-1], 8.3e-2, tol=2e-3)
    check(s, "gap monotone non-decreasing as s shrinks", bool(np.all(np.diff(gap) >= -1e-9)), True)
    check(s, "mixture violations == 0 (all levels)", int(df["mixture_violations"].sum()), 0, tol=0, cmp="eq")

    # n_games x n_levels = 8 x 12
    detail = pd.read_csv(data("exp_14_per_game.csv"))
    check(s, "n_games", detail["game"].nunique(), 8, tol=0, cmp="eq")


# --------------------------------------------------------------------------- #
def verify_mpe():
    s = "9.4 MPE"
    files = {
        150: (e2("rollout_certificate_simple_spread_ippo_jensen_m150.json"), "100", 4.45),
        1500: (e2("rollout_certificate_simple_spread_ippo_jensen_m1500.json"), "8000", 0.612),
        3000: (e2("rollout_certificate_simple_spread_ippo_jensen_m3000k25000.json"), "25000", 0.358),
    }
    c2emp_lo, c2emp_hi = 1.0, 0.0
    for m, (path, K, target_ratio) in files.items():
        with open(path) as f:
            d = json.load(f)
        pk = d["per_K"][K]
        check(s, f"m={m} Bhat/Rmax", pk["ratio_Rmax"], target_ratio, tol=0.01)
        c2emp = pk["X_bar_over_Rmax"]
        c2emp_lo, c2emp_hi = min(c2emp_lo, c2emp), max(c2emp_hi, c2emp)
        if m == 150:
            check(s, "m=150 range_cap_active", bool(pk["range_cap_active"]), True)
    CHECKS.append(dict(section=s, name="C2_emp/Rmax in [0.07,0.14] (all m)",
                        actual=f"[{c2emp_lo:.3f},{c2emp_hi:.3f}]", expected="[0.07,0.14]",
                        tol="n/a", passed=(c2emp_lo >= 0.065 and c2emp_hi <= 0.145)))

    # range-term decomposition
    for m, (path, K, _) in files.items():
        with open(path) as f:
            d = json.load(f)
        n, T, delta_B = d["n"], d["T"], d["delta_B"]
        pk = d["per_K"][K]
        L = np.log(2.0 / delta_B)
        nH = n * T
        range_term = 7.0 * nH * L / (3.0 * (m - 1))
        target = {150: 4.333, 1500: 0.431, 3000: 0.215}[m]
        pct_target = {150: 97, 1500: 70, 3000: 60}[m]
        var_term = (pk["X_bar"] + np.sqrt(2 * pk["X_var"] * L / m)) / d["Rmax_hat"]
        total = var_term + range_term
        check(s, f"m={m} range_term/Rmax", range_term, target, tol=0.005)
        check(s, f"m={m} range_term pct_of_total", round(100 * range_term / total), pct_target, tol=1)


def verify_route3():
    s = "8 route3"
    with open(e2("route3_risk_attribution_summary.json")) as f:
        d = json.load(f)
    check(s, "episodes m", d["m"], 300, tol=0, cmp="eq")
    check(s, "early (t<12) share", d["late_vs_early_share"]["early"], 0.747, tol=0.002)
    check(s, "per_agent_share agent_0", d["per_agent_share"]["agent_0"], 0.30, tol=0.01)
    check(s, "per_agent_share agent_1", d["per_agent_share"]["agent_1"], 0.41, tol=0.01)
    check(s, "per_agent_share agent_2", d["per_agent_share"]["agent_2"], 0.28, tol=0.01)
    check(s, "top decile share", d["top10pct_bucket_share"], 0.231, tol=0.002)


# --------------------------------------------------------------------------- #
def verify_a3():
    s = "Appendix A3"
    df = pd.read_csv(data("exp_a3_rare_unit_scaling.csv"))
    target_m = {0.1: 2068, 0.03: 6919, 0.01: 20781}
    for p, m in target_m.items():
        row = df[np.isclose(df["p"], p)].iloc[0]
        check(s, f"p={p} m_required", int(row["m_required"]), m, tol=0, cmp="eq")
    x = np.log(df["inv_p"].values)
    y = np.log(df["m_required"].values)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    r2 = 1.0 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
    check(s, "log-log slope", slope, 1.002, tol=0.005)
    check(s, "R^2", r2, 1.000, tol=0.005)


def verify_a4():
    s = "Appendix A4"
    df = pd.read_csv(data("exp_a4_adjacent_baselines.csv"))
    check(s, "n_games", df["game"].nunique(), 8, tol=0, cmp="eq")
    check(s, "repeats", df["repeat"].nunique(), 50, tol=0, cmp="eq")
    g = df.groupby("method").agg(coverage=("covered", "mean"), median_ratio_L=("ratio_L", "median"))
    target = {"I2_B": 1.334, "PDIS": 1.692, "DR": 2.682, "robust_sim_lemma": 3.892}
    for method, val in target.items():
        check(s, f"{method} median bound/loss", g.loc[method, "median_ratio_L"], val, tol=0.005)
        check(s, f"{method} coverage", g.loc[method, "coverage"], 1.000, tol=0.001)


def verify_exp02():
    s = "Appendix exp_02"
    df = pd.read_csv(data("exp_02_eta_positive_success_term.csv"))
    check(s, "instances", len(df), 1200, tol=0, cmp="eq")
    g = df.groupby("eta")["viol_failure_only"].mean()
    check(s, "eta=0 violation rate", g[0.0], 0.0, tol=1e-9)
    for eta in [0.1, 0.2, 0.3]:
        check(s, f"eta={eta} violation rate", g[eta], 1.0, tol=1e-9)


def verify_exp05():
    s = "Appendix exp_05"
    df = pd.read_csv(data("exp_05_falsification.csv")).set_index("baseline")
    check(s, "C2 (adavote) violation rate", df.loc["adavote", "violation_rate"], 0.0, tol=1e-9)
    check(s, "unweighted violation rate", df.loc["unweighted", "violation_rate"], 1.000, tol=0.01)
    check(s, "plugin_alpha violation rate", df.loc["plugin_alpha", "violation_rate"], 0.177, tol=0.01)
    check(s, "no_logging violation rate", df.loc["no_logging", "violation_rate"], 0.0, tol=1e-9)
    check(s, "no_logging median slack", df.loc["no_logging", "median_slack_vs_Ctilde"], 0.44, tol=0.01)


def verify_exp07():
    s = "Appendix exp_07"
    df = pd.read_csv(data("exp_07_witness_sharpness.csv"))
    check(s, "witnesses", len(df), 85, tol=0, cmp="eq")
    check(s, "max occupancy reconstruction error", df["occupancy_max_err"].max(), 2.2e-16, tol=0.5e-16)


def verify_exp08_09():
    s = "Appendix exp_08/09"
    df8 = pd.read_csv(data("exp_08_rank_consistency.csv"))
    part_c = df8[df8["part"] == "c"]
    check(s, "median Spearman rho (part c)", part_c["spearman_rho"].median(), 1.0, tol=1e-6)
    check(s, "reversal fraction (part c)", part_c["has_reversal"].mean(), 0.000, tol=1e-6)

    df9 = pd.read_csv(data("exp_09_budget_increase.csv"))
    frac = df9["frac_non_increase_2b"].dropna().unique()
    check(s, "frac no true-loss increase (2b)", frac[0], 0.987, tol=0.001)


# --------------------------------------------------------------------------- #
def verify_figures():
    """§2.2: figure underlying data vs source CSV -- reuses the same
    computations as e1_tabular/make_paper_figures.py and
    compile_task9_consistency_table.py, re-derived here independently so this
    single script is sufficient on its own (no dependency on those files)."""
    s = "Figures"
    # fig:exp03 right (bound comparison), 4 groups
    dfb = pd.read_csv(data("exp_03_bound_types.csv"))
    gb = dfb.groupby("bound_type").agg(coverage=("covered", "mean"), tightness=("ratio_L", "median"))
    for bt, (cov, tight) in {"empirical_bernstein": (1.000, 1.201), "hoeffding": (1.000, 1.396),
                              "clipped_empirical_bernstein": (1.000, 1.180),
                              "naive_plugin": (0.650, 1.022)}.items():
        check(s, f"fig:exp03 {bt} (cov,tightness)", f"({gb.loc[bt,'coverage']:.3f},{gb.loc[bt,'tightness']:.3f})",
              f"({cov:.3f},{tight:.3f})", tol=0, cmp="eq")

    # fig:exp15bridge, 5 K groups
    df12 = pd.read_csv(data("exp_12_rollout_bridge.csv"))
    target_BL = dict(zip([25, 50, 100, 200, 400], [1.624, 1.637, 1.646, 1.614, 1.608]))
    for K, sub in df12.groupby("K"):
        check(s, f"fig:exp15bridge K={K} B/L", sub["ratio_L"].median(), target_BL[K], tol=0.005)

    # fig:correlated, 12 groups
    df14 = pd.read_csv(data("exp_14_correlated_committees.csv"))
    check(s, "fig:correlated n_levels", len(df14), 12, tol=0, cmp="eq")

    # fig:rareunit slope/R2
    dfa3 = pd.read_csv(data("exp_a3_rare_unit_scaling.csv"))
    x = np.log(dfa3["inv_p"].values)
    y = np.log(dfa3["m_required"].values)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    r2 = 1.0 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
    check(s, "fig:rareunit slope", slope, 1.002, tol=0.005)
    check(s, "fig:rareunit R^2", r2, 1.000, tol=0.005)

    # fig:exp01, 3 medians + slack quantiles
    df01 = pd.read_csv(data("exp_01_chain_ordering.csv"))
    check(s, "fig:exp01 median C0/L", df01["r_C0"].median(), 8.53, tol=0.005)
    check(s, "fig:exp01 median C1/L", df01["r_Cstar"].median(), 2.04, tol=0.005)
    check(s, "fig:exp01 median C2/L", df01["r_Ctilde"].median(), 1.02, tol=0.005)
    check(s, "fig:exp01 slack p95", df01["slack_Ctilde"].quantile(0.95), 0.030, tol=0.0005)


# --------------------------------------------------------------------------- #
def main():
    verify_exp01()
    verify_exp06()
    verify_exp13()
    verify_exp03()
    verify_exp12()
    verify_exp14()
    verify_mpe()
    verify_route3()
    verify_a3()
    verify_a4()
    verify_exp02()
    verify_exp05()
    verify_exp07()
    verify_exp08_09()
    verify_figures()

    df = pd.DataFrame(CHECKS)
    width_name = max(len(r["name"]) for r in CHECKS) + 2
    cur_section = None
    n_fail = 0
    for r in CHECKS:
        if r["section"] != cur_section:
            cur_section = r["section"]
            print(f"\n=== {cur_section} ===")
        status = "PASS" if r["passed"] else "FAIL"
        if not r["passed"]:
            n_fail += 1
        print(f"  [{status}] {r['name']:<{width_name}} actual={r['actual']!s:<16} "
              f"expected={r['expected']!s:<16} tol={r['tol']}")

    print(f"\n{'='*70}\nTOTAL: {len(CHECKS)} checks, {len(CHECKS)-n_fail} PASS, {n_fail} FAIL\n{'='*70}")
    if n_fail:
        print("\nFAILED CHECKS:")
        for r in CHECKS:
            if not r["passed"]:
                print(f"  [{r['section']}] {r['name']}: actual={r['actual']} "
                      f"expected={r['expected']} tol={r['tol']}")
        sys.exit(1)
    print("\nAll checks PASS.")
    sys.exit(0)


if __name__ == "__main__":
    main()
