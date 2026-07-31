"""
compile_task9_consistency_table.py — Task 9 deliverable: for every value that
appears in a figure, list the figure file, the value, its source CSV path +
field, and the paper's corresponding value, so all three can be checked
side by side. Recomputes every number directly from the source CSVs (same
computations as make_paper_figures.py) rather than hardcoding, so this table
itself is self-asserting.

Run with: python -m e1_tabular.compile_task9_consistency_table
Output:   figures/consistency_table.csv
"""
import numpy as np
import pandas as pd

from common.io_utils import data_path

rows = []


def add(figure, value_desc, value, source, paper_value):
    rows.append(dict(figure=figure, value_desc=value_desc, value=value,
                      source=source, paper_value=paper_value,
                      match=bool(np.isclose(value, paper_value, atol=1e-2))
                      if isinstance(paper_value, (int, float)) else "n/a"))


# fig:exp01
df = pd.read_csv(data_path("exp_01_chain_ordering.csv"))
add("chain_ratios_box.pdf", "median C0/L", round(df["r_C0"].median(), 3),
    "results/data/exp_01_chain_ordering.csv:r_C0.median()", 8.53)
add("chain_ratios_box.pdf", "median C1/L", round(df["r_Cstar"].median(), 3),
    "results/data/exp_01_chain_ordering.csv:r_Cstar.median()", 2.04)
add("chain_ratios_box.pdf", "median C2/L", round(df["r_Ctilde"].median(), 3),
    "results/data/exp_01_chain_ordering.csv:r_Ctilde.median()", 1.02)
add("chain_slack_hist.pdf", "slack median", round(df["slack_Ctilde"].median(), 4),
    "results/data/exp_01_chain_ordering.csv:slack_Ctilde.median()", 0.017)
add("chain_slack_hist.pdf", "slack p95", round(df["slack_Ctilde"].quantile(0.95), 4),
    "results/data/exp_01_chain_ordering.csv:slack_Ctilde.quantile(0.95)", 0.030)
add("chain_slack_hist.pdf", "slack max (axis range)", round(df["slack_Ctilde"].max(), 4),
    "results/data/exp_01_chain_ordering.csv:slack_Ctilde.max()", 0.042)

# fig:exp03
df3 = pd.read_csv(data_path("exp_03_estimator.csv"))
g3 = df3.groupby(["variant", "m"])["covered"].mean()
add("estimator_coverage.pdf", "I2 coverage m=500,1500,5000",
    f"{g3[('Pi2',500)]:.3f}/{g3[('Pi2',1500)]:.3f}/{g3[('Pi2',5000)]:.4f}",
    "results/data/exp_03_estimator.csv:groupby(Pi2,m).covered.mean()",
    "1.000/1.000/0.9994")
add("estimator_coverage.pdf", "I1 coverage m=500,1500,5000",
    f"{g3[('Pi1',500)]:.3f}/{g3[('Pi1',1500)]:.3f}/{g3[('Pi1',5000)]:.3f}",
    "results/data/exp_03_estimator.csv:groupby(Pi1,m).covered.mean()",
    "1.000/1.000/1.000")

dfb = pd.read_csv(data_path("exp_03_bound_types.csv"))
gb = dfb.groupby("bound_type").agg(coverage=("covered", "mean"), tightness=("ratio_L", "median"))
expect = {"empirical_bernstein": (1.000, 1.201), "hoeffding": (1.000, 1.396),
          "clipped_empirical_bernstein": (1.000, 1.180), "naive_plugin": (0.650, 1.022)}
for bt, (cov, tight) in expect.items():
    add("estimator_bound_comparison.pdf", f"{bt} coverage", round(float(gb.loc[bt, "coverage"]), 3),
        f"results/data/exp_03_bound_types.csv:groupby({bt}).covered.mean()", cov)
    add("estimator_bound_comparison.pdf", f"{bt} tightness", round(float(gb.loc[bt, "tightness"]), 3),
        f"results/data/exp_03_bound_types.csv:groupby({bt}).ratio_L.median()", tight)

# fig:exp15bridge (exp_12 K-sweep)
df12 = pd.read_csv(data_path("exp_12_rollout_bridge.csv"))
expect_BL = {25: 1.624, 50: 1.637, 100: 1.646, 200: 1.614, 400: 1.608}
expect_BR = {25: 0.148, 50: 0.149, 100: 0.147, 200: 0.145, 400: 0.145}
expect_XC2 = {25: 0.9982, 50: 1.0124, 100: 1.0097, 200: 0.9922, 400: 0.9871}
for K, sub in df12.groupby("K"):
    add("consbound_tightness_vs_K.pdf", f"K={K} median B/L", round(float(sub['ratio_L'].median()), 3),
        f"results/data/exp_12_rollout_bridge.csv:groupby(K={K}).ratio_L.median()", expect_BL[K])
    add("consbound_tightness_vs_K.pdf", f"K={K} median B/Rmax", round(float(sub['ratio_Rmax'].median()), 3),
        f"results/data/exp_12_rollout_bridge.csv:groupby(K={K}).ratio_Rmax.median()", expect_BR[K])
    add("consbound_rate_vs_K.pdf", f"K={K} median Xbar/C2", round(float(sub['ratio_XC2'].median()), 4),
        f"results/data/exp_12_rollout_bridge.csv:groupby(K={K}).ratio_XC2.median()", expect_XC2[K])
    add("consbound_tightness_vs_K.pdf", f"K={K} coverage", round(float(sub['covered'].mean()), 3),
        f"results/data/exp_12_rollout_bridge.csv:groupby(K={K}).covered.mean()", 1.000)

# fig:correlated
df14 = pd.read_csv(data_path("exp_14_correlated_committees.csv"))
expect_undercov = {500: 0.00, 300: 0.12, 200: 0.50, 150: 0.62, 100: 0.62, 75: 0.88,
                    50: 1.00, 30: 1.00, 20: 1.00, 10: 1.00, 5: 1.00, 2: 1.00}
expect_trueloss = {500: 0.351, 300: 0.352, 200: 0.354, 150: 0.355, 100: 0.358, 75: 0.361,
                    50: 0.366, 30: 0.377, 20: 0.388, 10: 0.418, 5: 0.461, 2: 0.523}
for _, r in df14.iterrows():
    s = int(r["s"])
    add("correlated_committee.pdf", f"s={s} plug-in undercoverage", round(float(r["plugin_under_coverage_rate"]), 2),
        f"results/data/exp_14_correlated_committees.csv:s={s}.plugin_under_coverage_rate", expect_undercov[s])
    add("correlated_true_loss.pdf", f"s={s} mean true_loss", round(float(r["mean_true_loss"]), 3),
        f"results/data/exp_14_correlated_committees.csv:s={s}.mean_true_loss", expect_trueloss[s])

# fig:rareunit
dfa3 = pd.read_csv(data_path("exp_a3_rare_unit_scaling.csv"))
x = np.log(dfa3["inv_p"].values)
y = np.log(dfa3["m_required"].values)
slope, intercept = np.polyfit(x, y, 1)
y_pred = slope * x + intercept
r2 = 1.0 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
add("rareunit_m_vs_inv_p.pdf", "fit slope", round(float(slope), 3),
    "results/data/exp_a3_rare_unit_scaling.csv:polyfit(log(inv_p),log(m_required))", 1.002)
add("rareunit_m_vs_inv_p.pdf", "R^2", round(float(r2), 3),
    "results/data/exp_a3_rare_unit_scaling.csv:polyfit residual R^2", 1.000)
for _, r in dfa3.iterrows():
    add("rareunit_m_vs_inv_p.pdf", f"p={r['p']} m_required", int(r["m_required"]),
        f"results/data/exp_a3_rare_unit_scaling.csv:p={r['p']}.m_required",
        {0.1: 2068, 0.03: 6919, 0.01: 20781}[r["p"]])

# fig:exp02, fig:exp05, fig:exp06 -- instance counts + qualitative
df2 = pd.read_csv(data_path("exp_02_eta_positive_success_term.csv"))
add("eta_success_term_violation.pdf", "instances", len(df2),
    "results/data/exp_02_eta_positive_success_term.csv:len(df)", 1200)
df6 = pd.read_csv(data_path("exp_06_prefix_drift.csv"))
add("exp_06_naive_vs_true_scatter.pdf", "instances", len(df6),
    "results/data/exp_06_prefix_drift.csv:len(df)", 3600)
df5 = pd.read_csv(data_path("exp_05_falsification.csv")).set_index("baseline")
add("falsification_violation.pdf", "unweighted violation rate", round(float(df5.loc["unweighted", "violation_rate"]), 3),
    "results/data/exp_05_falsification.csv:baseline=unweighted.violation_rate", 1.000)
add("falsification_violation.pdf", "plugin_alpha violation rate", round(float(df5.loc["plugin_alpha", "violation_rate"]), 3),
    "results/data/exp_05_falsification.csv:baseline=plugin_alpha.violation_rate", 0.177)

out = pd.DataFrame(rows)
out_path = data_path("task9_consistency_table.csv")
out.to_csv(out_path, index=False)
import shutil
shutil.copy(out_path, "figures/consistency_table.csv")

n_mismatch = (out["match"] == False).sum()
print(out.to_string(index=False))
print(f"\n{len(out)} rows, {n_mismatch} mismatches, written to {out_path} and figures/consistency_table.csv")
assert n_mismatch == 0, "consistency table has mismatches -- see rows with match=False"
