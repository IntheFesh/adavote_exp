"""
make_paper_figures.py — Task 9: regenerate the paper's figures/ PDFs from the
already-committed (Task 4/5/7/8, ddof + make_alpha_fn fixed) CSVs, matching
the ACTUALLY-SUBMITTED version's visual style exactly (not the possibly-
drifted dev-style some results/figs/*.pdf scripts currently produce -- this
was checked directly, pixel-by-pixel, against the original supplementary
bundle's figures/ before writing any code here; see the Task 9 report for
specifics of where the two diverged: several results/figs/ scripts add a
title, use different legend wording, or a different chart type than what
was actually submitted).

No experiment is rerun here. Every number that appears in a figure is
computed directly from its source CSV in this script (assert-checked before
plotting), never hardcoded -- per Task 9 §3's self-assertion requirement.

Run with: python -m e1_tabular.make_paper_figures
Writes:   figures/*.pdf  (13 files; see FIGURES_MANIFEST at the bottom)
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from scipy.stats import binom  # noqa: F401 (kept for parity with common.certificates)

from common.io_utils import data_path
from common import certificates as C

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 11
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["grid.alpha"] = 0.3

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def fpath(name):
    return os.path.join(FIG_DIR, name)


def save(fig, name):
    fig.savefig(fpath(name), format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fpath(name)}")


# --------------------------------------------------------------------------- #
def fig_exp01():
    df = pd.read_csv(data_path("exp_01_chain_ordering.csv"))
    r_C0, r_C1, r_C2 = df["r_C0"], df["r_Cstar"], df["r_Ctilde"]
    slack = df["slack_Ctilde"]

    med = dict(C0=r_C0.median(), C1=r_C1.median(), C2=r_C2.median())
    assert abs(med["C0"] - 8.53) < 0.05, med
    assert abs(med["C1"] - 2.04) < 0.05, med
    assert abs(med["C2"] - 1.02) < 0.05, med
    slack_med, slack_p95, slack_max = slack.median(), slack.quantile(0.95), slack.max()
    assert abs(slack_med - 0.017) < 0.003, slack_med
    assert abs(slack_p95 - 0.030) < 0.003, slack_p95
    assert abs(slack_max - 0.042) < 0.003, slack_max

    # --- left: chain_ratios_box.pdf ---
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.boxplot([r_C0, r_C1, r_C2], tick_labels=[r"$C_0/L$", r"$C_1/L$", r"$C_2/L$"],
               showfliers=False)
    ax.axhline(1.0, color="red", ls="-.", lw=1.2, label="=L (tight)")
    ax.set_ylabel("certificate / true loss")
    ax.legend()
    save(fig, "chain_ratios_box.pdf")

    # --- right: chain_slack_hist.pdf (axis tightened per Task 9 §2.1: new
    # max is 0.042, not the old ~0.06, so the upper bound is pulled in to 0.05
    # to avoid a large empty tail implying a more distant outlier than exists) ---
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.hist(slack, bins=30, range=(0.0, 0.05), color="steelblue", edgecolor="k", linewidth=0.5)
    ax.set_xlim(0.0, 0.05)
    ax.set_xlabel(r"$(C_2-L)/L$")
    ax.set_ylabel("count")
    save(fig, "chain_slack_hist.pdf")


# --------------------------------------------------------------------------- #
def fig_exp02():
    df = pd.read_csv(data_path("exp_02_eta_positive_success_term.csv"))
    g = df.groupby("eta").agg(
        fail=("viol_failure_only", "mean"), succ=("viol_success_inclusive", "mean"))
    assert np.isclose(g.loc[0.0, "fail"], 0.0)
    assert np.isclose(g.loc[0.0, "succ"], 0.0)
    for eta in [0.1, 0.2, 0.3]:
        assert np.isclose(g.loc[eta, "fail"], 1.0)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    etas = g.index.values
    ax.plot(etas, g["fail"], "-o", color="C0", ms=8, label=r"$nH\,\mathbb{E}[gW]$ (failure-only)")
    ax.plot(etas, g["succ"], "-s", color="C1", ms=7,
            label=r"$nH\,\mathbb{E}[(1-g)w_\psi + gW]$ (success-inclusive)")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("violation rate")
    ax.legend()
    save(fig, "eta_success_term_violation.pdf")


# --------------------------------------------------------------------------- #
def fig_exp03():
    df = pd.read_csv(data_path("exp_03_estimator.csv"))
    g = df.groupby(["variant", "m"])["covered"].mean()
    ms = sorted(df["m"].unique())
    assert np.isclose(g[("Pi2", 5000)], 0.9994, atol=1e-3)
    assert np.isclose(g[("Pi1", 5000)], 1.0)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(ms, [g[("Pi2", m)] for m in ms], "-o", color="C0", ms=9, label=r"$\mathcal{I}_2$ (logged fallback)")
    ax.plot(ms, [g[("Pi1", m)] for m in ms], "-o", color="C1", ms=9, label=r"$\mathcal{I}_1$ (worst swing)")
    ax.axhline(0.95, color="red", ls="--", lw=1.2, label=r"$1-\delta = 0.95$")
    ax.set_xscale("log")
    ax.set_xticks(ms)
    ax.set_xticklabels([str(m) for m in ms])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("samples $m$")
    ax.set_ylabel("coverage")
    ax.set_ylim(0.90, 1.01)
    ax.legend(loc="lower left")
    save(fig, "estimator_coverage.pdf")

    # --- right: estimator_bound_comparison.pdf (from Task 8's new exp_03_bound_types) ---
    dfb = pd.read_csv(data_path("exp_03_bound_types.csv"))
    gb = dfb.groupby("bound_type").agg(coverage=("covered", "mean"), tightness=("ratio_L", "median"))
    expect = {"empirical_bernstein": (1.000, 1.201), "hoeffding": (1.000, 1.396),
              "clipped_empirical_bernstein": (1.000, 1.180), "naive_plugin": (0.650, 1.022)}
    for bt, (cov, tight) in expect.items():
        assert abs(gb.loc[bt, "coverage"] - cov) < 0.01, (bt, gb.loc[bt, "coverage"])
        assert abs(gb.loc[bt, "tightness"] - tight) < 0.01, (bt, gb.loc[bt, "tightness"])

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    style = {
        "empirical_bernstein": ("o", "C0", "Emp.-Bernstein"),
        "hoeffding": ("o", "C1", "Hoeffding"),
        "naive_plugin": ("X", "crimson", "plug-in mean"),
        "clipped_empirical_bernstein": ("o", "green", "quantile-clipped empirical Bernstein"),
    }
    for bt, (marker, color, label) in style.items():
        cov, tight = gb.loc[bt, "coverage"], gb.loc[bt, "tightness"]
        ax.scatter([tight], [cov], marker=marker, s=140, color=color, edgecolor="k",
                   linewidth=0.8,
                   label=f"{label} (cov={cov:.3f}, $\\hat B/L$={tight:.2f})"
                         + (" [biased, no proven bound]" if bt == "clipped_empirical_bernstein" else ""))
    ax.axhline(0.95, color="red", ls="--", lw=1.2, label=r"$1-\delta = 0.95$")
    ax.set_xlabel(r"median $\hat B/L$ (tightness, lower is better)")
    ax.set_ylabel("coverage (higher is better)")
    ax.set_ylim(0.55, 1.03)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
    save(fig, "estimator_bound_comparison.pdf")


# --------------------------------------------------------------------------- #
def fig_exp05():
    df = pd.read_csv(data_path("exp_05_falsification.csv"))
    order = ["adavote", "no_logging", "worst_prefix", "naive_marginal", "plugin_alpha", "unweighted"]
    labels = [r"$C_2$ full", r"$C_1$ no-log", "worst-prefix", "marginal", r"plug-in $\alpha$", "unweighted"]
    row = df.set_index("baseline")
    assert np.isclose(row.loc["adavote", "violation_rate"], 0.0)
    assert np.isclose(row.loc["unweighted", "violation_rate"], 1.000, atol=0.01)
    assert np.isclose(row.loc["plugin_alpha", "violation_rate"], 0.177, atol=0.01)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    vals = [row.loc[b, "violation_rate"] for b in order]
    colors = ["crimson" if v > 0 else "white" for v in vals]
    ax.barh(labels, vals, color=colors, edgecolor="k", linewidth=0.8)
    for i, (b, v) in enumerate(zip(order, vals)):
        if v == 0:
            note = "valid" if b == "adavote" else "valid (looser)"
            ax.text(0.01, i, note, va="center", fontsize=9, color="green")
    ax.set_xlabel("violation rate")
    ax.set_xlim(0, 1.05)
    save(fig, "falsification_violation.pdf")

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    looser = ["no_logging", "worst_prefix"]
    looser_labels = [r"$C_1$ no-log", "worst-prefix"]
    vals2 = [row.loc[b, "median_slack_vs_Ctilde"] for b in looser]
    ax.bar(looser_labels, vals2, color="orange", edgecolor="k", linewidth=0.8)
    ax.set_ylabel(r"median excess over $C_2$")
    save(fig, "falsification_excess.pdf")


# --------------------------------------------------------------------------- #
def fig_exp06():
    df = pd.read_csv(data_path("exp_06_prefix_drift.csv"))
    assert len(df) == 3600
    tl, naive = df["true_loss"].values, df["naive"].values
    viol = df["naive_violation"].astype(bool).values

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(tl[~viol], naive[~viol], s=10, alpha=0.4, color="C0")
    ax.scatter(tl[viol], naive[viol], marker="x", s=40, color="crimson")
    lim = max(tl.max(), naive.max()) * 1.02
    ax.plot([0, lim], [0, lim], "k--", lw=1.0, label="diagonal (bound = true)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("true loss")
    ax.set_ylabel("reference-prefix bound")
    ax.legend(loc="upper left")
    save(fig, "exp_06_naive_vs_true_scatter.pdf")


# --------------------------------------------------------------------------- #
def fig_exp12():
    df = pd.read_csv(data_path("exp_12_rollout_bridge.csv"))
    Ks = sorted(df["K"].unique())

    def boot_ci(vals, n_boot=2000, alpha=0.05, seed=0):
        rng = np.random.default_rng(seed)
        vals = np.asarray(vals)
        meds = np.array([np.median(rng.choice(vals, size=len(vals), replace=True))
                          for _ in range(n_boot)])
        return np.percentile(meds, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    med_BL, med_BR, med_XC2, lo_ci, hi_ci = [], [], [], [], []
    for K in Ks:
        sub = df[df.K == K]
        med_BL.append(sub["ratio_L"].median())
        med_BR.append(sub["ratio_Rmax"].median())
        xc2 = sub["ratio_XC2"].values
        med_XC2.append(np.median(xc2))
        lo, hi = boot_ci(xc2)
        lo_ci.append(lo)
        hi_ci.append(hi)

    expect_BL = [1.624, 1.637, 1.646, 1.614, 1.608]
    for a, b in zip(med_BL, expect_BL):
        assert abs(a - b) < 0.01, (a, b)

    # --- left: population conservatism (already matches the corrected
    # baseline from Task 4's rerun; re-derived here from CSV directly) ---
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    err = [np.array(med_XC2) - np.array(lo_ci), np.array(hi_ci) - np.array(med_XC2)]
    ax.errorbar(Ks, med_XC2, yerr=err, fmt="-o", color="C0", ms=8, capsize=4,
                label=r"median $\bar X/C_2$ (95% bootstrap CI)")
    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="population target = 1 (Jensen exact)")
    ax.set_xscale("log")
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("rollout budget $K$")
    ax.set_ylabel(r"median $\bar X/C_2$")
    ax.legend(loc="lower left", fontsize=9)
    save(fig, "consbound_rate_vs_K.pdf")

    # --- right: tightness vs K, twin axis (Task 9 §2.3: primary axis tightened
    # to resolve the [1.61,1.65] fluctuation the text now describes; B/Rmax
    # lives on its own natural scale on a secondary axis so both series stay
    # visible without changing colors/markers/legend content) ---
    fig, ax1 = plt.subplots(figsize=(6.0, 4.0))
    l1, = ax1.plot(Ks, med_BL, "-o", color="orange", ms=8, label=r"$\hat B_N/L$")
    ax1.set_ylim(1.55, 1.70)
    ax1.set_xscale("log")
    ax1.set_xticks(Ks)
    ax1.set_xticklabels([str(k) for k in Ks])
    ax1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.set_xlabel("rollout budget $K$")
    ax1.set_ylabel(r"median $\hat B_N/L$")

    ax2 = ax1.twinx()
    l2, = ax2.plot(Ks, med_BR, "--s", color="green", ms=7, label=r"$\hat B_N/R_{max}$")
    ax2.set_ylabel(r"median $\hat B_N/R_{max}$")
    ax2.grid(False)

    ax1.legend(handles=[l1, l2], loc="lower center")
    save(fig, "consbound_tightness_vs_K.pdf")


# --------------------------------------------------------------------------- #
def fig_correlated():
    df = pd.read_csv(data_path("exp_14_correlated_committees.csv")).sort_values("s", ascending=False)
    detail = pd.read_csv(data_path("exp_14_per_game.csv"))

    s_vals = df["s"].values
    undercov = df["plugin_under_coverage_rate"].values
    true_loss = df["mean_true_loss"].values
    mixture_viol_rate = 1.0 - df["mixture_coverage"].values
    plugin_cert_mean = detail.groupby("s")["Ctilde_plugin"].mean().reindex(s_vals).values
    mixture_cert_mean = df["mean_Ctilde_mixture"].values

    assert np.allclose(mixture_viol_rate, 0.0)
    assert abs(undercov[0] - 0.00) < 1e-6 and abs(undercov[-1] - 1.00) < 1e-6

    inv_s = 1.0 / s_vals

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(inv_s, undercov, "-o", color="crimson", ms=8, label=r"binomial plug-in $g_N(\hat\alpha)$")
    ax.plot(inv_s, mixture_viol_rate, "-s", color="steelblue", ms=7, label="mixture-$g$")
    ax.axhline(0.05, color="gray", ls=":", lw=1.2, label=r"$\delta=0.05$")
    ax.set_xscale("log")
    ax.set_xlabel(r"over-dispersion strength $1/(a_u+b_u)$")
    ax.set_ylabel("certificate violation rate")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "correlated_committee.pdf")

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(inv_s, true_loss, "-o", color="k", ms=7, label="true loss (exact-DP)")
    ax.plot(inv_s, mixture_cert_mean, "-s", color="steelblue", ms=7, label="mixture-$g$ cert")
    ax.plot(inv_s, plugin_cert_mean, "-^", color="crimson", ms=7, label="plug-in cert")
    ax.set_xscale("log")
    ax.set_xlabel(r"over-dispersion strength $1/(a_u+b_u)$")
    ax.set_ylabel("value")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "correlated_true_loss.pdf")


# --------------------------------------------------------------------------- #
def fig_rareunit():
    df = pd.read_csv(data_path("exp_a3_rare_unit_scaling.csv"))
    x = np.log(df["inv_p"].values)
    y = np.log(df["m_required"].values)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    r2 = 1.0 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
    assert abs(slope - 1.002) < 0.01, slope
    assert abs(r2 - 1.000) < 0.01, r2

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    inv_p = df["inv_p"].values
    m_req = df["m_required"].values
    xx = np.array([inv_p.min() * 0.9, inv_p.max() * 1.1])
    ax.plot(xx, np.exp(intercept) * xx ** slope, "-", color="gray", lw=2,
            label=f"fit: slope = {slope:.3f}")
    c0 = m_req[0] / inv_p[0]
    ax.plot(xx, c0 * xx, "--", color="salmon", lw=1.5, label="reference slope = 1")
    ax.scatter(inv_p, m_req, s=140, color="C0", edgecolor="k", zorder=5, label="empirical $m$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$1/p$ (inverse occupancy of rare unit $u^*$)")
    ax.set_ylabel(r"$m$ (required episode budget)")
    ax.set_title(f"rare-unit sample complexity: $m \\gtrsim 1/p$ (empirical slope {slope:.3f})")
    ax.legend(loc="lower right")
    save(fig, "rareunit_m_vs_inv_p.pdf")


FIGURES_MANIFEST = [
    "chain_ratios_box.pdf", "chain_slack_hist.pdf",
    "eta_success_term_violation.pdf",
    "estimator_coverage.pdf", "estimator_bound_comparison.pdf",
    "falsification_violation.pdf", "falsification_excess.pdf",
    "exp_06_naive_vs_true_scatter.pdf",
    "consbound_rate_vs_K.pdf", "consbound_tightness_vs_K.pdf",
    "correlated_committee.pdf", "correlated_true_loss.pdf",
    "rareunit_m_vs_inv_p.pdf",
]


def main():
    fig_exp01()
    fig_exp02()
    fig_exp03()
    fig_exp05()
    fig_exp06()
    fig_exp12()
    fig_correlated()
    fig_rareunit()

    missing = [f for f in FIGURES_MANIFEST if not os.path.exists(fpath(f))]
    assert not missing, f"missing figures: {missing}"
    print(f"\nAll {len(FIGURES_MANIFEST)} figures written to {FIG_DIR}/")


if __name__ == "__main__":
    main()
