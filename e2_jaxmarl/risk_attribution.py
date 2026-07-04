"""
e2_jaxmarl/risk_attribution.py — Route 3: risk-attribution demo on MPE
(paper Section 8: the certificate is additive over units, so
mu_bar(u)*g(u)*W_fb(u) is a risk-attribution score identifying which
state-agent-prefix units dominate the certified downside).

OPERATIONAL / MONITORING DEMO ONLY. This does NOT claim certified validity
and does NOT claim true-loss improvement -- it only ranks WHERE the
deployed committee's downside risk concentrates, for monitoring/hardening
purposes, exactly as Section 8 frames it ("diagnostic; does not by itself
claim true-loss improvement").

Unlike the certificate (rollout_certificate.py), this script does NOT need
large K: the swing estimate Q^ref-Q^fb is already near-noise-free at small K
in this environment (empirically confirmed: tail-rollout std ~1e-6, see
Pilot A decomposition), because only the CERTIFICATE's worst-case
concentration bound needs a large, inflated radius -- the underlying POINT
ESTIMATE of the swing itself does not. So this script uses a modest K and a
larger m (many episodes, ALL nH units logged per episode, not just one
subsampled unit per episode as in the strict certification protocol -- that
subsampling is required for Theorem 3's i.i.d. concentration argument, which
this descriptive analysis does not need).

Occupancy proxy: MPE's continuous state space means no two episodes ever
revisit the exact same tabular state, so mu_bar(u) is approximated by
treating each of the nH=(t,agent) coordinate "slots" as equally occupied
(each appears exactly once per deployed episode) -- i.e. mu_bar is uniform
over (t,agent), and per-slot risk is aggregated by averaging F*W_fb over the
m episodes that visited it. This is an explicit, documented approximation,
not an exact occupancy measure (only available in the tabular setting).

Usage:
    python -m e2_jaxmarl.risk_attribution --env simple_spread --algo ippo \
        --checkpoints_dir results/e2/checkpoints --m 500 --k 20 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

_JAX_AVAILABLE = False
_IMPORT_ERROR_MSG = ""
try:
    import jax
    import jax.numpy as jnp
    from jaxmarl import make as jaxmarl_make
    from e2_jaxmarl.train_committee import _resolve_env_name
    from e2_jaxmarl.rollout_certificate import (
        load_committee, build_logits_fn, build_single_logits_fn,
        build_ref_logits_batch_fn, deploy_episode, mc_tail_return_batch,
    )
    _JAX_AVAILABLE = True
except Exception as _e:
    _IMPORT_ERROR_MSG = str(_e)

from common.certificates import clip_pos
from common.io_utils import data_path, fig_path
from common.plotting import new_fig, save_pdf

DIAG = ("ROUTE-3 risk-attribution DEMO -- operational/monitoring use only. "
        "NOT a validity claim; does NOT claim true-loss improvement (paper Section 8).")


def _require_jax():
    if not _JAX_AVAILABLE:
        raise RuntimeError(f"JAX/JaxMARL not available: {_IMPORT_ERROR_MSG}")


def run_risk_attribution(env_name, algo, checkpoints_dir, ref_member, m, K, T, seed):
    _require_jax()
    env = jaxmarl_make(_resolve_env_name(env_name))
    members = load_committee(checkpoints_dir, env_name, algo)
    logits_fn = build_logits_fn(env, members, algo)
    agents = env.agents
    n = len(agents)
    N = len(members)
    stepped_fn = jax.jit(jax.vmap(env.step))
    ref_logits_batch_fn = build_ref_logits_batch_fn(logits_fn, members[ref_member]["params"], agents)
    single_logits_fn = build_single_logits_fn(logits_fn, agents)
    jitted_step = jax.jit(env.step)

    jax_rng = jax.random.PRNGKey(seed)
    k_rng = jax.random.PRNGKey(seed + 1)

    print(f"[risk_attribution] deploying {m} episodes (env={env_name} algo={algo} "
          f"N={N} n={n} T={T}), logging ALL nH={n*T} units per episode "
          f"(not subsampled -- this is a descriptive demo, not the strict "
          f"certification protocol) ...")

    # per (t, agent) bucket: running sums for aggregation
    bucket_F_sum = defaultdict(float)
    bucket_score_sum = defaultdict(float)
    bucket_count = defaultdict(int)
    all_records = []  # (episode, t, agent, F, score) for top-K instance ranking

    for ep in range(m):
        jax_rng, ep_key = jax.random.split(jax_rng)
        timestep_info, units, team_return, ep_rewards = deploy_episode(
            env, members, ref_member, single_logits_fn, jitted_step, T, ep_key)

        for u in units:
            t, i, agent, F, a_fb = u["t"], u["i"], u["agent"], u["F"], u["a_fb"]
            bucket_count[(t, agent)] += 1
            bucket_F_sum[(t, agent)] += F
            score = 0.0
            if F == 1:
                ti = timestep_info[t]
                k_rng, ref_key, fb_key = jax.random.split(k_rng, 3)
                ref_samples = mc_tail_return_batch(
                    stepped_fn, ref_logits_batch_fn, agents, T,
                    ti["state_before"], t, i, ti["ref_actions"][agent],
                    ti["joint_action_executed"], ti["ref_actions"], K, ref_key)
                fb_samples = mc_tail_return_batch(
                    stepped_fn, ref_logits_batch_fn, agents, T,
                    ti["state_before"], t, i, a_fb,
                    ti["joint_action_executed"], ti["ref_actions"], K, fb_key)
                swing = float(clip_pos(np.asarray(ref_samples.mean() - fb_samples.mean())))
                score = swing  # mu_bar(u)=1/m-normalized later; F=1 here so g(u)*W_fb(u)=swing
            bucket_score_sum[(t, agent)] += score
            all_records.append(dict(episode=ep, t=t, agent=agent, F=F, score=score))
        if (ep + 1) % max(1, m // 10) == 0:
            print(f"  ... {ep+1}/{m} episodes logged")

    return bucket_F_sum, bucket_score_sum, bucket_count, all_records, agents


def main():
    ap = argparse.ArgumentParser(description="Route-3 risk-attribution operational demo (not a validity claim)")
    ap.add_argument("--env", default="simple_spread")
    ap.add_argument("--algo", default="ippo", choices=["ippo", "mappo"])
    ap.add_argument("--checkpoints_dir", default="results/e2/checkpoints")
    ap.add_argument("--ref_member", type=int, default=0)
    ap.add_argument("--m", type=int, default=500, help="deployment episodes (ALL units logged per episode)")
    ap.add_argument("--k", type=int, default=20, help="rollout budget for the swing POINT ESTIMATE (not a certificate; small K suffices since dynamics are near-deterministic)")
    ap.add_argument("--T", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bucket_F_sum, bucket_score_sum, bucket_count, all_records, agents = run_risk_attribution(
        args.env, args.algo, args.checkpoints_dir, args.ref_member, args.m, args.k, args.T, args.seed)

    n = len(agents)
    T = args.T

    # ---- Bucket-level aggregation: mean g(u)*W_fb(u) per (t, agent) ----
    rows = []
    for t in range(T):
        for agent in agents:
            key = (t, agent)
            cnt = bucket_count.get(key, 0)
            if cnt == 0:
                continue
            mean_g = bucket_F_sum[key] / cnt
            mean_score = bucket_score_sum[key] / cnt  # = mean(F*swing) over episodes = g(u)*E[W_fb(u)|bucket]
            rows.append(dict(t=t, agent=agent, mean_failure_rate=mean_g,
                              mean_risk_score=mean_score, n_visits=cnt))

    import pandas as pd
    df = pd.DataFrame(rows)
    total_risk = df["mean_risk_score"].sum()
    df["risk_share"] = df["mean_risk_score"] / total_risk if total_risk > 0 else 0.0
    df = df.sort_values("mean_risk_score", ascending=False).reset_index(drop=True)

    csv_path = data_path("route3_risk_attribution_buckets.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path}")
    print(df.head(15).to_string(float_format=lambda x: f"{x:.5f}"))

    # top-10% concentration stat
    n_top = max(1, len(df) // 10)
    top_share = df.head(n_top)["risk_share"].sum()
    print(f"\ntop-10% buckets ({n_top}/{len(df)}) account for {top_share*100:.1f}% of total attributed risk")

    # ---- per-agent aggregation ----
    agent_agg = df.groupby("agent")["mean_risk_score"].sum().sort_values(ascending=False)
    print("\nper-agent total risk contribution:")
    print(agent_agg.to_string(float_format=lambda x: f"{x:.5f}"))

    # ---- per-timestep aggregation ----
    t_agg = df.groupby("t")["mean_risk_score"].sum()

    # ---- Figure 1: risk score heatmap-style bar, by (t, agent), stacked ----
    fig, ax = new_fig(figsize=(8, 4.5))
    bottom = np.zeros(T)
    colors = plt_colors = ["steelblue", "darkorange", "seagreen", "tomato", "purple"]
    for ai, agent in enumerate(agents):
        vals = np.array([bucket_score_sum.get((t, agent), 0.0) / max(bucket_count.get((t, agent), 1), 1)
                          for t in range(T)])
        ax.bar(range(T), vals, bottom=bottom, label=agent, color=colors[ai % len(colors)])
        bottom += vals
    ax.set_xlabel("timestep t")
    ax.set_ylabel(r"mean risk score  $\bar{g}(u) \cdot \bar{W}_{fb}(u)$")
    ax.set_title("Route 3 (operational demo): risk attribution by timestep and agent\n"
                 "NOT a validity claim -- monitoring/hardening use only")
    ax.legend(fontsize=8, ncol=n)
    save_pdf(fig, fig_path("route3_risk_by_timestep_agent.pdf"))

    # ---- Figure 2: top-K individual (episode,t,agent) instances ----
    rec_df = pd.DataFrame(all_records)
    rec_df = rec_df.sort_values("score", ascending=False).reset_index(drop=True)
    top20 = rec_df.head(20)
    fig, ax = new_fig(figsize=(7, 5))
    labels = [f"ep{r.episode} t={r.t} {r.agent}" for r in top20.itertuples()]
    ax.barh(range(len(top20)), top20["score"].values[::-1], color="tomato")
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(labels[::-1], fontsize=6)
    ax.set_xlabel("risk score (realized swing, F=1 instances)")
    ax.set_title("Route 3 (operational demo): top-20 individual risk-dominating instances\n"
                 "NOT a validity claim -- monitoring/hardening use only")
    save_pdf(fig, fig_path("route3_top_instances.pdf"))

    rec_csv = data_path("route3_risk_attribution_instances.csv")
    rec_df.to_csv(rec_csv, index=False)
    print(f"Instance-level CSV saved: {rec_csv}")

    summary = dict(
        m=args.m, K=args.k, T=T, n_agents=n,
        total_risk_score=float(total_risk),
        top10pct_bucket_share=float(top_share),
        per_agent_share=(agent_agg / agent_agg.sum()).to_dict(),
        late_vs_early_share=dict(
            early=float(t_agg[t_agg.index < T // 2].sum() / t_agg.sum()) if t_agg.sum() > 0 else 0.0,
            late=float(t_agg[t_agg.index >= T // 2].sum() / t_agg.sum()) if t_agg.sum() > 0 else 0.0,
        ),
    )
    out_path = os.path.join("results", "e2", "route3_risk_attribution_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved: {out_path}")
    print(f"\n[risk_attribution] {DIAG}")


if __name__ == "__main__":
    main()
