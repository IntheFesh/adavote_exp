"""
e2_jaxmarl/rollout_certificate.py — Pilot A / Route 2: Theorem-4 conservative
rollout value bound + Theorem-3 finite-sample certificate, instantiated on a
REAL environment (MPE) via actual resettable Monte-Carlo rollouts.

This is NOT eval_committee.py's critic-based diagnostic (which only estimates
endorsement-rate stability using a one-step log-prob-gap proxy for Q). This
script actually DEPLOYS the agreement-gated committee controller on MPE and
computes a certificate whose validity is BY CONSTRUCTION (Theorem 4 + Theorem
3), the same as Assumption 2 discharged exactly, with NO exact-DP and NO
learned critic anywhere in the loop -- only resettable rollouts.

Reuses the EXACT SAME shared math as e1_tabular/exp_12_rollout_bridge.py
(common.certificates.wfb_plus_from_rollouts / rad_hoeffding /
delta_prime_union / empirical_bernstein), so Route 1 (tabular) and Route 2
(MPE) certificates are computed by identical formulas.

HONESTY BOUNDARY (see repo README / task brief):
  - MPE has no ground truth (no exact DP), so this certificate can be
    produced and its OPERATIONALITY (does it run, is it non-vacuous) can be
    checked, but its TIGHTNESS cannot be AUDITED against ground truth here.
    Tightness auditing is done only in e1_tabular/exp_12 against exact loss.
  - B_Q (per-step reward range) is NOT an exact analytic constant in MPE
    (unlike tabular Delta_r); it is estimated empirically from observed
    rollout rewards and is clearly logged as such.
  - Modeling choice: MPE agents act from local observations without
    mid-timestep communication. We adopt the same coordinate-telescoping
    convention as the tabular controller (Definition 3/Theorem 1): agents
    are ordered by env.agents, and unit (t, agent_i) uses prefix = the
    ACTUALLY-EXECUTED actions of agents before i this same timestep, with
    agents after i assigned their reference action (not yet "decided") when
    constructing the coordinate counterfactual for Q^ref/Q^fb. This is a
    valid instantiation of the general theory (Corollary 2: the carrier is
    agnostic to committee dependence structure) adapted to a real
    decentralized-execution simulator that requires one full joint action
    per env.step call.
  - Fallback rule: a^fb = committee PLURALITY vote among all N members'
    proposals (not synthetic uniform-random) -- a realistic deployable
    fallback.

Usage:
    python -m e2_jaxmarl.rollout_certificate \\
        --env simple_spread --algo ippo --ref_member 0 \\
        --checkpoints_dir results/e2/checkpoints \\
        --m 200 --k_grid 100,200,400,800 --seed 0
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
from typing import Any, Dict, List

import numpy as np

from common.certificates import (
    wfb_plus_from_rollouts, delta_prime_union, empirical_bernstein, clip_pos,
)

_JAX_AVAILABLE = False
_IMPORT_ERROR_MSG = ""
try:
    import jax
    import jax.numpy as jnp
    from jaxmarl import make as jaxmarl_make
    from e2_jaxmarl.train_committee import _build_network, _resolve_env_name, ActorMLP
    _JAX_AVAILABLE = True
except Exception as _e:
    _IMPORT_ERROR_MSG = str(_e)

DIAG = "ROUTE-2 by-construction certificate (Theorem 3 + Theorem 4) -- NOT audited for tightness (no MPE ground truth)"


def _require_jax():
    if not _JAX_AVAILABLE:
        raise RuntimeError(f"JAX/JaxMARL not available: {_IMPORT_ERROR_MSG}")


def load_committee(checkpoints_dir: str, env_name: str, algo: str) -> List[Dict[str, Any]]:
    pattern = os.path.join(checkpoints_dir, f"{env_name}__{algo}__member*__seed*.pkl")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No checkpoints found matching {pattern}")
    members = []
    for p in paths:
        with open(p, "rb") as f:
            members.append(pickle.load(f))
    return members


def build_logits_fn(env, members, algo):
    config = members[0]["config"]
    is_mappo = (algo == "mappo")
    if is_mappo:
        actor_hidden = tuple(config["ACTOR_HIDDEN"])
        actor_nets = {a: ActorMLP(hidden_sizes=actor_hidden, action_dim=env.action_space(a).n)
                      for a in env.agents}

        def logits_fn(params, agent, obs):
            return actor_nets[agent].apply(params["actor"][agent], obs)
    else:
        nets = _build_network(env, config)

        def logits_fn(params, agent, obs):
            l, _v = nets[agent].apply(params[agent], obs)
            return l
    return logits_fn


def plurality_action(proposals: np.ndarray, A: int) -> int:
    """Mode of proposals, ties broken toward the lowest action index."""
    counts = np.bincount(proposals, minlength=A)
    return int(np.argmax(counts))


def deploy_episode(env, members, ref_member, logits_fn, T, rng):
    """One deployment episode under the actual agreement-gated committee
    controller (eta=0, plurality-vote fallback). Returns:
      timestep_info: list of dicts per t (state_before, obs_before,
        ref_actions, joint_action_executed)
      units: list of dicts per (t, agent-index) (t, i, agent, F, a_fb)
      team_return: realized episode return (mean-of-agents reward summed)
      all_rewards: list of per-step mean rewards (for empirical B_Q estimate)
    """
    agents = env.agents
    N = len(members)
    A = env.action_space(agents[0]).n
    ref_params = members[ref_member]["params"]
    other_params = [m["params"] for k, m in enumerate(members) if k != ref_member]

    rng, rk = jax.random.split(rng)
    obs, state = env.reset(rk)

    timestep_info = []
    units = []
    team_return = 0.0
    all_rewards = []

    for t in range(T):
        state_before = state
        obs_before = obs
        ref_actions_this_t = {}
        for agent in agents:
            o = jnp.asarray(obs_before[agent])
            ref_actions_this_t[agent] = int(jnp.argmax(logits_fn(ref_params, agent, o)))

        joint_action = {}
        for i, agent in enumerate(agents):
            o = jnp.asarray(obs_before[agent])
            proposals = [ref_actions_this_t[agent]]
            for op in other_params:
                proposals.append(int(jnp.argmax(logits_fn(op, agent, o))))
            proposals = np.array(proposals)
            n_endorse = int(np.sum(proposals == ref_actions_this_t[agent]))
            F = 1 if n_endorse <= N // 2 else 0
            if F == 0:
                chosen = ref_actions_this_t[agent]
                a_fb = None
            else:
                chosen = plurality_action(proposals, A)
                a_fb = chosen
            joint_action[agent] = chosen
            units.append(dict(t=t, i=i, agent=agent, F=F, a_fb=a_fb))

        timestep_info.append(dict(
            state_before=state_before, obs_before=obs_before,
            ref_actions=ref_actions_this_t, joint_action_executed=dict(joint_action),
        ))

        rng, sk = jax.random.split(rng)
        actions_j = {a: jnp.asarray(joint_action[a]) for a in agents}
        obs, state, reward, done, info = env.step(sk, state_before, actions_j)
        r = float(np.mean([float(reward[a]) for a in agents]))
        team_return += r
        all_rewards.append(r)
        if bool(done.get("__all__", False)):
            break

    return timestep_info, units, team_return, all_rewards


def mc_tail_return_batch(stepped_fn, ref_params, logits_fn, agents, T,
                          state0, t0, agent_idx, coordinate_action,
                          prefix_actions, ref_actions_this_t, K, rng):
    """K parallel resettable rollouts: apply the hypothetical joint action at
    (state0, t0) -- prefix (already-decided agents < i), coordinate_action
    (agent i), reference (agents > i, not yet decided) -- then follow pi^ref
    (all agents, greedy) for t0+1..T-1. Returns an array of K realized tail
    returns (team-mean reward, summed over the tail).
    """
    joint0 = {}
    for k, a in enumerate(agents):
        if k < agent_idx:
            joint0[a] = prefix_actions[a]
        elif k == agent_idx:
            joint0[a] = coordinate_action
        else:
            joint0[a] = ref_actions_this_t[a]

    state_b = jax.tree_util.tree_map(lambda x: jnp.stack([x] * K), state0)
    joint0_b = {a: jnp.full((K,), v, dtype=jnp.int32) for a, v in joint0.items()}

    rng, key0 = jax.random.split(rng)
    keys0 = jax.random.split(key0, K)
    obs_b, state_b, reward_b, done_b, info_b = stepped_fn(keys0, state_b, joint0_b)
    total = jnp.mean(jnp.stack([reward_b[a] for a in agents], axis=0), axis=0)

    for t in range(t0 + 1, T):
        actions_b = {}
        for a in agents:
            logits_b = jax.vmap(lambda o: logits_fn(ref_params, a, o))(obs_b[a])
            actions_b[a] = jnp.argmax(logits_b, axis=-1).astype(jnp.int32)
        rng, key_t = jax.random.split(rng)
        keys_t = jax.random.split(key_t, K)
        obs_b, state_b, reward_b, done_b, info_b = stepped_fn(keys_t, state_b, actions_b)
        total = total + jnp.mean(jnp.stack([reward_b[a] for a in agents], axis=0), axis=0)

    return np.asarray(total)


def run_certification(env_name, algo, checkpoints_dir, ref_member, m, k_grid,
                       delta_B, delta_G, T, seed):
    _require_jax()
    env = jaxmarl_make(_resolve_env_name(env_name))
    members = load_committee(checkpoints_dir, env_name, algo)
    logits_fn = build_logits_fn(env, members, algo)
    agents = env.agents
    n = len(agents)
    N = len(members)
    stepped_fn = jax.jit(jax.vmap(env.step))

    master_rng = np.random.default_rng(seed)
    jax_rng = jax.random.PRNGKey(seed)

    # ---- Phase 1: deploy m episodes, sample one unit each ----
    print(f"[rollout_certificate] deploying {m} certification episodes "
          f"(env={env_name} algo={algo} N={N} n={n} T={T}) ...")
    sampled = []  # list of (timestep_info_t, units_t, agent_idx, F, a_fb) per episode
    all_rewards = []
    returns = []
    n_fail_logged = 0
    for j in range(m):
        jax_rng, ep_key = jax.random.split(jax_rng)
        timestep_info, units, team_return, ep_rewards = deploy_episode(
            env, members, ref_member, logits_fn, T, ep_key)
        all_rewards.extend(ep_rewards)
        returns.append(team_return)
        actual_T = len(timestep_info)
        idx = master_rng.integers(0, len(units))
        u = units[idx]
        ti = timestep_info[u["t"]]
        sampled.append((ti, u))
        if u["F"] == 1:
            n_fail_logged += 1

    alpha_hat_endorse = 1.0 - (n_fail_logged / m)
    print(f"[rollout_certificate] m={m} episodes; sampled-unit failure rate "
          f"(m_F/m) = {n_fail_logged}/{m} = {n_fail_logged/m:.4f}  "
          f"(diagnostic endorsement proxy = {alpha_hat_endorse:.4f})")

    # ---- Empirical per-step reward range -> B_Q, Rmax ----
    all_rewards = np.array(all_rewards)
    r_min, r_max = float(all_rewards.min()), float(all_rewards.max())
    delta_r_hat = r_max - r_min
    Rmax_hat = T * delta_r_hat
    B_Q = Rmax_hat  # global tail-return range bound (Theorem 4 convention)
    print(f"[rollout_certificate] empirical per-step reward range: "
          f"[{r_min:.4f}, {r_max:.4f}]  delta_r_hat={delta_r_hat:.4f}  "
          f"Rmax_hat = T*delta_r_hat = {Rmax_hat:.4f}  (B_Q uses this empirical range -- "
          f"MPE has no analytic reward bound, unlike tabular Delta_r)")
    print(f"[rollout_certificate] mean team return over {m} episodes: {np.mean(returns):.4f}")

    nH = n * T
    b0 = nH * B_Q
    m_F = n_fail_logged
    delta_prime = delta_prime_union(delta_G, m_F)

    results = {}
    for K in k_grid:
        jax_rng, k_key = jax.random.split(jax_rng)
        X = np.zeros(m)
        for j, (ti, u) in enumerate(sampled):
            if u["F"] == 0:
                continue
            k_key, ref_key, fb_key = jax.random.split(k_key, 3)
            ref_samples = mc_tail_return_batch(
                stepped_fn, members[ref_member]["params"], logits_fn, agents, T,
                ti["state_before"], u["t"], u["i"], ti["ref_actions"][u["agent"]],
                ti["joint_action_executed"], ti["ref_actions"], K, ref_key)
            fb_samples = mc_tail_return_batch(
                stepped_fn, members[ref_member]["params"], logits_fn, agents, T,
                ti["state_before"], u["t"], u["i"], u["a_fb"],
                ti["joint_action_executed"], ti["ref_actions"], K, fb_key)
            Qref_hat = float(ref_samples.mean())
            Qfb_hat = float(fb_samples.mean())
            w_fb_plus = wfb_plus_from_rollouts(Qref_hat, Qfb_hat, K, delta_prime, B_Q)
            X[j] = nH * w_fb_plus * 1.0

        Bhat = empirical_bernstein(X, delta_B, b0)
        Bhat_capped = min(Rmax_hat, Bhat)
        results[K] = dict(Bhat=Bhat, Bhat_capped=Bhat_capped, Rmax=Rmax_hat,
                           ratio_Rmax=Bhat / Rmax_hat, ratio_Rmax_capped=Bhat_capped / Rmax_hat,
                           m_F=m_F, delta_prime=delta_prime)
        print(f"[rollout_certificate] K={K:4d}  Bhat={Bhat:.4f}  Bhat_capped={Bhat_capped:.4f}  "
              f"Rmax={Rmax_hat:.4f}  Bhat/Rmax={Bhat/Rmax_hat:.4f}  "
              f"(cap {'ACTIVE' if Bhat > Rmax_hat else 'inactive'})")

    return dict(env=env_name, algo=algo, N=N, n=n, T=T, m=m, m_F=m_F,
                delta_B=delta_B, delta_G=delta_G, delta_r_hat=delta_r_hat,
                Rmax_hat=Rmax_hat, mean_return=float(np.mean(returns)),
                alpha_hat_endorse=alpha_hat_endorse, per_K=results)


def main():
    ap = argparse.ArgumentParser(description="Route-2 Theorem-3/4 rollout certificate on MPE (by-construction, not tightness-audited)")
    ap.add_argument("--env", default="simple_spread")
    ap.add_argument("--algo", default="ippo", choices=["ippo", "mappo"])
    ap.add_argument("--checkpoints_dir", default="results/e2/checkpoints")
    ap.add_argument("--ref_member", type=int, default=0)
    ap.add_argument("--m", type=int, default=200, help="certification episodes")
    ap.add_argument("--k_grid", default="100,200,400,800")
    ap.add_argument("--delta_B", type=float, default=0.05)
    ap.add_argument("--delta_G", type=float, default=0.05)
    ap.add_argument("--T", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    k_grid = [int(x) for x in args.k_grid.split(",")]
    res = run_certification(args.env, args.algo, args.checkpoints_dir, args.ref_member,
                             args.m, k_grid, args.delta_B, args.delta_G, args.T, args.seed)

    import json
    out_dir = "results/e2"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"rollout_certificate_{args.env}_{args.algo}.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\n[rollout_certificate] {DIAG}")
    print(f"[rollout_certificate] saved -> {out_path}")


if __name__ == "__main__":
    main()
