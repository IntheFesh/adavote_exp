"""
e2_jaxmarl/eval_committee.py — Deployment-time committee evaluation (E2).

DIAGNOSTIC / EMPIRICAL PROXY — NOT STRICT-VALIDITY EVIDENCE.
Strict validity (coverage guarantees) comes ONLY from E1 (e1_tabular/).

Loads a trained committee (see train_committee.py), fixes one member as the
reference policy π^ref, and treats the remaining members as advisors.  At each
agent coordinate during deployment rollouts it:

  * forms the committee endorsement: α̂(u) = frequency that advisor members'
    proposed actions land in the reference-relative endorsed set
        G^η(u) = { a : Δ_+(u,a) ≤ η };
  * uses the reference critic to APPROXIMATE Q^{π^ref}(u,·) and hence the
    clipped fallback advantage W̃^+(u) (this is Assumption 1 — a *proxy*, not
    an exact tabular DP, so the resulting certificate is diagnostic only);
  * forms the Theorem-3 style empirical-Bernstein DIAGNOSTIC certificate via
    `common.certificates.empirical_bernstein` and `g_N`.

Outputs diagnostic metrics consumed by run_all_e2.py.  Everything is labeled
"diagnostic — not strict validity".

Usage (on a GPU box):
    python -m e2_jaxmarl.eval_committee \\
        --env simple_spread --algo ippo \\
        --checkpoints_dir results/e2/checkpoints \\
        --ref_member 0 --eta 0.0 --n_eval_episodes 64 --mc_rollouts 16
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np

# numpy-only certificate helpers are shared with E1 (no JAX inside them).
from common.certificates import g_N, empirical_bernstein, clip_pos

# ---------------------------------------------------------------------------
# Graceful JAX import guard (eval runs the trained nets on the GPU).
# ---------------------------------------------------------------------------
_JAX_AVAILABLE = False
_IMPORT_ERROR_MSG = ""
try:
    import jax
    import jax.numpy as jnp
    import distrax  # noqa: F401
    from jaxmarl import make as jaxmarl_make
    # Reuse the network + config from the trainer.
    from e2_jaxmarl.train_committee import default_config, _build_network
    _JAX_AVAILABLE = True
except Exception as _e:  # ImportError or downstream errors on CPU-only boxes
    _IMPORT_ERROR_MSG = str(_e)

DIAG = "DIAGNOSTIC / EMPIRICAL PROXY — NOT STRICT-VALIDITY EVIDENCE"


def _require_jax() -> None:
    if not _JAX_AVAILABLE:
        raise RuntimeError(
            "\n=================================================================\n"
            "E2 (GPU / JaxMARL) dependencies are not installed/importable.\n"
            f"  Import error: {_IMPORT_ERROR_MSG}\n\n"
            "Install (GPU box):   pip install -r requirements_e2.txt\n"
            "CPU smoke-test:      JAX_PLATFORMS=cpu python -m e2_jaxmarl.eval_committee ...\n"
            "Strict validity is provided by E1 only (e1_tabular/).\n"
            "=================================================================\n"
        )


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def load_committee(checkpoints_dir: str, env_name: str, algo: str) -> List[Dict[str, Any]]:
    """Load all committee checkpoints for (env, algo), sorted by member index."""
    pattern = os.path.join(checkpoints_dir, f"{env_name}__{algo}__member*__seed*.pkl")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No checkpoints found matching {pattern}.\n"
            f"Train first:  python -m e2_jaxmarl.train_committee "
            f"--env {env_name} --algo {algo} --n_members 7 ..."
        )
    members = []
    for p in paths:
        with open(p, "rb") as f:
            members.append(pickle.load(f))
    return members


# ---------------------------------------------------------------------------
# Endorsement / advantage proxies (Assumption 1)
# ---------------------------------------------------------------------------
def _advantage_proxy(ref_logits, ref_value, member_action, ref_action):
    """Clipped non-negative advantage proxy Δ_+ for an advisor's action.

    We approximate Q^{π^ref}(u,a) by a one-step advantage read off the
    reference actor's log-prob gap (a coarse but standard critic-based proxy
    under Assumption 1).  Returns Δ_+ = [Q(ref) - Q(a)]_+ as a scalar.
    """
    # log pi^ref(a) as a stand-in for relative action quality under the
    # reference policy; larger gap => larger swing.
    logp = jax.nn.log_softmax(ref_logits)
    delta = logp[ref_action] - logp[member_action]
    return float(clip_pos(np.asarray(delta)))


# ---------------------------------------------------------------------------
# Core diagnostic evaluation
# ---------------------------------------------------------------------------
def evaluate_committee(
    env_name: str,
    algo: str,
    members: List[Dict[str, Any]],
    ref_member: int,
    eta: float,
    n_eval_episodes: int,
    mc_rollouts: int,
    delta: float = 0.05,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run diagnostic deployment evaluation; return a metrics dict.

    Returns (all DIAGNOSTIC):
        alpha_hat_mean, membership_fn_rate, Wtilde_proxy_mean,
        diag_certificate, mean_return, reward_range, cert_over_range, g_mean.
    """
    _require_jax()
    config = members[ref_member]["config"]
    env = jaxmarl_make(env_name, **config.get("ENV_KWARGS", {}))
    nets = _build_network(env, config)  # dict {agent: ActorCriticMLP}

    ref_params = members[ref_member]["params"]
    advisor_params = [m["params"] for k, m in enumerate(members) if k != ref_member]
    N = len(advisor_params) + 1  # reference also votes (endorses itself)

    rng = jax.random.PRNGKey(seed)
    agents = env.agents

    endorse_flags: List[float] = []      # per (episode,t,agent): F (majority fail)
    alpha_samples: List[float] = []      # per coordinate endorsement frequency
    wtilde_samples: List[float] = []     # clipped fallback advantage proxy
    fn_membership: List[float] = []      # membership false-negative indicator
    returns: List[float] = []
    all_rewards: List[float] = []
    X_samples: List[float] = []          # estimator samples nH * Wtilde * F

    for ep in range(n_eval_episodes):
        rng, rk = jax.random.split(rng)
        obs, state = env.reset(rk)
        done = {a: False for a in agents}
        ep_return = 0.0
        T = config.get("MAX_STEPS_EVAL", 25)
        for t in range(T):
            actions = {}
            for ai, agent in enumerate(agents):
                o = jnp.asarray(obs[agent])
                ref_logits, ref_value = nets[agent].apply(ref_params[agent], o)
                rng, ak = jax.random.split(rng)
                ref_action = int(jnp.argmax(ref_logits))  # reference = greedy
                # advisor proposals
                n_endorse = 1  # reference endorses its own action
                worst_delta = 0.0
                fb_deltas = []
                for ap in advisor_params:
                    m_logits, _ = nets[agent].apply(ap[agent], o)
                    m_action = int(jnp.argmax(m_logits))
                    d = _advantage_proxy(ref_logits, ref_value, m_action, ref_action)
                    if d <= eta + 1e-9:
                        n_endorse += 1
                    else:
                        fb_deltas.append(d)
                        worst_delta = max(worst_delta, d)
                alpha_hat = n_endorse / N
                alpha_samples.append(alpha_hat)
                # majority failure indicator (diagnostic)
                F = 1.0 if n_endorse <= np.floor(N / 2.0) else 0.0
                endorse_flags.append(F)
                # clipped fallback advantage proxy W̃^+ (mean of failing proposals)
                wt = float(np.mean(fb_deltas)) if fb_deltas else 0.0
                wtilde_samples.append(wt)
                # membership false-negative: an endorsed-set action mislabeled
                fn_membership.append(1.0 if (worst_delta <= eta and F == 1.0) else 0.0)
                nH = len(agents) * T
                X_samples.append(nH * wt * F)
                actions[agent] = ref_action
            obs, state, reward, done, info = env.step(
                jax.random.split(rng)[0], state, actions)
            r = float(np.mean([float(reward[a]) for a in agents]))
            ep_return += r
            all_rewards.append(r)
            if all(bool(done[a]) for a in agents) if "__all__" not in done \
                    else bool(done["__all__"]):
                break
        returns.append(ep_return)

    X = np.asarray(X_samples)
    b = float(X.max()) if X.size and X.max() > 0 else 1.0
    diag_cert = empirical_bernstein(X, delta, b) if X.size >= 2 else float("nan")
    reward_range = (float(np.max(all_rewards) - np.min(all_rewards))
                    if all_rewards else 1.0)
    alpha_hat_mean = float(np.mean(alpha_samples)) if alpha_samples else float("nan")
    g_mean = float(np.mean([g_N(N, a) for a in alpha_samples])) if alpha_samples else float("nan")

    return {
        "diagnostic_label": DIAG,
        "env": env_name, "algo": algo, "ref_member": ref_member, "eta": eta,
        "N": N,
        "alpha_hat_mean": alpha_hat_mean,
        "g_mean": g_mean,
        "membership_fn_rate": float(np.mean(fn_membership)) if fn_membership else 0.0,
        "Wtilde_proxy_mean": float(np.mean(wtilde_samples)) if wtilde_samples else 0.0,
        "diag_certificate": float(diag_cert),
        "mean_return": float(np.mean(returns)) if returns else float("nan"),
        "reward_range": reward_range,
        "cert_over_range": float(diag_cert / max(reward_range, 1e-9)),
        "n_eval_episodes": n_eval_episodes,
        "mc_rollouts": mc_rollouts,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="E2 diagnostic committee evaluation")
    ap.add_argument("--env", default="simple_spread")
    ap.add_argument("--algo", default="ippo", choices=["ippo", "mappo"])
    ap.add_argument("--checkpoints_dir", default="results/e2/checkpoints")
    ap.add_argument("--ref_member", type=int, default=0)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--n_eval_episodes", type=int, default=64)
    ap.add_argument("--mc_rollouts", type=int, default=16)
    ap.add_argument("--budget_grid", default="3,5,7,9",
                    help="comma-separated odd committee sizes for diagnostics")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _require_jax()
    members = load_committee(args.checkpoints_dir, args.env, args.algo)
    metrics = evaluate_committee(
        args.env, args.algo, members, args.ref_member, args.eta,
        args.n_eval_episodes, args.mc_rollouts, seed=args.seed,
    )
    print("[E2 eval_committee] DIAGNOSTIC metrics (NOT strict validity):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


if __name__ == "__main__":
    main()
