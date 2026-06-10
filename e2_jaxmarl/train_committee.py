"""
e2_jaxmarl/train_committee.py — Committee trainer for AdaVote E2 diagnostics.

DIAGNOSTIC / EMPIRICAL PROXY — NOT STRICT-VALIDITY EVIDENCE.
Strict validity (coverage guarantees) comes ONLY from E1 (e1_tabular/).
This module provides GPU training of IPPO / MAPPO committee members for
*diagnostic* purposes: the trained policies are used downstream in
eval_committee.py to estimate empirical endorsement frequencies and
clipped-fallback advantage proxies.

Implements IPPO and MAPPO in PureJaxRL / JaxMARL end-to-end JIT style:
  * A single jitted `make_train(config)` returns a `train_fn` that, given a
    PRNGKey, runs the full training loop on the GPU and returns the final
    carry (params + metrics).
  * Supports MPE `simple_spread` and `simple_reference` (primary),
    `overcooked_v0` (secondary), `smax` (optional, flag-gated, default OFF).
  * Network: actor-critic MLP (actor = diagonal Gaussian or categorical
    depending on action space; critic = shared-trunk baseline).
  * Optimization: GAE with lambda=0.95, clipped PPO surrogate, Adam.

Usage (on a GPU box):
    pip install -r requirements_e2.txt
    python -m e2_jaxmarl.train_committee \\
        --env simple_spread --algo ippo --n_members 7 \\
        --total_timesteps 10_000_000 --seed 0 --out_dir results/e2/checkpoints

Requirements: JAX with GPU support, Flax, Optax, Distrax, JaxMARL.
If JAX is not installed, a friendly error is printed (no crash).
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from typing import Any, Dict, NamedTuple, Tuple

# ---------------------------------------------------------------------------
# Graceful import guard — JAX / JaxMARL are optional at import time on CPU.
# ---------------------------------------------------------------------------
_JAX_AVAILABLE = False
_IMPORT_ERROR_MSG = ""

try:
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    import flax.serialization as flax_serial
    import optax
    import distrax
    from jaxmarl import make as jaxmarl_make
    from jaxmarl.wrappers.baselines import (
        LogWrapper,
        MPELogWrapper,
    )
    _JAX_AVAILABLE = True
except ImportError as _e:
    _IMPORT_ERROR_MSG = str(_e)


# ---------------------------------------------------------------------------
# Environment name mapping: short names (used throughout the codebase / CLI)
# -> actual jaxmarl registered names in the installed jaxmarl version.
# ---------------------------------------------------------------------------
_JAXMARL_ENV_MAP = {
    "simple_spread": "MPE_simple_spread_v3",
    "simple_reference": "MPE_simple_reference_v3",
    "overcooked_v0": "overcooked",
    "smax": "SMAX",
}


def _resolve_env_name(name):
    """Translate a short env name to the jaxmarl registered name.
    If `name` is already a registered name, it is returned unchanged."""
    return _JAXMARL_ENV_MAP.get(name, name)




def _require_jax() -> None:
    """Raise a friendly RuntimeError if JAX / JaxMARL are not installed."""
    if not _JAX_AVAILABLE:
        raise RuntimeError(
            "\n"
            "=================================================================\n"
            "E2 (GPU / JaxMARL) dependencies are not installed.\n"
            f"  Import error: {_IMPORT_ERROR_MSG}\n"
            "\n"
            "To install (on a GPU box):\n"
            "    pip install -r requirements_e2.txt\n"
            "\n"
            "To run on CPU (slow, for smoke-testing only):\n"
            "    JAX_PLATFORMS=cpu python -m e2_jaxmarl.train_committee ...\n"
            "=================================================================\n"
        )


# ---------------------------------------------------------------------------
# Supported environments
# ---------------------------------------------------------------------------
MPE_ENVS = {"simple_spread", "simple_reference"}
OVERCOOKED_ENVS = {"overcooked_v0"}
SMAX_ENVS = {"smax"}

ALL_PRIMARY = MPE_ENVS | OVERCOOKED_ENVS

# ---------------------------------------------------------------------------
# Default hyper-parameter configs
# ---------------------------------------------------------------------------

def default_config(env_name: str, algo: str) -> Dict[str, Any]:
    """Return a sensible default config dict for the given env + algo."""
    cfg: Dict[str, Any] = {
        # Environment
        "ENV_NAME": env_name,
        "ALGO": algo,
        # Network
        "ACTOR_HIDDEN": [256, 256],
        "CRITIC_HIDDEN": [256, 256],
        # PPO
        "TOTAL_TIMESTEPS": 10_000_000,
        "NUM_ENVS": 128,
        "NUM_STEPS": 128,           # rollout horizon per update
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 8,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        # Optimiser
        "LR": 2.5e-4,
        "ANNEAL_LR": True,
        # MAPPO-specific
        "CENTRAL_CRITIC": algo == "mappo",
        # Logging
        "LOG_INTERVAL": 10,
    }
    # Env-specific overrides
    if env_name in OVERCOOKED_ENVS:
        cfg["NUM_ENVS"] = 64
        cfg["NUM_STEPS"] = 256
    elif env_name in SMAX_ENVS:
        cfg["NUM_ENVS"] = 64
        cfg["NUM_STEPS"] = 128
        cfg["TOTAL_TIMESTEPS"] = 20_000_000
    return cfg


# ---------------------------------------------------------------------------
# Neural networks
# ---------------------------------------------------------------------------

if _JAX_AVAILABLE:

    class ActorMLP(nn.Module):
        """Categorical actor for discrete action spaces."""
        hidden_sizes: Tuple[int, ...]
        action_dim: int

        @nn.compact
        def __call__(self, obs):
            x = obs
            for h in self.hidden_sizes:
                x = nn.Dense(h)(x)
                x = nn.tanh(x)
            logits = nn.Dense(self.action_dim)(x)
            return logits

    class CriticMLP(nn.Module):
        """Scalar value critic."""
        hidden_sizes: Tuple[int, ...]

        @nn.compact
        def __call__(self, obs):
            x = obs
            for h in self.hidden_sizes:
                x = nn.Dense(h)(x)
                x = nn.tanh(x)
            return nn.Dense(1)(x).squeeze(-1)

    class ActorCriticMLP(nn.Module):
        """Combined actor-critic (shared first layer optional)."""
        actor_hidden: Tuple[int, ...]
        critic_hidden: Tuple[int, ...]
        action_dim: int

        def setup(self):
            self.actor = ActorMLP(
                hidden_sizes=self.actor_hidden, action_dim=self.action_dim
            )
            self.critic = CriticMLP(hidden_sizes=self.critic_hidden)

        def __call__(self, obs):
            logits = self.actor(obs)
            value = self.critic(obs)
            return logits, value

        def get_action_and_value(self, obs, key):
            logits, value = self(obs)
            pi = distrax.Categorical(logits=logits)
            action = pi.sample(seed=key)
            log_prob = pi.log_prob(action)
            return action, log_prob, value, pi

        def get_logprob_and_value(self, obs, action):
            logits, value = self(obs)
            pi = distrax.Categorical(logits=logits)
            log_prob = pi.log_prob(action)
            entropy = pi.entropy()
            return log_prob, value, entropy

    def _build_network(env, config):
        """Build a dict {agent: ActorCriticMLP} matching the trained IPPO actor-
        critic architecture, so eval_committee can re-instantiate the network
        and apply saved per-agent params.  Returns the per-agent networks; each
        `net.apply(params[agent], obs) -> (logits, value)`.
        """
        actor_hidden = tuple(config["ACTOR_HIDDEN"])
        critic_hidden = tuple(config["CRITIC_HIDDEN"])
        nets = {}
        for a in env.agents:
            nets[a] = ActorCriticMLP(
                actor_hidden=actor_hidden,
                critic_hidden=critic_hidden,
                action_dim=env.action_space(a).n,
            )
        return nets


# ---------------------------------------------------------------------------
# Transition storage
# ---------------------------------------------------------------------------

if _JAX_AVAILABLE:

    class Transition(NamedTuple):
        obs: Any
        action: Any
        reward: Any
        done: Any
        log_prob: Any
        value: Any
        info: Any


# ---------------------------------------------------------------------------
# GAE computation
# ---------------------------------------------------------------------------

def _compute_gae(traj_batch, last_val, gamma, gae_lambda):
    """Vectorised GAE over a trajectory batch (JaxMARL convention)."""
    import jax
    import jax.numpy as jnp

    def _gae_step(carry, transition):
        gae, next_value = carry
        done, value, reward = transition.done, transition.value, transition.reward
        delta = reward + gamma * next_value * (1.0 - done) - value
        gae = delta + gamma * gae_lambda * (1.0 - done) * gae
        return (gae, value), (gae, gae + value)

    _, (advantages, targets) = jax.lax.scan(
        _gae_step,
        (jnp.zeros_like(last_val), last_val),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return advantages, targets


# ---------------------------------------------------------------------------
# IPPO make_train
# ---------------------------------------------------------------------------

def make_ippo_train(config: Dict[str, Any]):
    """Return a jitted IPPO train function.

    Usage:
        train_fn = jax.jit(make_ippo_train(config))
        final_carry, metrics = train_fn(jax.random.PRNGKey(seed))

    The returned `final_carry` contains:
        ("params", params_dict) — actor-critic params per agent.
        ("metrics", last_metrics_dict)
    """
    _require_jax()
    import jax
    import jax.numpy as jnp
    import optax
    import distrax

    # Build env
    env_name = config["ENV_NAME"]
    env = jaxmarl_make(_resolve_env_name(env_name))
    try:
        env = MPELogWrapper(env)
    except Exception:
        env = LogWrapper(env)

    num_agents = len(env.agents)
    obs_sizes = {a: env.observation_space(a).shape[0] for a in env.agents}
    act_sizes = {a: env.action_space(a).n for a in env.agents}

    total_timesteps = config["TOTAL_TIMESTEPS"]
    num_envs = config["NUM_ENVS"]
    num_steps = config["NUM_STEPS"]
    num_updates = total_timesteps // (num_envs * num_steps)
    num_minibatches = config["NUM_MINIBATCHES"]
    update_epochs = config["UPDATE_EPOCHS"]
    gamma = config["GAMMA"]
    gae_lambda = config["GAE_LAMBDA"]
    clip_eps = config["CLIP_EPS"]
    ent_coef = config["ENT_COEF"]
    vf_coef = config["VF_COEF"]
    max_grad_norm = config["MAX_GRAD_NORM"]
    lr = config["LR"]
    anneal_lr = config["ANNEAL_LR"]
    actor_hidden = tuple(config["ACTOR_HIDDEN"])
    critic_hidden = tuple(config["CRITIC_HIDDEN"])

    def _make_optimizer():
        if anneal_lr:
            schedule = optax.linear_schedule(lr, 0.0, num_updates * update_epochs)
        else:
            schedule = lr
        return optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(schedule, eps=1e-5),
        )

    def train(rng):
        # Initialise per-agent networks + optimisers
        agent_keys = env.agents
        rng, *init_keys = jax.random.split(rng, num_agents + 1)

        networks = {}
        params_dict = {}
        opt_states_dict = {}

        for agent, ikey in zip(agent_keys, init_keys):
            obs_dim = obs_sizes[agent]
            act_dim = act_sizes[agent]
            net = ActorCriticMLP(
                actor_hidden=actor_hidden,
                critic_hidden=critic_hidden,
                action_dim=act_dim,
            )
            dummy_obs = jnp.zeros((obs_dim,))
            params = net.init(ikey, dummy_obs)
            networks[agent] = net
            params_dict[agent] = params
            opt = _make_optimizer()
            opt_states_dict[agent] = opt.init(params)

        # Environment reset
        rng, reset_key = jax.random.split(rng)
        reset_keys = jax.random.split(reset_key, num_envs)
        obs, env_state = jax.vmap(env.reset)(reset_keys)

        def _env_step(carry, _):
            params_d, opt_d, obs_d, env_state_d, rng_step = carry
            rng_step, act_key, step_key = jax.random.split(rng_step, 3)

            # Collect actions from all agents
            actions = {}
            log_probs = {}
            values = {}
            for agent in agent_keys:
                a_obs = obs_d[agent]  # (num_envs, obs_dim)
                net = networks[agent]
                p_agent = params_d[agent]
                act_keys = jax.random.split(act_key, num_envs)

                def _act(o, k, _net=net, _p=p_agent):
                    logits, value = _net.apply(_p, o)
                    pi = distrax.Categorical(logits=logits)
                    a = pi.sample(seed=k)
                    return a, pi.log_prob(a), value

                action, log_prob, value = jax.vmap(_act)(a_obs, act_keys)
                actions[agent] = action
                log_probs[agent] = log_prob
                values[agent] = value

            # Step environment
            step_keys = jax.random.split(step_key, num_envs)
            obs_next, env_state_next, rewards, dones, infos = jax.vmap(
                env.step
            )(step_keys, env_state_d, actions)

            transitions = {}
            for agent in agent_keys:
                transitions[agent] = Transition(
                    obs=obs_d[agent],
                    action=actions[agent],
                    reward=rewards[agent],
                    done=dones[agent],
                    log_prob=log_probs[agent],
                    value=values[agent],
                    info=infos,
                )
            return (
                params_d, opt_d, obs_next, env_state_next, rng_step
            ), transitions

        def _update_epoch(carry, _):
            params_d, opt_d, traj, last_obs, rng_upd = carry
            rng_upd, perm_key = jax.random.split(rng_upd)

            loss_dict = {}
            new_params_d = {}
            new_opt_d = {}

            for agent in agent_keys:
                net = networks[agent]
                traj_a = traj[agent]
                last_obs_a = last_obs[agent]

                # Compute last value for GAE bootstrap
                last_val = jax.vmap(lambda o: net.apply(params_d[agent], o)[1])(
                    last_obs_a
                )
                advantages, targets = _compute_gae(
                    traj_a, last_val, gamma, gae_lambda
                )
                # Flatten (num_steps, num_envs) -> (batch,)
                batch_size = num_steps * num_envs
                obs_b = traj_a.obs.reshape((batch_size, -1))
                act_b = traj_a.action.reshape((batch_size,))
                logp_b = traj_a.log_prob.reshape((batch_size,))
                adv_b = advantages.reshape((batch_size,))
                tgt_b = targets.reshape((batch_size,))

                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                # Minibatch permutation
                perm = jax.random.permutation(perm_key, batch_size)
                mb_size = batch_size // num_minibatches

                def _ppo_loss(params, obs_mb, act_mb, logp_old_mb, adv_mb, tgt_mb):
                    def _single(o, a, _net=net, _p=params):
                        logits, val = _net.apply(_p, o)
                        pi = distrax.Categorical(logits=logits)
                        return pi.log_prob(a), val, pi.entropy()

                    new_logp, new_val, entropy = jax.vmap(_single)(obs_mb, act_mb)

                    ratio = jnp.exp(new_logp - logp_old_mb)
                    pg_loss1 = -adv_mb * ratio
                    pg_loss2 = -adv_mb * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
                    vf_loss = 0.5 * jnp.square(new_val - tgt_mb).mean()
                    ent_loss = -entropy.mean()
                    total = pg_loss + vf_coef * vf_loss + ent_coef * ent_loss
                    return total, {
                        "pg_loss": pg_loss,
                        "vf_loss": vf_loss,
                        "entropy": -ent_loss,
                    }

                params_a = params_d[agent]
                opt_a = opt_d[agent]
                opt_a_inst = _make_optimizer()

                def _mb_update(carry_mb, mb_idx):
                    p, o_state = carry_mb
                    start = mb_idx * mb_size
                    idx = jax.lax.dynamic_slice(perm, (start,), (mb_size,))
                    obs_mb = obs_b[idx]
                    act_mb = act_b[idx]
                    logp_mb = logp_b[idx]
                    adv_mb = adv_b[idx]
                    tgt_mb = tgt_b[idx]
                    (loss_val, aux), grads = jax.value_and_grad(
                        _ppo_loss, has_aux=True
                    )(p, obs_mb, act_mb, logp_mb, adv_mb, tgt_mb)
                    updates, new_o = opt_a_inst.update(grads, o_state, p)
                    new_p = optax.apply_updates(p, updates)
                    return (new_p, new_o), aux

                (params_a, opt_a), aux = jax.lax.scan(
                    _mb_update,
                    (params_a, opt_a),
                    jnp.arange(num_minibatches),
                )
                new_params_d[agent] = params_a
                new_opt_d[agent] = opt_a
                loss_dict[agent] = aux

            return (new_params_d, new_opt_d, traj, last_obs, rng_upd), loss_dict

        def _update_step(carry, _):
            params_d, opt_d, obs_d, env_state_d, rng_u = carry
            # Collect rollout
            rng_u, rollout_key = jax.random.split(rng_u)
            (params_d, opt_d, obs_next, env_state_next, rollout_key), traj = (
                jax.lax.scan(
                    _env_step,
                    (params_d, opt_d, obs_d, env_state_d, rollout_key),
                    None,
                    length=num_steps,
                )
            )
            # PPO update over epochs
            rng_u, epoch_key = jax.random.split(rng_u)
            (params_d, opt_d, _, _, _), loss_dict = jax.lax.scan(
                _update_epoch,
                (params_d, opt_d, traj, obs_next, epoch_key),
                None,
                length=update_epochs,
            )
            # Aggregate metrics (mean over epochs + envs)
            metrics = {
                agent: {
                    k: v.mean() for k, v in loss_dict[agent].items()
                }
                for agent in agent_keys
            }
            return (params_d, opt_d, obs_next, env_state_next, rng_u), metrics

        # Full training loop via scan
        (final_params, final_opt, _, _, _), all_metrics = jax.lax.scan(
            _update_step,
            (params_dict, opt_states_dict, obs, env_state, rng),
            None,
            length=num_updates,
        )
        return final_params, all_metrics

    return train


# ---------------------------------------------------------------------------
# MAPPO make_train (central critic sees concatenated obs)
# ---------------------------------------------------------------------------

def make_mappo_train(config: Dict[str, Any]):
    """Return a jitted MAPPO train function.

    MAPPO differs from IPPO only in the critic: it receives the concatenation
    of all agents' observations (global state proxy) instead of the local obs.
    The actor remains agent-local (decentralised execution).
    """
    _require_jax()
    import jax
    import jax.numpy as jnp
    import optax
    import distrax

    env_name = config["ENV_NAME"]
    env = jaxmarl_make(_resolve_env_name(env_name))
    try:
        env = MPELogWrapper(env)
    except Exception:
        env = LogWrapper(env)

    num_agents = len(env.agents)
    agent_keys = env.agents
    obs_sizes = {a: env.observation_space(a).shape[0] for a in agent_keys}
    act_sizes = {a: env.action_space(a).n for a in agent_keys}
    global_obs_dim = sum(obs_sizes[a] for a in agent_keys)

    total_timesteps = config["TOTAL_TIMESTEPS"]
    num_envs = config["NUM_ENVS"]
    num_steps = config["NUM_STEPS"]
    num_updates = total_timesteps // (num_envs * num_steps)
    num_minibatches = config["NUM_MINIBATCHES"]
    update_epochs = config["UPDATE_EPOCHS"]
    gamma = config["GAMMA"]
    gae_lambda = config["GAE_LAMBDA"]
    clip_eps = config["CLIP_EPS"]
    ent_coef = config["ENT_COEF"]
    vf_coef = config["VF_COEF"]
    max_grad_norm = config["MAX_GRAD_NORM"]
    lr = config["LR"]
    anneal_lr = config["ANNEAL_LR"]
    actor_hidden = tuple(config["ACTOR_HIDDEN"])
    critic_hidden = tuple(config["CRITIC_HIDDEN"])

    def _make_optimizer():
        if anneal_lr:
            schedule = optax.linear_schedule(lr, 0.0, num_updates * update_epochs)
        else:
            schedule = lr
        return optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(schedule, eps=1e-5),
        )

    def train(rng):
        rng, *init_keys = jax.random.split(rng, num_agents * 2 + 1)
        actor_keys = init_keys[:num_agents]
        critic_keys = init_keys[num_agents: 2 * num_agents]

        actors = {}
        critics = {}
        actor_params = {}
        critic_params = {}
        actor_opts = {}
        critic_opts = {}

        for agent, akey, ckey in zip(agent_keys, actor_keys, critic_keys):
            obs_dim = obs_sizes[agent]
            act_dim = act_sizes[agent]
            actor = ActorMLP(hidden_sizes=actor_hidden, action_dim=act_dim)
            critic = CriticMLP(hidden_sizes=critic_hidden)
            ap = actor.init(akey, jnp.zeros(obs_dim))
            cp = critic.init(ckey, jnp.zeros(global_obs_dim))
            actors[agent] = actor
            critics[agent] = critic
            actor_params[agent] = ap
            critic_params[agent] = cp
            ao = _make_optimizer()
            co = _make_optimizer()
            actor_opts[agent] = ao.init(ap)
            critic_opts[agent] = co.init(cp)

        # Reset
        rng, reset_key = jax.random.split(rng)
        reset_keys = jax.random.split(reset_key, num_envs)
        obs, env_state = jax.vmap(env.reset)(reset_keys)

        def _global_obs(obs_d):
            return jnp.concatenate(
                [obs_d[a] for a in agent_keys], axis=-1
            )  # (num_envs, global_obs_dim)

        def _env_step(carry, _):
            ap, cp, ao, co, obs_d, env_state_d, rng_s = carry
            rng_s, act_key, step_key = jax.random.split(rng_s, 3)
            glob_obs = _global_obs(obs_d)

            actions = {}
            log_probs = {}
            values = {}

            for agent in agent_keys:
                a_obs = obs_d[agent]
                actor = actors[agent]
                critic = critics[agent]
                act_keys = jax.random.split(act_key, num_envs)

                def _act(o, k):
                    logits = actor.apply(ap[agent], o)
                    pi = distrax.Categorical(logits=logits)
                    a = pi.sample(seed=k)
                    return a, pi.log_prob(a)

                action, log_prob = jax.vmap(_act)(a_obs, act_keys)
                value = jax.vmap(lambda g: critic.apply(cp[agent], g))(glob_obs)

                actions[agent] = action
                log_probs[agent] = log_prob
                values[agent] = value

            step_keys = jax.random.split(step_key, num_envs)
            obs_next, env_state_next, rewards, dones, infos = jax.vmap(
                env.step
            )(step_keys, env_state_d, actions)

            transitions = {}
            for agent in agent_keys:
                transitions[agent] = Transition(
                    obs=obs_d[agent],
                    action=actions[agent],
                    reward=rewards[agent],
                    done=dones[agent],
                    log_prob=log_probs[agent],
                    value=values[agent],
                    info=infos,
                )
            # Store global obs for central critic updates
            transitions["__global_obs__"] = glob_obs

            return (ap, cp, ao, co, obs_next, env_state_next, rng_s), transitions

        def _update_epoch(carry, _):
            ap, cp, ao, co, traj, last_glob_obs, rng_e = carry
            rng_e, perm_key = jax.random.split(rng_e)

            new_ap = {}
            new_cp = {}
            new_ao = {}
            new_co = {}
            loss_dict = {}

            glob_traj = traj["__global_obs__"]  # (num_steps, num_envs, global_obs_dim)

            for agent in agent_keys:
                actor = actors[agent]
                critic = critics[agent]
                traj_a = traj[agent]

                # Bootstrap last value using central critic
                last_val = jax.vmap(lambda g: critic.apply(cp[agent], g))(
                    last_glob_obs
                )
                advantages, targets = _compute_gae(
                    traj_a, last_val, gamma, gae_lambda
                )

                batch_size = num_steps * num_envs
                obs_b = traj_a.obs.reshape((batch_size, -1))
                act_b = traj_a.action.reshape((batch_size,))
                logp_b = traj_a.log_prob.reshape((batch_size,))
                adv_b = advantages.reshape((batch_size,))
                tgt_b = targets.reshape((batch_size,))
                glob_b = glob_traj.reshape((batch_size, -1))

                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                perm = jax.random.permutation(perm_key, batch_size)
                mb_size = batch_size // num_minibatches

                opt_a_inst = _make_optimizer()
                opt_c_inst = _make_optimizer()

                def _mb_update(carry_mb, mb_idx):
                    pa, pc, oa, oc = carry_mb
                    start = mb_idx * mb_size
                    idx = jax.lax.dynamic_slice(perm, (start,), (mb_size,))
                    obs_mb = obs_b[idx]
                    act_mb = act_b[idx]
                    logp_mb = logp_b[idx]
                    adv_mb = adv_b[idx]
                    tgt_mb = tgt_b[idx]
                    glob_mb = glob_b[idx]

                    def actor_loss(pa_inner):
                        def _single(o, a):
                            logits = actor.apply(pa_inner, o)
                            pi = distrax.Categorical(logits=logits)
                            return pi.log_prob(a), pi.entropy()
                        lp, ent = jax.vmap(_single)(obs_mb, act_mb)
                        ratio = jnp.exp(lp - logp_mb)
                        pg1 = -adv_mb * ratio
                        pg2 = -adv_mb * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
                        pg = jnp.maximum(pg1, pg2).mean()
                        e_loss = -ent.mean()
                        return pg + ent_coef * e_loss, {"pg": pg, "ent": -e_loss}

                    def critic_loss(pc_inner):
                        vals = jax.vmap(lambda g: critic.apply(pc_inner, g))(glob_mb)
                        return 0.5 * jnp.square(vals - tgt_mb).mean()

                    (al, a_aux), ag = jax.value_and_grad(actor_loss, has_aux=True)(pa)
                    cl, cg = jax.value_and_grad(critic_loss)(pc)

                    a_updates, new_oa = opt_a_inst.update(ag, oa, pa)
                    c_updates, new_oc = opt_c_inst.update(cg, oc, pc)
                    new_pa = optax.apply_updates(pa, a_updates)
                    new_pc = optax.apply_updates(pc, c_updates)
                    return (new_pa, new_pc, new_oa, new_oc), {
                        "actor_loss": al,
                        "critic_loss": cl,
                        "entropy": a_aux["ent"],
                    }

                (ap_a, cp_a, ao_a, co_a), aux = jax.lax.scan(
                    _mb_update,
                    (ap[agent], cp[agent], ao[agent], co[agent]),
                    jnp.arange(num_minibatches),
                )
                new_ap[agent] = ap_a
                new_cp[agent] = cp_a
                new_ao[agent] = ao_a
                new_co[agent] = co_a
                loss_dict[agent] = aux

            return (new_ap, new_cp, new_ao, new_co, traj, last_glob_obs, rng_e), loss_dict

        def _update_step(carry, _):
            ap, cp, ao, co, obs_d, env_state_d, rng_u = carry
            rng_u, rk = jax.random.split(rng_u)
            (ap, cp, ao, co, obs_next, env_state_next, rk), traj = jax.lax.scan(
                _env_step,
                (ap, cp, ao, co, obs_d, env_state_d, rk),
                None,
                length=num_steps,
            )
            glob_last = _global_obs(obs_next)
            rng_u, ek = jax.random.split(rng_u)
            (ap, cp, ao, co, _, _, _), loss_dict = jax.lax.scan(
                _update_epoch,
                (ap, cp, ao, co, traj, glob_last, ek),
                None,
                length=update_epochs,
            )
            metrics = {
                agent: {k: v.mean() for k, v in loss_dict[agent].items()}
                for agent in agent_keys
            }
            return (ap, cp, ao, co, obs_next, env_state_next, rng_u), metrics

        (final_ap, final_cp, _, _, _, _, _), all_metrics = jax.lax.scan(
            _update_step,
            (actor_params, critic_params, actor_opts, critic_opts, obs, env_state, rng),
            None,
            length=num_updates,
        )
        return {"actor": final_ap, "critic": final_cp}, all_metrics

    return train


# ---------------------------------------------------------------------------
# train_committee: main entry point
# ---------------------------------------------------------------------------

def train_committee(
    env_name: str,
    algo: str,
    n_members: int,
    seed: int,
    total_timesteps: int,
    out_dir: str,
    enable_smax: bool = False,
) -> None:
    """Train `n_members` committee checkpoints (one per seed) and save them.

    DIAGNOSTIC use only — NOT strict-validity evidence.

    Each member is trained with a distinct PRNGKey derived from `seed` by
    shifting: member k uses seed + k.  Parameters are saved as pickle files
    (compatible with flax.serialization for JAX pytrees) in `out_dir`.

    Parameters
    ----------
    env_name : str
        JaxMARL environment name, e.g. "simple_spread".
    algo : {"ippo", "mappo"}
        Multi-agent RL algorithm.
    n_members : int
        Number of committee members to train.
    seed : int
        Base random seed; member k gets seed+k.
    total_timesteps : int
        Total env steps per member.
    out_dir : str
        Directory to write checkpoint files.
    enable_smax : bool
        If False (default) and env_name is in SMAX_ENVS, raises ValueError.
    """
    _require_jax()
    import jax

    if env_name in SMAX_ENVS and not enable_smax:
        raise ValueError(
            f"SMAX env '{env_name}' is disabled by default.  "
            "Pass --enable_smax to opt in."
        )

    os.makedirs(out_dir, exist_ok=True)

    config = default_config(env_name, algo)
    config["TOTAL_TIMESTEPS"] = total_timesteps

    if algo == "ippo":
        make_train = make_ippo_train
    elif algo == "mappo":
        make_train = make_mappo_train
    else:
        raise ValueError(f"Unknown algo: {algo!r}.  Choose 'ippo' or 'mappo'.")

    print(
        f"[E2 train_committee] env={env_name}  algo={algo}  "
        f"n_members={n_members}  timesteps={total_timesteps:,}  base_seed={seed}"
    )
    print("[E2 DIAGNOSTIC] This run produces diagnostic data only. "
          "Strict validity comes from E1.")

    train_fn = jax.jit(make_train(config))

    for k in range(n_members):
        member_seed = seed + k
        rng = jax.random.PRNGKey(member_seed)
        t0 = time.time()
        print(f"  Training member {k} (seed={member_seed}) ...", flush=True)
        params, metrics = train_fn(rng)
        elapsed = time.time() - t0

        # Save checkpoint
        ckpt_path = os.path.join(
            out_dir, f"{env_name}__{algo}__member{k:02d}__seed{member_seed}.pkl"
        )
        # Convert JAX arrays to numpy for serialisation portability
        params_np = jax.device_get(params)
        with open(ckpt_path, "wb") as f:
            pickle.dump(
                {
                    "params": params_np,
                    "config": config,
                    "env_name": env_name,
                    "algo": algo,
                    "member_idx": k,
                    "seed": member_seed,
                    "total_timesteps": total_timesteps,
                    "diagnostic_label": (
                        "DIAGNOSTIC / EMPIRICAL PROXY — NOT STRICT-VALIDITY EVIDENCE"
                    ),
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        # Log final metrics (mean over agents + last update window)
        agent_metrics = {}
        for agent, m in metrics.items():
            agent_metrics[agent] = {
                k2: float(v[-1].mean()) for k2, v in m.items()
            }
        print(
            f"  [member {k}] done in {elapsed:.1f}s  "
            f"  saved -> {ckpt_path}"
        )
        for agent, am in agent_metrics.items():
            print(f"    {agent}: " + "  ".join(f"{k2}={v:.4f}" for k2, v in am.items()))

    print(f"[E2 train_committee] All {n_members} checkpoints saved to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "AdaVote E2 — DIAGNOSTIC committee trainer (IPPO / MAPPO / JaxMARL). "
            "DIAGNOSTIC / EMPIRICAL PROXY — NOT STRICT-VALIDITY EVIDENCE. "
            "Requires JAX with GPU support."
        )
    )
    p.add_argument(
        "--env",
        default="simple_spread",
        choices=sorted(MPE_ENVS | OVERCOOKED_ENVS | SMAX_ENVS),
        help="JaxMARL environment name.",
    )
    p.add_argument(
        "--algo",
        default="ippo",
        choices=["ippo", "mappo"],
        help="Multi-agent RL algorithm.",
    )
    p.add_argument(
        "--n_members",
        type=int,
        default=7,
        help="Number of committee members to train (default: 7).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed (member k uses seed+k).",
    )
    p.add_argument(
        "--total_timesteps",
        type=int,
        default=10_000_000,
        help="Total environment steps per member.",
    )
    p.add_argument(
        "--out_dir",
        default="results/e2/checkpoints",
        help="Output directory for checkpoint files.",
    )
    p.add_argument(
        "--enable_smax",
        action="store_true",
        default=False,
        help="Enable SMAX environments (optional, disabled by default).",
    )
    return p


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if not _JAX_AVAILABLE:
        print(
            "\n[E2 train_committee] JAX / JaxMARL is NOT installed.\n"
            f"  Error: {_IMPORT_ERROR_MSG}\n"
            "\n"
            "  To train on a GPU box:\n"
            "    pip install -r requirements_e2.txt\n"
            "    python -m e2_jaxmarl.train_committee \\\n"
            f"        --env {args.env} --algo {args.algo} "
            f"--n_members {args.n_members} \\\n"
            f"        --total_timesteps {args.total_timesteps} "
            f"--seed {args.seed} --out_dir {args.out_dir}\n"
            "\n"
            "  CPU smoke-test (slow):\n"
            "    JAX_PLATFORMS=cpu python -m e2_jaxmarl.train_committee ...\n"
        )
        sys.exit(0)

    train_committee(
        env_name=args.env,
        algo=args.algo,
        n_members=args.n_members,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        out_dir=args.out_dir,
        enable_smax=args.enable_smax,
    )


if __name__ == "__main__":
    main()
