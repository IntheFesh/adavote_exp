"""
common/baselines.py — bound functions and OPE estimators for A1 (concentration
comparison) and A4 (OPE strawman baselines). All functions are sampling-agnostic
(callers feed X or trajectories), so the same draws can be reused across bounds.
"""
import numpy as np

# -------- A1: bound functions ----------------------------------------------
def hoeffding_bound(X, delta, b):
    """Hoeffding's inequality, X in [0, b]."""
    X = np.asarray(X, float); m = len(X)
    return float(X.mean() + b * np.sqrt(np.log(1.0/delta) / (2*m)))

def emp_bernstein_bound(X, delta, b):
    """Empirical-Bernstein (Maurer & Pontil 2009). Matches certificates.empirical_bernstein."""
    X = np.asarray(X, float); m = len(X)
    L = np.log(2.0/delta)
    return float(X.mean() + np.sqrt(2*X.var()*L/m) + 7*b*L/(3*(m-1)))

def naive_plugin_mean(X, delta, b):
    """Empirical mean only; NO concentration term. Expected to under-cover."""
    return float(np.asarray(X, float).mean())

def clipped_emp_bernstein_bound(X, delta, b, clip_quantile=0.99):
    """EB after clipping X at the empirical clip_quantile (variance reduction
    with a clipping bias)."""
    X = np.asarray(X, float)
    c = np.quantile(X, clip_quantile) if len(X) > 1 else b
    Xc = np.clip(X, 0, c)
    return emp_bernstein_bound(Xc, delta, c)

# -------- A4: OPE estimators -----------------------------------------------
def ope_pdis_upper(trajs, pi_ref_probs, pi_ctrl_probs, J_ctrl_hat, delta, R_max, H):
    """Per-decision importance sampling (Precup 2000). Returns (upper_bound,
    point_estimate) for J(pi_ref) - J(pi_ctrl)."""
    m = len(trajs)
    pdis_returns = []
    for ep, traj in enumerate(trajs):
        cum_rho = 1.0
        ep_pdis = 0.0
        for t, (_, _, r) in enumerate(traj):
            cum_rho *= pi_ref_probs[ep][t] / max(pi_ctrl_probs[ep][t], 1e-12)
            ep_pdis += cum_rho * r
        pdis_returns.append(ep_pdis)
    J_ref_hat = np.mean(pdis_returns)
    loss_hat = J_ref_hat - J_ctrl_hat
    rad = 2 * R_max * H * np.sqrt(np.log(2.0/delta) / (2*m))
    return float(loss_hat + rad), float(loss_hat)

def ope_dr_upper(trajs, pi_ref_probs, pi_ctrl_probs, V_hat_ref, Q_hat_ref,
                 J_ctrl_hat, delta, R_max, H):
    """Doubly-robust OPE (Jiang & Li 2016).
    V_hat_ref: dict (t,s) -> V at time t state s (every-visit MC).
    Q_hat_ref: dict (t,s,a) -> Q at time t state s joint action a.
    Returns (upper_bound, point_estimate) for J(pi_ref) - J(pi_ctrl)."""
    m = len(trajs)
    dr_returns = []
    for ep, traj in enumerate(trajs):
        s0 = traj[0][0]
        ep_dr = V_hat_ref.get((0, s0), 0.0)
        cum_rho = 1.0
        for t, (s, a, r) in enumerate(traj):
            rho_t = pi_ref_probs[ep][t] / max(pi_ctrl_probs[ep][t], 1e-12)
            s_next = traj[t+1][0] if t+1 < len(traj) else None
            V_next = V_hat_ref.get((t+1, s_next), 0.0) if s_next is not None else 0.0
            Q_st_a = Q_hat_ref.get((t, s, a), 0.0)
            ep_dr += cum_rho * (rho_t * (r + V_next) - Q_st_a)
            cum_rho *= rho_t
        dr_returns.append(ep_dr)
    J_ref_hat = np.mean(dr_returns)
    loss_hat = J_ref_hat - J_ctrl_hat
    rad = 2 * R_max * H * np.sqrt(np.log(2.0/delta) / (2*m))
    return float(loss_hat + rad), float(loss_hat)

def simulation_lemma_upper(epsilon_pi_max, R_max, H):
    """Kakade & Langford 2002 simulation lemma: deterministic upper bound
    J(ref) - J(ctrl) <= H * epsilon_pi_max * R_max where epsilon_pi_max is the
    worst-state TV distance ||pi_ref(.|s) - pi_ctrl(.|s)||_1 / 2."""
    return float(H * epsilon_pi_max * R_max)
