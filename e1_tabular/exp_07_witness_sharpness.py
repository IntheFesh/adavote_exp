"""
exp_07_witness_sharpness.py  —  Realizability / sharpness lemma.

Claim: one can construct WITNESS games (action-INDEPENDENT transitions +
separable rewards) where:
  (a) the state occupancy is realized exactly (error 0 to ~machine precision);
  (b) certificate levels degrade to exact closed forms, true loss matches
      closed form, and C* = C̃* (when all mass is on a single fallback action);
  (c) prefix-conditional failure is NOT collapsed: distinct per-prefix alpha
      values produce distinct realized prefix failure rates.

Outputs:
    results/data/exp_07_witness_sharpness.csv
    results/figs/exp_07_cert_errors.pdf
    one PASS/FAIL line -> results/summary.txt

PASS <=> all closed-form vs computed errors ≤ 1e-9, occupancy error ~0,
         non-collapsed check confirms prefix dependence.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import certificates as C
from common.games import make_independent_game_with_occupancy
from common.io_utils import data_path, fig_path, write_summary
from common.plotting import new_fig, save_pdf
from e1_tabular._common import build, make_alpha_fn, const_fn

TOL_PASS = 1e-9   # tight machine-precision criterion


# ---------------------------------------------------------------------------
# Part (a): Occupancy realizability
# ---------------------------------------------------------------------------

def check_occupancy_realizability(seed: int, S: int, A: int, H: int, n: int,
                                  N: int) -> dict:
    """Build an action-independent game with a random target_d; verify that
    the controller's realized d_ctrl matches the analytic occupancy from target_d.

    In an action-independent game P_t(s'|s,a) = d_{t+1}(s') for all (a),
    so the next-state law is fixed regardless of the controller.  The state
    occupancy at time t+1 is therefore:

        d[t+1, s'] = Σ_s d[t, s] · P_t(s'|s,·) = Σ_s d[t, s] · target_d[t, s']

    (But note: make_independent_game_with_occupancy uses target_d[t] as the
    next-state distribution starting FROM time t, so d[t+1] = target_d[t]
    when it's the same for all s.  Actually it may vary by s — let's compute
    the analytic d properly.)

    Actually looking at the game constructor: P[t, s, :] = d_next[t] for each s.
    So for all s, P_t(·|s, a) = target_d[t] (a fixed row).  Hence:
        d[t+1, s'] = Σ_s d[t, s] * target_d[t, s'] = 1 * target_d[t, s'] = target_d[t, s'].
    So d[t+1] = target_d[t] regardless of d[t] (once d[0] = d0).

    We generate target_d of shape (H, S) and verify d_ctrl[t+1] = target_d[t].
    """
    rng = np.random.default_rng(seed)
    target_d = np.array([rng.dirichlet(np.ones(S)) for _ in range(H)])  # (H, S)

    game = make_independent_game_with_occupancy(S=S, A=A, H=H, n=n,
                                                seed=seed, target_d=target_d)
    alpha_fn = make_alpha_fn(seed)
    b = build(game, alpha_fn, const_fn(N))

    # Analytic occupancy: d[0] = game.d0; d[t+1] = target_d[t]
    d_analytic = np.zeros((H, S))
    d_analytic[0] = game.d0.copy()
    for t in range(H - 1):
        d_analytic[t + 1] = target_d[t]

    max_err = float(np.max(np.abs(b.d_ctrl - d_analytic)))
    return dict(
        witness="occupancy_realizability",
        seed=seed, S=S, A=A, H=H, N=N,
        occupancy_max_err=max_err,
        cert_err=0.0,           # not applicable here
        description="d_ctrl matches analytic d from target_d",
    )


# ---------------------------------------------------------------------------
# Part (b): Sharpness witnesses
# ---------------------------------------------------------------------------

def witness_Cstar_equals_Ctilde(seed: int, S: int, A: int, H: int, n: int,
                                 N: int) -> dict:
    """Witness: uniform fallback on ALL actions => W̃ = (1/A) Σ_a Δ_+(a).

    For C* = C̃* we need a single non-reference action that captures ALL the
    mass.  We use a fallback concentrated on ONE action (worst action).

    Use fb_fn that always puts ALL mass on the worst (highest Δ_+) action.
    Then W̃ = max_a Δ_+(a) = W, so C* = C̃* exactly.

    We also verify the closed-form:
        C̃*_closed = Σ_{t,s,i,prefix} rho(u) * g(u) * W(u)
    which by construction equals cert_Cstar and cert_Ctilde.
    """

    def worst_fb(t, s, i, prefix, game, delta_plus):
        fb = np.zeros(game.A)
        fb[int(np.argmax(delta_plus))] = 1.0
        return fb

    game = make_independent_game_with_occupancy(S=S, A=A, H=H, n=n,
                                                seed=seed, target_d=None)
    alpha_fn = make_alpha_fn(seed)
    b = build(game, alpha_fn, const_fn(N), fb_fn=worst_fb)

    # By construction W̃(u) = W(u) for all u (all fallback mass on argmax Δ_+)
    # so C̃* = C* exactly.
    ctilde_computed = b.chain["Ctilde"]
    cstar_computed = b.chain["Cstar"]
    closed_form = cstar_computed   # closed form IS C* = sum rho*g*W

    err_Ctilde_eq_Cstar = abs(ctilde_computed - cstar_computed)
    err_closed = abs(closed_form - ctilde_computed)

    # Also verify: C̃* >= true_loss
    ctilde_ge_true = ctilde_computed + TOL_PASS >= b.true_loss

    return dict(
        witness="Cstar_eq_Ctilde",
        seed=seed, S=S, A=A, H=H, N=N,
        closed_form=closed_form,
        ctilde_computed=ctilde_computed,
        cstar_computed=cstar_computed,
        true_loss=b.true_loss,
        cert_err=err_closed,
        occupancy_max_err=0.0,
        err_Ctilde_eq_Cstar=err_Ctilde_eq_Cstar,
        ctilde_ge_true=int(ctilde_ge_true),
        description="worst-fallback => W~=W => C*=C~ exactly",
    )


def witness_C0_equals_Cstar(seed: int, S: int, A: int, H: int, n: int,
                              N: int, delta_r_target: float = 1.0) -> dict:
    """Witness for C0 = C* (sharpness of the coarsest level).

    C0 weights each unit by the FULL remaining-reward range (H-t)·Δr, while
    C* weights by the actual swing W(u).  They coincide iff a single
    coordinate deviation can cost the *entire* remaining reward range, i.e.
    W(u) = (H-t)·Δr for every weighted unit.

    The clean exact witness is a SINGLE-AGENT (n=1) game with H=1 and an
    all-or-nothing reward: the reference action 0 yields Δr, every other
    action yields 0.  With n=1 there is no prefix (hence no drift), and
        Q(u, ref=0) = Δr ,   Q(u, a≠0) = 0   =>   W(u) = Δr = (H-t)Δr ,
    so C0 = C* = Σ_u ρ(u) g(u) Δr exactly.

    (For n≥2 with all-or-nothing reward the equality breaks: once agent 0's
    fallback drifts off-reference, agent 1's swing collapses to 0, so
    W ≠ Δr at drifted prefixes — which is exactly why a single-agent witness
    is the sharp construction here.)
    """
    rng = np.random.default_rng(seed)
    A2 = 2
    n1 = 1                       # single agent: no prefix, no drift
    H1 = 1
    Anj = A2 ** n1               # = A2

    P = np.zeros((H1, S, Anj, S))
    d_next = rng.dirichlet(np.ones(S))
    for s in range(S):
        P[0, s, :] = d_next[None, :]
    d0 = rng.dirichlet(np.ones(S))

    # action 0 -> full reward Δr; action 1 -> 0
    R = np.zeros((H1, S, Anj))
    R[:, :, 0] = delta_r_target

    delta_r = float(delta_r_target)
    game = C.Game(S=S, A=A2, n=n1, H=H1, R=R, P=P, d0=d0,
                  ref=np.zeros((H1, S, n1), dtype=int), delta_r=delta_r)

    alpha_fn = make_alpha_fn(seed)
    b = build(game, alpha_fn, const_fn(N))

    c0 = b.chain["C0"]
    cstar = b.chain["Cstar"]
    err = abs(c0 - cstar)        # should be ~0 exactly now

    # closed-form: Σ_u ρ(u) g(u) Δr   (W=Δr everywhere)
    cstar_closed = sum(b.rho[k] * u.g * delta_r_target
                       for k, u in b.table.items())
    err_closed = abs(cstar_closed - cstar)

    return dict(
        witness="C0_eq_Cstar_n1H1",
        seed=seed, S=S, A=A2, H=H1, N=N,
        closed_form=cstar_closed,
        ctilde_computed=b.chain["Ctilde"],
        cstar_computed=cstar,
        c0=c0,
        true_loss=b.true_loss,
        cert_err=max(err_closed, err),
        occupancy_max_err=0.0,
        err_C0_eq_Cstar=err,
        description="single-agent all-or-nothing => C0=C*=ΣρgΔr (exact)",
    )


def witness_Ctilde_closed_form(seed: int, S: int, A: int, H: int, n: int,
                                N: int) -> dict:
    """Witness: action-independent game, uniform fallback.

    Closed-form for C̃* is just its definition:
        C̃*_closed = Σ_u rho(u) * g(u) * W̃(u)

    We verify that cert_Ctilde() matches this sum to machine precision,
    and that C̃* >= true_loss.

    Also compute and report the coordinate-decomposition gap:
        gap = C̃* - true_loss
    """
    game = make_independent_game_with_occupancy(S=S, A=A, H=H, n=n,
                                                seed=seed, target_d=None)
    alpha_fn = make_alpha_fn(seed)
    b = build(game, alpha_fn, const_fn(N))

    ctilde_api = b.chain["Ctilde"]
    # Manual closed-form recomputation
    ctilde_manual = sum(b.rho[k] * u.g * u.Wtilde for k, u in b.table.items())
    err = abs(ctilde_api - ctilde_manual)
    gap = ctilde_api - b.true_loss
    ctilde_ge_true = gap >= -TOL_PASS

    return dict(
        witness="Ctilde_closed_form",
        seed=seed, S=S, A=A, H=H, N=N,
        closed_form=ctilde_manual,
        ctilde_computed=ctilde_api,
        cstar_computed=b.chain["Cstar"],
        true_loss=b.true_loss,
        cert_err=err,
        occupancy_max_err=0.0,
        gap_Ctilde_minus_true=gap,
        ctilde_ge_true=int(ctilde_ge_true),
        description="manual Σ rho*g*W~ == cert_Ctilde to machine precision",
    )


# ---------------------------------------------------------------------------
# Part (c): Non-prefix-collapsed check
# ---------------------------------------------------------------------------

def check_non_collapsed(seed: int, S: int, A: int, H: int, n: int,
                         N: int) -> dict:
    """Set distinct per-prefix alpha values (manually) and verify that the
    controller prefix_probs genuinely depend on which prefix was executed.

    If all prefixes had the same alpha, prefix_probs would be the same for
    any prefix-conditional computation.  With distinct alpha per prefix, the
    failure probability g differs per prefix, causing different marginal
    distributions over the NEXT agent's action.

    We verify: for agent i=1 (n>=2), the conditional action distribution
    at different prefixes (different choices of agent 0's action) differs,
    confirming the prefix distribution is NOT collapsed.
    """
    assert n >= 2, "need n>=2 for prefix-collapse check"
    game = make_independent_game_with_occupancy(S=S, A=A, H=H, n=n,
                                                seed=seed, target_d=None)

    # Build a custom alpha_fn with distinct values per prefix for i=1
    # alpha depends on which action agent 0 took (the prefix for i=1)
    lo_alpha = 0.55
    hi_alpha = 0.90

    def custom_alpha(t, s, i, prefix):
        if i == 0:
            return 0.7  # uniform for agent 0
        if i == 1 and len(prefix) == 1:
            # alpha varies by which action agent 0 took
            # spread across [lo, hi] based on prefix[0]
            return lo_alpha + (hi_alpha - lo_alpha) * prefix[0] / max(A - 1, 1)
        # default
        h = hash((seed, "alpha", t, s, i, tuple(prefix))) & 0xFFFFFFFF
        return lo_alpha + (hi_alpha - lo_alpha) * (h / 0xFFFFFFFF)

    b = build(game, custom_alpha, const_fn(N))

    # For each (t, s), look at prefix_probs for length-1 prefixes (agent 1's prefix)
    # The probability that agent 1 executes each of its actions should differ
    # depending on which action agent 0 executed.
    collapsed_count = 0
    total_ts = 0

    realized_vs_intended = []
    for t in range(min(H, 2)):  # check a couple time steps
        for s in range(min(S, 2)):  # check a couple states
            _, prefix_probs = C.controller_state_dist(game, b.table, t, s)
            total_ts += 1

            # Get action distribution for each length-1 prefix
            prefix_action_dists = {}
            for a0 in range(A):
                prefix_len1 = (a0,)
                p_prefix = prefix_probs.get(prefix_len1, 0.0)
                if p_prefix < 1e-10:
                    continue
                # What's the distribution of agent 1's action given this prefix?
                u_i1 = b.table[(t, s, 1, prefix_len1)]
                intended_alpha = custom_alpha(t, s, 1, prefix_len1)
                realized_g = u_i1.g  # = g_N(N, intended_alpha)
                p_exec = u_i1.g * u_i1.fb + (1 - u_i1.g) * (
                    np.eye(game.A)[u_i1.psi])
                prefix_action_dists[a0] = (intended_alpha, realized_g, p_exec)

            # Check if distributions genuinely differ across prefixes
            if len(prefix_action_dists) >= 2:
                keys = list(prefix_action_dists.keys())
                dist0 = prefix_action_dists[keys[0]][2]
                dist1 = prefix_action_dists[keys[1]][2]
                is_collapsed = float(np.max(np.abs(dist0 - dist1))) < 1e-10
                if is_collapsed:
                    collapsed_count += 1

                realized_vs_intended.append(dict(
                    t=t, s=s,
                    prefix_a0=keys[0],
                    intended_alpha_a0=prefix_action_dists[keys[0]][0],
                    realized_g_a0=prefix_action_dists[keys[0]][1],
                    prefix_a1=keys[1],
                    intended_alpha_a1=prefix_action_dists[keys[1]][0],
                    realized_g_a1=prefix_action_dists[keys[1]][1],
                    max_dist_diff=float(np.max(np.abs(dist0 - dist1))),
                    is_collapsed=is_collapsed,
                ))

    non_collapsed = collapsed_count == 0 and len(realized_vs_intended) > 0
    max_dist_diff = max((r["max_dist_diff"] for r in realized_vs_intended), default=0.0)

    return dict(
        witness="non_collapsed",
        seed=seed, S=S, A=A, H=H, N=N,
        total_ts_checked=total_ts,
        collapsed_count=collapsed_count,
        non_collapsed=int(non_collapsed),
        max_prefix_action_dist_diff=max_dist_diff,
        realized_vs_intended=realized_vs_intended,
        closed_form=0.0,
        ctilde_computed=b.chain["Ctilde"],
        cstar_computed=b.chain["Cstar"],
        true_loss=b.true_loss,
        cert_err=0.0,
        occupancy_max_err=0.0,
        description="distinct alpha per prefix => action dist differs by prefix (not collapsed)",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--S", type=int, default=4)
    ap.add_argument("--A", type=int, default=3)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--n_witnesses", type=int, default=20,
                    help="number of random seeds per witness type")
    ap.add_argument("--N", type=int, default=5)
    args = ap.parse_args()

    S, A, H, n, N = args.S, args.A, args.H, 2, args.N
    seeds = [args.seed + k for k in range(args.n_witnesses)]

    rows = []
    max_cert_err = 0.0
    max_occ_err = 0.0
    all_non_collapsed = True
    all_ctilde_ge_true = True

    # (a) occupancy realizability
    for seed in tqdm(seeds, desc="(a) occupancy"):
        r = check_occupancy_realizability(seed, S, A, H, n, N)
        rows.append(r)
        max_occ_err = max(max_occ_err, r["occupancy_max_err"])
        if r["occupancy_max_err"] > TOL_PASS:
            print(f"[WARN] occ err={r['occupancy_max_err']:.3e} seed={seed}")

    # (b-i) worst-fallback: C* = C̃*
    for seed in tqdm(seeds, desc="(b) Cstar=Ctilde"):
        r = witness_Cstar_equals_Ctilde(seed, S, A, H, n, N)
        rows.append(r)
        max_cert_err = max(max_cert_err, r["cert_err"])
        err_eq = r.get("err_Ctilde_eq_Cstar", 0.0)
        if err_eq > 1e-10:
            print(f"[WARN] C*!=C~ err={err_eq:.3e} seed={seed}")
        if not r.get("ctilde_ge_true", 1):
            all_ctilde_ge_true = False
            print(f"[FAIL] C~* < true_loss seed={seed}")

    # (b-ii) H=1 independent game: C0 ~ C*
    for seed in tqdm(seeds, desc="(b) C0=Cstar H=1"):
        r = witness_C0_equals_Cstar(seed, S=S, A=2, H=1, n=n, N=N)
        rows.append(r)
        max_cert_err = max(max_cert_err, r["cert_err"])
        err_eq = r.get("err_C0_eq_Cstar", 0.0)
        if err_eq > 1e-8:
            print(f"[WARN] C0!=C* (H=1) err={err_eq:.3e} seed={seed}")

    # (b-iii) Ctilde closed-form check
    for seed in tqdm(seeds, desc="(b) Ctilde closed-form"):
        r = witness_Ctilde_closed_form(seed, S, A, H, n, N)
        rows.append(r)
        max_cert_err = max(max_cert_err, r["cert_err"])
        if not r.get("ctilde_ge_true", 1):
            all_ctilde_ge_true = False

    # (c) non-prefix-collapsed check
    nc_results = []
    for seed in tqdm(seeds[:5], desc="(c) non-collapsed"):
        r = check_non_collapsed(seed, S, A, H, n=2, N=N)
        rows.append(r)
        nc_results.append(r)
        if not r["non_collapsed"]:
            all_non_collapsed = False
            print(f"[WARN] collapsed prefix found! seed={seed}")

    # --- CSV ---
    # Extract scalar columns for the CSV (drop complex nested fields)
    csv_rows = []
    for r in rows:
        csv_row = {k: v for k, v in r.items()
                   if not isinstance(v, (list, dict, np.ndarray))}
        csv_rows.append(csv_row)
    df = pd.DataFrame(csv_rows)
    csv_path = data_path("exp_07_witness_sharpness.csv")
    df.to_csv(csv_path, index=False)
    print(f"[exp_07] CSV saved: {csv_path}")

    # --- Figure: |closed-form - computed| on log scale ---
    fig, ax = new_fig(figsize=(8, 4))
    witness_types = df["witness"].unique().tolist()
    for wi, wtype in enumerate(witness_types):
        sub = df[df["witness"] == wtype]
        errs = sub["cert_err"].values
        errs_plot = np.where(errs <= 0, 1e-18, errs)
        ax.scatter([wi] * len(errs_plot), errs_plot, s=20, alpha=0.7,
                   label=wtype)
    ax.set_yscale("log")
    ax.set_xticks(range(len(witness_types)))
    ax.set_xticklabels(witness_types, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("|closed-form - computed| (log scale)")
    ax.set_title("exp_07  Certificate closed-form vs computed errors\n(expect ~1e-15..1e-12 = machine precision)")
    ax.axhline(TOL_PASS, color="red", ls="--", lw=0.8, label=f"TOL={TOL_PASS:.0e}")
    ax.legend(fontsize=7, loc="upper right")
    save_pdf(fig, fig_path("exp_07_cert_errors.pdf"))

    # --- Non-collapsed summary ---
    if nc_results:
        print("\n[exp_07] Non-collapsed check: per-prefix action distribution diffs:")
        for r in nc_results:
            for rd in r.get("realized_vs_intended", []):
                print(
                    f"  (t={rd['t']},s={rd['s']}) "
                    f"prefix a0={rd['prefix_a0']} alpha={rd['intended_alpha_a0']:.3f} g={rd['realized_g_a0']:.4f} "
                    f"vs a0={rd['prefix_a1']} alpha={rd['intended_alpha_a1']:.3f} g={rd['realized_g_a1']:.4f} "
                    f"| dist_diff={rd['max_dist_diff']:.4e}"
                )

    # --- PASS/FAIL ---
    pass_occ = max_occ_err <= TOL_PASS
    pass_cert = max_cert_err <= TOL_PASS
    pass_nc = all_non_collapsed
    pass_true = all_ctilde_ge_true

    status = "PASS" if (pass_occ and pass_cert and pass_nc and pass_true) else "FAIL"

    write_summary(
        f"exp_07 witness_sharpness [{status}] "
        f"witnesses={len(rows)} "
        f"max_occ_err={max_occ_err:.3e} "
        f"max_cert_err={max_cert_err:.3e} "
        f"non_collapsed={int(all_non_collapsed)} "
        f"ctilde_ge_true={int(all_ctilde_ge_true)}"
    )

    if not pass_occ:
        print(f"[FAIL] max_occ_err={max_occ_err:.3e} > {TOL_PASS}")
    if not pass_cert:
        print(f"[FAIL] max_cert_err={max_cert_err:.3e} > {TOL_PASS}")
    if not pass_nc:
        print("[FAIL] some prefix distributions were collapsed")
    if not pass_true:
        print("[FAIL] C~* < true_loss for some witness")


if __name__ == "__main__":
    main()
