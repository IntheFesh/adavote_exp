"""
e2_jaxmarl/run_all_e2.py — top-level E2 diagnostic driver.

DIAGNOSTIC / EMPIRICAL PROXY — NOT STRICT-VALIDITY EVIDENCE.
Strict validity comes ONLY from E1 (e1_tabular/).

Given trained committees (see train_committee.py), this driver produces three
DIAGNOSTIC artifacts to results/e2/:
  (a) compute–value Pareto frontier: true return vs diagnostic certificate at
      a grid of committee budgets;
  (b) conservativeness diagnostics: W̃^+ proxy vs #MC rollouts, membership
      false-negative rate, and certificate inflation relative to reward range;
  (c) ranking-validity diagnostic: committee budget-allocation ordering.

Every CSV/figure is labeled diagnostic.  If JAX / checkpoints are missing
(e.g. on a no-GPU box) the driver prints the exact GPU commands and exits
cleanly WITHOUT crashing.

Usage (GPU box):
    python -m e2_jaxmarl.run_all_e2 \\
        --envs simple_spread,simple_reference,overcooked_v0 \\
        --algos ippo,mappo --n_members 7 --budget_grid 1,3,5,7,9
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np
import pandas as pd

from common.io_utils import e2_path
from common.plotting import new_fig, save_pdf

DIAG = "DIAGNOSTIC — not strict validity (E1 only)"


def _try_imports():
    """Import the JAX-dependent eval pipeline; return (ok, err)."""
    try:
        from e2_jaxmarl.eval_committee import load_committee, evaluate_committee  # noqa
        import jax  # noqa
        return True, ""
    except Exception as e:  # noqa
        return False, str(e)


def _gpu_instructions(args):
    print("\n" + "=" * 70)
    print("E2 cannot run here (missing JAX/JaxMARL or no trained checkpoints).")
    print("E2 is DIAGNOSTIC ONLY — strict validity comes from E1 (e1_tabular/).")
    print("=" * 70)
    print("On a GPU box (see README_E2.md):")
    print("  pip install -r requirements_e2.txt   # match your CUDA wheel")
    for env in args.envs.split(","):
        for algo in args.algos.split(","):
            print(f"  python -m e2_jaxmarl.train_committee --env {env} "
                  f"--algo {algo} --n_members {args.n_members} "
                  f"--out_dir {args.checkpoints_dir}")
    print(f"  python -m e2_jaxmarl.run_all_e2 --envs {args.envs} "
          f"--algos {args.algos} --budget_grid {args.budget_grid}")
    print("CPU smoke-test (slow):  JAX_PLATFORMS=cpu python -m e2_jaxmarl.run_all_e2 ...")
    print("=" * 70)


def run(args):
    from e2_jaxmarl.eval_committee import load_committee, evaluate_committee

    envs = args.envs.split(",")
    algos = args.algos.split(",")
    budgets = [int(x) for x in args.budget_grid.split(",")]
    if args.enable_smax:
        envs = envs + ["smax"]

    pareto_rows = []
    conserv_rows = []
    ranking_rows = []

    for env in envs:
        for algo in algos:
            try:
                members = load_committee(args.checkpoints_dir, env, algo)
            except FileNotFoundError as e:
                print(f"[skip] {env}/{algo}: {e}")
                continue

            # (a) Pareto frontier across budgets (use first `budget` members).
            for budget in budgets:
                sub = members[:max(budget, 1)]
                if len(sub) < 1:
                    continue
                m = evaluate_committee(
                    env, algo, sub, ref_member=0, eta=args.eta,
                    n_eval_episodes=args.n_eval_episodes,
                    mc_rollouts=args.mc_rollouts, seed=args.seed,
                )
                pareto_rows.append(dict(env=env, algo=algo, budget=budget,
                                        mean_return=m["mean_return"],
                                        diag_certificate=m["diag_certificate"],
                                        cert_over_range=m["cert_over_range"],
                                        note=DIAG))

            # (b) conservativeness vs #MC rollouts.
            for mc in [1, 4, 8, 16, 32]:
                m = evaluate_committee(
                    env, algo, members, ref_member=0, eta=args.eta,
                    n_eval_episodes=max(args.n_eval_episodes // 2, 8),
                    mc_rollouts=mc, seed=args.seed,
                )
                conserv_rows.append(dict(env=env, algo=algo, mc_rollouts=mc,
                                         Wtilde_proxy_mean=m["Wtilde_proxy_mean"],
                                         membership_fn_rate=m["membership_fn_rate"],
                                         cert_over_range=m["cert_over_range"],
                                         note=DIAG))

            # (c) ranking-validity: diagnostic ordering of g_mean vs budget.
            for budget in budgets:
                sub = members[:max(budget, 1)]
                m = evaluate_committee(
                    env, algo, sub, ref_member=0, eta=args.eta,
                    n_eval_episodes=max(args.n_eval_episodes // 2, 8),
                    mc_rollouts=args.mc_rollouts, seed=args.seed,
                )
                ranking_rows.append(dict(env=env, algo=algo, budget=budget,
                                         g_mean=m["g_mean"],
                                         alpha_hat_mean=m["alpha_hat_mean"],
                                         note=DIAG))

    # ---- Save CSVs ----
    if pareto_rows:
        df = pd.DataFrame(pareto_rows)
        df.to_csv(e2_path("e2_pareto.csv"), index=False)
        fig, ax = new_fig()
        for (env, algo), sub in df.groupby(["env", "algo"]):
            ax.plot(sub["diag_certificate"], sub["mean_return"], "o-",
                    label=f"{env}/{algo}")
        ax.set_xlabel("diagnostic certificate")
        ax.set_ylabel("mean return")
        ax.set_title("E2 (a) compute–value Pareto  [%s]" % DIAG)
        ax.legend(fontsize=7)
        save_pdf(fig, e2_path("e2_pareto.pdf"))

    if conserv_rows:
        df = pd.DataFrame(conserv_rows)
        df.to_csv(e2_path("e2_conservativeness.csv"), index=False)
        fig, ax = new_fig()
        for (env, algo), sub in df.groupby(["env", "algo"]):
            ax.plot(sub["mc_rollouts"], sub["Wtilde_proxy_mean"], "o-",
                    label=f"{env}/{algo}")
        ax.set_xlabel("# MC rollouts")
        ax.set_ylabel(r"$\tilde{W}^+$ proxy (conservativeness)")
        ax.set_title("E2 (b) conservativeness vs MC rollouts  [%s]" % DIAG)
        ax.legend(fontsize=7)
        save_pdf(fig, e2_path("e2_conservativeness.pdf"))

    if ranking_rows:
        df = pd.DataFrame(ranking_rows)
        df.to_csv(e2_path("e2_ranking.csv"), index=False)
        fig, ax = new_fig()
        for (env, algo), sub in df.groupby(["env", "algo"]):
            ax.plot(sub["budget"], sub["g_mean"], "o-", label=f"{env}/{algo}")
        ax.set_xlabel("committee budget N")
        ax.set_ylabel("mean majority-failure g")
        ax.set_title("E2 (c) ranking diagnostic: g vs budget  [%s]" % DIAG)
        ax.legend(fontsize=7)
        save_pdf(fig, e2_path("e2_ranking.pdf"))

    print(f"[E2 run_all] DIAGNOSTIC outputs written to {e2_path('')}")
    print("REMINDER: strict validity is provided by E1 only; E2 is diagnostic.")


def main():
    ap = argparse.ArgumentParser(description="E2 diagnostic driver (NOT strict validity)")
    ap.add_argument("--envs", default="simple_spread,simple_reference,overcooked_v0")
    ap.add_argument("--algos", default="ippo,mappo")
    ap.add_argument("--n_members", type=int, default=7)
    ap.add_argument("--budget_grid", default="1,3,5,7")
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--n_eval_episodes", type=int, default=64)
    ap.add_argument("--mc_rollouts", type=int, default=16)
    ap.add_argument("--checkpoints_dir", default="results/e2/checkpoints")
    ap.add_argument("--enable_smax", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ok, err = _try_imports()
    if not ok:
        print(f"[E2 run_all] JAX/JaxMARL unavailable: {err}")
        _gpu_instructions(args)
        return
    try:
        run(args)
    except FileNotFoundError:
        _gpu_instructions(args)
    except Exception:
        print("[E2 run_all] unexpected error during diagnostics:")
        traceback.print_exc()
        _gpu_instructions(args)


if __name__ == "__main__":
    main()
