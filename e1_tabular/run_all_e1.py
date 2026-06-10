"""
run_all_e1.py — top-level driver for the E1 tabular suite (CPU-only).

Runs exp_01 .. exp_11 in order with a fixed global seed, resets
results/summary.txt at the start, and prints the aggregated PASS/FAIL summary
at the end.  Each experiment writes its own CSV (results/data/), PDF
(results/figs/) and one summary line.

Usage:
    python e1_tabular/run_all_e1.py --seed 0
    python -m e1_tabular.run_all_e1 --seed 0 --quick   # smaller sizes, fast

Expected CPU wall-clock: tens of minutes (a couple of hours at large sizes).
This script NEVER imports jax — E1 is strictly CPU/numpy/scipy.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

# Allow `python e1_tabular/run_all_e1.py` (script dir != repo root) by putting
# the repo root on sys.path before importing the `common` package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common.io_utils import summary_file, results_root

# (module, extra-args-for-quick-mode) — every exp accepts --seed.
EXPERIMENTS = [
    ("e1_tabular.exp_01_chain_ordering", ["--n_games", "60"]),
    ("e1_tabular.exp_02_eta_positive_success_term", ["--n_games", "60"]),
    ("e1_tabular.exp_03_estimator", ["--n_games", "4", "--repeats", "50"]),
    ("e1_tabular.exp_04_marginal_fails", []),
    ("e1_tabular.exp_05_falsification", ["--n_games", "60"]),
    ("e1_tabular.exp_06_prefix_drift", ["--n_games", "80"]),
    ("e1_tabular.exp_07_witness_sharpness", []),
    ("e1_tabular.exp_08_rank_consistency", ["--n_games", "60"]),
    ("e1_tabular.exp_09_budget_increase", ["--n_games", "60"]),
    ("e1_tabular.exp_10_eligibility", []),
    ("e1_tabular.exp_11_wrapper_vs_naive", ["--n_games", "60"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="use smaller sizes for a fast smoke run")
    args = ap.parse_args()

    # Reset summary file.
    os.makedirs(results_root(), exist_ok=True)
    open(summary_file(), "w").close()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    # Ensure imports resolve from repo root and stay CPU-only.
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    t0 = time.time()
    failures = []
    for mod, quick_args in EXPERIMENTS:
        extra = quick_args if args.quick else []
        cmd = [sys.executable, "-m", mod, "--seed", str(args.seed)] + extra
        print(f"\n=== running {mod} {' '.join(extra)} ===", flush=True)
        rc = subprocess.call(cmd, cwd=repo_root, env=env)
        if rc != 0:
            failures.append((mod, rc))
            print(f"[ERROR] {mod} exited with code {rc}", flush=True)

    dt = time.time() - t0
    print("\n" + "=" * 70)
    print(f"E1 suite finished in {dt/60:.1f} min")
    print("Summary (results/summary.txt):")
    print("=" * 70)
    with open(summary_file()) as f:
        sys.stdout.write(f.read())

    if failures:
        print("\nNON-ZERO EXIT CODES:")
        for mod, rc in failures:
            print(f"  {mod}: {rc}")
        sys.exit(1)
    print("\nAll experiments completed (see PASS/FAIL lines above).")


if __name__ == "__main__":
    main()
