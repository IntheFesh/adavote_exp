"""
独立 E2 诊断脚本 — 只跑 4 个 MPE 组合,绕过会卡的 run_all_e2。
直接循环调 evaluate_committee,每个 (组合,budget) 跑完即时写 CSV。
DIAGNOSTIC ONLY — strict validity 来自 E1。
"""
import os
import time
import numpy as np
import pandas as pd

os.environ.setdefault("ADAVOTE_RESULTS", "/root/autodl-tmp/adavote_exp/results")

from e2_jaxmarl.eval_committee import load_committee, evaluate_committee
from common.io_utils import e2_path

COMBOS = [
    ("MPE_simple_spread_v3_short", "simple_spread", "ippo"),
    ("simple_spread", "simple_spread", "mappo"),
    ("simple_reference", "simple_reference", "ippo"),
    ("simple_reference", "simple_reference", "mappo"),
]
# 上面第一列只是注释占位,实际用 env/algo 两列
COMBOS = [
    ("simple_spread", "ippo"),
    ("simple_spread", "mappo"),
    ("simple_reference", "ippo"),
    ("simple_reference", "mappo"),
]

BUDGETS = [1, 3, 5, 7]
CKPT_DIR = "results/e2/checkpoints"
N_EVAL_EPISODES = 16
MC_ROLLOUTS = 8
ETA = 0.0
SEED = 0

CSV_PATH = e2_path("e2_diagnostics.csv")

rows = []
t_start = time.time()

for env, algo in COMBOS:
    print(f"\n{'='*60}\n=== {env} / {algo} ===\n{'='*60}", flush=True)
    try:
        members = load_committee(CKPT_DIR, env, algo)
    except FileNotFoundError as e:
        print(f"[skip] {env}/{algo}: {e}", flush=True)
        continue

    for budget in BUDGETS:
        sub = members[:max(budget, 1)]
        if len(sub) < 1:
            continue
        t0 = time.time()
        m = evaluate_committee(
            env, algo, sub, ref_member=0, eta=ETA,
            n_eval_episodes=N_EVAL_EPISODES, mc_rollouts=MC_ROLLOUTS, seed=SEED,
        )
        dt = time.time() - t0
        row = dict(
            env=env, algo=algo, budget=budget,
            N=m["N"],
            alpha_hat_mean=m["alpha_hat_mean"],
            g_mean=m["g_mean"],
            p_hat_advisor=m["p_hat_advisor"],
            h_anchored=m["h_anchored"],
            diag_certificate=m["diag_certificate"],
            cert_over_range=m["cert_over_range"],
            mean_return=m["mean_return"],
            Wtilde_proxy_mean=m["Wtilde_proxy_mean"],
            reward_range=m["reward_range"],
            eval_seconds=dt,
        )
        rows.append(row)
        print(f"[{env}/{algo} budget={budget}] "
              f"alpha={m['alpha_hat_mean']:.3f} g={m['g_mean']:.3f} "
              f"cert={m['diag_certificate']:.2f} return={m['mean_return']:.2f} "
              f"({dt:.0f}s)", flush=True)
        # 即时落盘:每跑完一个就写,卡住也不丢已有结果
        pd.DataFrame(rows).to_csv(CSV_PATH, index=False)

print(f"\n=== ALL DIAGNOSTICS DONE in {(time.time()-t_start)/60:.1f} min ===", flush=True)
print(f"CSV saved: {CSV_PATH}", flush=True)
