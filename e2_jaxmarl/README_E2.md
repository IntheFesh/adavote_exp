# E2 — JaxMARL diagnostics (GPU)

> **DIAGNOSTIC / EMPIRICAL PROXY — NOT STRICT-VALIDITY EVIDENCE.**
> All strict-validity (coverage) evidence in the paper comes from **E1**
> (`e1_tabular/`, exact tabular, CPU). E2 provides **scalability and
> diagnostic** signal only. E2 does **not** claim coverage guarantees and does
> **not** benchmark against SOTA MARL — this paper is about deployment-time
> downside certification, not policy improvement.

## Contents
- `train_committee.py` — IPPO / MAPPO committee trainers (PureJaxRL / JaxMARL
  end-to-end-JIT style). Trains `--n_members` (default 7) seeds and saves a
  checkpoint per member.
- `eval_committee.py` — deployment evaluation: fixes one member as reference
  π^ref, treats the rest as advisors, estimates the endorsement rate α̂ (advisor
  proposals landing in `G^η`), uses the reference critic to approximate
  `Q^{π^ref}` and the clipped fallback advantage `W̃^+` (Assumption 1), and forms
  a **diagnostic** empirical-Bernstein certificate (reusing the numpy helpers in
  `common/certificates.py`).
- `run_all_e2.py` — driver producing the three diagnostic artifacts to
  `results/e2/`.

## Environments / algorithms
- MPE `simple_spread`, `simple_reference` (primary)
- `overcooked_v0` (secondary)
- `smax` (optional, `--enable_smax`, default OFF)
- IPPO and MAPPO.

Default config: `{simple_spread, simple_reference, overcooked_v0} × {ippo, mappo}
× 7 seeds`.

## Diagnostic outputs (`results/e2/`)
- (a) **compute–value Pareto frontier**: true return vs diagnostic certificate
  across committee budgets (`e2_pareto.{csv,pdf}`).
- (b) **conservativeness diagnostics** (honestly reported): `W̃^+` proxy vs #MC
  rollouts, membership false-negative rate, certificate inflation relative to
  the reward range (`e2_conservativeness.{csv,pdf}`).
- (c) **ranking-validity diagnostic**: committee budget-allocation ordering
  (`e2_ranking.{csv,pdf}`).

All titles/CSVs carry a "diagnostic — not strict validity" label.

## GPU setup (AutoDL: RTX 5090 32GB / vGPU-48GB)
JAX auto-detects the GPU. If the prebuilt `jaxlib` mismatches the CUDA toolkit,
install the matching CUDA wheel:
```bash
pip install -U "jax[cuda12]"        # CUDA 12.x: RTX 5090 (drv 595.x) & vGPU-48GB (drv 580.x)
python -c "import jax; print(jax.devices())"
```
CPU fallback (debug only, slow):
```bash
pip install -U "jax[cpu]"
JAX_PLATFORMS=cpu python -m e2_jaxmarl.run_all_e2 ...
```

## Reproduction
```bash
pip install -r ../requirements_e2.txt
# 1) train committees
python -m e2_jaxmarl.train_committee --env simple_spread --algo ippo \
    --n_members 7 --total_timesteps 10_000_000 --out_dir results/e2/checkpoints
# ... repeat per env/algo, or script it ...
# 2) diagnostics
python -m e2_jaxmarl.run_all_e2 --envs simple_spread,simple_reference,overcooked_v0 \
    --algos ippo,mappo --n_members 7 --budget_grid 1,3,5,7,9
```

## Expected GPU wall-clock
- MPE (`simple_spread` / `simple_reference`): ~5–20 min / seed
- Overcooked: ~15–40 min / seed
- SMAX: ~30–90 min / seed
- Default config ≈ **15–30 GPU-hours**; with SMAX enabled ≈ **40–60 GPU-hours**.

## No-GPU-mode discipline (AutoDL)
When the training window ends and the **GPU has been idle > 30 minutes, switch
the instance back to no-GPU (CPU-only) mode** to save credits. (E1 always runs
in no-GPU mode.)
