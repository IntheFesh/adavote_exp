# AdaVote — Experiment Suite (E1 exact-tabular + E2 JaxMARL)

Reproducible experiment code for the TMLR submission on **cooperative-MARL
deployment-time downside-value certification** (working title *AdaVote*).

> **Validity boundary (read first).** *All strict-validity evidence in the
> paper comes from **E1** (exact tabular, CPU).* **E2** (JaxMARL, GPU) is a
> **diagnostic / scalability proxy only** and makes **no** strict-validity
> claim. The code is split accordingly: E1 never imports JAX and runs
> standalone; E2 is a separate package that E1 never depends on.

---

## 1. What's here

```
adavote_exp/
  common/
    certificates.py   # core math: units, Δ_+, W, W̃, g_N, occupancy, the
                      #            three-level certificate chain, estimator
    games.py          # random Markov-game generators (dependent/independent)
    plotting.py       # matplotlib PDF style (vector, pdf.fonttype=42)
    io_utils.py       # results paths + summary.txt writer
  e1_tabular/
    exp_01_chain_ordering.py          # Thm 1b: C0 ≥ C* ≥ C̃* ≥ true
    exp_02_eta_positive_success_term.py  # §5 Remark: η>0 needs success term
    exp_03_estimator.py               # Thm 3: empirical-Bernstein Π2 vs Π1
    exp_04_marginal_fails.py          # Prop 1/1b: marginal under-bound, n-amplification
    exp_05_falsification.py           # ablation falsification of each component
    exp_06_prefix_drift.py            # Thm 1: prefix drift; naive under-bounds
    exp_07_witness_sharpness.py       # realizability/sharpness witnesses
    exp_08_rank_consistency.py        # Prop 3: rank consistency + reversals
    exp_09_budget_increase.py         # Cor 2.1/2.2 + honest counterexamples
    exp_10_eligibility.py             # Cor 2.1 premise: g_N ↓ in N iff α>1/2
    exp_11_wrapper_vs_naive.py        # Prop 0: wrapper == naive voting at η=0
    run_all_e1.py                     # runs exp_01..11, aggregates summary
    API_REFERENCE.md                  # internal API doc
  e2_jaxmarl/
    train_committee.py  eval_committee.py  run_all_e2.py  README_E2.md
  tests/test_certificates.py          # pytest unit tests for the core
  results/{data,figs,e2,summary.txt}  # outputs
  requirements_e1.txt   requirements_e2.txt
```

## 2. Core objects (see `common/certificates.py`)

- **Unit** `u=(t,s,i,a_{<i})`: a single coordinate decision (agent `i` at time
  `t`, state `s`, given the executed prefix `a_{<i}`).
- **Non-negative coordinate amplitude** `Δ_+(u,a)=[Q^ref(u,a_i^ref)−Q^ref(u,a)]_+`,
  `W=max_a Δ_+`, `W̃=E[Δ_+(u,a^fb)|F=1]`.
- **Majority-failure** `g_N(α)=Pr[Bin(N,α)≤⌊N/2⌋]` (`N` odd).
- **Measure convention** `μ` is a law over units; certificates are
  `C = nH·E_μ[ω g] = Σ_u ρ(u) ω(u) g(u)` with controller unit-occupancy
  `ρ(u)=d^ctrl_t(s)·P(a_{<i}|t,s)` and `Σ_u ρ(u)=nH`.
- **Three-level chain (η=0)**
  `C0 = nH·E_μ[g(H−t)Δr] ≥ C* = nH·E_μ[gW] ≥ C̃* = nH·E_μ[gW̃] ≥ true loss`.
  The chain holds *by construction* of the occupancy weighting (the
  performance-difference / coordinate-telescoping derivation is implemented
  exactly in `controller_value`, `unit_occupancy`, `cert_*`).

## 3. E1 — exact tabular (CPU-only; strict evidence)

E1 runs **entirely in no-GPU mode** (CPU only: numpy + scipy). It does **not**
import any GPU/JAX library.

```bash
# from repo root
pip install -r requirements_e1.txt
pytest tests/ -q                       # unit tests must be all-green

python e1_tabular/run_all_e1.py --seed 0          # full suite
python e1_tabular/run_all_e1.py --seed 0 --quick  # fast smoke run

# or a single experiment
python -m e1_tabular.exp_01_chain_ordering --n_games 300 --seed 0
```

Each experiment writes `results/data/exp_XX_*.csv`, `results/figs/exp_XX_*.pdf`
(vector, embedded Type-42 fonts), and one `PASS/FAIL` line to
`results/summary.txt`.

**Honest-boundary experiments.** Some experiments are designed to *exhibit
failure modes* and report them faithfully — their "PASS" means *the predicted
counterexample/violation appeared*, not "zero violations":
- `exp_02` pure-failure term under-bounds once `η>0`;
- `exp_05` naive / unweighted / plug-in-α̂ ablations are invalid (nonzero
  violation rates); worst-prefix and Π1 are valid-but-looser;
- `exp_06` the naive single-agent bound under-bounds under prefix drift;
- `exp_08c` order-reversal counterexamples under occupancy coupling;
- `exp_09` group-2c / group-3 counterexamples (true loss can rise via
  coordinate coupling / occupancy drift).

Expected CPU wall-clock for the full suite: **tens of minutes to ~1–2 hours**
on the AutoDL 25-core Xeon (no GPU needed).

## 4. E2 — JaxMARL diagnostics (GPU; NOT strict evidence)

See `e2_jaxmarl/README_E2.md` for the full GPU runbook. Summary:

```bash
pip install -r requirements_e2.txt     # pick the right jax/jaxlib CUDA wheel!
python e2_jaxmarl/run_all_e2.py --envs simple_spread,simple_reference,overcooked \
       --algos ippo,mappo --n_members 7
```

- Envs: MPE `simple_spread`, `simple_reference` (primary), `overcooked`
  (secondary), `smax` (optional, `--enable_smax`, default OFF).
- Algos: IPPO, MAPPO (PureJaxRL-style end-to-end JIT).
- Outputs go to `results/e2/` and are **all labeled diagnostic / empirical
  proxy — NOT strict-validity evidence.**

### AutoDL GPU setup (RTX 5090 32GB / vGPU-48GB)

JAX auto-detects the GPU. If the pre-built `jaxlib` does not match the
installed CUDA toolkit, install the matching CUDA wheel:

```bash
# CUDA 12.x (works for RTX 5090 driver 595.x and vGPU-48GB driver 580.x):
pip install -U "jax[cuda12]"
# verify:
python -c "import jax; print(jax.devices())"
```

CPU fallback (no card / debugging on CPU):

```bash
pip install -U "jax[cpu]"
JAX_PLATFORMS=cpu python e2_jaxmarl/run_all_e2.py ...
```

### Expected GPU wall-clock
- MPE (`simple_spread`/`simple_reference`): ~5–20 min / seed
- Overcooked: ~15–40 min / seed
- SMAX: ~30–90 min / seed
- Default config `{3 envs} × {2 algos} × 7 seeds` ≈ **15–30 GPU-hours**;
  enabling SMAX can push this to **40–60 GPU-hours**.

### No-GPU-mode discipline (AutoDL)
- **E1**: always run in **no-GPU (CPU-only) mode**.
- **E2**: once a training window ends and the **GPU has been idle > 30 min,
  switch the instance back to no-GPU mode** to save credits.

## 5. Data layout / AutoDL paths

On AutoDL place the repo on the data disk and write outputs there:

```bash
# code on the data disk:
/root/autodl-tmp/adavote_exp/
# outputs (overridable via ADAVOTE_RESULTS):
export ADAVOTE_RESULTS=/root/autodl-tmp/adavote_exp/results
```

By default outputs go to `<repo>/results/`. Set `ADAVOTE_RESULTS` to redirect.

## 6. Reproducibility

- Every script exposes `--seed` (and sizes / repeats) via `argparse`; the full
  suite is one command (`run_all_e1.py --seed 0`).
- No expected result is hard-coded: every PASS/FAIL is computed from data the
  code actually produced, and inequality claims are asserted **per instance**
  with violation counts reported (expected 0 for valid claims, nonzero only at
  the known counterexamples).
- All figures are vector PDFs with embedded fonts (`pdf.fonttype=42`); all CSVs
  carry headers and reproducible data.
