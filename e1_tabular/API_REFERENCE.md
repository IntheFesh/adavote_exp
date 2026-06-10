# AdaVote E1 — internal API reference (for experiment authors)

Working dir: `/home/user/adavote_exp`. Run experiments as modules from repo root:
`python -m e1_tabular.exp_XX_name [args]`.  Outputs via `common.io_utils`.
CPU-only: **never import jax**.

## Math recap (see `common/certificates.py` docstring for full detail)

- Unit `u=(t,s,i,prefix)`: coordinate decision for agent `i` at time `t`,
  state `s`, given executed prefix `a_{<i}` (tuple length `i`, 0-indexed agents).
- `Q^{ref}(u,a)`: continuation value with prefix=`prefix`, agent i=`a`, agents>i=ref.
- `Δ_+(u,a) = [Q(u, ref_i) - Q(u,a)]_+`.  `W=max_a Δ_+`, `W̃ = Σ_a fb(a)Δ_+(a)`.
- `g_N(α) = binom.cdf(floor(N/2), N, α)`. N odd. Eligible iff α>1/2 (then g_N ↓ in odd N).
- Occupancy `ρ(u)=d^ctrl_t(s)·P(prefix|t,s)`, `Σ_u ρ(u)=nH`. `μ=ρ/(nH)`.
- Chain (η=0): `C0 = Σ ρ g (H-t)Δr ≥ C* = Σ ρ g W ≥ C̃* = Σ ρ g W̃ ≥ true_loss`.
- η>0: pure-failure `Σ ρ g W̃` may UNDERBOUND; success-inclusive
  `Σ ρ [(1-g)w_ψ + g W̃] ≥ true_loss`.

## Key functions / classes (`from common import certificates as C`)

- `C.g_N(N, alpha) -> float`
- `C.clip_pos(x)`
- `C.Game(S,A,n,H,R,P,d0,ref,delta_r)`  dataclass. `R:(H,S,A**n)`, `P:(H,S,A**n,S)`,
  `d0:(S,)`, `ref:(H,S,n) int`. `game.ref_joint(t,s)->tuple`.
- `C.joint_to_index(tuple,A)`, `C.index_to_joint(idx,A,n)`
- `C.ref_value_iteration(game) -> V_ref (H+1,S)`
- `C.coordinate_Q(game,V_ref,t,s,i,prefix,a_i) -> float`
- `C.coordinate_Q_vec(game,V_ref,t,s,i,prefix) -> (A,)`
- `C.build_unit_table(game,V_ref,alpha_fn,N_fn,fb_fn=C.uniform_fb,eta=0.0,psi_rule="ref")`
  -> dict {(t,s,i,prefix): UnitInfo}.  `psi_rule in {"ref","worst_in_Geta"}`.
  `alpha_fn(t,s,i,prefix)->float`, `N_fn(...)->int`,
  `fb_fn(t,s,i,prefix,game,delta_plus)->(A,) dist`.
- `UnitInfo` fields: `t,s,i,prefix,ref_i,Q,delta_plus,W,Wtilde,fb,psi,w_psi,Geta,alpha,N,g`.
- `C.controller_value(game,table) -> (V_ctrl (H+1,S), d_ctrl (H,S))`
- `C.unit_occupancy(game,table,d_ctrl) -> {key: rho}`
- `C.controller_state_dist(game,table,t,s) -> (joint_dist (A**n,), prefix_probs dict)`
- `C.true_loss(game,V_ref,V_ctrl) -> float`
- `C.cert_C0/cert_Cstar/cert_Ctilde/cert_failure_only/cert_success_inclusive(game,table,rho)`
- `C.cert_chain(game,table,rho) -> {"C0","Cstar","Ctilde","success_inclusive"}`
- `C.max_W(table)`
- `C.empirical_bernstein(X, delta, b) -> float`  (X array, X∈[0,b])
- `C.estimator_samples(game,table,rho,m,rng,variant="Pi2"|"Pi1") -> (m,)`
  E[X]=C̃* (Pi2) or C* (Pi1). Both ≥ true_loss.

## Games (`from common.games import ...`)

- `make_game(S=4,A=3,H=4,n=2,seed=0,transition="dependent"|"independent",
   eta=0.0,ref_mode="greedy"|"random"|"given",ref_actions=None,
   reward_scale=1.0,dirichlet_alpha=1.0) -> Game`
- `make_independent_game_with_occupancy(S,A,H,n,seed,target_d=None) -> Game`
   action-independent transitions (fixed occupancy subclass).

## E1 helpers (`from e1_tabular._common import build, make_alpha_fn, const_fn`)

- `make_alpha_fn(seed, lo=0.55, hi=0.9)` -> per-unit α function (reproducible).
- `const_fn(value)` -> returns a function ignoring args returning value.
- `build(game, alpha_fn, N_fn, fb_fn=C.uniform_fb, eta=0.0, psi_rule="ref") -> Built`
  with fields `game,V_ref,table,V_ctrl,d_ctrl,rho,true_loss,chain`.

## Output conventions (`from common.io_utils import data_path, fig_path, write_summary`)
`from common.plotting import new_fig, save_pdf`
- `data_path("exp_XX_name.csv")`, `fig_path("exp_XX_name.pdf")`.
- `df.to_csv(data_path(...), index=False)`.
- `fig, ax = new_fig(); ...; save_pdf(fig, fig_path("..."))`. Boxplot: use `tick_labels=`.
- `write_summary("exp_XX name [PASS|FAIL] key=val ...")`  (one line).

## Discipline
- argparse exposes seed / scale / repeats. tqdm progress. Assert + print counterexamples.
- Honest PASS/FAIL: "should hold" experiments PASS at 0 violations; counterexample
  experiments (02 pure-failure, 05 naive, 06 naive, 08c, 09 group3) must HONESTLY
  REPORT the expected nonzero violation/counterexample counts (their PASS criterion
  is "counterexample found as predicted", not "zero violations").
