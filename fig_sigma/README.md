# fig_sigma — Fig. 9 of the paper (`fig:sigma`), retarded self-energy of the strongly damped impurity

Reproduces: all six panels — −Im Σ^R(ω) and Re Σ^R(ω) of the impurity (spin ↑) in the steady state reached from the
empty impurity, at γ = 4, 8, 16, for exact, MF, TPPX, TPPX₀, DSL — and the numbers of Sec. VII C and the caption
(shape of Σ^R via `quoted_numbers/postprocess.py`; high-frequency limit γ n_d↓).

Runs: 18 driver (MF, 2B, TPPX, TPPX₀, mGW, DSL × γ ∈ {4, 8, 16}; RTDE from the detected t′ with the automatic RTDE
end, tight tolerances), 3 exact (NESS by propagation, G^R(τ ≤ 400) by quantum regression, and G^R(ω) at 25
frequencies by the resolvent as an independent check of the Fourier transform).

Cost (measured, AMD EPYC 7502, `-P 21`): 1.35 core-h (driver 0.01, exact 1.34); 48 min wall at `-P 21`
(set by the longest run, 48 min: exact γ = 4); peak RAM 41 MB per driver run, 0.7 GB per exact run, < 3 GB in total.

## Steps (from this directory)

1. `python make_runcards.py` — optional (cards committed): rewrites `runcards/` (21 cards + `index.csv`).
2. `./run.sh -P 21` — runs every card into `output/<run_id>/` (~50 min, dominated by the three exact runs).
3. `python postprocess.py` — `output/` → `data/` (~1 min): G^R(ω) by piecewise-cubic Filon quadrature of G^R(τ)
   (η = 0, to the run's own τ_end), Σ^R = 1/G₀ − 1/G^R with the exact γ = 0 propagator G₀.
4. `jupyter nbconvert --execute --to notebook --inplace fig_sigma.ipynb` — draws `fig_sigma.pdf` (5 s).

## Data files
- `sigma.h5`: groups `/<method>/c3_ic00_g<γ>/` with `tau`, `GR` [n, 2] (G^R(t′+τ, t′), both spins, every 10th
  sample), `omega`, `GR_w`, `Sigma` [m, 2]; exact also `omega_resolvent`, `GR_resolvent`; attributes γ, t′, τ_end,
  dt, η (layout in `data/meta.json`).
- `summary.csv`: per run: status, t′, τ_end, n_d per spin at the steady state, γ n_d↓, Σ↑(ω → ∞) from G^R′(0).

Expected agreement with the shipped data: T_tight (1e-8 on G^R(τ), 1e-6 on Σ(ω)) on a different compiler or machine;
t′ and τ_end within the detector windows (T_detect).
