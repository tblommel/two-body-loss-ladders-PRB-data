# fig_impss — Fig. 8 of the paper (`fig:impss`), impurity steady state and spin relaxation

Reproduces: all panels — steady-state impurity occupation n_d versus γ from the empty impurity (a, c) and the
impurity magnetization M_z^d(t) from the singly occupied (↑) impurity at γ = 24 (b, d), for the strongly damped
model (top row, system token `c3`) and the weakly damped model (bottom row, `c3G0.2`: all four mode rates Γ = 0.2) —
and the per-run record used for the numbers of Sec. VII C (`quoted_numbers/sec7C.ipynb`).

Runs: 330 driver (MF, 2B, TPPX, mGW, DSL × 2 models × 11 γ ∈ {0, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24} × 3 impurity
starts: empty, doubly occupied, singly occupied ↑; RTDE runs with the automatic t′ and RTDE end, tight tolerances),
66 exact (per model and γ: NESS, transient and G^R(τ) from the empty impurity; transients from the other two
starts). One driver run stops early with status `Diverged` (mGW, weakly damped model, doubly occupied start, γ = 3),
as in the paper's data.

Cost (measured, AMD EPYC 7713, `-P 100`): 27 core-h (driver 1.1, exact 26); 72 min wall at `-P 100`
(set by the longest run, 72 min: exact NESS + G^R of the weakly damped model at γ = 8); peak RAM 0.3 GB per driver
run and 0.9 GB per exact run, ≤ 63 GB in total at `-P 100` (use `-P` ≤ RAM / 0.9 GB). Output on disk ≈ 30 GB.

## Steps (from this directory)

1. `python make_runcards.py` — optional (cards committed): rewrites `runcards/` (396 cards + `index.csv`).
2. `./run.sh -P 100` — runs every card into `output/<run_id>/` (~1.2 h at `-P 100`; with fewer cores the 66 exact
   runs of 10–70 min dominate; finished runs are skipped on a rerun).
3. `python postprocess.py` — `output/` → `data/` (~15 min: the P/Q/G′ minima from the driver checkpoints).
4. `jupyter nbconvert --execute --to notebook --inplace fig_impss.ipynb` — draws `fig_impss.pdf` (5 s).

## Data files (columns and units in `data/meta.json`)
- `steady.csv`: one row per run: steady state (driver: frame at t′; exact: NESS for the empty start, state at t = 150
  otherwise), t′, τ_end, minima of D and n, extreme eigenvalues of ρ, P/Q/G′ minima, M_z 1/e time and last time with
  M_z > 0.05 (singly occupied start).
- `mz_curves.csv`: n_d↑, n_d↓, M_z versus t ≤ 20 from the singly occupied start, every 5th output frame.

Expected agreement with the shipped data: T_tight (1e-8) on a different compiler or machine; t′ and τ_end within the
detector windows (T_detect).
