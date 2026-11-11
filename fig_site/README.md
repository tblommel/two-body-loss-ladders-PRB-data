# fig_site — Fig. 5 of the paper (`fig:site`), single lossy site

Reproduces: all three panels — (a) n_σ(γt) from n₀ = 0.9, (b) D(γt) from n₀ = 0.5, (c) n_σ(γt = 2000) versus n₀ — and
the per-run summary used for the numbers of Sec. VII A (`quoted_numbers/sec7A.ipynb`).

Runs: 110 driver (MF, 2B, TPPX, mGW, DSL × γ = 1 × 22 initial fillings n₀), 43 exact (n₀ of the driver runs and the
grid n₀ = 0, 0.025, …, 1 for the exact curve of panel c).

Cost (measured, AMD EPYC 7643, `-P 32`): 0.31 core-h in total (driver 0.18, exact 0.14); 0.8 min wall
at `-P 32`; longest run 13 s (exact, n₀ = 1); peak RAM 55 MB per driver run, 81 MB per exact run, ≤ 2.5 GB in total
at `-P 32`. Output on disk ≈ 2.7 GB (`output/`, not needed after step 3).

## Steps (from this directory)

1. `python make_runcards.py` — optional: the cards in `runcards/` are committed. Rewrites `runcards/` (153 cards +
   `index.csv`) from the parameter block at the top of the script (< 10 s).
2. `./run.sh -P 32` — runs every card into `output/<run_id>/` (needs `../code_diagrammatics/build/dp5drive`, see
   `../code_diagrammatics/README.md`, and Python with numpy, scipy, h5py; set `PYTHON=...` if `python3` is not that
   interpreter). Finished runs are skipped when you rerun it; `./run.sh --dry-run` lists the runs. ~1 min at `-P 32`.
3. `python postprocess.py` — reads `output/`, writes `data/` (≈ 0.7 MB, ~1 min).
4. `jupyter nbconvert --execute --to notebook --inplace fig_site.ipynb` — draws `fig_site.pdf` from `data/` (5 s).
   Set `USE_OUTPUT = True` in its first cell to draw directly from a fresh `output/`.

## Data files (columns and units in `data/meta.json`)
- `curves.csv`: n_σ and D versus γt for n₀ = 0.5, 0.9, all methods and exact (every frame to γt = 10, then 700
  log-spaced frames).
- `endpoints.csv`: n_σ at γt = 2000 versus n₀ (panel c).
- `summary.csv`: per driver run: status, max deviation from the exact run (n, D, on identical frames), minima of n and
  D, extreme eigenvalues of ρ, max dN/dt, n at γt = 100, 500, 2000.
- `exact_ninf.csv`: exact t → ∞ occupation n₀(1 − n₀) from the zero-mode projection.

Expected agreement with the shipped data: T_default (≤ 1e-6 in n, D; a different compiler or machine changes the
driver results at the level of the integration tolerance). The exact runs agree with the analytic solution
n_σ(t) = n₀ − n₀²(1 − e^{−2γt}) to 1e-13.
