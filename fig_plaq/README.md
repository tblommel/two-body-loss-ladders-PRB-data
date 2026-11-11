# fig_plaq — Fig. 6 of the paper (`fig:plaq`), lossy 2×2 plaquette dynamics

Reproduces: all panels — N(t) from the mixed state at γ/J = 3 with the ⟨S²⟩(t) inset (a), N(t) from the half-filled
Slater determinant at γ/J = 10 (b), total double occupancy D(t) from one on-site pair at γ/J = 1 (c) and from the CDW
at γ/J = 3 (d).

Runs: 20 driver (MF, 2B, TPPX, mGW, DSL × the four (initial state, γ) pairs of the panels), 4 exact.

Cost (measured, AMD EPYC 7643, `-P 24`): 0.01 core-h; 0.2 min wall at `-P 24`; longest run 5 s (exact,
mixed γ/J = 3); peak RAM 29 MB per driver run, 135 MB per exact run, < 1 GB in total.

## Steps (from this directory)

1. `python make_runcards.py` — optional (cards committed): rewrites `runcards/` (24 cards + `index.csv`).
2. `./run.sh -P 24` — runs every card into `output/<run_id>/` (needs `../code_diagrammatics/build/dp5drive` and a
   Python with numpy, scipy, h5py; `PYTHON=...` selects it). < 1 min.
3. `python postprocess.py` — `output/` → `data/curves.csv` (≈ 4 MB, ~10 s).
4. `jupyter nbconvert --execute --to notebook --inplace fig_plaq.ipynb` — draws `fig_plaq.pdf` (5 s);
   `USE_OUTPUT = True` in the first cell draws from a fresh `output/`.

## Data files (columns and units in `data/meta.json`)
- `curves.csv`: N, D = Σ_r D_r and ⟨S²⟩ versus t (every output frame, Δt = 0.1/J, t ≤ 200/J) for each panel's
  initial state and γ, all methods and exact.

Expected agreement with the shipped data: T_default (≤ 1e-6; driver results change at the level of the integration
tolerance with the compiler/machine); exact runs are reproducible to ~1e-12.
