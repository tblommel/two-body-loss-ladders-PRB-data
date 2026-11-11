# tab_plaqerr — Table II of the paper (`tab:plaqerr`), plaquette error table and Sec. VII B numbers

Reproduces: every cell of the table — the time-averaged relative error ⟨|N(t) − N_ex(t)|⟩_t / N(0) over
0 ≤ tJ ≤ 200, averaged over the five loss rates for each scheme and initial state, and the mean over all trajectories
of a scheme — including the two excluded diverged trajectories (footnotes). The per-run record (`data/runs.csv`) also
holds the plaquette numbers quoted in Sec. VII B (errors of D, final N, minima of D, positivity of ρ and of the
two-body matrices P, Q, G′), read by `quoted_numbers/sec7B.ipynb`.

Runs: 175 driver (MF, 2B, TPPX, mGW, DSL × γ/J ∈ {0.1, 0.3, 1, 3, 10} × 7 initial states: Slater, CDW, mixed,
one pair, polarized, 3↑1↓, Néel), 35 exact (with the P/Q/G′ minima every 1/J). Two driver runs stop early with status
`Diverged` (exit 7: mGW 3↑1↓ at γ/J = 1, DSL Néel at γ/J = 10); they are data of the paper, kept with their status.

Cost (measured, AMD EPYC 7643, `-P 48`): 0.09 core-h (driver 0.05, exact 0.04); 0.3 min wall at `-P 48`;
longest run 10 s (exact, mixed γ/J = 10); peak RAM 29 MB per driver run, 150 MB per exact run, ≤ 4 GB in total at
`-P 48`.

## Steps (from this directory)

1. `python make_runcards.py` — optional (cards committed): rewrites `runcards/` (210 cards + `index.csv`).
2. `./run.sh -P 48` — runs every card into `output/<run_id>/` (≈ 0.5 min; finished runs are skipped on a rerun).
3. `python postprocess.py` — `output/` → `data/` (≈ 1 min: the P/Q/G′ minima are computed from the driver
   checkpoints).
4. `jupyter nbconvert --execute --to notebook --inplace tab_plaqerr.ipynb` — writes `tab_plaqerr.tex` (the tabular of
   the paper) and prints it.

## Data files (columns and units in `data/meta.json`)
- `runs.csv`: one row per driver run: status, err_N (table metric), err_D (mean over tJ ≤ 20 of |D − D_ex| / D_ex(0)),
  final N, minima of D, extreme eigenvalues of ρ, max dN/dt, ⟨S²⟩ drift, minima of the P, Q, G′ eigenvalues.
- `table.csv`: the table cells with the number of averaged and excluded trajectories.
- `exact.csv`: exact N(0), N(200), D(0), ⟨S²⟩ drift and P/Q/G′ minima per trajectory.

Expected agreement with the shipped data: T_default (≤ 1e-6 on trajectories, 1e-5 on P/Q/G′ minima); the table cells
reproduce the printed two significant digits. The two diverging runs agree up to the last frames before the stop.
