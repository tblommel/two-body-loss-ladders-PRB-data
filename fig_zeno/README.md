# fig_zeno — Fig. 7 of the paper (`fig:zeno`), quantum Zeno crossover on the plaquette

Reproduces: all panels — loss coefficient κ versus γ/J from the mixed (a) and Néel (b) starts, the correlation-hole
factor D/O from the mixed start (c) and O = Σ_r n_r↑ n_r↓ from the Néel start (d), as means over the κ window — and
the κ and window values quoted in Sec. VII B (`quoted_numbers/sec7B.ipynb`).

Runs: 168 driver (MF, TPPX, DSL × 2 initial states × 14 γ/J ∈ {0.1, …, 30} × 2 segments: `fine`, t ≤ 20/J with
Δt = 0.005, and `long`, t ≤ 3000/J with Δt = 1), 28 exact (transient on a grid containing both driver grids, the
t → ∞ limit by zero-mode projection, and for the mixed start at γ/J ≥ 10 a short fine transient for D/O). Six DSL
runs from the Néel start at γ/J ≥ 10 stop early (exit 5 or 7: the DSL diverges there, as stated in the paper).

Cost (measured, AMD EPYC 7643, `-P 48`): 1.3 core-h (driver 0.07, exact 1.2); 6.1 min wall at `-P 48`;
longest run 6 min (exact, mixed start, γ/J = 30: the dense zero-mode projection of a 4900-dimensional space); peak RAM
27 MB per driver run and 2.0 GB per exact run (all exact runs), ≤ 32 GB in total at `-P 48` (use a smaller `-P` on a
machine with less memory: the 28 exact runs need 2 GB each).

## Steps (from this directory)

1. `python make_runcards.py` — optional (cards committed): rewrites `runcards/` (196 cards + `index.csv`).
2. `./run.sh -P 48` — runs every card into `output/<run_id>/` (~6 min at `-P 48`; finished runs are skipped).
3. `python postprocess.py` — `output/` → `data/` (~1 min): N_∞, κ and the window means (definitions in the docstring
   of `postprocess.py`).
4. `jupyter nbconvert --execute --to notebook --inplace fig_zeno.ipynb` — draws `fig_zeno.pdf` (5 s).

## Data files (columns and units in `data/meta.json`)
- `kappa.csv`: N_∞ (and how it was obtained) and κ per method, initial state, γ.
- `window.csv`: window bounds t₁, t₂ and the window means of O, D, D/O, the staggering S and A = O/(N²/16).

Expected agreement with the shipped data: T_default — relative changes of κ and of the window means ≤ 1e-6 for a
different compiler or machine; exact N_∞ to 1e-9.
