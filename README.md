# two-body-loss-ladders-PRB-data

Code and data reproducing every figure, table and quoted number of

> T. Blommel and V. Vlcek, *Ladder Resummations and Time-Linear Green's Functions for Two-Body Loss and Gain in
> Open Quantum Systems* (2026).

License: MIT (see `LICENSE`). Citation: see `CITATION.cff`.

## Contents

| directory | reproduces |
|---|---|
| `code_diagrammatics/` | C++ solver `dp5drive` for the G1-G2 and RTDE equations (HF, 2B, TPPX, TPPX_0, mGW, DSL) |
| `code_exact/` | exact Lindblad solver in Python (sector-restricted superoperators) |
| `common/` | shared Python: models, run-card writers, readers, Σ tools, plot style, card runner |
| `fig_site/` | Fig. 5 (single lossy site, Sec. VII A) |
| `fig_plaq/` | Fig. 6 (plaquette dynamics, Sec. VII B) |
| `tab_plaqerr/` | Table III (plaquette errors, Sec. VII B) |
| `fig_zeno/` | Fig. 7 (Zeno crossover, Sec. VII B) |
| `fig_impss/` | Fig. 8 (impurity steady state and spin relaxation, Sec. VII C) |
| `fig_sigma/` | Fig. 9 (impurity self-energy, Sec. VII C) |
| `quoted_numbers/` | every number quoted in Secs. VII A–C and App. A (one notebook per section) |

Each figure/table directory holds committed run cards (`runcards/`), the processed data shown in the paper
(`data/`), a notebook that draws the figure (or prints the numbers) from `data/`, the figure itself, and the scripts
that regenerate everything from scratch (`make_runcards.py`, `run.sh`, `postprocess.py`). Its own `README.md` lists
the runs, the data files and the measured cost.

## 1. Environment

Install conda (e.g. [Miniforge](https://github.com/conda-forge/miniforge)), then from this directory:

    conda env create -f environment.yml     # once, ~5 min; environment.yml lists only conda-forge
    conda activate two-body-loss-ladders

The environment contains Python (numpy, scipy, matplotlib, h5py, jupyter) and the C++ toolchain for the solver
(compiler, CMake, HDF5, Eigen). Linux and macOS; x86-64 or arm64.

## 2. Redraw a figure from the shipped data (seconds)

    cd fig_plaq && jupyter nbconvert --execute --to notebook --inplace fig_plaq.ipynb      # -> fig_plaq.pdf

Likewise for every directory; the notebooks in `quoted_numbers/` print each quoted number next to its value in the
paper.

## 3. Build and test the solver (~2 min + ~15 min of tests)

With the environment active, from this directory:

    cmake -S code_diagrammatics -B code_diagrammatics/build -DCMAKE_BUILD_TYPE=Release
    cmake --build code_diagrammatics/build -j
    code_diagrammatics/build/dp5drive --version

Tests (each ends with `ALL ... PASSED`; `test_dp5` uses every hardware thread unless `OMP_NUM_THREADS` is set):

    export OMP_NUM_THREADS=4
    cd code_diagrammatics
    build/test_dp5
    for t in run_tests test_step2 test_step3 test_step3b test_step7a test_step7b test_step8 test_step9; do
        python -I tests/$t.py || break; done
    cd ..
    python -I code_exact/tests/test_exact.py

The default build is portable (no `-march=native`). See `code_diagrammatics/README.md` for build options, run-card
keys, the output format and system-package builds; `code_exact/README.md` for the exact solver.

## 4. Regenerate everything from scratch

    ./run_all.sh -P 32        # 32 runs in parallel; default: all cores

For each directory this runs every card (`run.sh` → `output/`), processes the outputs (`postprocess.py` → `data/`,
overwriting the shipped data) and executes the notebooks (→ figures and printed numbers). `./run_all.sh -P 32 fig_site`
does one directory. Finished runs are skipped when the script is restarted. Directories run one after another; to
fill idle cores during the long `fig_impss` tail, `run.sh` of a *different* directory can run alongside. Never run two
`run.sh` on the same directory at once: each restarts (and deletes) the other's unfinished outputs.

Cost (measured on AMD EPYC 7xx3/7502 cores, one core per run):

| directory | driver + exact runs | core-hours | longest single run | peak RAM per run |
|---|---|---|---|---|
| `fig_site` | 110 + 43 | 0.3 | 13 s | 0.1 GB |
| `fig_plaq` | 20 + 4 | 0.01 | 5 s | 0.1 GB |
| `tab_plaqerr` | 175 + 35 | 0.1 | 10 s | 0.2 GB |
| `fig_zeno` | 168 + 28 | 1.3 | 6 min | 2.0 GB |
| `fig_impss` | 330 + 66 | 27 | 72 min | 0.9 GB |
| `fig_sigma` | 18 + 3 | 1.4 | 48 min | 0.7 GB |
| `quoted_numbers` | 24 + 4 | 0.4 | 13 min | 2.1 GB |
| total | 845 + 183 | ≈ 31 | | |

The long exact impurity runs are memory-bandwidth bound: on an Intel Xeon Gold 6254 node with 34 runs in parallel the
`fig_impss` exact runs took 1.4–3 h each, the total was about 60 core-hours, and the full regeneration took 3.6 h.
`fig_impss/postprocess.py` reads 18 GB of outputs and takes about 15 min on a network file system.

Wall time is set by the longest runs: about 1.5–2 h with ≥ 64 cores, about a day on 8 cores. Keep `-P` ≤ (free RAM)
/ 2.1 GB. The outputs take about 22 GB of disk (`*/output/`, not part of the repository; `fig_impss` alone 18 GB) and can be
deleted after the postprocessing.

**Expected agreement with the shipped data.** On a different machine or compiler the results agree with the shipped
`data/` to the integrator tolerances, not bitwise: about 1e-6 for the plaquette and single-site cards, 1e-8 for the
impurity cards, 1e-10 for the exact solver. Every number printed in the paper is reproduced to its printed digits.
Two kinds of runs stop early by design (exit codes 5 and 7: divergence or loss of positivity, as stated in the
paper); `run.sh` counts them as finished. The steady-state detection time t′ of the impurity runs can shift between
machines when the state is already stationary to round-off; the values at t′ are unaffected.

## 5. Conventions

Fermions, Lindblad equation dρ/dt = −i[H, ρ] + Σ_L (2LρL† − {L†L, ρ}), pair loss L_r = √(γ w_r) c_{r↓}c_{r↑}; hopping
J = 1 (plaquette) and the impurity units of App. A set the energy unit. Method labels: `HF` (mean field, MF in the
paper), `2B`, `TPPX`, `TPPX_0`, `mGW`, `DSL`, `exact`. Data files are CSV (curves, tables) and HDF5 (G^R, Σ^R), with
columns and units in each `data/meta.json`.

## Versions

Every output file records the release version (`VERSION`) and the git commit of the build.
