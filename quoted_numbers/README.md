# quoted_numbers — every number quoted in the text of Sec. VII and Appendix A

One notebook per results section prints each quoted number next to the paper's printed string, with a match flag:

| notebook | section | reads |
|---|---|---|
| `sec7A.ipynb` | VII A, single lossy site (and the caption of Fig. 5) | `../fig_site/data` |
| `sec7B.ipynb` | VII B, plaquette (and the captions of Figs. 6, 7) | `../tab_plaqerr/data`, `../fig_plaq/data`, `../fig_zeno/data`, `data/` |
| `sec7C.ipynb` | VII C, impurity (and the captions of Figs. 8, 9) | `../fig_impss/data`, `../fig_sigma/data`, `data/` |
| `appA.ipynb` | Appendix A, Table of the impurity models | `../common/models.py` |

The numbers that no figure or table needs come from this directory's own runs:

Runs: 24 driver (stability of the MF and TPPX steady states of the strongly damped impurity: G1-G2 to t = 600 from a
slightly polarized impurity, M_z(0) = 5×10⁻⁴, and from the polarized start; MF at γ ∈ {4, 5, 6, 7, 8, 12, 24}, TPPX
at γ ∈ {12, 16, 20, 22, 24}), 4 exact (plaquette, mixed and Néel starts, γ/J = 50 and 100, for γκ/J²).

Cost (measured, AMD EPYC 7643, `-P 28`): 0.41 core-h (driver 0.005, exact 0.40); 12.6 min wall at
`-P 28` (set by the longest run, 12.6 min: exact mixed start, γ/J = 100); peak RAM 24 MB per driver run, 2.1 GB per
exact run, ≤ 5.2 GB in total.

## Steps (from this directory)

1. Produce the data of the other directories first (their `data/` is shipped, so this is only needed for a full
   rerun).
2. `python make_runcards.py` — optional (cards committed).
3. `./run.sh -P 28` — runs the 28 cards into `output/`.
4. `python postprocess.py` — `output/` (and `../fig_sigma/data`) → `data/`: `stability.csv`, `kappa_biggamma.csv`,
   `sigma_features.csv`, `sigma_extrema.csv` (columns in `data/meta.json`).
5. `jupyter nbconvert --execute --to notebook --inplace sec7A.ipynb sec7B.ipynb sec7C.ipynb appA.ipynb`.

Expected agreement: every quoted number reproduces its printed digits (the notebooks print `match`).
