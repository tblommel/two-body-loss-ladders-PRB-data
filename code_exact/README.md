# code_exact — exact Lindblad solver

Exact reference dynamics for the paper's open Hubbard-type models: a single site, the 2×2 plaquette, and impurity models
with pseudomodes. One run = one JSON run card in, one HDF5 result file out. Pure Python (numpy, scipy, h5py). It needs
nothing else from this repository.

## 1. Install

```
conda env create -f environment.yml    # from the release root
conda activate two-body-loss-ladders
```
Any Python ≥ 3.10 with numpy ≥ 1.26, scipy ≥ 1.12 (needs `expm_multiply` with time grids and `gmres(rtol=...)`) and
h5py also works. The pinned versions (numpy 2.4.1, scipy 1.17.0, h5py 3.15) are the ones the shipped data were made with.

## 2. Run one card

```
python -I code_exact/run_exact.py path/to/card.json path/to/outdir/
```
- writes `outdir/result.h5` (the name is `output.file` in the card) and `outdir/progress.txt` (one line, rewritten
  every few seconds: task, percent done, elapsed time, ETA);
- prints one line per task to stdout;
- exit code 0 = Ok; 1 = runtime failure; 2 = invalid card or command line. For 1 and 2 the result file is still
  written, with root attribute `status = "Error"` and the reason in `message`.

Use one thread per run (the figure directories' `run.sh` does this): `export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.

## 3. What is solved

$$\dot\rho = -i[H,\rho] + \sum_L \big(2L\rho L^\dagger - \{L^\dagger L,\rho\}\big),\qquad
H = \sum_\sigma c^\dagger_\sigma h_1 c_\sigma ,$$
with jumps $\sqrt{G_r}\,c_{r\sigma}$ (one-body loss), $\sqrt{G_r}\,c^\dagger_{r\sigma}$ (one-body gain), both spins, and the
pair loss $\sqrt{\gamma w_r}\,c_{r\downarrow}c_{r\uparrow}$ on the listed sites. Fermions; 1RDM
$\rho^\sigma_{sa} = \langle c^\dagger_{a\sigma} c_{s\sigma}\rangle$; spin index 0 = ↑, 1 = ↓.

Method. Every jump shifts the sector label $(N_\uparrow, N_\downarrow)$ by a fixed vector, so the density matrix of a
state that starts block diagonal in $(N_\uparrow, N_\downarrow)$ stays block diagonal. The solver works only with
those blocks (the *even space*), and only with the sectors reachable from the initial state; the Green's function
works in the *odd space* of blocks $(a + e_\sigma, a)$. The Lindblad superoperator is assembled block by block as a
sparse matrix; nothing of size (Fock dimension)² is ever built. Time evolution is the exact exponential
(`scipy.sparse.linalg.expm_multiply`, no time step error) or a dense eigendecomposition.

| system | n_orb | Fock dim | even dim (all sectors) | odd dim |
|---|---|---|---|---|
| single site | 1 | 4 | 4 | 2 |
| toy impurity (d + 1 greater + 1 lesser mode) | 3 | 64 | 400 | 300 |
| plaquette (4 sites) | 4 | 256 | 4900 | 3920 |
| impurity c3 and its weakly damped variant (d + 2 + 2 modes) | 5 | 1024 | 63504 | 52920 |

## 4. Run card reference (schema `exact_card/1`)

All keys are required unless marked optional; any unknown key is an error (exit 2). Arrays are nested lists.

```json
{
  "schema": "exact_card/1",
  "run_id": "plaq_mixed_g1_exact",
  "model": {
    "name": "plaq",
    "n_orb": 4,
    "h1": [[0, -1, 0, -1], [-1, 0, -1, 0], [0, -1, 0, -1], [-1, 0, -1, 0]],
    "onebody_loss": [0, 0, 0, 0],
    "onebody_gain": [0, 0, 0, 0],
    "pair_loss": {"sites": [0, 1, 2, 3], "gamma": 1.0, "weights": [1, 1, 1, 1]}
  },
  "initial_state": {"type": "gaussian",
                    "rho1_up": [[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 0.5]],
                    "rho1_dn": [[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 0.5]]},
  "tasks": [ ... see below ... ],
  "output": {"file": "result.h5"}
}
```

| key | meaning |
|---|---|
| `model.n_orb` | orbitals per spin (1–6) |
| `model.h1` | real symmetric `n_orb × n_orb`, the same for both spins |
| `model.onebody_loss`, `onebody_gain` | rate $G_r \ge 0$ per orbital |
| `model.pair_loss` | `sites` (orbital indices), `gamma`, `weights` (optional, default 1) |
| `initial_state` | `gaussian` only: the uncorrelated state with spin-diagonal 1RDM blocks `rho1_up`, `rho1_dn` (real symmetric, spectrum in [0, 1]); all cumulants beyond the 1RDM vanish |
| `tasks` | list, run in order; each has a unique `name` (= its HDF5 group) |

### Observables (`observables` lists of the tasks)

| name | shape (transient) | meaning |
|---|---|---|
| `N` | [T] | total particle number |
| `N_sigma` | [T, 2] | $N_\uparrow, N_\downarrow$ |
| `n` | [T, 2, N] | $\rho^\sigma_{rr}$ |
| `D` | [T, N] | $\langle n_{r\uparrow} n_{r\downarrow}\rangle$ for every orbital |
| `S2` | [T] | total spin $\langle \mathbf S^2\rangle$ |
| `Mz` | [T] | $(N_\uparrow - N_\downarrow)/2$ |
| `rho1` | complex [T, 2, N, N] | $\rho^\sigma_{sa}$ |
| `rho1_eig` | [T, 2N] | eigenvalues of both spin blocks, ascending |

NESS and n_inf store the same names without the T axis.

### Task `transient`
```json
{"type": "transient", "name": "transient",
 "grid": [{"t_end": 20.0, "dt": 0.005}, {"t_end": 3000.0, "dt": 1.0}],
 "observables": ["N", "n", "D", "S2", "rho1_eig"],
 "pqg_every": 1.0,
 "method": "expm_multiply", "chunk": 200}
```
- `grid`: consecutive segments from t = 0; each segment's (t_end − t_start)/dt must be an integer.
- `pqg_every` (optional, default 0 = off): every that many time units the minimum eigenvalues of the two-body Gram
  matrices P (pairs), Q (holes), G′ (particle–hole) and Var N (`code_exact/rdm_tools.py`). Must be a multiple of
  every segment's `dt`.
- `method` (optional): `expm_multiply` (default and recommended for every card; exact exponential on the grid,
  error ~1e-13, any γ; the plaquette Zeno grid to t = 3000 takes 0.5–2 min) or `eig` (dense eigendecomposition of
  the even space, ≤ 8000). `eig` is kept as a cross-check only: its error is set by the conditioning of the
  eigenvector matrix and reaches ~1e-9 on the plaquette (mixed start, small γ).
- `chunk` (optional, default 200): frames per `expm_multiply` call (memory: chunk × even dim × 16 bytes).

Output group: `t` [T], one dataset per observable, `pqg/{t, P_min, Q_min, Gp_min, varN}`; attrs `max_trace_err`, `dim`.

### Task `ness` (non-equilibrium steady state)
```json
{"type": "ness", "name": "ness", "method": "propagate",
 "propagate": {"block": 50.0, "tol": 1e-15, "t_max": 3000.0},
 "observables": ["N", "N_sigma", "n", "D", "rho1_eig"], "pqg": true, "spectrum": {"k": 8}}
```
- `method`: `lu` (sparse LU with one equation replaced by the trace condition; fast for the small models, too
  slow above an even dimension of ~10⁴), `propagate` ($e^{St}$ from the initial state in blocks of `block` until the
  largest change of the state over one block is below `tol`, at most `t_max`), or `auto` (`lu` up to even dimension
  10⁴, else `propagate`; `propagate` must then be given).
- `pqg` (optional, default false): P/Q/G′ minima and Var N of the NESS.
- `spectrum` (optional): the `k` Liouvillian eigenvalues with the largest real part (dense below even dimension 5000,
  Arnoldi above), stored as `spectrum/eigs`; attrs `n_zero_modes`, `gap`.
- The NESS must be unique (one-body loss/gain present); with pair loss only use `n_inf`.

Output: observables, `pqg/…`; attrs `residual` (max |S x|), `method`, `t_prop`, `converged`, `mb_eig_min`
(smallest eigenvalue of the many-body density matrix).

### Task `n_inf` (exact t → ∞ limit)
```json
{"type": "n_inf", "name": "n_inf", "observables": ["N", "D", "S2"]}
```
Spectral projection of the initial state onto the zero modes of the Liouvillian, $P = R(L^\dagger R)^{-1}L^\dagger$
with the right/left null spaces $R$, $L$ from one dense SVD of the even space (≤ 8000; the zero eigenvalue of a
Lindblad generator is semisimple). Used for the plaquette, whose pair-loss-only dynamics keeps dark states. Agrees with
the t = 3000 value of the propagated state to ~1e-12. Attrs `n_zero_modes` (|λ| < 1e-9), `n_oscillating_modes`
(undamped modes with Re λ = 0, Im λ ≠ 0: present on the plaquette; they do not enter N, D, S² but mean the state
itself keeps oscillating), `gap` (smallest −Re λ among the damped modes), `residual`, `sv_null_max`, `sv_next`.

### Task `greens_R` (retarded Green's function at the NESS)
```json
{"type": "greens_R", "name": "GR", "state": "ness", "orbital": 0, "spins": ["up", "dn"],
 "tau_max": 400.0, "dtau": 0.01,
 "resolvent": {"omega_min": -6.0, "omega_max": 6.0, "n": 25, "solver": "gmres", "rtol": 1e-13}}
```
$G^R_{r\sigma}(\tau) = -i\,\mathrm{Tr}\big[c_{r\sigma}\,e^{S_{\rm odd}\tau}\{c^\dagger_{r\sigma},\rho_{\rm NESS}\}\big]$
(quantum regression; in the odd space the one-body jumps enter with a minus sign). Needs a `ness` task earlier in the
list. `resolvent` (optional): $G^R(\omega) = i\,w\cdot(S_{\rm odd} + i\omega)^{-1}y$ on a uniform ω grid, an
independent check of a Fourier transform of $G^R(\tau)$; `solver` `eig` (dense, small models) or `gmres` (`rtol`
optional, default 1e-13). Output: `tau` [n], `GR` complex [n, n_spins], `resolvent/{omega, GR, info}`.

### Output file `result.h5` (schema `exact_result/1`)
Root attrs: `schema`, `run_id`, `release_version` (`<VERSION>+<git describe>` or `+nogit`), `status` (`Ok` | `Error`),
`message`, `wall_seconds`, `card_json` (the card verbatim), `python_versions`. Group `/model` attrs `n_orb`,
`dim_fock`, `dim_even`, `dim_odd`, `even_dim_used`. One group per task with its card parameters and `wall_seconds`
as attributes.

Read it with h5py:
```python
import h5py
with h5py.File("outdir/result.h5") as f:
    assert f.attrs["status"] == "Ok"
    t, N = f["transient/t"][()], f["transient/N"][()]
```

## 5. Tests
```
OMP_NUM_THREADS=1 python -I code_exact/tests/test_exact.py   # about 15 s, prints ALL PASSED
```
1. single site against the analytic solution (every observable, Var N, G′, n_inf; several n₀ and γ up to 100; both
   time-stepping methods), to 1e-12;
2. Gaussian initial state: 1RDM reproduced, two- and three-body cumulants zero, equal to an independent full-Fock-space
   construction;
3. trace preservation and Hermiticity;
4. sector-restricted solver vs a dense full-space Liouvillian on a two-orbital model with all jump types (transient,
   NESS by LU and by propagation, G^R(τ), resolvent), to 1e-12;
5. resolvent: eig vs gmres, and vs the Fourier transform of G^R(τ);
6. card validation and the command line (exit codes, file layout).

## 6. Cost (one core, AMD EPYC 7643; measured)

Measured on an AMD EPYC 7643 host, one core per run, with 80–94 such runs in parallel on the host (memory
bandwidth shared; an idle machine is faster). Peak RAM = maximum resident set size of the process.

| card (paper system) | even dim | wall per card | peak RAM |
|---|---|---|---|
| single site: transient to γt = 2000 (40001 frames, all observables, P/Q/G′) + n_inf | 4 | 15 s | 85 MB |
| plaquette, Table II / Fig. 6: transient T = 200, dt 0.1, all observables, P/Q/G′ every 1 (γ ≤ 10) | 257–4900 | 3–12 s | 150 MB |
| plaquette, γ = 100: same, plus fine segment dt 0.001 to t = 1 | 4900 | 55 s | 150 MB |
| plaquette, Zeno: transient fine (dt 0.005; 0.001 for γ ≥ 30) to 20 + long (dt 1) to 3000 + n_inf | 1553 (Néel, Slater), 4900 (mixed) | 20 s – 13 min (median 40 s; mixed: n_inf 2.5–5.5 min) | 2.1 GB (n_inf of the mixed start) |
| toy impurity, IC 00: NESS (lu) + transient 150 + G^R(τ ≤ 800, 2 spins) + resolvent (eig) | 400 | 15–22 s | 85 MB |
| toy impurity, IC 11 / 10: transient 150 | 400 | 3 s | 80 MB |
| c3, IC 00: NESS (propagate, converges at T = 100–150) + transient 150 (dt 0.01) + G^R(τ ≤ 400, 2 spins) + 25 gmres frequencies | 63504 (odd 52920) | 32–61 min | 0.9 GB |
| c3, IC 11 / 10: transient 150 (dt 0.01), P/Q/G′ every 1 | 63504 | 7–23 min (slower at small γ) | 0.9 GB |
| c3 weakly damped (all mode rates 0.3), IC 00: as c3 IC 00 with τ ≤ 800 | 63504 | 34 min | 0.9 GB |

Paper total (site 22 cards, plaquette 35 + 42 Zeno cards, toy 48, c3 33): about 16 core-hours, dominated by c3.
The sparse LU NESS is impractical for c3 (even dimension 63504: did not finish in 10 min, fill-in); use `propagate`
(or `auto`).

## 7. Files
- `run_exact.py` — command line, result file, version stamp.
- `lindblad_exact/fock.py` — Fock space, model operators, sector labels.
- `lindblad_exact/superop.py` — block spaces, sparse superoperators, linear functionals.
- `lindblad_exact/states.py` — Gaussian initial states.
- `lindblad_exact/observables.py` — observables, 2RDM, P/Q/G′.
- `lindblad_exact/card.py` — strict card reader.
- `lindblad_exact/tasks.py` — the four tasks, progress file.
- `rdm_tools.py` — positivity matrices P, Q, G′ (also used by `common/positivity.py`).
- `tests/` — tests and small card builders.
