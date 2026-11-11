# code_diagrammatics — the `dp5drive` driver

`dp5drive` integrates the equations of motion of the paper for a fermionic open system with two-body (pair) loss:

* the **G1-G2 scheme** (equal-time one-body density matrix ρ plus a two-particle correlator), and
* the **RTDE** (real-time Dyson equation) for the retarded Green's function G^R(t′ + τ, t′), co-propagated with the
  G1-G2 system from a time t′ (fixed, or detected automatically when the G1-G2 state becomes stationary),

for the approximations

| label | run card | correlator blocks |
|---|---|---|
| `HF` | `approximation = HF` | — (mean field) |
| `2B` | `approximation = TPPX`, `flag_2B = 1` | particle-particle, no ladder memory |
| `TPPX` | `approximation = TPPX` | particle-particle T-matrix |
| `TPPX_0` | `approximation = TPPX`, `bare_propagators = 1` | as TPPX, correlator drifts with the bare h_o |
| `mGW` | `approximation = mGW` | particle-hole chain + its second-order counterterm |
| `DSL` | `approximation = DSL` | particle-particle + particle-hole rungs (C₃ = 0 closure) |

Fermions (ξ = −1), Lindblad form dρ/dt = −i[H, ρ] + Σ_L (2LρL† − {L†L, ρ}), pair loss L_r = √(γ w_r) c_{r↓}c_{r↑} on the
lossy sites of the model, one-body loss/gain on any orbitals. One output HDF5 file per run.

This directory is self-contained: the Dormand–Prince integrator headers are vendored in `integrator/`, the test
references in `tests/ref/`, and CMake builds from this directory alone.

```
CMakeLists.txt          build (portable flags by default); cmake/gen_version.cmake: version stamp
src/                    the driver (main.cpp: CLI and run loop; inputs: model + run card; physics.cpp, tensor.hpp:
                        equations; observables.cpp; steady_state.cpp: t' detector and automatic RTDE end;
                        output.cpp, h5util.cpp: HDF5 output; version.cpp)
integrator/             dp5_integrator.hpp, dp5_profile.hpp (header-only), DESIGN.md, tests/test_dp5.cpp
python/                 dp5model.py (model files: writer, validator, content hash), runcard.py (run-card writer),
                        read_output.py (status / profile summary, spectral function)
tests/                  test suites (Python) and tests/ref/ (independent full spin-orbital reference equations)
```

---------------------------------------------------------------------------------------------------------------------

## 1. Build

You need a C++17 compiler with OpenMP, CMake ≥ 3.16, HDF5 with the C++ bindings, and Eigen ≥ 3.3.
For the tests also Python 3 with numpy, scipy, h5py.

### Option A: conda (recommended; same versions as the paper's runs)

From the release root (the directory that holds `environment.yml`):

```bash
conda env create -f environment.yml --override-channels -c conda-forge     # once, ~5 min
conda activate two-body-loss-ladders
cmake -S code_diagrammatics -B code_diagrammatics/build -DCMAKE_BUILD_TYPE=Release
cmake --build code_diagrammatics/build -j
code_diagrammatics/build/dp5drive --version
```

The build takes about 1–2 minutes. It produces `code_diagrammatics/build/dp5drive` (the path the figure
directories' `run.sh` use by default) and the integrator unit test `code_diagrammatics/build/test_dp5`.

Note: activating a conda environment with `cxx-compiler` sets `CXXFLAGS` (e.g. `-march=nocona -mtune=haswell -O2 …`),
which CMake picks up. These flags are portable to every x86-64 machine and are recorded in every output file
(`/provenance/build_flags`). To build with exactly `-O3 -DNDEBUG`, run `unset CXXFLAGS` before the first `cmake`.

### Option B: system packages

```bash
# Debian / Ubuntu
sudo apt install build-essential cmake libhdf5-dev libeigen3-dev python3-numpy python3-scipy python3-h5py
# macOS (Homebrew; Apple clang needs libomp)
brew install cmake hdf5 eigen libomp
```

then the same two `cmake` commands as above (from the release root). If CMake does not find a package, point it
there, e.g. `-DCMAKE_PREFIX_PATH=/opt/homebrew` or `-DEigen3_DIR=/usr/share/eigen3/cmake`.

### Build options

| option | effect |
|---|---|
| (default) | `-O3 -DNDEBUG`, no `-march`, no `-ffast-math`: runs on any x86-64 (or arm64) machine |
| `-DCMAKE_CXX_FLAGS=-march=x86-64-v3` | AVX2/FMA build for recent x86-64 CPUs |
| `-DDP5_NATIVE=ON` | `-march=native` (only runs on CPUs like the build host) |
| `-DDP5_BUILD_TESTS=OFF` | skip the integrator unit test |

Different compilers or flags change results only at the level of the integration tolerances (FMA contraction and
vectorization change the round-off); the same build with the same run card is bitwise reproducible.

### Version stamp

At every build (not only at configure time) CMake writes `build/generated/version_info.hpp` with
`<VERSION>+<git describe --always --dirty>` (the file `../VERSION`; `+nogit` outside a git checkout, e.g. in the
Zenodo tarball), the compiler and the full compile flags. `dp5drive --version` prints them, and every output file
carries them: root attribute `release_version`, and `/provenance` attributes `release_version`, `git_describe`,
`build_flags`, `compiler`, `eigen_version`, `hdf5_version`.

---------------------------------------------------------------------------------------------------------------------

## 2. Run one card

A run needs a **model file** (Hamiltonian, one-body jump rates, lossy sites; immutable, content-hashed) and a
**run card** (approximation, mode, grid, tolerances, initial state) that refers to the model by a path relative to
the card. Both are HDF5 files written by `python/dp5model.py` and `python/runcard.py`. In the release the figure
directories write them with `common/runcards.py`; by hand, e.g. a single site with pair loss:

```python
import sys, numpy as np
sys.path.insert(0, "code_diagrammatics/python")
import dp5model, runcard

N = 1                                                  # orbitals per spin
h = np.zeros((2, N, N), complex)                       # one-body h per spin [2, N, N]
z = np.zeros_like(h)                                   # no one-body loss (ell_greater) or gain (ell_lesser)
rho = np.array([[[0.5]], [[0.5]]], complex)            # default rho(0) per spin, rho_sa = <c_a^dag c_s>
dp5model.write_model("site.h5", h, z, z, rho, lossy_sites=[0], site_weight=[1.0])
runcard.write_runcard("rc.h5", "site.h5", gamma=1.0, t_end=20.0, dt_out=0.05, approximation="DSL", mode="g1g2",
                      rho0=rho, ic_mode="full", threads=1)
```

```bash
code_diagrammatics/build/dp5drive --runcard rc.h5 --output out.h5 > run.log 2>&1; echo "exit $?"
python3 code_diagrammatics/python/read_output.py out.h5          # status and integrator profile summary
```

The run prints a profile summary and `status Ok at t = 20 -> out.h5`. The output file must not exist (the driver
never overwrites); runs are single-threaded unless the card sets `threads` (OpenMP, worth it only for large N).

### Command line

```
dp5drive --runcard RC.h5 --output OUT.h5 [--threads N]    run one card
dp5drive --runcard RC.h5 --print-layout                    method label and state layout (block sizes), no run
dp5drive --version                                         release version, git describe, compiler, flags
dp5drive --runcard RC.h5 --eval-rhs STATE.h5 RESULT.h5     test hook: one right-hand-side evaluation (tests/)
```

`--threads` overrides the card's `threads` (recorded in `/input`).

### Exit codes

| code | meaning | output file |
|---|---|---|
| 0 | Ok | complete |
| 2 | input error (bad or missing file, unknown run-card key, refused value, existing output) | none, or `/status` = Error |
| 3 | not implemented | `/status` = NotImplemented |
| 5 | integrator failure (`StepTooSmall`, `NonFinite`, `MaxSteps`) | complete up to `t_reached`, state in `/final` |
| 6 | `NoSteadyState`: `t_prime = auto` timed out with `ss_on_timeout = fail` | complete up to the timeout |
| 7 | `Diverged`: an eigenvalue of ρ^σ left [`diverge_rho_lo`, `diverge_rho_hi`] | complete up to `t_reached` |

Exit codes 5 and 7 are outcomes of the physics (an unstable closure), not crashes: the file holds every frame up
to the stop, the status, and the message.

---------------------------------------------------------------------------------------------------------------------

## 3. Run-card keys

All values are attributes of the group `/run` (root attribute `dp5_runcard_schema = 1`). **The key set is closed:**
any other attribute in `/run`, or any object other than the dataset `rho0`, is refused with exit code 2 and a
message naming it. The release's card writer (`common/runcards.py`) writes every key explicitly.

**Always** (both modes):

| key | value / default | meaning |
|---|---|---|
| `model_path`, `model_hash` | required | model file (relative to the card's directory) and its content hash; a mismatch is refused |
| `gamma` | required, ≥ 0 | pair-loss rate γ (L_r = √(γ w_r) c_{r↓}c_{r↑}) |
| `approximation` | required: `HF`, `TPPX`, `mGW`, `DSL` | (`BARE` is refused: use `TPPX` with `bare_propagators = 1`) |
| `flag_2B` | 0 | 1: 2B (TPPX only) |
| `bare_propagators` | 0 | 1: correlator drifts with the bare h_o = h − i(ℓ^> + ℓ^<) (methods X_0); refused for HF |
| `mode` | required: `g1g2`, `rtde` | `rtde`: G1-G2 plus the RTDE from t′ |
| `spin_mode` | `full` | the only layout (both spin blocks stored) |
| `vertex_impl` | `general` | `general` (any set of lossy sites) or `impurity` (fast path, requires lossy sites = {0}; equal to `general` up to round-off) |
| `ic_mode` | `model` | `model`: ρ(0) = the model's `/initial/rho`; `full`: ρ(0) = dataset `/run/rho0` [2, N, N] complex |
| `t0`, `t_end`, `dt_out` | 0, required, required | output grid t0 + k·dt_out ((t_end − t0)/dt_out integer) |
| `probe_columns` | [0] | exactly one probe orbital j of G^R_{xj} |
| `tol_rho`, `tol_tensor`, `tol_rtde` | [1e-9, 1e-7], [1e-8, 1e-6], [1e-10, 1e-6] | [atol, rtol] of the integrator error norm for ρ, the G1-G2 tensors, the RTDE blocks |
| `checkpoint_interval` | 0 | > 0: full state in `/checkpoints/periodic_<j>` at t0 + j·interval (used for the positivity analysis) |
| `checkpoint_keep` | 0 | > 0: keep only the newest that many periodic checkpoints |
| `rho_stride` | 1 | store ρ in `/rho` every that many frames (observables on every frame) |
| `diverge_rho_lo`, `diverge_rho_hi` | −1, 2 | divergence guard (exit 7) |
| `threads` | 0 (OpenMP default) | OpenMP threads |

**Mode `rtde`** (start time t′ and duration of the RTDE):

| key | value / default | meaning |
|---|---|---|
| `t_prime` | required: one number, or `"auto"` | RTDE start; `auto`: when the G1-G2 state is stationary (windowed-drift detector) |
| `ss_on_timeout` | `fail` | `auto` without detection by c_max/λ_ref: `fail` (exit 6) or `start` (open the RTDE anyway, `warning` attribute) |
| `tol_ss` | 1e-6 | stationarity: windowed drift < tol_ss·λ_used |
| `c_hold`, `c_min`, `c_max` | 1, 1, 50 | hold window, earliest trigger, timeout in units of 1/λ_ref (λ_ref = min |Im ε| of h_o^MF) |
| `ss_fit_rtol`, `ss_window_c` | 0.2, 1 | agreement of the half-window decay rates; drift-window scale |
| `ss_hold_records` | 1 | 1: hold window extended until each half holds ≥ 3 drift windows; the paper's impurity runs use 0 |
| `rtde_tau` | required: a number, or `"auto"` | RTDE duration after t′ (multiple of dt_out); `auto`: stop when the spectral function is converged |
| `tol_A` | 1e-4 | `auto`: bound on the truncation error of A(ω) |
| `rtde_tau_min`, `c_tau_min` | 0, 3 | earliest stop: rtde_tau_min if > 0, else c_tau_min/λ_ref(t′) |
| `rtde_tau_max` | required with `auto` | cap of the RTDE duration (multiple of dt_out); reaching it is status Ok with `cap_reached = 1` |
| `rtde_window_beats` | 8 | monitor window ≥ that many beat periods (0: off); the paper's impurity runs use 0 |
| `rtde_monitor` | `auto` | G^R entries monitored for the stop: `impurity` (orbitals of the impurity), `all`, `auto` (impurity if the model has `N_greater`/`N_lesser`) |

With `t_prime = auto` the card's `t_end` is not used (the run ends at t′ + rtde_tau, or t′ + τ_end with `auto`).

**Model file** (`dp5_model_schema = 1`, written only by `python/dp5model.py`): `/model/h`, `/model/ell_greater`
(one-body loss), `/model/ell_lesser` (one-body gain), `/initial/rho`, all complex [2, N, N] (spin blocks, index
0 = ↑); `/model/vertex/lossy_sites` int64 [M], `/model/vertex/site_weight` [M] (w_r); root attribute `content_hash`
(FNV-1a 64 over the arrays; checked by the driver). Optional descriptive attributes of `/model` (`N_greater`,
`N_lesser`, `ordering`, `units`, …).

---------------------------------------------------------------------------------------------------------------------

## 4. Output file

One HDF5 file per run (`dp5_driver_output_schema = 1`), created before any physics runs. Complex data use the
h5py `{r, i}` compound convention (read as complex128 by h5py). T = number of output frames.

| path | content |
|---|---|
| `/` attrs | `dp5_driver_output_schema`, `run_id`, `created`, `release_version` |
| `/input` | `run/` = exact copy of the card's `/run`; attrs `model_path`, `model_hash`, `N`, `model_*`, `approximation`, `flag_2B`, `bare_propagators`, `method` (label, e.g. `TPPX_0`), `command_line` |
| `/time/t` | [T] frame times |
| `/rho/rho`, `/rho/t` | complex [T_ρ, 2, N, N] ρ^σ_{sa} = ⟨c†_{aσ} c_{sσ}⟩, and its times (every `rho_stride`-th frame) |
| `/observables/{n_d, N}` | [T, 2]: ρ^σ_00, tr ρ^σ |
| `/observables/{D, loss_rate, dN_dt, energy, rho_dot_max, rho_eig_min, rho_eig_max}` | double occupancy per lossy site [T, M]; −4γΣ w_r D_r; d tr ρ/dt; Σ tr hρ; max|dρ/dt|; spectrum extremes of ρ^σ |
| `/observables/equal_time/` | [T, …]: `N_tot`, `N_sigma`, `Mz`, `n_site` [T,2,N], `D_a` [T,M,2], `D_b` [T,N], `D_b_exchange` (mGW), `S2_total`, `Sz2_total`, `local_moment_a/_b/_2rdm`, `SS_nn`, `nn_conn_nn` (bonds), energies, Galitskii–Migdal traces; each dataset described by the group attribute of the same name |
| `/rtde/tp=0/` | `t` [T_R], `GR` complex [T_R, 1, 2, N] = G^{Rσ}_{xj}(t, t′) (column j = probe), attrs `t_prime`, `probe_columns` |
| `/steady_state` attrs | `t_prime = auto`: detected t′, λ_ref, λ_fit, λ_used, drift history, `timeout`, … |
| `/rtde_end` attrs + `m_tau`, `m` | `rtde_tau = auto`: `tau_end`, `t_end`, `lambda_GR`, `truncation_error`, `cap_reached`, monitor history |
| `/checkpoints/tp0`, `/checkpoints/periodic_<j>`, `/final` | full integrator state `y` with attrs `t`, `block_names`, `block_sizes` (state at t′, periodic, at the end) |
| `/status` attrs | `status` (Ok, StepTooSmall, NonFinite, MaxSteps, NoSteadyState, Diverged, Error), `message`, `t_reached`, `wall_seconds`, step counts, run-wide diagnostics (`diag_min_D`, `diag_min_rho_eig`, …) |
| `/profile` | integrator profile (timings, step statistics, memory) |
| `/provenance` attrs | `release_version`, `git_describe`, `build_flags`, `compiler`, `eigen_version`, `hdf5_version`, `host`, `threads` |

The state vector of a checkpoint is the concatenation of the blocks in `block_names` (sizes `block_sizes`): `rho`
[2·N²] first, then the correlator blocks of the approximation (`Kpp`; `GW_P, GW_A, GW_UU, GW_DD, GW2_P`; `G_opp,
G_UU, G_DD` (packed)), then the RTDE blocks after t′. `dp5drive --print-layout` lists them for a card.

---------------------------------------------------------------------------------------------------------------------

## 5. Tests

From the release root, after building (each suite recreates its own `tests/work*/` directory):

```bash
cd code_diagrammatics
build/test_dp5                          # integrator unit tests            (~1 min)
python3 -I tests/run_tests.py           # step 1: models, HF vs solve_ivp, output schema, bad input files
python3 -I tests/test_step2.py          # G1-G2 RHS of every method vs full spin-orbital references; trajectories;
                                        #   single site exact (DSL, TPPX_0); plaquette; bare-propagator flag; inputs
python3 -I tests/test_step3.py          # TPPX/2B/TPPX_0 RTDE vs full references; single-site G^R
python3 -I tests/test_step3b.py         # equal-time observables vs exact Lindblad (ED); GM trace
python3 -I tests/test_step7a.py         # mGW RTDE with counterterm
python3 -I tests/test_step7b.py         # DSL RTDE (single-site G^R exact)
python3 -I tests/test_step8.py          # steady-state detection (t_prime = auto)
python3 -I tests/test_step9.py          # automatic RTDE end, checkpoint_keep, divergence guard
```

Each suite prints one line per check (`ok` / `FAIL`) and ends with `ALL ... TESTS PASSED` (exit 0) or the number
of failures (exit 1). They use at most 4 OpenMP threads. On a 2020s workstation core they take ≈ 1 min each
(`test_step3b.py` ≈ 6 min). `DP5_EXE=/path/to/dp5drive` tests another binary. Optional: `DP5_REF_EXE` (step 2 [7]) and
`DP5_STEP9_REF_EXE` (step 9 [3]) name a second build to check bitwise identity against.

The references in `tests/ref/` are independent implementations of the equations in full spin-orbital form
(`item0_check.py`: G1-G2 right-hand sides of TPPX, mGW, DSL; `impurity_blocks_check.py`: TPPX RTDE); the
suites also compare against exact Lindblad dynamics (exact diagonalization, quantum regression) written in the tests.

---------------------------------------------------------------------------------------------------------------------

## 6. Cost

Single-threaded runs of the paper's settings (one EPYC 7643 core of a shared host, default build; peak RSS from
`/profile/memory`):

| run class | settings | wall time per run | peak RSS | output |
|---|---|---|---|---|
| single site (`fig_site`) | g1g2, γt ≤ 2000, dt_out 0.05/γ | 6–9 s | 55 MB | ≈ 18 MB |
| plaquette (`fig_plaq`, `tab_plaqerr`) | g1g2, t ≤ 200, dt_out 0.1, checkpoints every 1.0 | 1–2 s | 30 MB | ≤ 8 MB |
| Zeno (`fig_zeno`) | g1g2, fine (t ≤ 20, dt 0.005) and long (t ≤ 3000, dt 1) | ≤ 1 s / ≤ 15 s | 25 MB | ≤ 5 MB |
| impurity c3 (`fig_impss`, `fig_sigma`) | rtde, t′ = auto, τ = auto, tight tolerances, dt_out 0.01 | 2 s typical, ≤ 70 s | ≤ 130 MB | ≤ 170 MB |
| toy impurity | rtde, t′ = auto, τ = auto, default tolerances | ≈ 1 s | 30 MB | ≤ 5 MB |

Runs that stop early (exit 5, 7) take less. The cost grows like N⁵ per right-hand-side call for the correlated
methods (N = orbitals per spin); mGW is the most expensive (five N⁴ blocks), then DSL, TPPX, HF.

---------------------------------------------------------------------------------------------------------------------

## 7. Relation to the development driver

This is the driver the paper's data were produced with, reduced to the features the paper uses. The physics code
(right-hand sides, observables, integrator, detector, automatic RTDE end) is unchanged: for identical run cards and
build flags the release binary reproduces the development driver bitwise (every dataset of 73 runs covering all
figures; checked by the authors). Removed: the paramagnetic one-spin layout, several probe columns or t′ per run,
frozen-state RTDE runs, restart/extension/stitching, wall-time and signal handling, the Jacobian (Arnoldi), second-t′
and freeze checks of the detector, debugging hooks, and the run-card keys `output`, `walltime_*`, `corr_matrix`,
`block_norms`, `impurity_occupation`, `frozen_checkpoint_*`, `ss_check_dt`, `ss_arnoldi`, `ss_freeze`, `debug_*`.
The internal spin-count generality of the equations' code (one or two stored spin blocks) is kept unchanged; only
the two-block layout can be selected. Output files of the development driver read the same way (same dataset
names and attributes); `approximation = BARE` in old files means `TPPX_0`.
