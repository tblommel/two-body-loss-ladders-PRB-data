# dp5_integrator: design note (revision 5)

Dormand-Prince 5(4) integrator for block-structured complex states, written to replace an earlier RK45 header of
the development code after a code review. (Vendored unchanged into the release; references to files of the
development repository below are historical.)

| file | content |
|---|---|
| `dp5_integrator.hpp` | header-only integrator (C++17 standard library + optional OpenMP; no HDF5, Eigen or physics headers) |
| `dp5_profile.hpp` | profiling data (`dp5::Profile`), text summary and flat export; included by the integrator (§10) |
| `tests/test_dp5.cpp` | correctness tests (12 groups), scipy export, benchmark with a faithful re-implementation of the old scheme |
| `tests/scipy_crosscheck.py` | cross-check against `scipy.integrate.solve_ivp` (RK45, DOP853) |
| `tests/run_all.sh` | builds with `g++ -O3 -march=native -fopenmp` (and once with `-DDP5_PROFILE=0`) and runs everything with 12 threads |
| `tests/profile_checksum.txt`, `bench_profile.txt` | profiling on/off checksums and overhead benchmark |
| `tests/profile_summary.py`, `profile_dump.json`, `profile_summary_output.txt` | analysis from the flat export alone (§10.5) |
| `tests/test_output.txt`, `scipy_output.txt`, `bench_results.txt` | outputs of the last run |

## 1. Method choice: registers and passes

A *register* is one state-sized complex vector. A *touch* is one full read or
write of one register. Touches per step are the integrator's memory traffic.

**Old scheme (`RK45.hpp`, Cash–Karp).**
- Registers: `y`, `k1..k6`, `y_tmp`, `y5`, `y4`, `y_old`, `k1_old` = **12**. On
  top of those, `g_interp` and `final_state_G` each hold the tensor part.
- Touches per accepted step:
  - stage inputs: 3+4+5+6+7;
  - y5 and y4: 9;
  - error: 3;
  - three copies: 6;
  - **total 43**, all serial.
- Additional cost per output frame: a Hermite pass over the tensor part.

**New scheme (Dormand–Prince 5(4), FSAL, Hairer/Shampine dense output).**
- Registers: `Y`, `T` (stage input, which becomes the new state), `K1..K6` =
  **8**.
- k7 reuses k2's slot. k2 is dead after stage 6 (a72 = b2 = e2 = 0, and the
  dense output has no k2 term).
- Touches per accepted step: stage inputs 3+4+5+6+7, y5 7, fused error pass 8 =
  **40**.
- Zero copies: an accepted step rotates pointers (Y↔T, K1↔K2).
- 6 RHS calls per step, as before.
- Dense output covers only the `save` blocks and needs no extra register.

**Low-storage schemes were rejected.**
- 2R/3S* schemes need an extra register for the error estimate and two more for
  dense output.
- Their 4(3)/5(4) pairs need 0–50% more RHS calls.
- The RHS (O(N⁵)) dominates. At N = 25 the state of the largest layout is about
  2×10⁶ entries, so 8 registers are 0.27 GB (table in §6). Memory is not
  binding.

## 2. API

```cpp
#include "dp5_integrator.hpp"
using dp5::cplx;

dp5::Integrator in(blocks, rhs, opts, t0);
// blocks: std::vector<dp5::BlockSpec>{name, size, atol, rtol, weight, norm, active, save, rms_count}
//         helpers: dp5::tensor_block(name, n, atol, rtol)      -> RMS norm
//                  dp5::observable_block(name, n, atol, rtol)  -> Max norm, saved
// rhs:    void(double t, const cplx* y, cplx* dy)                         or
//         void(double t, const cplx* y, cplx* dy, const dp5::Layout& L)   (recommended)
in.integrate(t_end) -> dp5::Status   // Ok | Interrupted | NonFinite | StepTooSmall | MaxSteps
in.set_output_grid(t0, t1, n, obs)   // obs(const dp5::Frame&): f.k, f.t, f[block_id] (save blocks)
in.set_output_grid_dt(t0, dt, n, obs) // same with t_k = t0 + k*dt for every k (frame times independent of the grid length)
in.add_stop(ts, cb)                  // land exactly on ts, run cb(Integrator&), then emit the frame at ts
in.set_interrupt(fn); in.set_step_observer(fn);
in.append_block(spec) / remove_block(id) / extract_block(id) / set_active(id, on)
in.append_blocks(specs)             // many blocks, ONE reallocation (same ids/values as one-by-one)
in.set_tolerances(id, atol, rtol) / set_norm(id, norm) / set_save(id, on)
in.block(id), in.block_id(name), in.offset(id), in.layout(), in.layout_version()
in.slope()                          // f(t, y) at the current state if cached (always right after an accepted step: FSAL), else nullptr
in.checkpoint() / in.restore(cp, y)  // bitwise continuation
in.stats(), in.diagnostics()         // per-block errors of the last attempted step, worst block
```

### 2.1 Layout stability: the RHS idiom

Block **ids are stable** for the life of the integrator. Removed ids stay
reserved. **Offsets are not stable**: `append_block`, `remove_block` and
`extract_block` reallocate the registers, move blocks, and increment
`Layout::version`. The pointers `y` and `dy` change at every call anyway, since
they point to a different register for each stage. Each `append_block` reallocates and copies all 8 registers, so appending
many blocks one by one costs O(#blocks x state) (seen in the driver: 225 RTDE
blocks at N = 25 took 14 s); use `append_blocks` (one reallocation, identical
layout, ids and values; tested bitwise in test [8b]). So the RHS must never cache
pointers, and should read offsets from the `Layout` it receives:

```cpp
auto rhs = [&](double t, const cplx* y, cplx* dy, const dp5::Layout& L) {
  const cplx* G = y + L.offset[idG];  cplx* dG = dy + L.offset[idG];   // cheap, every call
  if (L.active[idF]) { const cplx* F = y + L.offset[idF]; ... }
};
```

If an RHS precomputes something expensive per layout, such as index maps, it
keys it on `L.version`. Only active blocks need `dy` written. Inactive and
removed blocks are never read from `dy`.

### 2.2 Frozen inputs live outside the state

**Read-only inputs must not be blocks.** Examples are the NESS ρ and 𝒢 for a
frozen RTDE, and parameters. Keep them in **one user-owned buffer** and capture
it in the RHS closure. The integrator never touches it: no registers, no passes,
no error.

For a handoff inside one run (G1-G2 → frozen RTDE at t′), use
`extract_block(id)` in the stop callback. It returns the block's data as a
`std::vector<cplx>`, removes the block from all 8 registers, and moves the
remaining blocks. Tested in §7, group 9(c): the state shrinks from 2 015 625 to
15 625 entries, and the result is bitwise equal to co-propagating a stationary
G.

**What inactive blocks are for.** An inactive block stays in all 8 registers.
It takes no passes and no error, and the RHS can read it from `y`. Use it only
for small or short-lived blocks: a block switched off temporarily, or a
reference run that carries a block from the start (test 4). It is *not* meant
for large frozen inputs. Values are synced Y→T only when the state is changed
from outside (`state_changed`, stops, `set_active`, restore), **not per stage**:
the passes never write inactive entries.

### 2.3 Threading and NUMA contract

- One model: OpenMP. Each pass is one parallel region over the active blocks.
- Thread t of n gets the same static slice of each block in every pass
  (`dp5::static_range`). Registers are first-touched with that slice.
- The same split is public:
  - `dp5::parallel_for_block(L, id, f)` opens a region and calls `f(i)` for
    this thread's local indices of block `id`;
  - `dp5::for_static(lo, hi, f)` does the same inside an existing region.
- An RHS loop written this way touches the same pages with the same threads as
  the integrator. That applies to **elementwise RHS work**: diagonal pieces of
  drifts, sources, masks, sparse impurity-index terms, copies.
- **Contractions** done by Eigen or BLAS (the O(N⁵) drifts) use their own
  partition and get no NUMA benefit. Run them with OpenMP threads (not a
  separate `Eigen::ThreadPool`) so that the integrator and the RHS share one
  thread team.
- Reductions are combined in thread order, so results are bitwise reproducible
  for a fixed thread count.
- Passes run serially below `par_threshold` = 32768 active entries.

## 3. Usage sketches

Recommended defaults are used throughout: RMS for tensor blocks, Max for
observables (§5).

**A. G1-G2 alone**
```cpp
dp5::Integrator in({dp5::observable_block("rho", 2*N*N),
                    dp5::tensor_block("G2", N*N*N*N)}, rhs_g1g2);
write_rho0(in.block(0)); in.state_changed();                    // G2(0) = 0 already
in.set_output_grid(0.0, t_end, n_out, [&](const dp5::Frame& f){ h5.append_rho(f.t, f[0]); });
for (double tc = dT; tc < t_end; tc += dT)
  in.add_stop(tc, [&](dp5::Integrator& it){ h5.write_checkpoint(it.checkpoint(), it.y(), it.size()); });
in.set_interrupt([&]{ return walltime_left() < 15*60; });
dp5::Status st = in.integrate(t_end);
if (st != dp5::Status::Ok && st != dp5::Status::Interrupted)
  log("failed in block", in.spec(in.diagnostics().worst_block).name);
h5.write_status(st, in.stats()); h5.write_checkpoint(in.checkpoint(), in.y(), in.size());
```

**B. G1-G2, then RTDE co-propagated from t′**
```cpp
const double tp = in.grid_time(k_p);                             // t' on the output grid
in.add_stop(tp, [&](dp5::Integrator& it) {
  h5.write_checkpoint(it.checkpoint(), it.y(), it.size());       // NESS checkpoint first
  model.idGR = it.append_block(dp5::observable_block("GR", 2*N, 1e-10, 1e-6)); // column s=0, both spins
  model.idF  = it.append_block(dp5::tensor_block("F", n_F, 1e-10, 1e-6));      // F(t',t') = 0
  cplx* g = it.block(model.idGR); g[0] = cplx(0,-1); g[N] = cplx(0,-1);         // G^R(t'+,t') = -i e_0
  model.rtde_on = true;
});
in.integrate(t_end);   // the frame at t' shows G^R = -i e_0, emitted exactly once
```

**C. Frozen-NESS RTDE (separate run, or the handoff variant)**
```cpp
NessInputs ness = load_checkpoint_blocks("rho", "G2");          // one buffer each, user-owned
dp5::Integrator in({dp5::observable_block("GR", 2*N, 1e-10, 1e-6),
                    dp5::tensor_block("F", n_F, 1e-10, 1e-6)},
                   [&](double t, const cplx* y, cplx* dy, const dp5::Layout& L) {
                     rtde_rhs(ness, y + L.offset[0], y + L.offset[1], dy + L.offset[0], dy + L.offset[1]);
                   }, opts, tp);
set_GR(in.block(0)); in.state_changed();
in.set_output_grid(tp, tp + tau_max, n_tau, obs);
in.integrate(tp + tau_max);
// handoff variant inside one run, in the stop callback at t':
//   ness.G2 = it.extract_block(idG2); ness.rho = it.extract_block(idrho); then append GR, F
```

## 4. How D1–D4 are fixed (and the minor items)

**D1 (stale slope).** Dense output uses the step's k1 and FSAL k7 =
f(t_new, y5), which are always computed, including at stops and at t_end. After
every stop callback the slope is recomputed.

**D2 (lost or duplicate grid points).**
- Grid times are `t0 + k*dt` by integer index, with the last point exactly `t1`.
- The integrator lands exactly on stops and t_end (`t = t_stop` is assigned).
- A grid point within `land_tol(t)` = 16 ε max(1, |t|) of a landing is emitted
  from the exact state; otherwise it comes from dense output. Each index is
  emitted once, in order. At a stop the callback runs first, then the frame.
- **Grid points cannot merge.** `set_output_grid` rejects spacings below
  8 × land_tol(max|t|). Two grid points are therefore always further apart than
  any snapping window, and the dense-output test (`tg < t_new − tol`) can never
  skip one.
- The tolerance does not need to scale with h. h ≥ 10 ulp(t) is enforced, a
  clipped step is either > tol or treated as "already there", and snapping a
  grid point by < tol changes its value by < tol·|y′|, which is far below any
  tolerance.

**D3 (NaN, tiny steps, no cap).**
- NaN in the stages or in y5 is detected in the fused error pass, **per block**.
  It returns `NonFinite`, optionally after retries with h/4, leaving the last
  accepted (finite) state.
- Inf in the error estimate counts as a huge error and the step is rejected.
- `StepTooSmall` triggers below max(h_min, 10 ulp(t)). `MaxSteps` caps steps per
  call.
- **Diagnostics:** `in.diagnostics()` gives every block's weighted error and
  non-finite flag for the last attempted step, plus `worst_block`. In the tests
  it pinpoints a NaN or a blow-up in block "G" out of three blocks.

**D4 (validation).** `std::invalid_argument` is thrown for invalid sizes, names,
tolerances, weights, `rms_count` > size, options, grids (including unresolvable
spacing), times, stops, access to removed blocks, and restore with a different
layout.

**Complex states.** Every quantity is per complex entry z:
- the error is |e| (modulus), not separate real and imaginary parts;
- the scale is atol + rtol · max(|y|, |y5|), so atol applies to |z|;
- the RMS uses |z|².

**Antisymmetric and redundant storage (`rms_count`).**
- Replicated entries do not bias an RMS: each independent entry appears equally
  often.
- Structural zeros do bias it, by sqrt(nonzero/size). Examples are diagonals of
  an antisymmetric pair stored in full, or padding.
- Set `rms_count` = the number of structurally nonzero entries. The RMS is then
  over those entries, i.e. equal to the RMS over the independent entries.
- Packed storage needs nothing.
- Test 8: packed takes 86 steps; full storage with 75% zeros takes 75 steps
  (biased, too loose); full + `rms_count` takes 86 steps.
- Alternatively, `weight` can compensate in Max-over-blocks mode, but
  `rms_count` is exact and also correct in GlobalRMS mode.

**Controller.**
- PI as in Hairer's dopri5 (β = 0.04) by default, or `Standard`, which
  reproduces scipy's RK45 exactly.
- Automatic h₀ by Hairer II.4.
- h carries across stops (the requested h, not the clipped one).
  `reset_h_at_stop` forces a new h₀.
- `Checkpoint` holds t, h, the PI memory, the grid index, active and removed
  flags, and stats.

## 5. Error-norm recommendation

From test 2 (RTDE-sized damped oscillators, 535 vs 684 steps):
- An RMS norm bounds the block RMS. Its worst entry can sit about 77× above the
  local tolerance after many steps.
- A Max norm bounds every entry (worst 23×) at 28% more steps.

**Recommended defaults**, used in all sketches:
- **G1-G2 tensor blocks** (𝒢, P, A, U, D, counterterms; N⁴):
  `tensor_block` → **RMS**, rtol 1e-6, atol 1e-8. They are not observables
  entry by entry. Their influence on ρ and G^R is through contractions, which
  average.
- **Observable blocks** (ρ, G^R): `observable_block` → **Max**, saved. Choose
  atol from the dynamic range needed. For spectra from G^R_00(t′+τ, t′), which
  decays by many orders, set G^R/F atol about 1e-10, separately from the tensor
  blocks; test 9(b) shows why.
- **RTDE tensor blocks** (F; N³): RMS, with the same small atol as G^R. They
  directly feed G^R.
- Keep `Combine::MaxOverBlocks`, so that each block meets its own tolerance.

## 6. Memory at production sizes (reduced layouts)

- 8 registers × 16 B × entries.
- The state counts 2N² for ρ, the G1-G2 tensors, 2N for G^R(·,0) of both probe
  spins, and 2 × the per-probe-spin RTDE blocks.
- "Frozen RTDE" keeps only G^R + F in the registers. The frozen NESS inputs
  (ρ + tensors) sit once in a user buffer (last column).
- DSL antisymmetric blocks are counted packed.

| N | model | G1-G2: entries / registers | G1-G2 + RTDE: entries / registers | frozen RTDE: entries / registers | frozen input buffer |
|---:|---|---:|---:|---:|---:|
| 25 | TPPX/BARE | 3.92e5 / 50 MB | 4.23e5 / 54 MB | 3.13e4 / 4.0 MB | 6 MB |
| 25 | mGW + ct | 1.95e6 / 250 MB | 2.08e6 / 266 MB | 1.25e5 / 16 MB | 31 MB |
| 25 | DSL | 5.72e5 / 73 MB | 6.18e5 / 79 MB | 4.63e4 / 5.9 MB | 9 MB |
| 37 | TPPX/BARE | 1.88e6 / 240 MB | 1.98e6 / 253 MB | 1.01e5 / 13 MB | 30 MB |
| 37 | mGW + ct | 9.37e6 / 1.20 GB | 9.78e6 / 1.25 GB | 4.05e5 / 52 MB | 150 MB |
| 37 | DSL | 2.76e6 / 354 MB | 2.91e6 / 373 MB | 1.51e5 / 19 MB | 44 MB |

For comparison, the current (full-s) GW-RTDE at N = 25 keeps 3.9×10⁶ entries in
12 registers, i.e. 0.75 GB.

## 7. Test results

All tests ran on this laptop (22 cores, 12 OpenMP threads used).
`tests/test_output.txt`, `scipy_output.txt` and `bench_results.txt` hold the
full output.

| test | result |
|---|---|
| 1. order, fixed step | 5.01 for the step, 5.00 for interior dense frames |
| 1. tolerance proportionality (PI), rtol 1e-4…1e-10 | err(T)/rtol = 1.12–1.18 |
| 2. RTDE-sized damped oscillators (2.6e5 entries) | PI/RMS: 535 steps, err/tol RMS 10, max 77. Standard: 505 steps. PI/Max: 684 steps, max 23. |
| 3. landing, t_end = 40, n = 3001, t′ = 30 (+ stop at 12.3456) | 3001 frames, each k once and in order; times bitwise `grid_time(k)` (grid_time(2250) = 30.000000000000004 is merged with t′ = 30 into one frame); callbacks at exactly 30 and 12.3456; t_final = 40 exactly; frame 2250 shows G^R = −i after the callback; frame errors 1.9e-11 / 5.3e-10 |
| 4. activation at t′: appended / inactive→active / held-at-zero reference | bitwise identical; B error 1.9e-10 against exact |
| 5. failures | NaN → `NonFinite` with a finite state; Inf with retries → `NonFinite`; y′ = y² → `StepTooSmall`; cap → `MaxSteps` |
| 6. validation | 15/15 invalid inputs rejected; n = 1 grid gives one frame |
| 7. checkpoint and interrupt | bitwise continuation (51 frames and the final state) |
| 8. layout tools | `rms_count`: 86 / 75 (biased) / 86 steps; NaN and blow-up diagnostics name block "G"; `extract_block`: state 60 → 10 entries, version incremented, data preserved, removed id throws; unresolvable grid rejected |
| 9. coupled G1-G2 (2×10⁶) → RTDE (15 625 = 25³), t′ = 15, T = 35, F decays by e^−12 | see below |
| 10. m*GW* counterterm: propagate (𝒢^W, 𝒢^{W,2}) or (D = 𝒢^W − 𝒢^{W,2}, 𝒢^{W,2}) | equivalent: D agrees to 1.4e-9 relative, same errors (9.7e-7) and step counts; RK is equivariant under linear variable changes, so no reformulation is needed (§9.1) |
| 11. profiling | counters consistent (attempts 164 = 159 + 5 + 0; RHS 987 = 6·164 + 3 slope + 0 h₀ = `Stats::n_rhs`; per-stage calls = attempts; frames, segment, histogram and limiting-block sums match); timing closure exact (−1e-16 s, untimed 0.25%); bitwise identical on/off, and across `DP5_PROFILE=0/1` builds (checksum); trace decimated to ≤ 64; 245 export keys |
| 12. profile on failure paths and memory | RHS exception: status Exception, message recorded, closure exact, 2 RHS calls of the aborted attempt counted; stop-callback exception: profile consistent, the run continues to Ok; NonFinite: status and worst block recorded; memory: registers = 8·state·16, transient peak = (old + 8 new)·16, dense buffer, 2 layout changes, peak RSS |

**Test 9 (physics-shaped).**
- G′ = λ(G − G∞) with |Im λ| ≤ 2. After t′, F_j′ = (μ_j + c G_j) F_j with
  Re μ = −0.6 and |Im μ| ≤ 8.
- Output comes from F only.
- The solution is exact, so the error can be checked against the exact
  solution.

(a) **Step size.**
- Before t′ the mean h is 0.147, limited by G (the only block).
- After t′ the mean h is 0.028–0.030, and F is the worst block in 100% of steps.
- The step is set by whichever block limits, through `diagnostics().worst_block`.

(b) **Separate atol for F.**
- F with atol = 1e-10 (G at 1e-8): max over all 401 frames and 15 625 entries of
  |F − exact| / (atol + rtol|exact|) = **15**. The final relative error is
  **4e-5**, at 787 steps.
- With F atol = 1e-8: ratio 7.5, but the final relative error is **1.6e-3**.
  Once F has decayed, atol dominates and spectra tails lose accuracy (the review's
  O4 point).

(c) **Frozen input vs co-propagation (stationary G).**
- Frozen external G from t′: state 15 625 entries, **0.24 s**. Co-propagating G
  from 0: state 2 015 625 entries, **22 s**.
- Same F to 1.4e-7: tolerance-level, different step histories.
- Co-propagating from t′ with the same h₀ is **bitwise identical** to the frozen
  run.
- The `extract_block` handoff at t′ is **bitwise identical** to co-propagating
  from 0.

**scipy cross-check** (nonlinear complex ODE, rtol 1e-6, atol 1e-9):
- With `Controller::Standard`, the accepted step sequence is identical to scipy
  RK45 (117 steps, max |Δt| = 5.6e-12).
- Dense frames equal scipy's RK45 dense output to **3.2e-15**.
- Both are 2.9e-6 from a DOP853 reference.

**Benchmark** (integrator ms per attempted step, excluding the RHS; 1% of the
state saved, 201 frames):

| n | old | new, 1 thread | new, 12 threads | registers old → new | max RSS old → new |
|---:|---:|---:|---:|---:|---:|
| 1e4 | 0.22 | 0.12 | 0.13 (serial below threshold) | 1.9 → 1.3 MB | 7 → 6 MB |
| 1e5 | 3.0 | 1.4 | 0.58 | 19 → 13 MB | 35 → 24 MB |
| 1e6 | 42 | 24 | 8.8 | 192 → 128 MB | 310 → 218 MB |
| 4e6 | 169 | 101 | 36 | 768 → 512 MB | 1225 → 859 MB |

The passes are bandwidth-bound: 12 threads give only 2.8× on this laptop. The
max RSS includes the test's own arrays.

## 8. Not done / caveats

- Not yet run with the real physics RHS.
- The NUMA benefit requires the RHS to use `parallel_for_block` (or the same
  split) for its elementwise loops. Contractions inside Eigen or BLAS are not
  covered.
- No stiffness detection or exponential integrator (review item O8).
- Inactive blocks still occupy all 8 registers by design. Use frozen buffers or
  `extract_block` for large frozen data.
- Bitwise reproducibility needs a deterministic RHS and a fixed thread count.
- The benchmarks are laptop numbers. Rerun `tests/run_all.sh` on the production
  node type.

## 9. Mapping to Implementation_verification

Source: the implementation-verification notes of the development repository:
- notation paragraph;
- sec:symm, including the counterterm identities;
- TPPX secs. 1–2 (R1–R9);
- m*GW* impurity: sec:mGWG1G2imp (R36–R39) and sec:mGWRTDEimp (R40–R42);
- DSL impurity: sec:DSLG1G2imp (R43–R46) and sec:DSLRTDEimp (R47–R50).

### 9.1 Decision: no restructuring of the integrator

The spec changes the RHS, not the time stepping. Every requirement it raises
maps onto the existing API.

1. **Block inventories.**
   - Every approximation is a short list of named N⁴ (G1-G2) and N³ (RTDE,
     column s = 0, R7/R40/R47) blocks, plus ρ (2N²) and G^R(·,0) (N per probe
     spin).
   - Blocks are coupled only inside the RHS.
   - The `BlockSpec`/`Layout` model holds exactly that.
   - Spin sectors never mix block sizes: S_z leaves at most six blocks per
     tensor, and the item sections leave fewer. No nested or ragged structure is
     needed.
2. **Symmetries.** The integrator never needs to know a block's symmetry. It
   only needs the norm to count the right entries. The equations preserve every
   symmetry exactly (sec:symm, "What is claimed"), and the integrator's passes
   are elementwise linear combinations, which preserve any linear symmetry
   bitwise when the stage slopes have it. Two consequences:
   - **Antisymmetric same-spin blocks.** DSL 𝒢^{σσσσ} (G1-G2) and
     ℱ^{σσσσ}_{k0ab} (RTDE) appear in R43/R47 as "full arrays that preserve the
     antisymmetry". Storing them packed or full is an RHS/driver choice:
     - Packed storage: `size` = independent entries; no further setting.
     - Full storage: set `rms_count` to the structurally nonzero entries.
       G1-G2: N²(N−1)² (all s ≠ l and a ≠ b). RTDE at s = 0: N·N(N−1)
       (all a ≠ b).
     - Then the RMS equals the RMS over the independent entries (§4).
     - Full storage doubles or triples these blocks' memory (§9.3) but keeps
       the drifts as plain contractions.
   - **Pair-Hermitian blocks** (TPPX 𝒦^{pp}, R2/R4; the m*GW* counterterm). They
     can be stored as full arrays, as now. Halving them would need a
     packed-Hermitian RHS. The integrator is indifferent.
   - Roundoff drift of a symmetry stays at the 1e-15 level, because nothing
     amplifies it.
3. **Paramagnetic collapses** (R6, sec:symm-model, the "paramagnetic case"
   paragraphs).
   - A paramagnetic run is a different, smaller layout: one TPPX RTDE tensor,
     three m*GW* G1-G2 tensors and three RTDE tensors, one DSL tensor per layer.
     It is chosen by the driver at construction.
   - Spin-broken starts (needed in general, per the author) use the full layout.
   - Switching mid-run, for example when a polarized start relaxes to a
     paramagnetic NESS (the DSL star runs in sec:DSLimp-num), needs nothing new:
     a stop callback can `remove_block` the redundant partners and set an RHS
     flag (§2.1–2.2).
   - I recommend **not** collapsing automatically. Paramagnetic equality holds
     only to roundoff (3e-15 in the spec's runs). Collapse is safe only when the
     inputs are exactly paramagnetic (R6), which the driver knows and the
     integrator does not.
4. **Frozen-NESS RTDE reads only N² slices** (R42 for m*GW*, R49 for DSL, R8 for
   TPPX).
   - At the handoff, the stop callback reads those slices directly from
     `it.block(id)` into the user's frozen-input buffer, then calls
     `remove_block(id)` for each G1-G2 block.
   - No full copy is needed (`extract_block` is for when the whole block is
     needed).
   - Frozen inputs are then tiny:
     - m*GW*: 2 slices × 3 blocks per probe spin, i.e. 12 N² numbers;
     - DSL: 8 N² slices;
     - TPPX: one slice 𝒦_{ab00};
     - plus h_o and row/column 0 of ρ.
   - So the frozen RTDE state is only G^R and the ℱ blocks (§6, "frozen RTDE"
     column), and memory is far below the G1-G2 run's.
5. **The m*GW* counterterm** (Eqs. symm-G2pp, symm-F2pp).
   - 𝒢^{W,2} = 2𝒢^{pp}|_{𝔥^{pp}=0} and ℱ^{W,2} = 2ξℱ^{pp}|_{𝔥^{pp}=0,𝒢^{pp}=0}, driven by
     the **same** ρ(t) and h_o(t) within the run.
   - That is **RHS code sharing**: one "pp pair tensor" routine (drifts, source,
     optional 𝔥^{pp} memory, optional 𝒢 coupling) serves:
     - TPPX (full);
     - BARE (bare h_o in the pair drifts);
     - the DSL opposite-spin block (TPPX part, lines 1–2 of Eq. DSLimp-P);
     - the m*GW* counterterm (no memory, factor 2 on the source; slot map
       (s,l,a,b) = (s,o,p,j)).
   - **No storage sharing** is possible: m*GW* has no TPPX tensor of its own, and
     a separate TPPX run has a different ρ(t) (stated in the spec).
   - **No integrator implication.** In particular there is no need to propagate
     D = 𝒢^{W,↑↓↑↓} − 𝒢^{W,2,↑↓↑↓}, although the two blocks share drift and
     source (Eqs. mGWimp-P/P2, mGWRimp-UP/UP2) and I and I^R read their
     difference. Runge–Kutta methods are equivariant under linear changes of
     variables, and both blocks see identical ρ and h_o at every stage. Test 10
     confirms it: (𝒢, 𝒢₂) and (D, 𝒢₂) give the same D to 1.4e-9 relative, with
     errors 9.74e-7 vs 9.76e-7 and 61 steps each. Keep the natural variables.
6. **Cost remarks** (R38, R41, R45, R48: drifts 4N⁵ per N⁴ block and 3N⁴ per N³
   block dominate; sources, kernels and couplings are O(N³–N⁴)).
   - Per entry, the drifts cost about 4N complex multiply-adds per RHS call,
     i.e. about 24N ≈ 600 per step at N = 25 (≈ 4800 real flops). The
     integrator costs 40 touches = 640 B per entry and step.
   - The drift runs at compute speed and the integrator at bandwidth, so the
     integrator is a modest but non-negligible fraction, of order 10–50% (to be
     measured, §2.4 of the review). This is why it is fused and parallel (§1).
   - The coupling is one-way, G1-G2 → RTDE. That is already used: frozen-NESS
     mode, separate RTDE tolerances (§5), and the RTDE adds only 2–4% of the
     state (§6).
   - Elementwise pieces of the RHS (sources, kernels writing N³ slices, the
     `ℓ^{>,MF}` diagonal part of h_o) are where `parallel_for_block` helps.
     Contractions are not.

### 9.2 Error-control defaults derived from the spec

- **DSL G1-G2 same-spin blocks** (1–3% of the opposite-spin block; dropping
  them moves n₀σ by ≤ 5e-4, R46).
  - Keep them with the **same** atol/rtol and weight 1 as 𝒢^{↑↓↑↓}.
  - They are cheap, and, being small, they are controlled mainly by atol. In
    the Max-over-blocks combination they will practically never limit the step,
    so lowering their weight buys nothing.
  - Do not drop them in a DSL run that claims to be DSL. R46 calls dropping
    "an approximation beyond the DSL".
- **DSL RTDE same-spin block** (13–57% of the opposite-spin block; up to 3.4%
  in A₀₀(ω) at γ = 5, R50).
  - Treat it exactly like the opposite-spin RTDE block: same small atol (about
    1e-10, separately from the G1-G2 tensors), weight 1.
  - It can limit the step, and it should.
- **m*GW* same-spin blocks** (𝒢^{W,↑↑↑↑}, 𝒢^{W,↓↓↓↓}; 2–10×10⁻³ against
  2–4×10⁻² for the opposite-spin blocks at t = 2 in the spec's RK4 runs). Same tolerance as the other
  resummed blocks.
- **m*GW* counterterm.** Same tolerance as 𝒢^{W,↑↓↑↓}. Its error enters I
  only through the difference, which RK treats consistently (item 5 above).
- **G^R and ρ** are observables: Max norm. G^R uses a small atol, because
  A(ω) needs G^R₀₀ over many decades (the spec truncates at |G^R₀₀| ≈ 2e-5).
  ρ is O(1), with small off-diagonal pieces.
- **All inhomogeneous RTDE terms ∝ G^R₀₀(t,t′)** (spec: "one driving scalar").
  The ℱ blocks decay with G^R₀₀, so a fixed atol on ℱ at the G^R level (test 9)
  is the consistent choice. A relative-only control would chase noise as
  everything decays.

### 9.3 Approximation → blocks

Ids are per run, and names are suggestions. N is the number of orbitals per
spin (impurity + pseudomodes). Probe spins: ↑ uses the "probe ↑" set and ↓ its
spin flip (sec:mGWRTDEimp, sec:DSLRTDEimp). "obs" = `observable_block` (Max,
saved). "tens" = `tensor_block` (RMS). Suggested tolerances (atol/rtol):
- ρ: 1e-9 / 1e-7;
- G1-G2 tensors: 1e-8 / 1e-6;
- G^R and RTDE tensors: 1e-10 / 1e-6.

The memory column counts 8 registers at N = 25 (N = 37 in brackets). Same-spin
DSL blocks are stored packed.

**TPPX / BARE** (secs. 1–2, R1–R9; BARE = same blocks, bare h_o in the pair
drifts)

| run | blocks: name (size, norm, tol) | registers |
|---|---|---|
| G1-G2 | `rho` (2N², obs, 1e-9/1e-7); `Kpp` = 𝒦^{pp,↑↓↑↓} (N⁴, tens, 1e-8/1e-6; pair-Hermitian, R2/R4) | 50 MB (240 MB) |
| + RTDE at t′ | + `GR` (2N, obs, 1e-10/1e-6); `Dpp` = 𝒟^{pp}_{k0ab} (N³, tens, 1e-10/1e-6, feeds G^{R↑}); `Upp` = 𝒰^{pp}_{k0ab} (N³, tens, idem, feeds G^{R↓}) | 54 MB (253 MB) |
| frozen RTDE | `GR`, `Dpp`, `Upp`. Inputs: h_o^σ, ρ^σ row/column 0, slice 𝒦^{pp}_{ab00} (N²) | 4 MB (13 MB) |
| paramagnetic | RTDE: only `Dpp` (𝒰 = ξ𝒟 swapped, R6) and one `GR` column | — |

**m*GW* with counterterms** (sec:mGWG1G2imp, sec:mGWRTDEimp)

| run | blocks | registers |
|---|---|---|
| G1-G2 | `rho` (2N², obs); `GW_P` = 𝒢^{W,↑↓↑↓}, `GW_A` = 𝒢^{W,↑↓↓↑}, `GW_UU` = 𝒢^{W,↑↑↑↑}, `GW_DD` = 𝒢^{W,↓↓↓↓}, `GW2_P` = 𝒢^{W,2,↑↓↑↓} (N⁴ each, tens, 1e-8/1e-6) | 250 MB (1.20 GB) |
| + RTDE at t′ | + `GR` (2N, obs, 1e-10/1e-6); probe ↑: `FW_UU` = ℱ^{W,↑↑↑↑}, `FW_UA` = ℱ^{W,↑↓↓↑}, `FW_UP` = ℱ^{W,↑↓↑↓}, `FW2_UP` = ℱ^{W,2,↑↓↑↓}; probe ↓: the four flipped blocks (N³ each, tens, 1e-10/1e-6) | 266 MB (1.25 GB) |
| frozen RTDE | `GR` + the eight ℱ blocks. Inputs: h_o^σ, ρ^σ row/column 0, slices 𝒢^W_{p00o}, 𝒢^W_{0po0} of `GW_P`, `GW_A`, `GW_DD` (probe ↑) and `GW_P`, `GW_A`, `GW_UU` (probe ↓); 𝒢^{W,2} is not read (R42) | 16 MB (52 MB) |
| paramagnetic | G1-G2: `GW_P`, `GW_A`, `GW2_P` (`GW_UU` = `GW_DD` = A + P). RTDE: three N³ tensors, one probe spin | — |

**DSL** (sec:DSLG1G2imp, sec:DSLRTDEimp)

| run | blocks | registers |
|---|---|---|
| G1-G2 | `rho` (2N², obs); `G_opp` = 𝒢^{↑↓↑↓} (N⁴, tens, 1e-8/1e-6); `G_UU`, `G_DD` = 𝒢^{σσσσ}: packed (N(N−1)/2)², or full N⁴ with `rms_count` = N²(N−1)²; tens, same tol, weight 1 | 73 MB (354 MB) packed; 150 MB (720 MB) full |
| + RTDE at t′ | + `GR` (2N, obs, 1e-10/1e-6); probe ↑: `F_opp_u` = ℱ^{↓↑↑↓}_{k0ab} (N³), `F_UU` = ℱ^{↑↑↑↑}_{k0ab} (packed N·N(N−1)/2, or full N³ with `rms_count` = N·N(N−1)); probe ↓: `F_opp_d` = ℱ^{↑↓↓↑}, `F_DD`. All tens, 1e-10/1e-6, weight 1 (R50) | 79 MB (373 MB) |
| frozen RTDE | `GR` + the four ℱ blocks. Inputs: h_o^σ, ρ^σ row/column 0, eight N² slices (probe ↑: 𝒢^{↑↓↑↓}_{ab00}, _{a00k}, _{a0k0}, 𝒢^{↓↓↓↓}_{0bk0}; probe ↓: 𝒢^{↑↓↑↓}_{ba00}, _{0ak0}, _{0a0k}, 𝒢^{↑↑↑↑}_{0bk0}) (R49) | 6 MB (19 MB) |
| paramagnetic | G1-G2: `G_opp` only (𝒢^{σσσσ} = 𝒢^{↑↓↑↓} + ξ𝒢^{↑↓↑↓}_{slba}). RTDE: `F_opp` only, one probe spin | — |

**HF.** `rho` (2N², obs), plus `GR` (2N) in the RTDE. The state is below
`par_threshold`, so passes run serially. The integrator overhead is negligible.

### 9.4 What the drivers must do (not the integrator)

- **Rebuild h_o^σ every stage** from ρ (R1, R8, R37): h − i[ℓ^> + ℓ^<] −
  iγρ^{σ̄}₀₀ on orbital 0. Add +2ℓ^< in ρ̇.
- **RTDE activation at t′** (stop callback):
  - set G^R(t′⁺, t′) = −i e₀;
  - append the zeroed ℱ blocks;
  - choose co-propagation, or frozen inputs (slices plus `remove_block`).
- **Antisymmetric storage choice** and the matching `rms_count`.
- **Spin layout choice** (full vs paramagnetic) at construction. Optionally
  collapse mid-run with `remove_block` when the inputs are exactly paramagnetic.


## 10. Profiling

The profile is collected during the run and handed to the driver at the end.
It never changes results: test 11 compares states and frames bitwise with
profiling on and off, and `run_all.sh` compares a checksum between the normal
build and a `-DDP5_PROFILE=0` build.

### 10.1 Switches and API

- **Compile time:** `-DDP5_PROFILE=0` turns `prof_on()` into a constant
  `false`, so every timer and counter is compiled out. The `Profile` struct
  still exists but stays empty.
- **Run time:** `Options::profile` (default `true`) turns it on or off.
- **Optional per-thread times:** `Options::profile_threads` adds two clock reads
  per thread per parallel pass and records load imbalance (max/mean thread
  time).
- **Optional step trace:** `Options::profile_trace_max` > 0 keeps the step
  sequence (t, h, err, worst block, accepted). When the trace is full it is
  decimated by 2 and the recording stride doubles.

```cpp
const dp5::Profile& P = in.profile();    // accumulates across integrate() calls and layout changes
dp5::Profile P2 = in.take_profile();     // copy + reset
in.reset_profile();
std::string txt = P.to_text();           // log summary (example in 10.4)
P.visit([&](const std::string& name, const double* d, std::size_t n) { h5.write("/profile/" + name, d, n); });
auto m = P.to_map();                     // std::map<string, vector<double>>, 245 keys in test 11
auto s = P.strings();                    // compiler, build flags, block names, section names
```

The driver can add its own compile flags with
`-DDP5_BUILD_FLAGS="\"-O3 -march=...\""`.

### 10.2 Fields

**Timers.** `sec[SecX]` holds `seconds` (inside `integrate`),
`seconds_outside`, `calls`, and `imb_max`/`imb_mean`. Times are **exclusive**:
a section nested in another is not counted twice, e.g. a layout change inside a
stop callback is counted under layout, not under the callback. The sections:
- RHS by stage: `rhs_k2`…`rhs_k7`; `rhs_slope` (k1 at the start of a run and
  after every stop); `rhs_init_step`.
- Fused passes: `pass_stage2`…`pass_stage6`; `pass_solution` (y5);
  `pass_error` (error norm + non-finite check); `pass_init_step`; `dense`
  (dense-output construction).
- `controller` (step-size logic, bookkeeping, register rotation).
- User code: `stop_callback`, `observer`, `step_observer`, `interrupt`.
- `layout` (append/remove/extract/set_active, including reallocation and first
  touch), `checkpoint`, `restore`.

**Totals.**
- `integrate_seconds`, and `untimed_seconds` = integrate − Σ sections (loop
  bookkeeping).
- Derived: `rhs_seconds()`, `user_seconds()`,
  `overhead_seconds()` = integrate − RHS − user callbacks, `pass_seconds()`.
- The closure Σ sections + untimed = integrate is exact by construction
  (test 11: −1e-16 s). The untimed part was 0.25% (test 11) and 0.04%
  (test 9).

**Steps.**
- `attempts` = `accepted` + `rejected_error` + `rejected_nonfinite`.
- `clipped`: accepted steps shortened to land on a stop or on t_end.
- `rhs_calls` = 6·attempts + `slope_recomputes` + `init_step_estimates`
  (DP5 with FSAL: every attempt evaluates k2…k7). Test 11 checks this against
  `Stats::n_rhs`.
- `frames_dense`, `frames_exact`, `stops`, and the automatic initial steps
  `h0_auto_t`/`h0_auto_h`.
- h statistics: `h_min`, `h_max`, geometric mean, and the log₁₀ histogram
  `h_hist` (10 bins per decade from 1e-12). `h_median()` is the bin-centre
  estimate.
- Error ratio of accepted steps: `err_hist` (log₁₀, 5 bins per decade from
  1e-8), plus mean and max.
- `limiting_block[id]`: accepted steps in which block `id` had the largest
  weighted error.

**Segments.**
- A new segment starts after each stop's callbacks; the callbacks belong to the
  segment that ends.
- Per segment: t range, integrate seconds, all timers, steps, rejections,
  clipped steps, RHS calls, h range/geomean, and the state size while stepping.
- This separates the G1-G2 phase from the post-t′ phase.

**Context.** Threads, `par_threshold`, controller, combine, compiler and flags,
and every block (name, size, offset, norm, atol, rtol, weight, rms_count,
active/removed/save).

**Segment split.** Each segment exports `rhs_seconds`, `user_seconds`,
`overhead_seconds` and `pass_seconds`, with the same definitions as the
totals. `to_text()` shows them as RHS% / user% / ovh% (passes%), so the
integrator share before t′ can be compared with the share after.

**Memory (bytes).**
- `register_bytes` = 8 × state × 16, `register_bytes_peak`.
- `layout_transient_peak_bytes`: during append/remove, (old y + 8 new
  registers) × 16.
- `dense_bytes`: the dense-output buffer.
- `maxrss_bytes`: process peak RSS from `getrusage`, read at the end of every
  `integrate()`. It is −1 where `getrusage` is unavailable.
- `layout_t` / `layout_register_bytes`: history of every layout change.

**End state and failures.**
- t, h, PI memory.
- `last_status`: `status_name()` gives Ok, Interrupted, NonFinite,
  StepTooSmall, MaxSteps, or −1 = Exception.
- `exceptions` and `last_exception` (the `what()` text).
- The last `diagnostics()`: `last_err`, `last_worst_block`, `last_block_err`.
- All timers are RAII, so an exception from the RHS or a callback closes every
  open section during unwinding. `integrate()` then records the profile and
  rethrows. Test 12 checks that the timing closure and the counters stay
  consistent after an RHS exception (the aborted attempt's completed RHS calls
  are counted, not the attempt), after a stop-callback exception (followed by
  a successful continuation), and after a NonFinite return.

### 10.3 Overhead

- **Granularity:** one RAII timer (two `steady_clock::now()` reads) per section
  call. That is about 15 timed sections per attempted step, outside all inner
  loops. Step statistics cost a few floating-point operations per step.
- **Measured clock read on this laptop:** 13–26 ns.
- **Estimated instrumentation cost** (`PROFILE-COST` lines in
  `tests/bench_results.txt`):

  | state size | share of integrate time | share of integrator overhead |
  |---:|---:|---:|
  | 1e4 entries | 0.17–0.27% | 0.37% |
  | 1e5 entries | 0.03–0.06% | |
  | 1e6 entries | 0.001–0.004% | |
  | 4e6 entries | 0.0003–0.0017% | 0.0005–0.0025% |

- **Wall-clock at 4×10⁶** (`tests/bench_profile.txt`, three repeats of each
  build, overhead per step in ms):

  | build | runs | mean |
  |---|---|---:|
  | compile-off | 41.4 / 37.7 / 42.0 | 40.4 |
  | runtime-off | 42.5 / 38.9 / 43.4 | 41.6 |
  | on | 37.8 / 38.7 / 40.2 | 38.9 |
  | on + threads | 39.5 / 39.8 / 39.3 | 39.5 |

  All four lie within run-to-run noise, which is about ±7% on this laptop.
  "On" is not slower. The analytic estimate above (about 0.001%) is the
  meaningful number.
- **Bitwise:** checksums of a profiled run (stops, append, rejections,
  per-thread timing and trace on) are identical between `DP5_PROFILE=1` and
  `DP5_PROFILE=0` builds (`tests/profile_checksum.txt`).

### 10.4 Intended analysis

1. **Integrator vs RHS.**
   - If `overhead_seconds()/integrate_seconds` is large, the vector passes
     matter. Look at `pass_*`, their `imb_max/imb_mean` (with
     `profile_threads`), and the threads/NUMA settings.
   - If it is small, the RHS dominates: optimize the contractions (the
     per-stage `rhs_k*` times should be nearly equal).
   - In the synthetic tests the RHS is a trivial diagonal, so the integrator
     share appears large (60–65%). With the real O(N⁵) drifts it will be much
     smaller.
2. **Which pass dominates.**
   - Stage-pass time grows with the number of k vectors read (stage2 < … <
     stage6 ≈ solution), and `pass_error` is the largest (8 reads) — the
     4×10⁶ benchmark shows exactly that ordering.
   - A pass far off that pattern indicates bandwidth or NUMA problems.
3. **Step-size limitation by block.**
   - `limiting_block` shares, per segment through the trace, or globally.
   - Test 9: G limits all steps before t′; after t′ F limits (87% overall), and
     the segment h_geomean drops from 0.145 to 0.029.
   - A block that limits unexpectedly suggests its tolerances or weight need
     revisiting (§5, §9.2).
4. **Rejections and clipping.**
   - A rejection rate above a few percent means the controller is struggling
     (stiffness, discontinuous inputs).
   - `rejected_nonfinite` > 0 is a bug or blow-up signal; combine it with
     `diagnostics()`.
   - `clipped` counts the stops and grid landings that shortened steps.
5. **Phase comparison.** The `segments` table shows the cost of the G1-G2 phase
   against the RTDE phase: steps, h, RHS% and passes%.

Sample `to_text()` from test 9(a) (2×10⁶ G block, F appended at t′ = 15, 12
threads). Segment 0 is the G1-G2 phase; segment 1 is G1-G2 + RTDE. The state
column is the size while stepping, and RHS% / user% / ovh% split each
segment's time:

```
dp5 profile: enabled, 12 threads, par_threshold 32768, controller PI, combine MaxOverBlocks
  build: gcc-compatible 14.4.0 | optimize avx2 fma openmp=201511 DP5_PROFILE=1
  state: 2015625 complex entries; G[2000000 RMS on atol=1e-08 rtol=1e-06] F[15625 Max on atol=1e-10 rtol=1e-06 save]
  integrate: 22.5115 s in 1 call(s); RHS 8.7183 s (38.7%), user callbacks 0.0420 s (0.2%), integrator overhead 13.7512 s (61.1%; passes 13.7114 s, untimed 0.0073 s)
  section             seconds   %total    calls   us/call   imbalance(max/mean)
  rhs_slope            0.0038    0.02%        2   1886.16   -
  rhs_init_step        0.0016    0.01%        1   1594.71   -
  rhs_k2               1.5187    6.75%      789   1924.81   -
  rhs_k3               1.4458    6.42%      789   1832.43   -
  rhs_k4               1.4388    6.39%      789   1823.57   -
  rhs_k5               1.4237    6.32%      789   1804.41   -
  rhs_k6               1.4530    6.45%      789   1841.52   -
  rhs_k7               1.4331    6.37%      789   1816.31   -
  pass_stage2          1.1198    4.97%      789   1419.28   -
  pass_stage3          1.4038    6.24%      789   1779.21   -
  pass_stage4          1.6837    7.48%      789   2134.02   -
  pass_stage5          1.9922    8.85%      789   2524.92   -
  pass_stage6          2.3427   10.41%      789   2969.16   -
  pass_solution        2.3422   10.40%      789   2968.59   -
  pass_error           2.7799   12.35%      789   3523.35   -
  pass_init_step       0.0037    0.02%        1   3724.36   -
  dense                0.0433    0.19%      399    108.55   -
  controller           0.0006    0.00%      789      0.79   -
  stop_callback        0.0000    0.00%        2     17.46   -
  observer             0.0415    0.18%      401    103.42   -
  step_observer        0.0005    0.00%      789      0.61   -
  layout               0.0319    0.14%        2  22269.97   -   (+0.0126 s outside integrate)
  steps: attempts 789 = accepted 787 + rejected(err>1) 2 + rejected(non-finite) 0; clipped to stops 2; RHS calls 4737 (slope 2, h0 1); frames 399 dense + 2 exact; stops 2
  h: min 2.207e-02  median~ 2.818e-02  geomean 3.578e-02  max 1.698e-01;  err of accepted steps: mean 0.439 max 0.645; rejection rate 0.25%
  automatic h0: 2.207e-02@t=0
  limiting block (share of accepted steps): G 13.0% F 87.0%
  segments (split at stops):
    #   t_begin     t_end   seconds   accepted  rej  clip    h_geomean   state    RHS%  user%  ovh%  (passes%)
    0           0        15    2.8673        102    0     1   1.445e-01  2000000   38.2    0.0  61.8  ( 60.7)
    1          15        35   19.6442        685    2     1   2.907e-02  2015625   38.8    0.2  61.0  ( 60.9)
  memory: registers 258.0 MB (peak 258.0 MB, layout-change transient peak 290.0 MB), dense buffer 0.25 MB, process peak RSS 555.9 MB
  end: t=35 h=4.165e-02 err_old=1.147e-01 status=Ok; last step err=1.147e-01 worst block 1
```

### 10.5 Export schema (stable, versioned)

`Profile::visit()` / `to_map()` export flat numeric arrays and `strings()`
exports strings. Key `profile/schema_version` (currently **1**) versions the
key set. Rules:
- keys are only ever **added**;
- renaming a key or changing its meaning bumps the version;
- analysis scripts check the version and ignore unknown keys.

Version-1 keys (all values are arrays; scalars have length 1):

| prefix | keys |
|---|---|
| `profile/` | `schema_version` |
| `context/` | `enabled`, `threads`, `par_threshold`, `controller` (0 PI, 1 Standard), `combine` (0 MaxOverBlocks, 1 GlobalRMS), `state_size`; strings `compiler`, `build_flags` |
| `blocks/` | per block id: `size`, `offset`, `rms_count`, `norm` (0 RMS, 1 Max), `atol`, `rtol`, `weight`, `active`, `removed`, `save`; string `names` (comma separated) |
| `time/<section>/` | `seconds`, `seconds_outside`, `calls`, `imb_max`, `imb_mean`, for every section in string `time/section_names` (order = `dp5::Sec`) |
| `time/` | `integrate_seconds`, `untimed_seconds`, `integrate_calls`, `rhs_seconds`, `user_seconds`, `overhead_seconds` |
| `steps/` | `attempts`, `accepted`, `rejected_error`, `rejected_nonfinite`, `clipped`, `rhs_calls`, `slope_recomputes`, `init_step_estimates`, `frames_dense`, `frames_exact`, `stops`, `h0_auto_t`, `h0_auto_h`, `h_min`, `h_max`, `h_median`, `h_geomean`, `h_hist`, `h_hist_lo_log10`, `h_bins_per_decade`, `err_hist`, `err_hist_lo_log10`, `err_bins_per_decade`, `err_mean`, `err_max`, `limiting_block` |
| `segments/` | per segment: `t_begin`, `t_end`, `integrate_seconds`, `accepted`, `rejected_error`, `rejected_nonfinite`, `clipped`, `rhs_calls`, `h_min`, `h_max`, `h_geomean`, `state_size`, `rhs_seconds`, `user_seconds`, `overhead_seconds`, `pass_seconds`; `segments/time/<section>/seconds`, `.../calls` |
| `trace/` | `stride`, `t`, `h`, `err`, `worst_block`, `accepted` (empty unless `profile_trace_max` > 0) |
| `end/` | `t`, `h`, `err_old`, `status`, `exceptions`, `last_err`, `last_worst_block`, `last_block_err`; strings `status_name`, `last_exception` |
| `memory/` | `register_bytes`, `register_bytes_peak`, `layout_transient_peak_bytes`, `dense_bytes`, `maxrss_bytes`, `layout_t`, `layout_register_bytes` |

For HDF5, write each key as a dataset under `/profile/` (the `/` in the names
maps onto groups) and each string as an attribute.

`tests/profile_summary.py` reads a JSON dump of exactly these keys (written by
test 11), recomputes every derived number of `to_text()` from the raw keys:
- the RHS / user / overhead split, total and per segment;
- the section table;
- step counts and RHS accounting;
- the h median from the histogram;
- the limiting blocks;
- memory.

It then checks the recomputed numbers against the exported derived keys and
prints `EXPORT CONSISTENT`. That shows the export is sufficient on its own;
the output is in `tests/profile_summary_output.txt`.
