// dp5_integrator.hpp -- model-agnostic adaptive Dormand-Prince 5(4) integrator
// for large complex ODE systems  dy/dt = f(t, y),  y in C^n.
//
// Design goals (see DESIGN.md):
//   * minimal state-sized registers (8) and fused, OpenMP-parallel vector passes;
//     no state-sized copies (pointer rotation only);
//   * block layout with named blocks and stable ids, per-block tolerances,
//     weights, norms (RMS/Max) and active flags; blocks can be appended,
//     activated, removed or extracted mid-run (layout version passed to the RHS);
//   * exact landing on stop times (with user callbacks) and on an output grid,
//     4th-order dense output in between (FSAL slopes, so end slopes are exact);
//   * PI step-size control, automatic initial step, step carried across stops;
//   * status codes and per-block diagnostics for non-finite values, too-small
//     steps and step caps;
//   * checkpoint/restart hooks (full state + controller state);
//   * the static thread partition is public (dp5::parallel_for_block), so the
//     RHS's elementwise loops can touch the same memory with the same threads.
//
// Complex states: every error and scale is per complex entry z: |z| is the
// modulus, RMS uses |z|^2, atol applies to |z|.
//
// Dependencies: C++17 standard library and (optionally) OpenMP. No HDF5, Eigen,
// or physics headers.
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "dp5_profile.hpp"
#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#define DP5_HAVE_GETRUSAGE 1
#endif

namespace dp5 {

using cplx = std::complex<double>;

enum class BlockNorm { RMS, Max };
enum class Combine { MaxOverBlocks, GlobalRMS };
enum class Controller { PI, Standard };  // Standard = scipy RK45 controller

enum class Status {
  Ok = 0,          // reached t_end
  Interrupted,     // user interrupt() returned true after an accepted step
  NonFinite,       // NaN/Inf in stages or new state (after retries)
  StepTooSmall,    // h fell below h_min (or 10 ulp of t)
  MaxSteps         // more than max_steps attempted steps in this integrate() call
};

inline const char* to_string(Status s) {
  switch (s) {
    case Status::Ok: return "Ok";
    case Status::Interrupted: return "Interrupted";
    case Status::NonFinite: return "NonFinite";
    case Status::StepTooSmall: return "StepTooSmall";
    case Status::MaxSteps: return "MaxSteps";
  }
  return "?";
}

struct BlockSpec {
  std::string name;
  std::size_t size = 0;      // number of complex entries
  double atol = 1e-8;        // absolute tolerance on |z| per entry
  double rtol = 1e-6;        // relative tolerance
  double weight = 1.0;       // multiplies this block's error norm (0 = excluded)
  BlockNorm norm = BlockNorm::RMS;
  bool active = true;        // inactive blocks are frozen: no passes, no error
  bool save = false;         // emitted to the output observer on the grid
  std::size_t rms_count = 0; // RMS denominator; 0 = size. Set it to the number of
                             // structurally nonzero entries for blocks stored with
                             // redundant (anti)symmetric copies or zero diagonals.
};

// Recommended block types (DESIGN.md sec. 5): large tensor blocks are controlled
// in RMS, small observable blocks (rho, G^R) entry-wise in Max and saved.
inline BlockSpec tensor_block(std::string name, std::size_t n, double atol = 1e-8, double rtol = 1e-6) {
  BlockSpec b; b.name = std::move(name); b.size = n; b.atol = atol; b.rtol = rtol; b.norm = BlockNorm::RMS;
  return b;
}
inline BlockSpec observable_block(std::string name, std::size_t n, double atol = 1e-8, double rtol = 1e-6) {
  BlockSpec b; b.name = std::move(name); b.size = n; b.atol = atol; b.rtol = rtol; b.norm = BlockNorm::Max;
  b.save = true;
  return b;
}

struct Options {
  Controller controller = Controller::PI;
  Combine combine = Combine::MaxOverBlocks;
  double h0 = 0.0;             // initial step; <= 0: automatic (Hairer II.4)
  double h_min = 0.0;          // absolute minimum step (in addition to 10 ulp(t))
  double h_max = std::numeric_limits<double>::infinity();
  double fixed_h = 0.0;        // > 0: fixed step, no error control (testing)
  double safety = 0.9;
  double fac_min = 0.2;        // smallest step-size factor
  double fac_max = 10.0;       // largest step-size factor
  double beta = 0.04;          // PI memory exponent (Hairer dopri5 default)
  double landing_slack = 1.01; // clip to a stop if t + slack*h >= t_stop
  bool carry_h_across_landing = true;  // keep requested h after a clipped step
  bool reset_h_at_stop = false;        // recompute h0 after every stop callback
  std::size_t max_steps = 100000000;   // attempted steps per integrate() call
  int nonfinite_retries = 0;           // retry with h/4 before NonFinite abort
  std::size_t par_threshold = 32768;   // passes run serially below this many active entries
  // profiling (DESIGN.md sec. 10); compile-time off with -DDP5_PROFILE=0
  bool profile = true;                 // section timers, step statistics, per-segment split
  bool profile_threads = false;        // per-thread times inside parallel passes (load imbalance)
  std::size_t profile_trace_max = 0;   // > 0: keep a (t, h, err, worst, accepted) trace, decimated to this length
};

struct Stats {
  std::uint64_t n_accept = 0, n_reject = 0, n_rhs = 0, n_nonfinite = 0;
  std::uint64_t n_frames = 0, n_stops = 0;
  double rhs_seconds = 0.0;     // wall time spent inside the user RHS
  double total_seconds = 0.0;   // wall time spent inside integrate()
};

// Per-block error information of the most recent attempted step.
struct Diagnostics {
  double t = 0.0, h = 0.0, err = 0.0;
  std::vector<double> block_err;    // weighted block norm (NaN if non-finite), 0 for inactive
  std::vector<char> block_nonfinite;
  std::size_t worst_block = 0;      // id with the largest error (or first non-finite)
};

// Block layout as seen by the RHS. Ids are stable for the life of the
// integrator; offsets change when blocks are appended or removed, and every
// such change increments `version`.
struct Layout {
  std::uint64_t version = 0;
  std::vector<std::size_t> offset, size;
  std::vector<char> active, removed;
  std::size_t total = 0;
  std::size_t num_blocks() const { return offset.size(); }
};

// --------------------------------------------------------------------------
// Thread partition shared by integrator and RHS. The integrator splits each
// block [offset, offset+size) of complex entries statically over the threads of
// a parallel region: thread `tid` of `nth` gets [a, b) below. Registers are
// first-touched with the same split, so an RHS loop that uses the same split
// over the same block reads and writes NUMA-local memory.
inline void static_range(std::ptrdiff_t lo, std::ptrdiff_t hi, int tid, int nth, std::ptrdiff_t& a,
                         std::ptrdiff_t& b) {
  const std::ptrdiff_t n = hi - lo, q = n / nth, r = n % nth;
  a = lo + tid * q + std::min<std::ptrdiff_t>(tid, r);
  b = a + q + (tid < r ? 1 : 0);
}
inline int thread_id() {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}
inline int num_threads_in_region() {
#ifdef _OPENMP
  return omp_get_num_threads();
#else
  return 1;
#endif
}
// Inside an existing parallel region: call f(i) for this thread's slice of [lo,hi).
template <class F>
inline void for_static(std::ptrdiff_t lo, std::ptrdiff_t hi, F&& f) {
  std::ptrdiff_t a, b;
  static_range(lo, hi, thread_id(), num_threads_in_region(), a, b);
  for (std::ptrdiff_t i = a; i < b; ++i) f(i);
}
// Open a parallel region and call f(i) for every local index i in [0, size) of
// block `id` (index relative to the block start), using the integrator's split.
// Use it for elementwise RHS work (drifts with diagonal pieces, sources, masks,
// copies); contractions done inside a BLAS/Eigen call use their own split.
template <class F>
inline void parallel_for_block(const Layout& L, std::size_t id, F&& f, std::size_t par_threshold = 32768) {
  const std::ptrdiff_t n = std::ptrdiff_t(L.size[id]);
#pragma omp parallel if (std::size_t(n) >= par_threshold)
  {
    std::ptrdiff_t a, b;
    static_range(0, n, thread_id(), num_threads_in_region(), a, b);
    for (std::ptrdiff_t i = a; i < b; ++i) f(i);
  }
}
// NOTE: the integrator splits [offset, offset+size); parallel_for_block splits
// [0, size). Both give thread t the same entries (the split depends only on n).

struct Checkpoint {
  double t = 0.0, h = 0.0, err_old = 1e-4;
  std::size_t grid_next = 0;
  std::vector<char> active, removed;
  std::vector<std::size_t> sizes;
  Stats stats;
};

// A frame handed to the output observer: pointers to the saved blocks (nullptr
// for blocks that are not saved or removed).
struct Frame {
  std::size_t k = 0;  // grid index
  double t = 0.0;     // nominal grid time
  std::vector<const cplx*> block;
  const cplx* operator[](std::size_t id) const { return block[id]; }
};

template <class F> struct is_nullable : std::is_pointer<F> {};
template <class S> struct is_nullable<std::function<S>> : std::true_type {};

class Integrator {
 public:
  using Rhs = std::function<void(double t, const cplx* y, cplx* dy)>;
  using RhsL = std::function<void(double t, const cplx* y, cplx* dy, const Layout& L)>;
  using StopCallback = std::function<void(Integrator&)>;
  using Observer = std::function<void(const Frame&)>;
  using StepObserver = std::function<void(double t_old, double h, double err, bool accepted)>;

  // F is either void(double, const cplx*, cplx*) or void(double, const cplx*,
  // cplx*, const Layout&). The second form receives the current layout.
  template <class F>
  Integrator(std::vector<BlockSpec> blocks, F rhs, Options opt = {}, double t0 = 0.0) : opt_(opt), t_(t0) {
    if constexpr (is_nullable<F>::value) {
      if (!rhs) throw std::invalid_argument("dp5: RHS must be set");
    }
    if constexpr (std::is_invocable_v<F&, double, const cplx*, cplx*, const Layout&>) {
      rhs_ = std::move(rhs);
    } else {
      static_assert(std::is_invocable_v<F&, double, const cplx*, cplx*>, "dp5: RHS signature");
      rhs_ = [f = std::move(rhs)](double t, const cplx* y, cplx* dy, const Layout&) mutable { f(t, y, dy); };
    }
    if (!std::isfinite(t0)) throw std::invalid_argument("dp5: t0 not finite");
    validate_options();
    if (blocks.empty()) throw std::invalid_argument("dp5: need at least one block");
    for (auto& b : blocks) add_spec(std::move(b));
    prof_reset_counters();
    PTimer tm(this, SecLayout);
    relayout(std::vector<std::size_t>(blocks_.size(), npos));
  }

  // ---------------------------------------------------------------- layout
  std::size_t size() const { return L_.total; }
  std::size_t num_blocks() const { return blocks_.size(); }
  const Layout& layout() const { return L_; }
  std::uint64_t layout_version() const { return L_.version; }
  const BlockSpec& spec(std::size_t id) const { return live(id).spec; }
  std::size_t offset(std::size_t id) const { live(id); return L_.offset[id]; }
  bool removed(std::size_t id) const { return blocks_.at(id).removed; }
  std::size_t block_id(const std::string& name) const {
    for (std::size_t i = 0; i < blocks_.size(); ++i)
      if (!blocks_[i].removed && blocks_[i].spec.name == name) return i;
    throw std::invalid_argument("dp5: unknown block '" + name + "'");
  }
  // Mutable access to the current state of a block (e.g. in a stop callback).
  // Call state_changed() afterwards if called outside a stop callback.
  cplx* block(std::size_t id) { live(id); return y() + L_.offset[id]; }
  const cplx* block(std::size_t id) const { live(id); return y() + L_.offset[id]; }
  cplx* y() { return reinterpret_cast<cplx*>(R_[Y].get()); }
  const cplx* y() const { return reinterpret_cast<const cplx*>(R_[Y].get()); }
  double t() const { return t_; }
  double h() const { return h_; }
  // f(t, y) at the current state if it is cached (always true right after an accepted step: FSAL slope),
  // else nullptr. Read-only; valid until the next step or layout change.
  const cplx* slope() const { return k1_valid_ ? reinterpret_cast<const cplx*>(R_[K1].get()) : nullptr; }
  const Stats& stats() const { return stats_; }
  const Diagnostics& diagnostics() const { return diag_; }

  // ---------------------------------------------------------------- profiling
  // Accumulates across integrate() calls and layout changes until reset.
  const Profile& profile() const { prof_context(); return prof_; }
  Profile take_profile() { prof_context(); Profile p = prof_; reset_profile(); return p; }
  void reset_profile() { prof_reset_counters(); }
  static constexpr std::size_t registers() { return NREG; }
  std::size_t register_bytes() const { return NREG * L_.total * sizeof(cplx); }

  void set_active(std::size_t id, bool on) {
    PTimer tm(this, SecLayout);
    live(id).spec.active = on;
    update_layout(false);
    state_changed();
  }
  void set_tolerances(std::size_t id, double atol, double rtol) {
    check_tol(atol, rtol);
    live(id).spec.atol = atol;
    live(id).spec.rtol = rtol;
  }
  void set_norm(std::size_t id, BlockNorm n) { live(id).spec.norm = n; }
  void set_save(std::size_t id, bool on) {
    live(id).spec.save = on; resize_dense();
    if (prof_on()) prof_.dense_bytes = dense_.size() * sizeof(cplx);
  }

  // Append a block at the end of the state. Registers are reallocated (only y
  // survives; peak memory = old y + 8 new registers); the new block is zero.
  // Offsets change -> layout version increments. Returns the new (stable) id.
  std::size_t append_block(BlockSpec b) {
    PTimer tm(this, SecLayout);
    std::vector<std::size_t> old = L_.offset;
    add_spec(std::move(b));
    old.push_back(npos);
    relayout(old);
    state_changed();
    return blocks_.size() - 1;
  }
  // Append several blocks with ONE register reallocation (append_block per block
  // reallocates and copies all 8 registers each time: O(#blocks x state) work).
  // The resulting layout, ids and values are identical to appending them one by
  // one in the same order (new blocks zero); the layout version increments once.
  // All-or-nothing: if any spec is invalid, nothing is appended.
  std::vector<std::size_t> append_blocks(std::vector<BlockSpec> bs) {
    PTimer tm(this, SecLayout);
    std::vector<std::size_t> ids;
    if (bs.empty()) return ids;
    const std::size_t n0 = blocks_.size();
    try {
      for (auto& b : bs) { add_spec(std::move(b)); ids.push_back(blocks_.size() - 1); }
    } catch (...) {
      blocks_.resize(n0);
      throw;
    }
    std::vector<std::size_t> old = L_.offset;
    old.resize(blocks_.size(), npos);
    relayout(old);
    state_changed();
    return ids;
  }
  // Remove a block from the state (its memory leaves all 8 registers). The id
  // stays reserved (removed); later blocks keep their ids but move.
  void remove_block(std::size_t id) {
    PTimer tm(this, SecLayout);
    live(id);
    std::vector<std::size_t> old = L_.offset;
    blocks_[id].removed = true;
    blocks_[id].spec.active = false;
    relayout(old);
    state_changed();
  }
  // Copy a block out (e.g. as a frozen RHS input) and remove it from the state.
  std::vector<cplx> extract_block(std::size_t id) {
    PTimer tm(this, SecLayout);
    std::vector<cplx> out(block(id), block(id) + spec(id).size);
    remove_block(id);
    return out;
  }

  // Invalidate the cached slope after the state was modified from outside.
  void state_changed() {
    k1_valid_ = false;
    sync_inactive();
  }
  void reset_step_size() { h_ = 0.0; }

  // ---------------------------------------------------------------- output
  // Uniform grid t_k = t0 + k (t1 - t0)/(n-1), k = 0..n-1, with t_{n-1} == t1
  // exactly. Points are emitted once each, in order.
  void set_output_grid(double t0, double t1, std::size_t n, Observer obs) {
    if (n == 0) throw std::invalid_argument("dp5: grid needs n >= 1");
    if (!std::isfinite(t0) || !std::isfinite(t1)) throw std::invalid_argument("dp5: grid not finite");
    if (n > 1 && !(t1 > t0)) throw std::invalid_argument("dp5: grid needs t1 > t0");
    if (t0 < t_ - land_tol(t_)) throw std::invalid_argument("dp5: grid starts before current time");
    if (!obs) throw std::invalid_argument("dp5: grid observer must be set");
    const double dt = (n > 1) ? (t1 - t0) / double(n - 1) : 0.0;
    // grid points must be resolvable: spacing well above the landing tolerance
    if (n > 1 && !(dt > 8.0 * land_tol(std::max(std::fabs(t0), std::fabs(t1)))))
      throw std::invalid_argument("dp5: grid spacing below time resolution");
    g_t0_ = t0; g_t1_ = (n > 1) ? t1 : t0; g_n_ = n; g_next_ = 0; obs_ = std::move(obs); g_dt_ = dt;
    resize_dense();
  }
  // Uniform grid with a given spacing: t_k = t0 + k dt for every k (also the last). The grid times then do not
  // depend on the grid length, so a run extended to a later end has the same frame times.
  // k_first > 0 skips the first points (a restart inside the grid); t0 + k_first dt must not lie before t().
  void set_output_grid_dt(double t0, double dt, std::size_t n, Observer obs, std::size_t k_first = 0) {
    if (!(dt > 0) || !std::isfinite(dt)) throw std::invalid_argument("dp5: grid spacing must be > 0");
    if (k_first >= n) throw std::invalid_argument("dp5: grid k_first beyond the grid");
    const double t1 = t0 + double(n - 1) * dt, tf = t0 + double(k_first) * dt;
    if (tf < t_ - land_tol(t_)) throw std::invalid_argument("dp5: grid starts before current time");
    if (!std::isfinite(t0) || !std::isfinite(t1)) throw std::invalid_argument("dp5: grid not finite");
    if (!obs) throw std::invalid_argument("dp5: grid observer must be set");
    if (n > 1 && !(dt > 8.0 * land_tol(std::max(std::fabs(t0), std::fabs(t1)))))
      throw std::invalid_argument("dp5: grid spacing below time resolution");
    g_t0_ = t0; g_t1_ = t1; g_n_ = n; g_dt_ = (n > 1) ? dt : 0.0; g_next_ = k_first; obs_ = std::move(obs);
    resize_dense();
  }
  double grid_time(std::size_t k) const {
    return (k + 1 == g_n_) ? g_t1_ : g_t0_ + double(k) * g_dt_;
  }
  std::size_t grid_next() const { return g_next_; }

  void set_step_observer(StepObserver s) { step_obs_ = std::move(s); }
  void set_interrupt(std::function<bool()> f) { interrupt_ = std::move(f); }

  // ---------------------------------------------------------------- stops
  // The integrator lands exactly on t_stop, runs cb (which may modify the
  // state, (de)activate, append, remove or extract blocks, change tolerances),
  // recomputes the slope, and then emits a grid frame at t_stop if there is
  // one (so the frame shows the post-callback state).
  void add_stop(double ts, StopCallback cb) {
    if (!std::isfinite(ts)) throw std::invalid_argument("dp5: stop time not finite");
    if (ts < t_ - land_tol(t_)) throw std::invalid_argument("dp5: stop time in the past");
    stops_.push_back({ts, std::move(cb), seq_++});
    std::stable_sort(stops_.begin(), stops_.end(),
                     [](const StopEntry& a, const StopEntry& b) { return a.t < b.t; });
  }

  // ---------------------------------------------------------------- checkpoint
  Checkpoint checkpoint() const {
    PTimer tm(this, SecCheckpoint);
    Checkpoint c;
    c.t = t_; c.h = h_; c.err_old = err_old_; c.grid_next = g_next_; c.stats = stats_;
    for (auto& b : blocks_) {
      c.active.push_back(b.spec.active); c.removed.push_back(b.removed);
      c.sizes.push_back(b.removed ? 0 : b.spec.size);
    }
    return c;
  }
  // The layout (blocks, appended/removed blocks) must already match.
  void restore(const Checkpoint& c, const cplx* state) {
    PTimer tm(this, SecRestore);
    if (c.sizes.size() != blocks_.size()) throw std::invalid_argument("dp5: restore: block count mismatch");
    for (std::size_t i = 0; i < blocks_.size(); ++i)
      if (c.sizes[i] != (blocks_[i].removed ? 0 : blocks_[i].spec.size) || bool(c.removed[i]) != blocks_[i].removed)
        throw std::invalid_argument("dp5: restore: block layout mismatch");
    if (!std::isfinite(c.t) || !(c.h >= 0.0)) throw std::invalid_argument("dp5: restore: bad controller state");
    for (std::size_t i = 0; i < blocks_.size(); ++i) blocks_[i].spec.active = c.active[i];
    update_layout(false);
    std::memcpy(R_[Y].get(), state, L_.total * sizeof(cplx));
    t_ = c.t; h_ = c.h; err_old_ = c.err_old; g_next_ = c.grid_next; stats_ = c.stats;
    state_changed();
  }

  // ---------------------------------------------------------------- integrate
  Status integrate(double t_end) {
    if (!std::isfinite(t_end)) throw std::invalid_argument("dp5: t_end not finite");
    if (t_end < t_ - land_tol(t_)) throw std::invalid_argument("dp5: t_end before current time");
    const double wall0 = now();
    const Clock::time_point c0 = Clock::now();
    const double acc0 = prof_acc_;
    in_integrate_ = true;
    seg_mark_ = c0;
    Status st = Status::Ok;
    try {
      st = run(t_end);
    } catch (const std::exception& e) {
      finish_integrate(wall0, c0, acc0, -1, e.what());
      throw;
    } catch (...) {
      finish_integrate(wall0, c0, acc0, -1, "unknown exception");
      throw;
    }
    finish_integrate(wall0, c0, acc0, int(st), nullptr);
    return st;
  }

  // Time tolerance used to identify grid points / stops with a landing time.
  static double land_tol(double t) {
    return 16.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::fabs(t));
  }

 private:
  // registers: Y, T(stage input / new state), K1..K6; K7 lives in K2's slot.
  enum { Y = 0, T = 1, K1 = 2, K2 = 3, K3 = 4, K4 = 5, K5 = 6, K6 = 7, NREG = 8 };
  static constexpr std::size_t npos = std::size_t(-1);

  struct Block { BlockSpec spec; bool removed = false; };
  struct StopEntry { double t; StopCallback cb; std::uint64_t seq; };
  struct AlignedDel { void operator()(double* p) const { ::operator delete[](p, std::align_val_t(64)); } };
  using Buf = std::unique_ptr<double[], AlignedDel>;
  struct Red { double sumsq = 0, mx = 0, chk = 0; };

  // ------------------------------------------------------------ profiling internals
  using Clock = std::chrono::steady_clock;
  static double secs(Clock::duration d) { return std::chrono::duration<double>(d).count(); }
  bool prof_on() const {
#if DP5_PROFILE
    return opt_.profile;
#else
    return false;
#endif
  }
  // RAII section timer. Exclusive time: nested timers are subtracted from the
  // enclosing one via the running total prof_acc_ of timed time.
  struct PTimer {
    const Integrator* it; int sec; bool on; Clock::time_point t0; double acc0 = 0.0;
    PTimer(const Integrator* i, int s) : it(i), sec(s), on(s >= 0 && i->prof_on()) {
      if (on) { acc0 = it->prof_acc_; t0 = Clock::now(); }
    }
    ~PTimer() { if (on) it->prof_stop(sec, t0, acc0); }
    PTimer(const PTimer&) = delete; PTimer& operator=(const PTimer&) = delete;
  };
  void prof_stop(int sec, Clock::time_point t0, double acc0) const {
    const double el = secs(Clock::now() - t0);
    const double excl = el - (prof_acc_ - acc0);
    prof_acc_ = acc0 + el;
    TimerStat& g = prof_.sec[sec];
    TimerStat& q = prof_.segments.back().sec[sec];
    if (in_integrate_) { g.seconds += excl; q.seconds += excl; }
    else { g.seconds_outside += excl; q.seconds_outside += excl; }
    ++g.calls; ++q.calls;
  }
  void prof_imbalance(int sec, int used) const {
    double mx = 0, sum = 0;
    for (int i = 0; i < used; ++i) { mx = std::max(mx, thr_time_[std::size_t(i)]); sum += thr_time_[std::size_t(i)]; }
    const double mean = used ? sum / used : 0.0;
    prof_.sec[sec].imb_max += mx; prof_.sec[sec].imb_mean += mean;
    prof_.segments.back().sec[sec].imb_max += mx; prof_.segments.back().sec[sec].imb_mean += mean;
  }
  // Bookkeeping at the end of integrate(), also on exceptions (all section timers
  // have been closed by their destructors during unwinding).
  void finish_integrate(double wall0, Clock::time_point c0, double acc0, int status, const char* what) {
    stats_.total_seconds += now() - wall0;
    in_integrate_ = false;
    if (!prof_on()) return;
    const auto c1 = Clock::now();
    const double el = secs(c1 - c0);
    prof_.integrate_seconds += el;
    prof_.untimed_seconds += el - (prof_acc_ - acc0);
    prof_acc_ = acc0 + el;
    ++prof_.integrate_calls;
    prof_.segments.back().integrate_seconds += secs(c1 - seg_mark_);
    prof_.t_last = t_; prof_.h_last = h_; prof_.err_old_last = err_old_; prof_.last_status = status;
    if (status == -1) { ++prof_.exceptions; prof_.last_exception = what ? what : ""; }
    prof_.last_err = diag_.err;
    prof_.last_worst_block = diag_.block_err.empty() ? -1 : int(diag_.worst_block);
    prof_.last_block_err = diag_.block_err;
#ifdef DP5_HAVE_GETRUSAGE
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
#ifdef __APPLE__
      prof_.maxrss_bytes = (long long)ru.ru_maxrss;
#else
      prof_.maxrss_bytes = (long long)ru.ru_maxrss * 1024;
#endif
    }
#endif
  }
  void prof_memory(std::size_t old_total) {
    if (!prof_on()) return;
    const std::size_t regs = NREG * L_.total * sizeof(cplx);
    const std::size_t transient = (old_total + NREG * L_.total) * sizeof(cplx);
    prof_.register_bytes = regs;
    prof_.register_bytes_peak = std::max(prof_.register_bytes_peak, regs);
    prof_.layout_transient_peak_bytes = std::max(prof_.layout_transient_peak_bytes, transient);
    prof_.dense_bytes = dense_.size() * sizeof(cplx);
    prof_.layout_t.push_back(t_);
    prof_.layout_register_bytes.push_back(double(regs));
  }
  void prof_reset_counters() {
    Profile fresh;
    prof_ = fresh;
    prof_.register_bytes = prof_.register_bytes_peak = NREG * L_.total * sizeof(cplx);
    prof_.dense_bytes = dense_.size() * sizeof(cplx);
    SegmentProfile seg; seg.t_begin = seg.t_end = t_; seg.state_size = L_.total;
    prof_.segments.push_back(seg);
    prof_.limiting_block.assign(blocks_.size(), 0);
    trace_count_ = 0;
  }
  void prof_new_segment() {
    if (!prof_on()) return;
    const auto c = Clock::now();
    SegmentProfile& old = prof_.segments.back();
    old.t_end = t_;
    if (in_integrate_) { old.integrate_seconds += secs(c - seg_mark_); seg_mark_ = c; }
    SegmentProfile seg; seg.t_begin = seg.t_end = t_; seg.state_size = L_.total;
    prof_.segments.push_back(seg);
  }
  void prof_context() const {
    prof_.enabled = prof_on();
    prof_.threads = max_threads();
    prof_.par_threshold = opt_.par_threshold;
    prof_.controller = int(opt_.controller);
    prof_.combine = int(opt_.combine);
    prof_.state_size = L_.total;
    if (!prof_.segments.empty()) prof_.segments.back().t_end = t_;
    prof_.blocks.clear();
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
      const auto& b = blocks_[i];
      BlockInfo bi; bi.name = b.spec.name; bi.size = b.removed ? 0 : b.spec.size; bi.offset = L_.offset[i];
      bi.rms_count = b.spec.rms_count; bi.norm = int(b.spec.norm); bi.atol = b.spec.atol; bi.rtol = b.spec.rtol;
      bi.weight = b.spec.weight; bi.active = L_.active[i]; bi.removed = b.removed; bi.save = b.spec.save;
      prof_.blocks.push_back(bi);
    }
#ifdef __VERSION__
    prof_.compiler = std::string("gcc-compatible ") + __VERSION__;
#endif
    std::string f;
#ifdef __OPTIMIZE__
    f += "optimize ";
#endif
#ifdef __FAST_MATH__
    f += "fast-math ";
#endif
#ifdef __AVX512F__
    f += "avx512f ";
#elif defined(__AVX2__)
    f += "avx2 ";
#endif
#ifdef __FMA__
    f += "fma ";
#endif
#ifdef _OPENMP
    f += "openmp=" + std::to_string(_OPENMP) + " ";
#endif
    f += "DP5_PROFILE=" + std::to_string(DP5_PROFILE);
#ifdef DP5_BUILD_FLAGS
    f += std::string(" | ") + DP5_BUILD_FLAGS;
#endif
    prof_.build_flags = f;
  }
  // per attempted step
  void prof_step(double h, double err, bool accepted, bool nonfinite, bool clipped) {
    if (!prof_on()) return;
    Profile& P = prof_;
    SegmentProfile& G = P.segments.back();
    ++P.attempts;
    G.state_size = L_.total;  // size while stepping in this segment
    if (nonfinite) { ++P.rejected_nonfinite; ++G.rejected_nonfinite; }
    else if (!accepted) { ++P.rejected_error; ++G.rejected_error; }
    else {
      ++P.accepted; ++G.accepted;
      if (clipped) { ++P.clipped; ++G.clipped; }
      P.h_min = std::min(P.h_min, h); P.h_max = std::max(P.h_max, h); P.log_h_sum += std::log(h);
      G.h_min = std::min(G.h_min, h); G.h_max = std::max(G.h_max, h); G.log_h_sum += std::log(h);
      int hb = int(std::floor((std::log10(h) - Profile::h_hist_lo) * Profile::h_bins_per_decade));
      hb = std::max(0, std::min(Profile::h_hist_bins - 1, hb));
      ++P.h_hist[std::size_t(hb)];
      int eb = (err > 0) ? int(std::floor((std::log10(err) - Profile::err_hist_lo) * Profile::err_bins_per_decade)) : 0;
      eb = std::max(0, std::min(Profile::err_hist_bins - 1, eb));
      ++P.err_hist[std::size_t(eb)];
      P.err_sum += err; P.err_max = std::max(P.err_max, err);
      if (P.limiting_block.size() < blocks_.size()) P.limiting_block.resize(blocks_.size(), 0);
      ++P.limiting_block[diag_.worst_block];
    }
    if (opt_.profile_trace_max > 0) {
      if (trace_count_ % P.trace_stride == 0) {
        P.trace_t.push_back(t_); P.trace_h.push_back(h); P.trace_err.push_back(err);
        P.trace_worst.push_back(double(diag_.worst_block)); P.trace_accepted.push_back(accepted ? 1.0 : 0.0);
        if (P.trace_t.size() >= opt_.profile_trace_max) {  // decimate: keep every other entry
          auto half = [](std::vector<double>& v) {
            std::size_t w = 0;
            for (std::size_t r = 0; r < v.size(); r += 2) v[w++] = v[r];
            v.resize(w);
          };
          half(P.trace_t); half(P.trace_h); half(P.trace_err); half(P.trace_worst); half(P.trace_accepted);
          P.trace_stride *= 2;
        }
      }
      ++trace_count_;
    }
  }

  RhsL rhs_;
  Options opt_;
  std::vector<Block> blocks_;
  Layout L_;
  std::size_t active_size_ = 0;
  Buf R_[NREG];
  double t_ = 0.0, h_ = 0.0, err_old_ = 1e-4;
  bool k1_valid_ = false;
  Stats stats_;
  Diagnostics diag_;
  mutable Profile prof_;
  mutable double prof_acc_ = 0.0;            // running total of timed (inclusive) time
  bool in_integrate_ = false;
  Clock::time_point seg_mark_{};
  mutable std::vector<double> thr_time_;    // per-thread pass times (profile_threads)
  std::uint64_t trace_count_ = 0;
  std::vector<StopEntry> stops_;
  std::uint64_t seq_ = 0;
  // grid
  double g_t0_ = 0, g_t1_ = 0, g_dt_ = 0;
  std::size_t g_n_ = 0, g_next_ = 0;
  Observer obs_;
  std::vector<cplx> dense_;
  StepObserver step_obs_;
  std::function<bool()> interrupt_;

  // ------------------------------------------------------------ helpers
  static double now() {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return 0.0;
#endif
  }
  static int max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
  }
  Block& live(std::size_t id) {
    Block& b = blocks_.at(id);
    if (b.removed) throw std::invalid_argument("dp5: block '" + b.spec.name + "' was removed");
    return b;
  }
  const Block& live(std::size_t id) const {
    const Block& b = blocks_.at(id);
    if (b.removed) throw std::invalid_argument("dp5: block '" + b.spec.name + "' was removed");
    return b;
  }
  static void check_tol(double atol, double rtol) {
    if (!(std::isfinite(atol) && std::isfinite(rtol) && atol >= 0 && rtol >= 0 && (atol > 0 || rtol > 0)))
      throw std::invalid_argument("dp5: tolerances must be finite, >= 0, not both 0");
  }
  void validate_options() const {
    const Options& o = opt_;
    auto bad = [](const char* m) { throw std::invalid_argument(std::string("dp5: option ") + m); };
    if (!(o.safety > 0 && o.safety <= 1)) bad("safety in (0,1]");
    if (!(o.fac_min > 0 && o.fac_min < 1)) bad("fac_min in (0,1)");
    if (!(o.fac_max > 1 && std::isfinite(o.fac_max))) bad("fac_max > 1");
    if (!(o.beta >= 0 && o.beta < 0.2)) bad("beta in [0,0.2)");
    if (!(o.h_min >= 0 && std::isfinite(o.h_min))) bad("h_min >= 0");
    if (!(o.h_max > 0)) bad("h_max > 0");
    if (!(o.h0 >= 0 && std::isfinite(o.h0))) bad("h0 >= 0");
    if (!(o.fixed_h >= 0 && std::isfinite(o.fixed_h))) bad("fixed_h >= 0");
    if (!(o.landing_slack >= 1 && o.landing_slack < 2)) bad("landing_slack in [1,2)");
    if (o.max_steps == 0) bad("max_steps > 0");
    if (o.nonfinite_retries < 0) bad("nonfinite_retries >= 0");
  }
  void add_spec(BlockSpec b) {
    if (b.size == 0) throw std::invalid_argument("dp5: block '" + b.name + "' has size 0");
    if (b.name.empty()) throw std::invalid_argument("dp5: block name empty");
    for (auto& x : blocks_)
      if (!x.removed && x.spec.name == b.name) throw std::invalid_argument("dp5: duplicate block name '" + b.name + "'");
    check_tol(b.atol, b.rtol);
    if (!(b.weight >= 0 && std::isfinite(b.weight))) throw std::invalid_argument("dp5: weight must be finite, >= 0");
    if (b.rms_count > b.size) throw std::invalid_argument("dp5: rms_count > size");
    Block blk; blk.spec = std::move(b);
    blocks_.push_back(std::move(blk));
  }
  // Recompute offsets (if `move`) and active size; bump version on moves.
  void update_layout(bool move) {
    const std::size_t nb = blocks_.size();
    L_.offset.resize(nb); L_.size.resize(nb); L_.active.resize(nb); L_.removed.resize(nb);
    if (move) {
      std::size_t off = 0;
      for (std::size_t i = 0; i < nb; ++i) {
        L_.removed[i] = blocks_[i].removed;
        L_.size[i] = blocks_[i].removed ? 0 : blocks_[i].spec.size;
        L_.offset[i] = off;
        off += L_.size[i];
      }
      L_.total = off;
      ++L_.version;
    }
    active_size_ = 0;
    for (std::size_t i = 0; i < nb; ++i) {
      L_.active[i] = !blocks_[i].removed && blocks_[i].spec.active;
      if (L_.active[i]) active_size_ += L_.size[i];
    }
  }
  bool par() const { return active_size_ >= opt_.par_threshold; }
  double* R(int r) { return R_[r].get(); }

  // New layout: allocate 8 registers with parallel first touch (same split as
  // every pass), copy y of surviving blocks from their old offsets (old_off[i]
  // == npos: new block, zero). Only y survives; the other old registers are
  // freed first, so the peak is old y + 8 new registers.
  void relayout(const std::vector<std::size_t>& old_off) {
    for (int r = 0; r < NREG; ++r) if (r != Y) R_[r].reset();
    Buf oldY = std::move(R_[Y]);
    const std::size_t old_total = oldY ? L_.total : 0;
    update_layout(true);
    const std::size_t n = std::max<std::size_t>(L_.total, 1);
    Buf fresh[NREG];
    for (int r = 0; r < NREG; ++r)
      fresh[r] = Buf(static_cast<double*>(::operator new[](2 * n * sizeof(double), std::align_val_t(64))));
    const double* oy = oldY ? oldY.get() : nullptr;
#pragma omp parallel if (L_.total >= opt_.par_threshold)
    {
      const int me = thread_id(), nth = num_threads_in_region();
      for (std::size_t q = 0; q < blocks_.size(); ++q) {
        if (blocks_[q].removed) continue;
        std::ptrdiff_t a, e;
        static_range(std::ptrdiff_t(L_.offset[q]), std::ptrdiff_t(L_.offset[q] + L_.size[q]), me, nth, a, e);
        const bool keep = oy && old_off[q] != npos;
        const std::ptrdiff_t shift = keep ? std::ptrdiff_t(old_off[q]) - std::ptrdiff_t(L_.offset[q]) : 0;
        for (std::ptrdiff_t i = a; i < e; ++i)
          for (int r = 0; r < NREG; ++r) {
            double* d = fresh[r].get() + 2 * i;
            if (keep && (r == Y || r == T)) { d[0] = oy[2 * (i + shift)]; d[1] = oy[2 * (i + shift) + 1]; }
            else { d[0] = 0.0; d[1] = 0.0; }
          }
      }
    }
    for (int r = 0; r < NREG; ++r) R_[r] = std::move(fresh[r]);
    k1_valid_ = false;
    resize_dense();
    diag_.block_err.assign(blocks_.size(), 0.0);
    diag_.block_nonfinite.assign(blocks_.size(), 0);
    if (!prof_.segments.empty()) {  // only if no step was taken in this segment yet
      SegmentProfile& g = prof_.segments.back();
      if (g.accepted + g.rejected_error + g.rejected_nonfinite == 0) g.state_size = L_.total;
    }
    prof_memory(old_total);
    if (prof_.limiting_block.size() < blocks_.size()) prof_.limiting_block.resize(blocks_.size(), 0);
  }
  void resize_dense() {
    std::size_t s = 0;
    for (auto& b : blocks_) if (!b.removed && b.spec.save) s += b.spec.size;
    dense_.assign(s, cplx(0.0, 0.0));
  }
  // Inactive blocks must read the same values in Y and in the stage register T.
  // Passes never write inactive entries, so this is needed only after the
  // state was modified from outside (state_changed), not per stage.
  void sync_inactive() {
    for (std::size_t q = 0; q < blocks_.size(); ++q)
      if (!blocks_[q].removed && !blocks_[q].spec.active)
        std::memcpy(R(T) + 2 * L_.offset[q], R(Y) + 2 * L_.offset[q], 2 * L_.size[q] * sizeof(double));
  }

  // Generic fused pass over all active entries (complex index i); the body
  // handles doubles 2i and 2i+1.
  template <class F>
  void pass(int sec, F&& body) {
    PTimer tm(this, sec);
    const bool thr = sec >= 0 && prof_on() && opt_.profile_threads && par();
    if (thr) thr_time_.assign(std::size_t(max_threads()), 0.0);
    int used = 1;
#pragma omp parallel if (par())
    {
      const Clock::time_point c0 = thr ? Clock::now() : Clock::time_point{};
      const int me = thread_id(), nth = num_threads_in_region();
      for (std::size_t q = 0; q < blocks_.size(); ++q) {
        if (!L_.active[q]) continue;
        std::ptrdiff_t a, e;
        static_range(std::ptrdiff_t(L_.offset[q]), std::ptrdiff_t(L_.offset[q] + L_.size[q]), me, nth, a, e);
#pragma omp simd
        for (std::ptrdiff_t i = a; i < e; ++i) { body(2 * i); body(2 * i + 1); }
      }
      if (thr) { thr_time_[std::size_t(me)] = secs(Clock::now() - c0); if (me == 0) used = nth; }
    }
    if (thr) prof_imbalance(sec, used);
  }

  void call_rhs(double t, int in, int out, int sec) {
    {
      PTimer tm(this, sec);
      const double w0 = now();
      rhs_(t, reinterpret_cast<const cplx*>(R(in)), reinterpret_cast<cplx*>(R(out)), L_);
      stats_.rhs_seconds += now() - w0;
    }
    ++stats_.n_rhs;
    if (prof_on()) {
      ++prof_.rhs_calls; ++prof_.segments.back().rhs_calls;
      if (sec == SecRhsSlope) ++prof_.slope_recomputes;
      if (sec == SecRhsInitStep) ++prof_.init_step_estimates;
    }
  }

  // Combine per-block reductions into the step error; fills diagnostics.
  double combine(const std::vector<Red>& red, bool fill_diag) {
    double out = 0.0, ssum = 0.0, nsum = 0.0;
    bool nan = false;
    std::size_t worst = 0;
    double worst_v = -1.0;
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
      double nb = 0.0;
      bool bad = false;
      if (L_.active[b]) {
        const auto& s = blocks_[b].spec;
        const double cnt = double(s.rms_count ? s.rms_count : s.size);
        bad = std::isnan(red[b].sumsq) || std::isnan(red[b].chk);
        nb = (s.norm == BlockNorm::RMS) ? std::sqrt(red[b].sumsq / cnt) : std::sqrt(red[b].mx);
        nb *= s.weight;
        if (bad) nb = std::numeric_limits<double>::quiet_NaN();
        if (s.weight > 0) {
          if (opt_.combine == Combine::MaxOverBlocks) out = std::max(out, nb);
          else { ssum += s.weight * s.weight * red[b].sumsq; nsum += cnt; }
        }
        if (bad && !nan) { nan = true; worst = b; }
        if (!nan && nb > worst_v) { worst_v = nb; worst = b; }
      }
      if (fill_diag) { diag_.block_err[b] = nb; diag_.block_nonfinite[b] = bad; }
    }
    if (opt_.combine == Combine::GlobalRMS) out = (nsum > 0) ? std::sqrt(ssum / nsum) : 0.0;
    if (nan) out = std::numeric_limits<double>::quiet_NaN();
    if (fill_diag) { diag_.worst_block = worst; diag_.err = out; }
    return out;
  }

  // Per-block reduction over active entries of r_i^2 = |v_i|^2 / sc_i^2 with
  // v_i = a_i + cb*b_i, sc_i = atol + rtol*max(|y_i|, |z_i|) (z optional), plus
  // a non-finite check on z. Deterministic: per-thread partials summed in order.
  template <class V>
  std::vector<Red> reduce(V&& vfun, const double* yv, const double* zv, int sec = -1) {
    const std::size_t nb = blocks_.size();
    const int TT = max_threads();
    std::vector<Red> loc(std::size_t(TT) * nb);
    int used = 1;
    const bool thr = sec >= 0 && prof_on() && opt_.profile_threads && par();
    if (thr) thr_time_.assign(std::size_t(TT), 0.0);
#pragma omp parallel if (par())
    {
      const Clock::time_point c0 = thr ? Clock::now() : Clock::time_point{};
      const int me = thread_id(), nth = num_threads_in_region();
#pragma omp single
      used = nth;
      for (std::size_t q = 0; q < nb; ++q) {
        if (!L_.active[q]) continue;
        const double at = blocks_[q].spec.atol, rt = blocks_[q].spec.rtol;
        std::ptrdiff_t lo, hi;
        static_range(std::ptrdiff_t(L_.offset[q]), std::ptrdiff_t(L_.offset[q] + L_.size[q]), me, nth, lo, hi);
        double s = 0, m = 0, c = 0;
#pragma omp simd reduction(+ : s, c) reduction(max : m)
        for (std::ptrdiff_t i = lo; i < hi; ++i) {
          const std::ptrdiff_t j = 2 * i;
          double vr, vi;
          vfun(j, vr, vi);
          const double ay = std::sqrt(yv[j] * yv[j] + yv[j + 1] * yv[j + 1]);
          double az = 0.0;
          if (zv) {
            az = std::sqrt(zv[j] * zv[j] + zv[j + 1] * zv[j + 1]);
            c += (zv[j] - zv[j]) + (zv[j + 1] - zv[j + 1]);  // NaN iff z non-finite
          }
          const double sc = at + rt * (ay > az ? ay : az);
          const double r2 = (vr * vr + vi * vi) / (sc * sc);
          s += r2;
          m = m > r2 ? m : r2;
        }
        loc[std::size_t(me) * nb + q] = {s, m, c};
      }
      if (thr) thr_time_[std::size_t(me)] = secs(Clock::now() - c0);
    }
    if (thr) prof_imbalance(sec, used);
    std::vector<Red> red(nb);
    for (int th = 0; th < used; ++th)
      for (std::size_t q = 0; q < nb; ++q) {
        const Red& l = loc[std::size_t(th) * nb + q];
        red[q].sumsq += l.sumsq; red[q].chk += l.chk; red[q].mx = std::max(red[q].mx, l.mx);
        if (std::isnan(l.mx)) red[q].mx = l.mx;
      }
    return red;
  }

  // Weighted norm of v = a + cb*b (b may be null), scale from y only.
  double wnorm(const double* a, const double* b, double cb, const double* yv) {
    auto red = reduce(
        [&](std::ptrdiff_t j, double& vr, double& vi) {
          vr = a[j]; vi = a[j + 1];
          if (b) { vr += cb * b[j]; vi += cb * b[j + 1]; }
        },
        yv, nullptr);
    return combine(red, false);
  }

  // Hairer/Wanner II.4 (identical to scipy.integrate select_initial_step).
  double initial_step(double t_bound) {
    const double interval = std::fabs(t_bound - t_);
    if (interval == 0.0) return 0.0;
    PTimer tm(this, SecPassInitStep);
    const double d0 = wnorm(R(Y), nullptr, 0, R(Y));
    const double d1 = wnorm(R(K1), nullptr, 0, R(Y));
    double h0 = (d0 < 1e-5 || d1 < 1e-5) ? 1e-6 : 0.01 * d0 / d1;
    h0 = std::min(h0, interval);
    double* y0 = R(Y); double* y1 = R(T); const double* f0 = R(K1);
    pass(-1, [&](std::ptrdiff_t j) { y1[j] = y0[j] + h0 * f0[j]; });
    call_rhs(t_ + h0, T, K2, SecRhsInitStep);
    const double d2 = wnorm(R(K2), R(K1), -1.0, R(Y)) / h0;
    double h1;
    if (!(d1 > 1e-15) && !(d2 > 1e-15)) h1 = std::max(1e-6, h0 * 1e-3);
    else h1 = std::pow(0.01 / std::max(d1, d2), 0.2);
    double h = std::min({100.0 * h0, h1, interval, opt_.h_max});
    if (!(h > 0)) h = std::min(1e-6, interval);  // non-finite slope: let the step loop report it
    if (prof_on()) { prof_.h0_auto_t.push_back(t_); prof_.h0_auto_h.push_back(h); }
    return h;
  }

  // ------------------------------------------------------------ DP5 tableau
  static constexpr double c2 = 1.0 / 5, c3 = 3.0 / 10, c4 = 4.0 / 5, c5 = 8.0 / 9;
  static constexpr double a21 = 1.0 / 5;
  static constexpr double a31 = 3.0 / 40, a32 = 9.0 / 40;
  static constexpr double a41 = 44.0 / 45, a42 = -56.0 / 15, a43 = 32.0 / 9;
  static constexpr double a51 = 19372.0 / 6561, a52 = -25360.0 / 2187, a53 = 64448.0 / 6561, a54 = -212.0 / 729;
  static constexpr double a61 = 9017.0 / 3168, a62 = -355.0 / 33, a63 = 46732.0 / 5247, a64 = 49.0 / 176,
                          a65 = -5103.0 / 18656;
  static constexpr double b1 = 35.0 / 384, b3 = 500.0 / 1113, b4 = 125.0 / 192, b5 = -2187.0 / 6784,
                          b6 = 11.0 / 84;
  static constexpr double e1 = 71.0 / 57600, e3 = -71.0 / 16695, e4 = 71.0 / 1920, e5 = -17253.0 / 339200,
                          e6 = 22.0 / 525, e7 = -1.0 / 40;
  static constexpr double d1 = -12715105075.0 / 11282082432, d3 = 87487479700.0 / 32700410799,
                          d4 = -10690763975.0 / 1880347072, d5 = 701980252875.0 / 199316789632,
                          d6 = -1453857185.0 / 822651844, d7 = 69997945.0 / 29380423;

  // One attempted step of size h from t_ to t_new. Afterwards: R(T) = y5,
  // R(K2) = k7 = f(t_new, y5). Returns the error norm (NaN if non-finite).
  double attempt(double h, double t_new) {
    const double* y = R(Y); double* yt = R(T);
    const double* k1 = R(K1); double* k2 = R(K2); double* k3 = R(K3);
    double* k4 = R(K4); double* k5 = R(K5); double* k6 = R(K6);
    { const double c = h * a21; pass(SecPassStage2, [&](std::ptrdiff_t j) { yt[j] = y[j] + c * k1[j]; }); }
    call_rhs(t_ + c2 * h, T, K2, SecRhsK2);
    { const double p1 = h * a31, p2 = h * a32;
      pass(SecPassStage3, [&](std::ptrdiff_t j) { yt[j] = y[j] + p1 * k1[j] + p2 * k2[j]; }); }
    call_rhs(t_ + c3 * h, T, K3, SecRhsK3);
    { const double p1 = h * a41, p2 = h * a42, p3 = h * a43;
      pass(SecPassStage4, [&](std::ptrdiff_t j) { yt[j] = y[j] + p1 * k1[j] + p2 * k2[j] + p3 * k3[j]; }); }
    call_rhs(t_ + c4 * h, T, K4, SecRhsK4);
    { const double p1 = h * a51, p2 = h * a52, p3 = h * a53, p4 = h * a54;
      pass(SecPassStage5, [&](std::ptrdiff_t j) { yt[j] = y[j] + p1 * k1[j] + p2 * k2[j] + p3 * k3[j] + p4 * k4[j]; }); }
    call_rhs(t_ + c5 * h, T, K5, SecRhsK5);
    { const double p1 = h * a61, p2 = h * a62, p3 = h * a63, p4 = h * a64, p5 = h * a65;
      pass(SecPassStage6, [&](std::ptrdiff_t j) {
        yt[j] = y[j] + p1 * k1[j] + p2 * k2[j] + p3 * k3[j] + p4 * k4[j] + p5 * k5[j]; }); }
    call_rhs(t_new, T, K6, SecRhsK6);
    { const double p1 = h * b1, p3 = h * b3, p4 = h * b4, p5 = h * b5, p6 = h * b6;
      pass(SecPassSolution, [&](std::ptrdiff_t j) {
        yt[j] = y[j] + p1 * k1[j] + p3 * k3[j] + p4 * k4[j] + p5 * k5[j] + p6 * k6[j]; }); }
    call_rhs(t_new, T, K2, SecRhsK7);  // k7 (FSAL), overwrites dead k2
    // fused error pass: e = h sum e_i k_i; scale = atol + rtol max(|y|,|y5|)
    const double* k7 = R(K2);
    const double q1 = h * e1, q3 = h * e3, q4 = h * e4, q5 = h * e5, q6 = h * e6, q7 = h * e7;
    PTimer tm(this, SecPassError);
    auto red = reduce(
        [&](std::ptrdiff_t j, double& er, double& ei) {
          er = q1 * k1[j] + q3 * k3[j] + q4 * k4[j] + q5 * k5[j] + q6 * k6[j] + q7 * k7[j];
          ei = q1 * k1[j + 1] + q3 * k3[j + 1] + q4 * k4[j + 1] + q5 * k5[j + 1] + q6 * k6[j + 1] + q7 * k7[j + 1];
        },
        y, yt, SecPassError);
    diag_.t = t_; diag_.h = h;
    return combine(red, true);
  }

  // Dense output at theta in (0,1) for the saved blocks (before rotation).
  void emit_dense(std::size_t k, double tg, double h, double theta) {
    const double* y0 = R(Y); const double* y1 = R(T);
    const double* k1 = R(K1); const double* k7 = R(K2); const double* k3 = R(K3);
    const double* k4 = R(K4); const double* k5 = R(K5); const double* k6 = R(K6);
    const double th = theta, th1 = 1.0 - theta;
    Frame f; f.k = k; f.t = tg; f.block.assign(blocks_.size(), nullptr);
    {
    PTimer tm(this, SecDense);
    std::size_t pos = 0;
    double* out = reinterpret_cast<double*>(dense_.data());
    for (std::size_t q = 0; q < blocks_.size(); ++q) {
      if (blocks_[q].removed || !blocks_[q].spec.save) continue;
      f.block[q] = dense_.data() + pos;
      const std::ptrdiff_t lo = std::ptrdiff_t(L_.offset[q]), n = std::ptrdiff_t(L_.size[q]);
      double* o = out + 2 * pos;
      if (!L_.active[q]) {
        std::memcpy(o, y0 + 2 * lo, 2 * n * sizeof(double));
      } else {
#pragma omp parallel for simd schedule(static) if (n >= std::ptrdiff_t(opt_.par_threshold))
        for (std::ptrdiff_t jj = 0; jj < 2 * n; ++jj) {
          const std::ptrdiff_t j = 2 * lo + jj;
          const double r2 = y1[j] - y0[j];
          const double r3 = h * k1[j] - r2;
          const double r4 = r2 - h * k7[j] - r3;
          const double r5 = h * (d1 * k1[j] + d3 * k3[j] + d4 * k4[j] + d5 * k5[j] + d6 * k6[j] + d7 * k7[j]);
          o[jj] = y0[j] + th * (r2 + th1 * (r3 + th * (r4 + th1 * r5)));
        }
      }
      pos += L_.size[q];
    }
    }
    ++stats_.n_frames;
    if (prof_on()) ++prof_.frames_dense;
    PTimer tm(this, SecObserver);
    obs_(f);
  }
  void emit_exact(std::size_t k, double tg) {
    Frame f; f.k = k; f.t = tg; f.block.assign(blocks_.size(), nullptr);
    for (std::size_t q = 0; q < blocks_.size(); ++q)
      if (!blocks_[q].removed && blocks_[q].spec.save) f.block[q] = y() + L_.offset[q];
    ++stats_.n_frames;
    if (prof_on()) ++prof_.frames_exact;
    PTimer tm(this, SecObserver);
    obs_(f);
  }
  // Emit pending grid points that coincide with the current time t_.
  void emit_at_current() {
    if (!obs_) return;
    while (g_next_ < g_n_ && grid_time(g_next_) <= t_ + land_tol(t_)) {
      emit_exact(g_next_, grid_time(g_next_));
      ++g_next_;
    }
  }

  bool run_stops_at_current() {
    bool any = false;
    while (!stops_.empty() && stops_.front().t <= t_ + land_tol(t_)) {
      StopEntry s = std::move(stops_.front());
      stops_.erase(stops_.begin());
      if (s.cb) { PTimer tm(this, SecStopCallback); s.cb(*this); }
      ++stats_.n_stops;
      if (prof_on()) ++prof_.stops;
      any = true;
    }
    if (any) {
      state_changed();
      if (opt_.reset_h_at_stop) h_ = 0.0;
      prof_new_segment();  // callbacks belong to the segment that ends here
    }
    return any;
  }

  bool call_interrupt() { PTimer tm(this, SecInterrupt); return interrupt_(); }
  void ensure_slope() {
    if (!k1_valid_) { call_rhs(t_, Y, K1, SecRhsSlope); k1_valid_ = true; }
  }

  double min_step() const {
    const double ulp = std::nextafter(t_, std::numeric_limits<double>::infinity()) - t_;
    return std::max(opt_.h_min, 10.0 * ulp);
  }

  // Step-size update; returns new h. err finite.
  double propose(double h, double err, bool accepted, bool rejected_before) {
    if (opt_.controller == Controller::Standard) {
      double f;
      if (accepted) {
        f = (err == 0.0) ? opt_.fac_max : std::min(opt_.fac_max, opt_.safety * std::pow(err, -0.2));
        if (rejected_before) f = std::min(1.0, f);
      } else {
        f = std::max(opt_.fac_min, opt_.safety * std::pow(err, -0.2));
      }
      return h * f;
    }
    // PI controller (Hairer dopri5)
    const double expo1 = 0.2 - opt_.beta * 0.75;
    const double fac11 = std::pow(err, expo1);
    const double inv_min = 1.0 / opt_.fac_min, inv_max = 1.0 / opt_.fac_max;
    if (accepted) {
      double fac = fac11 / std::pow(err_old_, opt_.beta);
      fac = std::max(inv_max, std::min(inv_min, fac / opt_.safety));
      double hn = h / fac;
      err_old_ = std::max(err, 1e-4);
      if (rejected_before) hn = std::min(hn, h);
      return hn;
    }
    return h / std::min(inv_min, fac11 / opt_.safety);
  }

  Status run(double t_end) {
    std::size_t attempts = 0;
    run_stops_at_current();
    ensure_slope();
    emit_at_current();
    if (opt_.fixed_h > 0) h_ = opt_.fixed_h;
    while (t_ < t_end - land_tol(t_end)) {
      const double t_stop = (!stops_.empty() && stops_.front().t < t_end) ? stops_.front().t : t_end;
      while (t_ < t_stop) {
        if (t_stop - t_ <= land_tol(t_stop)) { t_ = t_stop; break; }  // already there
        ensure_slope();
        if (!(h_ > 0)) h_ = (opt_.h0 > 0) ? opt_.h0 : initial_step(t_stop);
        bool rejected_before = false;
        int nonfinite_left = opt_.nonfinite_retries;
        for (;;) {
          if (++attempts > opt_.max_steps) return Status::MaxSteps;
          const double h_req = std::min(h_, opt_.h_max);
          double h = h_req;
          bool clipped = false;
          if (t_ + opt_.landing_slack * h >= t_stop) { h = t_stop - t_; clipped = true; }
          if (!clipped && h < min_step() && opt_.fixed_h <= 0) return Status::StepTooSmall;
          const double t_new = clipped ? t_stop : t_ + h;
          const double err = attempt(h, t_new);
          // NaN: something non-finite in the stages or y5. Inf (overflowed but
          // finite state) is just a huge error -> rejection in adaptive mode.
          const bool nonfinite = std::isnan(err) || (std::isinf(err) && opt_.fixed_h > 0);
          if (nonfinite) {
            ++stats_.n_nonfinite;
            prof_step(h, err, false, true, clipped);
            if (step_obs_) { PTimer tm(this, SecStepObserver); step_obs_(t_, h, err, false); }
            if (nonfinite_left-- > 0 && opt_.fixed_h <= 0) { h_ = 0.25 * h; rejected_before = true; continue; }
            return Status::NonFinite;
          }
          const bool accept = (opt_.fixed_h > 0) ? true
                              : (opt_.controller == Controller::Standard ? err < 1.0 : err <= 1.0);
          prof_step(h, err, accept, false, clipped);
          if (step_obs_) { PTimer tm(this, SecStepObserver); step_obs_(t_, h, err, accept); }
          if (!accept) {
            PTimer tm(this, SecController);
            ++stats_.n_reject;
            h_ = propose(h, err, false, rejected_before);
            rejected_before = true;
            if (h_ < min_step()) return Status::StepTooSmall;
            continue;
          }
          // ---- accepted: dense output for grid points strictly inside (t_, t_new)
          if (obs_) {
            const double tol = land_tol(t_new);
            while (g_next_ < g_n_) {
              const double tg = grid_time(g_next_);
              if (tg >= t_new - tol) break;
              emit_dense(g_next_, tg, h, (tg - t_) / h);
              ++g_next_;
            }
          }
          // rotate registers: Y <-> T (new state), K1 <-> K2 (k7 is new slope)
          PTimer tmc(this, SecController);
          std::swap(R_[Y], R_[T]);
          std::swap(R_[K1], R_[K2]);
          k1_valid_ = true;
          t_ = t_new;
          ++stats_.n_accept;
          if (opt_.fixed_h > 0) h_ = opt_.fixed_h;
          else {
            const double hn = propose(h, err, true, rejected_before);
            h_ = (clipped && opt_.carry_h_across_landing) ? std::max(hn, h_req) : hn;
          }
          break;
        }
        if (t_ < t_stop) {  // regular step: grid point at the landing time
          emit_at_current();
          if (interrupt_ && call_interrupt()) return Status::Interrupted;
        }
      }
      run_stops_at_current();  // callbacks first, then the frame (post-callback state)
      ensure_slope();
      emit_at_current();
      if (interrupt_ && call_interrupt() && t_ < t_end) return Status::Interrupted;
    }
    t_ = std::max(t_, t_end);
    return Status::Ok;
  }
};

}  // namespace dp5
