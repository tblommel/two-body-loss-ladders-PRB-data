// test_dp5.cpp -- tests and benchmark for dp5_integrator.hpp
//
// Build: g++ -O3 -march=native -fopenmp -std=c++17 -I.. test_dp5.cpp -o test_dp5
// Usage: ./test_dp5            all correctness tests
//        ./test_dp5 scipy DIR  write data for scipy_crosscheck.py into DIR
//        ./test_dp5 bench old|new N   integrator-overhead benchmark (one process per case)
#include "dp5_integrator.hpp"

#include <sys/resource.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

using dp5::cplx;
using dp5::Integrator;
using dp5::Status;

static int g_fail = 0;
#define CHECK(cond, ...)                                    \
  do {                                                      \
    if (!(cond)) {                                          \
      ++g_fail;                                             \
      std::printf("  FAIL %s:%d: %s  ", __FILE__, __LINE__, #cond); \
      std::printf(__VA_ARGS__);                             \
      std::printf("\n");                                    \
    }                                                       \
  } while (0)

static const cplx I1(0.0, 1.0);

// diagonal linear RHS dy_i = lam_i y_i (parallel for large n)
struct DiagRhs {
  std::vector<cplx> lam;
  void operator()(double, const cplx* y, cplx* dy) const {
    const std::ptrdiff_t n = std::ptrdiff_t(lam.size());
#pragma omp parallel for schedule(static) if (n > 8192)
    for (std::ptrdiff_t i = 0; i < n; ++i) dy[i] = lam[i] * y[i];
  }
};

static double maxabs(const std::vector<cplx>& a, const std::vector<cplx>& b) {
  double m = 0;
  for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

// ---------------------------------------------------------------- 1. order
static void test_order() {
  std::printf("[1] convergence order (linear complex ODE, fixed step)\n");
  const std::vector<cplx> lam = {cplx(-1, 2), cplx(-0.3, 5), cplx(0, 0.5), cplx(-2, 0)};
  const std::vector<cplx> y0 = {cplx(1, 0), cplx(0.5, 0.5), cplx(0, -1), cplx(0.3, 0.1)};
  const double T = 2.0;
  double prev_e = 0, prev_d = 0;
  for (int lev = 0; lev < 6; ++lev) {
    const int nsteps = 10 << lev;
    dp5::Options o; o.fixed_h = T / nsteps;
    Integrator in({{"y", 4, 1e-8, 1e-6, 1, dp5::BlockNorm::RMS, true, true}}, DiagRhs{lam}, o);
    std::copy(y0.begin(), y0.end(), in.block(0));
    in.state_changed();
    double dmax = 0;
    // 37 grid points never coincide with step ends except t=0 and t=T; measure
    // the interpolation error relative to the exact solution at interior points
    in.set_output_grid(0.0, T, 38, [&](const dp5::Frame& f) {
      if (f.k == 0 || f.k == 37) return;
      for (int i = 0; i < 4; ++i) dmax = std::max(dmax, std::abs(f[0][i] - std::exp(lam[i] * f.t) * y0[i]));
    });
    in.integrate(T);
    double e = 0;
    for (int i = 0; i < 4; ++i) e = std::max(e, std::abs(in.block(0)[i] - std::exp(lam[i] * T) * y0[i]));
    if (lev == 0) std::printf("   steps   err(T)      order   dense_err   order\n");
    std::printf("   %5d  %.3e  %5.2f   %.3e  %5.2f\n", nsteps, e, lev ? std::log2(prev_e / e) : 0.0, dmax,
                lev ? std::log2(prev_d / dmax) : 0.0);
    if (lev >= 2 && lev <= 4) {
      CHECK(std::log2(prev_e / e) > 4.7, "step order %.2f", std::log2(prev_e / e));
      CHECK(std::log2(prev_d / dmax) > 3.7, "dense order %.2f", std::log2(prev_d / dmax));
    }
    prev_e = e; prev_d = dmax;
  }
  std::printf("   adaptive (PI): tolerance proportionality, atol = 1e-3 rtol\n   rtol     steps  rej   err(T)/rtol\n");
  for (double rtol : {1e-4, 1e-6, 1e-8, 1e-10}) {
    Integrator in({{"y", 4, 1e-3 * rtol, rtol}}, DiagRhs{lam});
    std::copy(y0.begin(), y0.end(), in.block(0));
    in.state_changed();
    in.integrate(T);
    double e = 0;
    for (int i = 0; i < 4; ++i) e = std::max(e, std::abs(in.block(0)[i] - std::exp(lam[i] * T) * y0[i]));
    std::printf("   %.0e  %5llu %4llu   %.2f\n", rtol, (unsigned long long)in.stats().n_accept,
                (unsigned long long)in.stats().n_reject, e / rtol);
    CHECK(e < 30 * rtol, "adaptive error %.2e at rtol %.0e", e, rtol);
  }
}

// ---------------------------------------------------------------- 2. stiff-ish
static void test_stiffish() {
  std::printf("[2] damped oscillators, RTDE-sized (2 blocks, 625 + 262144 complex)\n");
  const std::size_t n1 = 625, n2 = 262144, n = n1 + n2;
  std::vector<cplx> lam(n), y0(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double u = double((i * 2654435761u) % 100003) / 100003.0;  // pseudo-random in [0,1)
    const double v = double((i * 40503u + 17) % 99991) / 99991.0;
    const double gam = (i < n1) ? 0.05 * u : 0.01 * std::pow(3000.0, u);  // up to 30
    lam[i] = cplx(-gam, 30.0 * (v - 0.5));
    y0[i] = cplx(std::cos(7.0 * u), std::sin(3.0 * v)) * ((i < n1) ? 1.0 : 0.1);
  }
  const double T = 10.0, atol = 1e-8, rtol = 1e-6;
  for (int variant = 0; variant < 3; ++variant) {
    const auto ctrl = variant == 1 ? dp5::Controller::Standard : dp5::Controller::PI;
    const auto nrm = variant == 2 ? dp5::BlockNorm::Max : dp5::BlockNorm::RMS;
    dp5::Options o; o.controller = ctrl;
    Integrator in({{"rho", n1, atol, rtol, 1, nrm}, {"F", n2, atol, rtol, 1, nrm}}, DiagRhs{lam}, o);
    std::copy(y0.begin(), y0.end(), in.y());
    in.state_changed();
    Status st = in.integrate(T);
    double worst = 0, maxabs_err = 0, rms = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const cplx ex = std::exp(lam[i] * T) * y0[i];
      const double e = std::abs(in.y()[i] - ex);
      maxabs_err = std::max(maxabs_err, e);
      const double r = e / (atol + rtol * std::abs(ex));
      worst = std::max(worst, r);
      rms += r * r;
    }
    rms = std::sqrt(rms / double(n));
    std::printf("   %-12s status=%s steps=%llu rej=%llu rhs=%llu max|err|=%.1e err/tol: rms=%.2f max=%.1f "
                "time=%.2fs (rhs %.2fs)\n",
                variant == 0 ? "PI,RMS" : variant == 1 ? "Standard,RMS" : "PI,Max", dp5::to_string(st),
                (unsigned long long)in.stats().n_accept, (unsigned long long)in.stats().n_reject,
                (unsigned long long)in.stats().n_rhs, maxabs_err, rms, worst, in.stats().total_seconds,
                in.stats().rhs_seconds);
    CHECK(st == Status::Ok, "status");
    // global error after 535 steps vs local tolerance: RMS control bounds the
    // RMS; the Max norm bounds every entry
    if (variant < 2) CHECK(rms < 20, "rms global error / tol = %.1f", rms);
    else CHECK(worst < 100, "max global error / tol = %.1f", worst);
  }
}

// ---------------------------------------------------------------- 3. landing
static void test_landing() {
  std::printf("[3] exact landing: grid t_end=40, n=3001, stop t'=30 (the D2 case) + stop at 12.3456\n");
  {
    const double dtg = 40.0 / 3000.0;
    std::printf("   grid_time(2250) = %.17g (t0 + k*dt), stop t' = 30 exactly: must give ONE frame k=2250\n",
                0.0 + 2250.0 * dtg);
  }
  const std::vector<cplx> lam = {cplx(-0.2, 1.0), cplx(-0.05, 3.0)};
  const cplx mu(-0.5, 2.0);
  bool on = false;
  auto rhs = [&](double, const cplx* y, cplx* dy) {
    dy[0] = lam[0] * y[0]; dy[1] = lam[1] * y[1];
    if (on) dy[2] = mu * y[2] + y[0];
  };
  Integrator in({{"A", 2, 1e-10, 1e-8, 1, dp5::BlockNorm::RMS, true, true},
                 {"GR", 1, 1e-10, 1e-8, 1, dp5::BlockNorm::RMS, false, true}}, rhs);
  in.block(0)[0] = 1.0; in.block(0)[1] = cplx(0, 1);
  in.state_changed();
  std::vector<std::size_t> ks; std::vector<double> ts; std::vector<cplx> gr, a0;
  in.set_output_grid(0.0, 40.0, 3001, [&](const dp5::Frame& f) {
    ks.push_back(f.k); ts.push_back(f.t); gr.push_back(f[1][0]); a0.push_back(f[0][0]);
  });
  double t_seen_30 = -1, t_seen_12 = -1;
  in.add_stop(30.0, [&](Integrator& it) {
    t_seen_30 = it.t();
    it.block(1)[0] = cplx(0, -1);  // G^R(t'+,t') = -i
    it.set_active(1, true);
    on = true;
  });
  in.add_stop(12.3456, [&](Integrator& it) { t_seen_12 = it.t(); });
  Status st = in.integrate(40.0);
  CHECK(st == Status::Ok, "status %s", dp5::to_string(st));
  CHECK(ks.size() == 3001, "frames %zu", ks.size());
  bool seq = true, times = true;
  for (std::size_t i = 0; i < ks.size(); ++i) {
    seq = seq && ks[i] == i;
    times = times && ts[i] == in.grid_time(i);
  }
  CHECK(seq, "frame indices not 0..3000 in order");
  CHECK(times, "frame times differ from grid_time(k)");
  CHECK(ts.back() == 40.0, "last frame time %.17g", ts.back());
  CHECK(in.t() == 40.0, "final t %.17g", in.t());
  CHECK(t_seen_30 == 30.0 && t_seen_12 == 12.3456, "stop times %.17g %.17g", t_seen_30, t_seen_12);
  CHECK(gr[2249] == cplx(0, 0) && gr[2250] == cplx(0, -1), "frame at t' must show post-callback state");
  // accuracy of every frame (dense and exact) for A, and of G^R after t'
  double ea = 0, eg = 0;
  for (std::size_t i = 0; i < ts.size(); ++i) {
    const double t = ts[i];
    ea = std::max(ea, std::abs(a0[i] - std::exp(lam[0] * t)));
    if (i >= 2250) {
      const double s = t - 30.0;
      const cplx a30 = std::exp(lam[0] * 30.0);
      const cplx ex = std::exp(mu * s) * cplx(0, -1) + a30 * (std::exp(lam[0] * s) - std::exp(mu * s)) / (lam[0] - mu);
      eg = std::max(eg, std::abs(gr[i] - ex));
    }
  }
  std::printf("   frames=%zu (k=0..3000 once each), t_final=%.17g, max frame err A=%.2e, G^R=%.2e, steps=%llu\n",
              ks.size(), in.t(), ea, eg, (unsigned long long)in.stats().n_accept);
  CHECK(ea < 1e-6 && eg < 1e-6, "frame errors %.2e %.2e", ea, eg);
  // last frame is the exact state (no interpolation), check bitwise
  CHECK(gr.back() == in.block(1)[0], "last frame not the exact final state");
}

// ---------------------------------------------------------------- 4. activation
struct ActResult { std::vector<cplx> y; std::vector<cplx> frames; std::uint64_t steps; };

static ActResult run_activation(int mode) {
  // A: 64 entries, dA = lamA A.  B: 64 entries, dB = mu B + c A after t' (=3).
  const std::size_t m = 64;
  std::vector<cplx> lamA(m), mu(m);
  for (std::size_t i = 0; i < m; ++i) {
    lamA[i] = cplx(-0.1 - 0.02 * i, 0.3 * i);
    mu[i] = cplx(-1.0 - 0.05 * i, -0.2 * i);
  }
  const double tp = 3.0, T = 8.0, c = 0.7;
  bool on = false;
  auto rhs = [&](double, const cplx* y, cplx* dy) {
    for (std::size_t i = 0; i < m; ++i) dy[i] = lamA[i] * y[i];
    if (mode == 2 || on)
      for (std::size_t i = 0; i < m; ++i) dy[m + i] = on ? mu[i] * y[m + i] + c * y[i] : cplx(0, 0);
  };
  std::vector<dp5::BlockSpec> blocks = {{"A", m, 1e-9, 1e-7, 1, dp5::BlockNorm::RMS, true, true}};
  if (mode == 1) blocks.push_back({"B", m, 1e-9, 1e-7, 1, dp5::BlockNorm::RMS, false, true});
  if (mode == 2) blocks.push_back({"B", m, 1e-9, 1e-7, 1, dp5::BlockNorm::RMS, true, true});
  Integrator in(blocks, rhs);
  for (std::size_t i = 0; i < m; ++i) in.block(0)[i] = cplx(1.0, 0.1 * i);
  in.state_changed();
  ActResult r;
  in.set_output_grid(0.0, T, 81, [&](const dp5::Frame& f) {
    if (f.block.size() > 1 && f[1]) r.frames.push_back(f[1][0]);
    else r.frames.push_back(cplx(0, 0));
  });
  in.add_stop(tp, [&](Integrator& it) {
    std::size_t b = (mode == 0) ? it.append_block({"B", m, 1e-9, 1e-7, 1, dp5::BlockNorm::RMS, true, true}) : 1;
    it.set_active(b, true);
    cplx* B = it.block(b);
    for (std::size_t i = 0; i < m; ++i) B[i] = 0.0;
    B[0] = cplx(0, -1);  // -i e_0
    on = true;
  });
  Status st = in.integrate(T);
  CHECK(st == Status::Ok, "activation mode %d status %s", mode, dp5::to_string(st));
  r.y.assign(in.y(), in.y() + in.size());
  r.steps = in.stats().n_accept;
  // exact check
  double e = 0;
  for (std::size_t i = 0; i < m; ++i) {
    const cplx A0 = cplx(1.0, 0.1 * i);
    const cplx At = std::exp(lamA[i] * tp) * A0;
    const double s = T - tp;
    const cplx Bp = (i == 0) ? cplx(0, -1) : cplx(0, 0);
    const cplx ex = std::exp(mu[i] * s) * Bp + c * At * (std::exp(lamA[i] * s) - std::exp(mu[i] * s)) / (lamA[i] - mu[i]);
    e = std::max(e, std::abs(r.y[m + i] - ex));
  }
  CHECK(e < 1e-6, "mode %d exact error %.2e", mode, e);
  if (mode == 0) std::printf("   B error vs exact: %.2e\n", e);
  return r;
}

static void test_activation() {
  std::printf("[4] block activation at t'=3: (0) appended, (1) inactive->active, (2) reference held at zero\n");
  ActResult r0 = run_activation(0), r1 = run_activation(1), r2 = run_activation(2);
  const double d01 = maxabs(r0.y, r1.y), d02 = maxabs(r0.y, r2.y);
  const double f01 = maxabs(r0.frames, r1.frames), f02 = maxabs(r0.frames, r2.frames);
  std::printf("   steps %llu/%llu/%llu; max|y0-y1|=%.1e max|y0-y2|=%.1e; frames: %.1e %.1e\n",
              (unsigned long long)r0.steps, (unsigned long long)r1.steps, (unsigned long long)r2.steps, d01, d02,
              f01, f02);
  CHECK(d01 == 0.0 && d02 == 0.0 && f01 == 0.0 && f02 == 0.0, "activation variants differ");
}

// ---------------------------------------------------------------- 5. failures
static void test_failures() {
  std::printf("[5] NaN abort, blow-up, step cap\n");
  {
    auto rhs = [](double t, const cplx* y, cplx* dy) {
      dy[0] = cplx(-0.1, 1) * y[0];
      dy[1] = -y[1];
      if (t > 2.0) dy[0] = cplx(std::nan(""), 0);
    };
    Integrator in({{"y", 2}}, rhs);
    in.block(0)[0] = 1; in.block(0)[1] = 1;
    in.state_changed();
    Status st = in.integrate(5.0);
    const bool fin = std::isfinite(in.y()[0].real()) && std::isfinite(in.y()[1].real());
    std::printf("   NaN for t>2: status=%s t=%.4f state finite=%d nonfinite=%llu\n", dp5::to_string(st), in.t(), fin,
                (unsigned long long)in.stats().n_nonfinite);
    CHECK(st == Status::NonFinite && in.t() <= 2.0 && fin, "NaN handling");
    // continuing from the last good state is possible after fixing the RHS
  }
  {
    dp5::Options o; o.nonfinite_retries = 3;
    auto rhs = [](double t, const cplx* y, cplx* dy) {
      dy[0] = -y[0];
      if (t > 2.0) dy[0] = cplx(std::numeric_limits<double>::infinity(), 0);
    };
    Integrator in({{"y", 1}}, rhs, o);
    in.block(0)[0] = 1; in.state_changed();
    Status st = in.integrate(5.0);
    std::printf("   Inf for t>2 with 3 retries: status=%s t=%.6f\n", dp5::to_string(st), in.t());
    CHECK((st == Status::NonFinite || st == Status::StepTooSmall) && in.t() <= 2.0, "Inf handling");
  }
  {
    auto rhs = [](double, const cplx* y, cplx* dy) { dy[0] = y[0] * y[0]; };  // blow-up at t=1
    Integrator in({{"y", 1}}, rhs);
    in.block(0)[0] = 1; in.state_changed();
    Status st = in.integrate(2.0);
    std::printf("   y'=y^2 (blow-up at t=1): status=%s t=%.12f\n", dp5::to_string(st), in.t());
    CHECK(st != Status::Ok && in.t() < 1.001, "blow-up must not report Ok");
  }
  {
    dp5::Options o; o.max_steps = 5;
    Integrator in({{"y", 1}}, [](double, const cplx* y, cplx* dy) { dy[0] = cplx(0, 50) * y[0]; }, o);
    in.block(0)[0] = 1; in.state_changed();
    Status st = in.integrate(100.0);
    std::printf("   max_steps=5: status=%s\n", dp5::to_string(st));
    CHECK(st == Status::MaxSteps, "step cap");
  }
}

// ---------------------------------------------------------------- 6. validation
static void test_validation() {
  std::printf("[6] input validation (D4)\n");
  auto rhs = [](double, const cplx* y, cplx* dy) { dy[0] = y[0]; };
  int thrown = 0, total = 0;
  auto expect_throw = [&](const char* what, auto&& f) {
    ++total;
    try { f(); std::printf("   no throw: %s\n", what); ++g_fail; }
    catch (const std::invalid_argument&) { ++thrown; }
  };
  expect_throw("zero-size block", [&] { Integrator in({{"y", 0}}, rhs); });
  expect_throw("duplicate names", [&] { Integrator in({{"y", 1}, {"y", 1}}, rhs); });
  expect_throw("negative atol", [&] { Integrator in({{"y", 1, -1e-8, 1e-6}}, rhs); });
  expect_throw("zero tolerances", [&] { Integrator in({{"y", 1, 0.0, 0.0}}, rhs); });
  expect_throw("NaN weight", [&] { Integrator in({{"y", 1, 1e-8, 1e-6, std::nan("")}}, rhs); });
  expect_throw("no RHS", [&] { Integrator in({{"y", 1}}, Integrator::Rhs()); });
  expect_throw("no blocks", [&] { Integrator in({}, rhs); });
  expect_throw("bad safety", [&] { dp5::Options o; o.safety = 1.5; Integrator in({{"y", 1}}, rhs, o); });
  expect_throw("grid n=0", [&] { Integrator in({{"y", 1}}, rhs); in.set_output_grid(0, 1, 0, [](const dp5::Frame&) {}); });
  expect_throw("grid t1<=t0", [&] { Integrator in({{"y", 1}}, rhs); in.set_output_grid(1, 1, 5, [](const dp5::Frame&) {}); });
  expect_throw("t_end < t", [&] { Integrator in({{"y", 1}}, rhs, {}, 2.0); in.integrate(1.0); });
  expect_throw("t_end NaN", [&] { Integrator in({{"y", 1}}, rhs); in.integrate(std::nan("")); });
  expect_throw("stop in past", [&] { Integrator in({{"y", 1}}, rhs, {}, 2.0); in.add_stop(1.0, nullptr); });
  expect_throw("unknown block", [&] { Integrator in({{"y", 1}}, rhs); in.block_id("x"); });
  expect_throw("restore mismatch", [&] {
    Integrator a({{"y", 2}}, rhs), b({{"y", 3}}, rhs);
    b.restore(a.checkpoint(), a.y());
  });
  std::printf("   %d/%d invalid inputs rejected with std::invalid_argument\n", thrown, total);
  {  // n=1 grid is legal: one frame at t0
    Integrator in({{"y", 1}}, [](double, const cplx* y, cplx* dy) { dy[0] = -y[0]; });
    in.block(0)[0] = 1; in.state_changed();
    int nf = 0;
    in.set_output_grid(0, 0, 1, [&](const dp5::Frame&) { ++nf; });
    in.integrate(1.0);
    CHECK(nf == 1, "n=1 grid frames %d", nf);
  }
}

// ---------------------------------------------------------------- 7. restart
static void nl_rhs(double, const cplx* y, cplx* dy, std::size_t n) {
#pragma omp parallel for schedule(static) if (n > 8192)
  for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(n); ++i)
    dy[i] = cplx(-0.05 - 0.001 * (i % 50), 0.5 + 0.01 * (i % 37)) * y[i] + cplx(0, 0.5) * std::norm(y[i]) * y[i];
}

static void test_restart() {
  std::printf("[7] checkpoint/restart and interrupt: bitwise continuation\n");
  const std::size_t n = 20000;
  auto mk = [&]() {
    Integrator in({{"rho", 1000, 1e-9, 1e-7, 1, dp5::BlockNorm::RMS, true, true}, {"G", n - 1000, 1e-9, 1e-7}},
                  [n](double t, const cplx* y, cplx* dy) { nl_rhs(t, y, dy, n); });
    for (std::size_t i = 0; i < n; ++i) in.y()[i] = cplx(std::cos(0.01 * i), 0.3 * std::sin(0.02 * i));
    in.state_changed();
    return in;
  };
  // run A: 0 -> 10 with a stop at 5 that takes a checkpoint
  std::map<std::size_t, std::vector<cplx>> fa, fb;
  dp5::Checkpoint cp; std::vector<cplx> ycp;
  Integrator a = mk();
  a.set_output_grid(0, 10, 101, [&](const dp5::Frame& f) { fa[f.k].assign(f[0], f[0] + 1000); });
  a.add_stop(5.0, [&](Integrator& it) { cp = it.checkpoint(); ycp.assign(it.y(), it.y() + it.size()); });
  a.integrate(10.0);
  // run B: fresh integrator restored at 5
  Integrator b = mk();
  b.set_output_grid(0, 10, 101, [&](const dp5::Frame& f) { fb[f.k].assign(f[0], f[0] + 1000); });
  b.restore(cp, ycp.data());
  b.integrate(10.0);
  double dy = 0; for (std::size_t i = 0; i < n; ++i) dy = std::max(dy, std::abs(a.y()[i] - b.y()[i]));
  double df = 0; std::size_t nf = 0;
  for (auto& kv : fb) { df = std::max(df, maxabs(kv.second, fa[kv.first])); ++nf; }
  std::printf("   restart at stop t=5: max|dy|=%.1e over %zu restored frames max|dframe|=%.1e (first k=%zu)\n", dy, nf,
              df, fb.begin()->first);
  CHECK(dy == 0.0 && df == 0.0 && fb.begin()->first == 50 && nf == 51, "restart not bitwise");
  // interrupt after 37 accepted steps, checkpoint, restore elsewhere, continue
  Integrator c = mk();
  std::size_t cnt = 0;
  c.set_interrupt([&] { return ++cnt == 37; });
  Status st = c.integrate(10.0);
  dp5::Checkpoint cp2 = c.checkpoint();
  std::vector<cplx> y2(c.y(), c.y() + c.size());
  Integrator d = mk();
  d.restore(cp2, y2.data());
  Status st2 = d.integrate(10.0);
  Status st3 = c.integrate(10.0);  // also continue c in place
  double dd = 0, dc = 0;
  for (std::size_t i = 0; i < n; ++i) { dd = std::max(dd, std::abs(a.y()[i] - d.y()[i])); dc = std::max(dc, std::abs(a.y()[i] - c.y()[i])); }
  std::printf("   interrupt at t=%.4f (%s); restored: %s max|dy|=%.1e ; continued in place: %s max|dy|=%.1e\n",
              cp2.t, dp5::to_string(st), dp5::to_string(st2), dd, dp5::to_string(st3), dc);
  // a (with stop at 5) and c/d (no stop) take different step sequences -> compare c vs d bitwise,
  // and both to a within tolerance
  double dcd = 0; for (std::size_t i = 0; i < n; ++i) dcd = std::max(dcd, std::abs(c.y()[i] - d.y()[i]));
  CHECK(st == Status::Interrupted && st2 == Status::Ok && st3 == Status::Ok && dcd == 0.0 && dd < 1e-5,
        "interrupt/restore: dcd=%.1e dd=%.1e", dcd, dd);
}

// ---------------------------------------------------------------- 8. layout tools
static void test_layout_tools() {
  std::printf("[8] rms_count, diagnostics, remove/extract, layout version, grid resolution\n");
  // (a) rms_count: M independent entries stored packed vs. stored 4x (+x,-x,-x,+x) plus K zeros
  {
    const std::size_t M = 4000, K = 48000;  // 75% structural zeros: RMS biased by 1/2
    std::vector<cplx> lam(M);
    for (std::size_t i = 0; i < M; ++i) lam[i] = cplx(-0.1 - 0.001 * (i % 97), 0.5 + 0.03 * (i % 61));
    auto run = [&](int mode) {  // 0 packed, 1 full w/o rms_count, 2 full with rms_count
      const std::size_t n = (mode == 0) ? M : 4 * M + K;
      dp5::BlockSpec b = dp5::tensor_block("G", n, 1e-8, 1e-6);
      if (mode == 2) b.rms_count = 4 * M;
      Integrator in({b}, [&](double, const cplx* y, cplx* dy) {
        for (std::size_t i = 0; i < M; ++i) {
          if (mode == 0) dy[i] = lam[i] * y[i];
          else for (int r = 0; r < 4; ++r) dy[4 * i + r] = lam[i] * y[4 * i + r];
        }
        if (mode) for (std::size_t i = 4 * M; i < n; ++i) dy[i] = 0.0;
      });
      for (std::size_t i = 0; i < M; ++i) {
        const cplx v(1.0, 0.01 * (i % 13));
        if (mode == 0) in.y()[i] = v;
        else { in.y()[4 * i] = v; in.y()[4 * i + 1] = -v; in.y()[4 * i + 2] = -v; in.y()[4 * i + 3] = v; }
      }
      in.state_changed();
      in.integrate(10.0);
      return in.stats().n_accept;
    };
    const auto s0 = run(0), s1 = run(1), s2 = run(2);
    std::printf("   antisymmetric block: steps packed=%llu, full=%llu (biased RMS), full+rms_count=%llu\n",
                (unsigned long long)s0, (unsigned long long)s1, (unsigned long long)s2);
    CHECK(s2 == s0 && s1 < s0, "rms_count does not restore the packed-norm step sequence");
  }
  // (b) diagnostics: NaN in one block of three
  {
    Integrator in({dp5::observable_block("rho", 10), dp5::tensor_block("G", 1000), dp5::tensor_block("F", 100)},
                  [](double t, const cplx* y, cplx* dy, const dp5::Layout& L) {
                    for (std::size_t i = 0; i < L.total; ++i) dy[i] = -0.1 * y[i];
                    if (t > 1.0) dy[L.offset[1] + 17] = cplx(std::nan(""), 0);
                  });
    for (std::size_t i = 0; i < in.size(); ++i) in.y()[i] = 1.0;
    in.state_changed();
    Status st = in.integrate(3.0);
    const auto& d = in.diagnostics();
    std::printf("   NaN in block G: status=%s worst_block=%s nonfinite flags=(%d,%d,%d)\n", dp5::to_string(st),
                in.spec(d.worst_block).name.c_str(), d.block_nonfinite[0], d.block_nonfinite[1], d.block_nonfinite[2]);
    CHECK(st == Status::NonFinite && d.worst_block == 1 && d.block_nonfinite[1] && !d.block_nonfinite[0] &&
              !d.block_nonfinite[2], "NaN diagnostics");
  }
  {  // blow-up in one block -> StepTooSmall, worst block reported
    Integrator in({dp5::observable_block("rho", 4), dp5::tensor_block("G", 100)},
                  [](double, const cplx* y, cplx* dy, const dp5::Layout& L) {
                    for (std::size_t i = 0; i < 4; ++i) dy[i] = -y[i];
                    for (std::size_t i = 0; i < 100; ++i) dy[L.offset[1] + i] = (i == 5) ? y[4 + i] * y[4 + i] : -y[4 + i];
                  });
    for (std::size_t i = 0; i < in.size(); ++i) in.y()[i] = 1.0;
    in.state_changed();
    Status st = in.integrate(3.0);
    std::printf("   blow-up in block G: status=%s at t=%.6f worst_block=%s err=%.2e\n", dp5::to_string(st), in.t(),
                in.spec(in.diagnostics().worst_block).name.c_str(), in.diagnostics().err);
    CHECK(st != Status::Ok && in.diagnostics().worst_block == 1, "blow-up diagnostics");
  }
  // (c) remove / extract and layout version
  {
    Integrator in({dp5::observable_block("rho", 3), dp5::tensor_block("G", 50), dp5::tensor_block("F", 7)},
                  [](double, const cplx* y, cplx* dy, const dp5::Layout& L) {
                    for (std::size_t q = 0; q < L.num_blocks(); ++q)
                      if (L.active[q])
                        for (std::size_t i = 0; i < L.size[q]; ++i) dy[L.offset[q] + i] = -y[L.offset[q] + i];
                  });
    for (std::size_t i = 0; i < in.size(); ++i) in.y()[i] = cplx(double(i), 0);
    in.state_changed();
    const auto v0 = in.layout_version();
    std::vector<cplx> G = in.extract_block(1);
    const bool ok = in.size() == 10 && in.layout_version() == v0 + 1 && G.size() == 50 && G[0] == 3.0 &&
                    in.block(2)[0] == 53.0 && in.offset(2) == 3 && in.block(0)[2] == 2.0 && in.removed(1);
    bool threw = false;
    try { in.block(1); } catch (const std::invalid_argument&) { threw = true; }
    in.integrate(1.0);
    std::printf("   extract G: state 60 -> %zu entries, version %llu -> %llu, data preserved=%d, access to removed throws=%d\n",
                in.size(), (unsigned long long)v0, (unsigned long long)in.layout_version(), ok, threw);
    CHECK(ok && threw && std::abs(in.block(2)[0] - 53.0 * std::exp(-1.0)) < 1e-4 * 53.0, "remove/extract");
  }
  {  // grid spacing below time resolution
    bool threw = false;
    try {
      Integrator in({{"y", 1}}, [](double, const cplx* y, cplx* dy) { dy[0] = -y[0]; }, {}, 1e6);
      in.set_output_grid(1e6, 1e6 + 1e-9, 11, [](const dp5::Frame&) {});
    } catch (const std::invalid_argument&) { threw = true; }
    std::printf("   grid spacing 1e-10 at t=1e6 (below 8 x landing tolerance %.1e): rejected=%d\n",
                8 * Integrator::land_tol(1e6), threw);
    CHECK(threw, "unresolvable grid accepted");
  }
}

// ---------------------------------------------------------------- 9. coupled G1-G2 -> RTDE
struct CoupledModel {
  std::size_t nG, nF;
  std::vector<cplx> lam, Ginf, G0, mu, Fp;
  double c = 0.2;
  // layout ids; idG = npos when G is not in the state (frozen external buffer)
  std::size_t idG = std::size_t(-1), idF = std::size_t(-1);
  const cplx* Gext = nullptr;
  bool rtde_on = false;
  void operator()(double, const cplx* y, cplx* dy, const dp5::Layout& L) const {
    const bool G_in_state = idG != std::size_t(-1) && !L.removed[idG];
    const cplx* G = G_in_state ? y + L.offset[idG] : Gext;
    if (G_in_state && L.active[idG]) {
      cplx* dG = dy + L.offset[idG];
      dp5::parallel_for_block(L, idG, [&](std::ptrdiff_t i) { dG[i] = lam[i] * (G[i] - Ginf[i]); });
    }
    if (rtde_on) {
      const cplx* F = y + L.offset[idF];
      cplx* dF = dy + L.offset[idF];
      dp5::parallel_for_block(L, idF, [&](std::ptrdiff_t j) { dF[j] = (mu[j] + c * G[j]) * F[j]; });
    }
  }
  cplx exactF(std::size_t j, double t, double tp, bool stationary) const {
    const double s = t - tp;
    if (stationary) return Fp[j] * std::exp((mu[j] + c * Ginf[j]) * s);
    const cplx intG = Ginf[j] * s + (G0[j] - Ginf[j]) * (std::exp(lam[j] * t) - std::exp(lam[j] * tp)) / lam[j];
    return Fp[j] * std::exp(mu[j] * s + c * intG);
  }
};

static CoupledModel make_model(std::size_t nG, std::size_t nF, bool stationary) {
  CoupledModel m; m.nG = nG; m.nF = nF;
  m.lam.resize(nG); m.Ginf.resize(nG); m.G0.resize(nG); m.mu.resize(nF); m.Fp.resize(nF);
  for (std::size_t i = 0; i < nG; ++i) {
    const double u = double((i * 2654435761u) % 100003) / 100003.0, v = double((i * 40503u + 17) % 99991) / 99991.0;
    m.lam[i] = cplx(-(0.05 + 0.45 * u), 4.0 * v - 2.0);
    m.Ginf[i] = 0.1 * cplx(std::cos(6.0 * v), std::sin(5.0 * u));
    m.G0[i] = stationary ? m.Ginf[i] : m.Ginf[i] + 0.5 * cplx(std::cos(3.0 * u), std::sin(2.0 * v));
  }
  for (std::size_t j = 0; j < nF; ++j) {
    const double w = double((j * 7919u + 3) % 10007) / 10007.0;
    m.mu[j] = cplx(-0.6, 16.0 * w - 8.0);
    m.Fp[j] = cplx(0, -1) * (1.0 + 0.5 * std::cos(0.37 * double(j)));
  }
  return m;
}

struct CoupledResult {
  Status st; std::vector<std::vector<cplx>> frames; std::vector<double> ft; double tol_ratio = 0, rel_final = 0;
  std::string prof_text;
  std::uint64_t steps = 0; double h_before = 0, h_after_early = 0, h_after_late = 0;
  double frac_F_worst_early = 0, frac_F_worst_late = 0; std::size_t state_size_after = 0; double seconds = 0;
};

// mode: 0 co-propagate (G in state, F appended at t'), 1 frozen external G from t',
//       2 co-propagate from t' (G in state from t'), 3 extract G at t' (handoff)
static CoupledResult run_coupled(CoupledModel m, int mode, bool stationary, double atolF, double h0 = 0) {
  const double tp = 15.0, T = 35.0, rtolF = 1e-6;
  CoupledResult r;
  dp5::Options o; o.h0 = h0;
  std::vector<cplx> Gbuf;
  auto F_spec = dp5::observable_block("F", m.nF, atolF, rtolF);
  std::vector<dp5::BlockSpec> blocks;
  double t0 = 0.0;
  if (mode == 1) {  // frozen: only F in the state, G in a user-owned buffer captured by the RHS
    Gbuf = m.Ginf; m.Gext = Gbuf.data(); m.idF = 0; m.rtde_on = true; t0 = tp;
    blocks.push_back(F_spec);
  } else {
    blocks.push_back(dp5::tensor_block("G", m.nG, 1e-8, 1e-6));
    m.idG = 0;
    if (mode == 2) { blocks.push_back(F_spec); m.idF = 1; m.rtde_on = true; t0 = tp; }
  }
  Integrator in(blocks, std::ref(m), o, t0);
  if (mode != 1) std::copy(m.G0.begin(), m.G0.end(), in.block(0));
  if (mode == 1 || mode == 2) std::copy(m.Fp.begin(), m.Fp.end(), in.block(m.idF));
  in.state_changed();
  std::vector<double> hs, ts; std::vector<std::size_t> worst;
  in.set_step_observer([&](double t, double h, double, bool acc) {
    if (acc) { ts.push_back(t); hs.push_back(h); worst.push_back(in.diagnostics().worst_block); }
  });
  if (mode == 0 || mode == 3) {
    in.add_stop(tp, [&](Integrator& it) {
      if (mode == 3) { Gbuf = it.extract_block(0); m.Gext = Gbuf.data(); }  // G1-G2 -> frozen handoff
      m.idF = it.append_block(F_spec);
      std::copy(m.Fp.begin(), m.Fp.end(), it.block(m.idF));
      m.rtde_on = true;
    });
  }
  // output only from the small block, on a grid starting at t'
  in.add_stop(tp, [&](Integrator& it) {
    it.set_output_grid(tp, T, 401, [&](const dp5::Frame& f) {
      r.frames.emplace_back(f[m.idF], f[m.idF] + m.nF); r.ft.push_back(f.t);
    });
  });
  const double w = omp_get_wtime();
  r.st = in.integrate(T);
  r.seconds = omp_get_wtime() - w;
  r.steps = in.stats().n_accept;
  r.state_size_after = in.size();
  r.prof_text = in.profile().to_text();
  // accuracy of every frame against the exact solution, in units of the F tolerance
  for (std::size_t f = 0; f < r.frames.size(); ++f)
    for (std::size_t j = 0; j < m.nF; ++j) {
      const cplx ex = m.exactF(j, r.ft[f], tp, stationary);
      r.tol_ratio = std::max(r.tol_ratio, std::abs(r.frames[f][j] - ex) / (atolF + rtolF * std::abs(ex)));
      if (f + 1 == r.frames.size()) r.rel_final = std::max(r.rel_final, std::abs(r.frames[f][j] - ex) / std::abs(ex));
    }
  // step-size statistics by window, and which block limited the step
  auto avg = [&](double a, double b, double& hav, double& fracF) {
    double s = 0; std::size_t n = 0, nf = 0;
    for (std::size_t i = 0; i < hs.size(); ++i)
      if (ts[i] >= a && ts[i] < b) { s += hs[i]; ++n; nf += (worst[i] == m.idF); }
    hav = n ? s / double(n) : 0; fracF = n ? double(nf) / double(n) : 0;
  };
  double dummy;
  avg(0, tp, r.h_before, dummy);
  avg(tp, tp + 5, r.h_after_early, r.frac_F_worst_early);
  avg(tp + 5, T, r.h_after_late, r.frac_F_worst_late);
  return r;
}

static void test_coupled() {
  const std::size_t nG = 2000000, nF = 15625;  // ~mGW G1-G2 at N=25; one N^3 RTDE block
  std::printf("[9] coupled G1-G2 (%zu) -> RTDE (%zu = 25^3), activation at t'=15, T=35, F decays ~e^-12\n", nG, nF);
  {
    CoupledModel m = make_model(nG, nF, false);
    CoupledResult a = run_coupled(m, 0, false, 1e-10);
    CoupledResult b = run_coupled(m, 0, false, 1e-8);
    std::printf("   (a) mean h: before t' %.4f (G limits), t'..t'+5 %.4f (F limits in %.0f%% of steps), "
                "after %.4f (F limits in %.0f%%)\n", a.h_before, a.h_after_early, 100 * a.frac_F_worst_early,
                a.h_after_late, 100 * a.frac_F_worst_late);
    std::printf("   (b) F atol=1e-10: max |F-exact|/(atol+rtol|exact|) = %.2f, final rel err %.1e, steps %llu, %.1fs\n",
                a.tol_ratio, a.rel_final, (unsigned long long)a.steps, a.seconds);
    std::printf("       F atol=1e-8 : max |F-exact|/(atol+rtol|exact|) = %.2f, final rel err %.1e, steps %llu\n",
                b.tol_ratio, b.rel_final, (unsigned long long)b.steps);
    std::printf("   profile of run (a), G1-G2 phase (segment 0) vs post-t' (segment 1+):\n%s", a.prof_text.c_str());
    CHECK(a.st == Status::Ok && b.st == Status::Ok, "coupled status");
    CHECK(a.frac_F_worst_early > 0.5, "F should limit the step right after t'");
    CHECK(a.h_after_early < a.h_before, "step should shrink after activation");
    CHECK(a.tol_ratio < 20 && b.tol_ratio < 20, "F error outside its own tolerance");
    CHECK(a.rel_final < b.rel_final, "tighter F atol should improve the late-time relative error");
  }
  {
    CoupledModel m = make_model(nG, nF, true);  // truly stationary G
    CoupledResult c1 = run_coupled(m, 0, true, 1e-10);
    CoupledResult c2 = run_coupled(m, 1, true, 1e-10, 1e-3);
    CoupledResult c3 = run_coupled(m, 2, true, 1e-10, 1e-3);
    CoupledResult c4 = run_coupled(m, 3, true, 1e-10);
    double d12 = 0, d23 = 0, d14 = 0;
    for (std::size_t f = 0; f < c1.frames.size(); ++f) {
      d12 = std::max(d12, maxabs(c1.frames[f], c2.frames[f]));
      d23 = std::max(d23, maxabs(c2.frames[f], c3.frames[f]));
      d14 = std::max(d14, maxabs(c1.frames[f], c4.frames[f]));
    }
    std::printf("   (c) stationary G: co-propagated from 0: tol ratio %.2f, state %zu, %.1fs\n", c1.tol_ratio,
                c1.state_size_after, c1.seconds);
    std::printf("       frozen external G from t' (state %zu): tol ratio %.2f, %.2fs; max|F_frozen - F_coprop| = %.1e\n",
                c2.state_size_after, c2.tol_ratio, c2.seconds, d12);
    std::printf("       co-propagated from t' with same h0: max|F - F_frozen| = %.1e (bitwise expected)\n", d23);
    std::printf("       extract_block handoff at t' (state after %zu): max|F - F_coprop| = %.1e, tol ratio %.2f\n",
                c4.state_size_after, d14, c4.tol_ratio);
    CHECK(c2.tol_ratio < 20 && c1.tol_ratio < 20 && c4.tol_ratio < 20, "stationary accuracy");
    CHECK(d23 == 0.0, "frozen vs co-propagated (same h0) not bitwise");
    CHECK(c2.state_size_after == nF && c4.state_size_after == nF, "frozen state size");
  }
}

// ---------------------------------------------------------------- 10. counterterm difference variable
// mGW: the resummed block G^{W,udud} and the counterterm G^{W,2,udud} have the same drift and the same
// source (Implementation_verification Eqs. mGWimp-P / mGWimp-P2), and I reads G^W - G^{W,2}. Toy: per
// entry G' = lam G + S(t) + eps*kap*G, G2' = lam G2 + S(t). Formulation A propagates (G, G2);
// formulation B propagates (D = G - G2, G2), with D' = lam D + eps*kap*(D + G2): source-free.
static std::vector<cplx> run_ct(int form, double atol, double rtol, std::uint64_t* steps) {
  const std::size_t n = 2000;
  const double eps = 1e-3;
  std::vector<cplx> lam(n), kap(n), s(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double u = double((i * 2654435761u) % 1009) / 1009.0;
    lam[i] = cplx(-0.2 - 0.3 * u, 3.0 * (u - 0.5));
    kap[i] = cplx(1.0, 0.5 + u);
    s[i] = cplx(std::cos(5.0 * u), std::sin(3.0 * u));
  }
  auto rhs = [&, form](double t, const cplx* y, cplx* dy) {
    const cplx* A = y; const cplx* G2 = y + n;
    for (std::size_t i = 0; i < n; ++i) {
      const cplx S = s[i] * std::exp(cplx(0, 1.3 * t));
      dy[n + i] = lam[i] * G2[i] + S;
      if (form == 0) dy[i] = lam[i] * A[i] + S + eps * kap[i] * A[i];          // A = G
      else           dy[i] = lam[i] * A[i] + eps * kap[i] * (A[i] + G2[i]);    // A = D
    }
  };
  Integrator in({dp5::tensor_block("A", n, atol, rtol), dp5::tensor_block("G2", n, atol, rtol)}, rhs);
  in.state_changed();
  in.integrate(10.0);
  if (steps) *steps = in.stats().n_accept;
  std::vector<cplx> D(n);
  for (std::size_t i = 0; i < n; ++i) D[i] = (form == 0) ? in.y()[i] - in.y()[n + i] : in.y()[i];
  return D;
}

static void test_counterterm_difference() {
  std::printf("[10] mGW counterterm: propagate G^W or D = G^W - G^{W,2} (same drift, same source)?\n");
  const std::vector<cplx> ref = run_ct(1, 1e-15, 1e-13, nullptr);
  double nref = 0; for (auto& z : ref) nref = std::max(nref, std::abs(z));
  double eA = 0, eB = 0, dAB = 0; std::uint64_t sA = 0, sB = 0;
  const auto A = run_ct(0, 1e-8, 1e-6, &sA), B = run_ct(1, 1e-8, 1e-6, &sB);
  for (std::size_t i = 0; i < ref.size(); ++i) {
    eA = std::max(eA, std::abs(A[i] - ref[i])); eB = std::max(eB, std::abs(B[i] - ref[i]));
    dAB = std::max(dAB, std::abs(A[i] - B[i]));
  }
  std::printf("   |D| ~ %.1e; rel. error of D: (G,G2) form %.2e (%llu steps), (D,G2) form %.2e (%llu steps); "
              "|D_A - D_B|/|D| = %.1e\n   -> RK is equivariant under linear changes of variables: no accuracy "
              "gain from the difference variable\n",
              nref, eA / nref, (unsigned long long)sA, eB / nref, (unsigned long long)sB, dAB / nref);
  // identical up to step-size decisions (the error norms see different variables): far below tolerance
  CHECK(dAB / nref < 1e-7 && eA < 2 * eB && eB < 2 * eA, "forms should be equivalent");
}

// JSON dump of the flat export (read by profile_summary.py)
static void write_profile_json(const dp5::Profile& P, const std::string& path) {
  std::ofstream f(path);
  f.precision(17);
  auto esc = [](const std::string& x) {
    std::string o;
    for (char c : x) { if (c == '"' || c == '\\') o += '\\'; o += c; }
    return o;
  };
  f << "{\"num\": {";
  bool first = true;
  P.visit([&](const std::string& n, const double* d, std::size_t k) {
    f << (first ? "" : ",") << "\n \"" << esc(n) << "\": [";
    for (std::size_t i = 0; i < k; ++i) {
      if (i) f << ",";
      if (std::isfinite(d[i])) f << d[i]; else f << "null";
    }
    f << "]";
    first = false;
  });
  f << "},\n\"str\": {";
  first = true;
  for (auto& kv : P.strings()) { f << (first ? "" : ",") << "\n \"" << esc(kv.first) << "\": \"" << esc(kv.second) << "\""; first = false; }
  f << "}}\n";
}

// ---------------------------------------------------------------- 12. profile on failure paths
static void test_profile_failures() {
  std::printf("[12] profile on failure paths (exceptions, non-finite) and memory accounting\n");
  auto closure_ok = [](const dp5::Profile& P) {
    double seg = 0; for (auto& g : P.segments) seg += g.integrate_seconds;
    return std::fabs(P.timed_seconds() + P.untimed_seconds - P.integrate_seconds) < 1e-9 &&
           std::fabs(seg - P.integrate_seconds) < 1e-6 * P.integrate_seconds + 1e-9;
  };
  const std::size_t n = 100000;
  {  // RHS throws in the middle of an attempt
    auto rhs = [&](double t, const cplx* y, cplx* dy) {
      if (t > 3.0) throw std::runtime_error("rhs failure at t>3");
#pragma omp parallel for schedule(static)
      for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(n); ++i) dy[i] = cplx(-0.1, 1.0) * y[i];
    };
    Integrator in({dp5::tensor_block("G", n)}, rhs);
    for (std::size_t i = 0; i < n; ++i) in.y()[i] = 1.0;
    in.state_changed();
    bool caught = false;
    try { in.integrate(10.0); } catch (const std::runtime_error&) { caught = true; }
    const dp5::Profile& P = in.profile();
    const std::uint64_t ev = P.accepted + P.rejected_error + P.rejected_nonfinite;
    const long long partial = (long long)P.rhs_calls - (long long)(6 * ev + P.slope_recomputes + P.init_step_estimates);
    std::printf("   RHS exception: caught=%d status=%s exceptions=%llu msg='%s' t=%.4f; RHS calls of the aborted attempt=%lld; closure ok=%d\n",
                caught, dp5::status_name(P.last_status), (unsigned long long)P.exceptions, P.last_exception.c_str(),
                P.t_last, partial, closure_ok(P));
    CHECK(caught && P.last_status == -1 && P.exceptions == 1 && P.last_exception == "rhs failure at t>3", "exception record");
    CHECK(partial >= 0 && partial <= 5 && closure_ok(P) && P.t_last <= 3.0, "profile consistent after exception");
  }
  {  // stop callback throws; the integrator can continue afterwards
    Integrator in({dp5::tensor_block("G", n)}, [&](double, const cplx* y, cplx* dy) {
#pragma omp parallel for schedule(static)
      for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(n); ++i) dy[i] = cplx(-0.1, 1.0) * y[i];
    });
    for (std::size_t i = 0; i < n; ++i) in.y()[i] = 1.0;
    in.state_changed();
    in.add_stop(2.0, [](Integrator&) { throw std::logic_error("callback failure"); });
    bool caught = false;
    try { in.integrate(5.0); } catch (const std::logic_error&) { caught = true; }
    const auto cb_calls = in.profile().sec[dp5::SecStopCallback].calls;
    const bool ok1 = closure_ok(in.profile()) && in.profile().last_status == -1;
    Status st = in.integrate(5.0);  // the stop was consumed; continue
    const dp5::Profile& P = in.profile();
    std::printf("   stop-callback exception: caught=%d, callback timed %llu x, closure ok=%d; continued: %s, "
                "integrate calls %llu, exceptions %llu, closure ok=%d\n",
                caught, (unsigned long long)cb_calls, ok1, dp5::to_string(st), (unsigned long long)P.integrate_calls,
                (unsigned long long)P.exceptions, closure_ok(P));
    CHECK(caught && cb_calls == 1 && ok1 && st == Status::Ok && P.last_status == 0 && P.exceptions == 1 && closure_ok(P),
          "callback exception path");
  }
  {  // NonFinite return: status + diagnostics in the profile
    Integrator in({dp5::observable_block("rho", 10), dp5::tensor_block("G", 1000)},
                  [](double t, const cplx* y, cplx* dy, const dp5::Layout& L) {
                    for (std::size_t i = 0; i < L.total; ++i) dy[i] = -0.1 * y[i];
                    if (t > 1.0) dy[L.offset[1] + 3] = cplx(std::nan(""), 0);
                  });
    for (std::size_t i = 0; i < in.size(); ++i) in.y()[i] = 1.0;
    in.state_changed();
    Status st = in.integrate(3.0);
    const dp5::Profile& P = in.profile();
    std::printf("   NonFinite: status %s, profile status %s, worst block %d, rejected_nonfinite %llu, closure ok=%d\n",
                dp5::to_string(st), dp5::status_name(P.last_status), P.last_worst_block,
                (unsigned long long)P.rejected_nonfinite, closure_ok(P));
    CHECK(P.last_status == int(Status::NonFinite) && P.last_worst_block == 1 && P.rejected_nonfinite == 1 && closure_ok(P),
          "NonFinite profile");
  }
  {  // memory accounting across an append
    Integrator in({dp5::tensor_block("G", n)}, [&](double, const cplx* y, cplx* dy, const dp5::Layout& L) {
      for (std::size_t i = 0; i < L.total; ++i) dy[i] = -0.1 * y[i];
    });
    for (std::size_t i = 0; i < n; ++i) in.y()[i] = 1.0;
    in.state_changed();
    in.add_stop(1.0, [](Integrator& it) { it.append_block(dp5::observable_block("F", 5000)); });
    in.integrate(2.0);
    const dp5::Profile& P = in.profile();
    const std::size_t regs = 8 * in.size() * sizeof(cplx);
    const std::size_t trans = (n + 8 * in.size()) * sizeof(cplx);
    std::printf("   memory: registers %zu B (= 8 x %zu x 16: %d), transient peak %zu B (= (old + 8 new) x 16: %d), "
                "dense %zu B, layout changes %zu, process peak RSS %.1f MB\n",
                P.register_bytes, in.size(), P.register_bytes == regs, P.layout_transient_peak_bytes,
                P.layout_transient_peak_bytes == trans, P.dense_bytes, P.layout_t.size(), P.maxrss_bytes / 1e6);
    CHECK(P.register_bytes == regs && P.register_bytes_peak == regs && P.layout_transient_peak_bytes == trans &&
              P.dense_bytes == 5000 * sizeof(cplx) && P.layout_t.size() == 2 && P.maxrss_bytes > 0,
          "memory accounting");
  }
}

// batch append (append_blocks): identical layout, ids and trajectory to block-by-block appends, one reallocation
static void test_append_batch() {
  std::printf("[8b] append_blocks: one reallocation, bitwise equal to sequential append_block\n");
  const std::size_t n = 20000, nb = 12, m = 3000;
  auto rhs = [&](double t, const cplx* y, cplx* dy, const dp5::Layout& L) {
    for (std::size_t i = 0; i < L.size[0]; ++i) dy[L.offset[0] + i] = cplx(-0.2, 0.7) * y[L.offset[0] + i];
    for (std::size_t q = 1; q < L.offset.size(); ++q) {
      if (L.removed[q]) continue;
      for (std::size_t i = 0; i < L.size[q]; ++i)
        dy[L.offset[q] + i] = cplx(-0.1 * double(q), 0.3) * y[L.offset[q] + i] + std::cos(t) * y[L.offset[0] + (i * 7 + q) % n];
    }
  };
  auto run = [&](bool batch, std::vector<std::size_t>& ids, std::uint64_t& ver, std::size_t& nlayout) {
    Integrator in({dp5::tensor_block("G", n)}, rhs);
    for (std::size_t i = 0; i < n; ++i) in.y()[i] = cplx(1.0 + 0.001 * double(i % 17), 0.01 * double(i % 5));
    in.state_changed();
    in.add_stop(1.0, [&](Integrator& it) {
      std::vector<dp5::BlockSpec> bs;
      for (std::size_t b = 0; b < nb; ++b) bs.push_back(dp5::tensor_block("F" + std::to_string(b), m + 100 * b, 1e-10, 1e-7));
      if (batch) ids = it.append_blocks(bs);
      else for (auto& b : bs) ids.push_back(it.append_block(b));
    });
    in.integrate(2.0);
    ver = in.layout_version();
    nlayout = in.profile().layout_t.size();
    return std::vector<cplx>(in.y(), in.y() + in.size());
  };
  std::vector<std::size_t> ia, ib;
  std::uint64_t va, vb;
  std::size_t la, lb;
  const auto ya = run(false, ia, va, la), yb = run(true, ib, vb, lb);
  std::printf("   sequential: %zu layout changes (version %llu); batch: %zu (version %llu)\n", la, (unsigned long long)va, lb,
              (unsigned long long)vb);
  CHECK(ia == ib && ya.size() == yb.size() && std::memcmp(ya.data(), yb.data(), ya.size() * sizeof(cplx)) == 0,
        "batch append differs from sequential append");
  CHECK(lb + nb - 1 == la, "batch append should reallocate once");
  {  // all-or-nothing on an invalid spec (duplicate name)
    Integrator in({dp5::tensor_block("G", 10)}, [](double, const cplx* y, cplx* dy, const dp5::Layout& L) {
      for (std::size_t i = 0; i < L.total; ++i) dy[i] = -y[i];
    });
    bool threw = false;
    try { in.append_blocks({dp5::tensor_block("A", 5), dp5::tensor_block("A", 5)}); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw && in.num_blocks() == 1 && in.size() == 10, "invalid batch must leave the layout unchanged");
  }
}

// cost of one steady_clock::now() call (seconds), for the profiling-overhead estimate
static double clock_read_cost() {
  using C = std::chrono::steady_clock;
  const int n = 2000000;
  volatile long long sink = 0;
  const auto t0 = C::now();
  for (int i = 0; i < n; ++i) sink = sink + C::now().time_since_epoch().count();
  return std::chrono::duration<double>(C::now() - t0).count() / n;
}
// estimated instrumentation time: 2 clock reads per timed section call (+2 per thread per pass if enabled)
static double profile_cost_estimate(const dp5::Profile& P, double tclk) {
  double calls = 0;
  for (int i = 0; i < dp5::SecCount; ++i) calls += double(P.sec[i].calls);
  return 2.0 * calls * tclk + double(P.attempts) * 4.0 * tclk;  // + prof_step bookkeeping (~ a few clock-reads worth)
}

// ---------------------------------------------------------------- 11. profiling
struct ProfRun { std::vector<cplx> y; std::vector<cplx> frames; dp5::Profile prof; dp5::Stats stats; std::size_t ngrid; };

static ProfRun run_profiled(bool profile, bool threads, std::size_t trace) {
  // 2 blocks + an RTDE-like block appended at t'=4; h0 deliberately too large -> early rejections
  const std::size_t n1 = 1000, n2 = 200000, n3 = 15625;
  std::vector<cplx> lam(n1 + n2), mu(n3);
  for (std::size_t i = 0; i < n1 + n2; ++i) lam[i] = cplx(-0.05 - 0.2 * double(i % 97) / 97.0, 3.0 * double(i % 89) / 89.0 - 1.5);
  for (std::size_t j = 0; j < n3; ++j) mu[j] = cplx(-0.5, 8.0 * double(j % 101) / 101.0 - 4.0);
  bool on = false;
  std::size_t idF = 0;
  auto rhs = [&](double t, const cplx* y, cplx* dy, const dp5::Layout& L) {
    const std::ptrdiff_t n = std::ptrdiff_t(n1 + n2);
#pragma omp parallel for schedule(static) if (n > 8192)
    for (std::ptrdiff_t i = 0; i < n; ++i) dy[i] = lam[std::size_t(i)] * y[i] + cplx(0.01 * std::cos(t), 0);
    if (on) {
      const cplx* F = y + L.offset[idF]; cplx* dF = dy + L.offset[idF];
      dp5::parallel_for_block(L, idF, [&](std::ptrdiff_t j) { dF[j] = mu[std::size_t(j)] * F[j] + 0.1 * y[j % n1]; });
    }
  };
  dp5::Options o;
  o.profile = profile; o.profile_threads = threads; o.profile_trace_max = trace; o.h0 = 2.0;
  Integrator in({dp5::observable_block("rho", n1, 1e-9, 1e-7), dp5::tensor_block("G", n2)}, rhs, o);
  for (std::size_t i = 0; i < n1 + n2; ++i) in.y()[i] = cplx(1.0, 0.001 * double(i % 13));
  in.state_changed();
  ProfRun r;
  r.ngrid = 101;
  in.set_output_grid(0, 10, r.ngrid, [&](const dp5::Frame& f) { r.frames.insert(r.frames.end(), f[0], f[0] + 8); });
  in.add_stop(4.0, [&](Integrator& it) {
    idF = it.append_block(dp5::observable_block("F", n3, 1e-10, 1e-6));
    it.block(idF)[0] = cplx(0, -1);
    on = true;
  });
  in.add_stop(7.0, [&](Integrator& it) { (void)it.checkpoint(); });
  in.integrate(6.0);   // two integrate() calls: profile accumulates
  in.integrate(10.0);
  r.y.assign(in.y(), in.y() + in.size());
  r.prof = in.profile();
  r.stats = in.stats();
  return r;
}

static void test_profiling() {
  std::printf("[11] profiling: counters, timing closure, bitwise on/off\n");
  ProfRun a = run_profiled(true, true, 64), b = run_profiled(false, false, 0);
  const dp5::Profile& P = a.prof;
  const std::uint64_t evaluated = P.accepted + P.rejected_error + P.rejected_nonfinite;
  const std::uint64_t rhs_expected = 6 * evaluated + P.slope_recomputes + P.init_step_estimates;
  bool stage_calls = true;
  for (int s = dp5::SecRhsK2; s <= dp5::SecRhsK7; ++s) stage_calls = stage_calls && P.sec[s].calls == P.attempts;
  std::uint64_t seg_acc = 0, lim = 0, hh = 0; double seg_time = 0;
  for (auto& g : P.segments) { seg_acc += g.accepted; seg_time += g.integrate_seconds; }
  for (auto c : P.limiting_block) lim += c;
  for (auto c : P.h_hist) hh += c;
  const double closure = P.timed_seconds() - P.sec[dp5::SecLayout].seconds_outside * 0 + P.untimed_seconds - P.integrate_seconds;
  std::printf("   attempts %llu = accepted %llu + rejected %llu + non-finite %llu; RHS %llu = 6*%llu + slope %llu + h0 %llu "
              "(Stats.n_rhs %llu); stage calls = attempts: %d\n",
              (unsigned long long)P.attempts, (unsigned long long)P.accepted, (unsigned long long)P.rejected_error,
              (unsigned long long)P.rejected_nonfinite, (unsigned long long)P.rhs_calls, (unsigned long long)evaluated,
              (unsigned long long)P.slope_recomputes, (unsigned long long)P.init_step_estimates,
              (unsigned long long)a.stats.n_rhs, stage_calls);
    std::printf("   timing closure: sections %.6f s + untimed %.6f s - integrate %.6f s = %.1e s (untimed %.2f%% of total); "
              "segments %zu (stops %llu), segment times sum %.6f s\n",
              P.timed_seconds(), P.untimed_seconds, P.integrate_seconds, closure, 100 * P.untimed_seconds / P.integrate_seconds,
              P.segments.size(), (unsigned long long)P.stops, seg_time);
  CHECK(P.attempts == evaluated && P.accepted == a.stats.n_accept && P.rejected_error == a.stats.n_reject, "step counters");
  CHECK(P.rejected_error > 0, "test should provoke rejections (h0 too large)");
  CHECK(P.rhs_calls == rhs_expected && P.rhs_calls == a.stats.n_rhs && stage_calls, "RHS accounting");
  CHECK(P.frames_dense + P.frames_exact == a.ngrid, "frame count");
  CHECK(seg_acc == P.accepted && lim == P.accepted && hh == P.accepted, "segment/histogram sums");
  CHECK(P.segments.size() == P.stops + 1 && P.integrate_calls == 2, "segments");
  CHECK(std::fabs(closure) < 1e-9 && std::fabs(seg_time - P.integrate_seconds) < 1e-6 * P.integrate_seconds + 1e-9,
        "timing closure");
  CHECK(P.untimed_seconds < 0.02 * P.integrate_seconds, "untimed residual too large");
  CHECK(P.trace_t.size() <= 64 && P.trace_stride > 1, "trace decimation");
  CHECK(P.sec[dp5::SecPassError].imb_mean > 0, "thread imbalance recorded");
  // bitwise identity with profiling off
  const bool same_y = a.y.size() == b.y.size() && std::memcmp(a.y.data(), b.y.data(), a.y.size() * sizeof(cplx)) == 0;
  const bool same_f = a.frames.size() == b.frames.size() &&
                      std::memcmp(a.frames.data(), b.frames.data(), a.frames.size() * sizeof(cplx)) == 0;
  std::printf("   profiling on (threads+trace) vs off: state bitwise equal %d, frames bitwise equal %d; off-profile attempts=%llu\n",
              same_y, same_f, (unsigned long long)b.prof.attempts);
  CHECK(same_y && same_f && b.prof.attempts == 0, "profiling changed the result");
  // export
  const auto m = P.to_map();
  const auto st = P.strings();
  std::printf("   export: %zu numeric keys (e.g. time/pass_error/seconds = %.4f), %zu string keys (blocks/names = %s)\n",
              m.size(), m.at("time/pass_error/seconds")[0], st.size(), st.at("blocks/names").c_str());
  const double tclk = clock_read_cost();
  std::printf("   instrumentation estimate: steady_clock::now() = %.1f ns, %.1f timed sections per attempted step -> "
              "%.4f%% of integrate time (%.4f%% of integrator overhead)\n",
              1e9 * tclk, [&] { double c = 0; for (int i = 0; i < dp5::SecCount; ++i) c += double(P.sec[i].calls); return c / double(P.attempts); }(),
              100 * profile_cost_estimate(P, tclk) / P.integrate_seconds,
              100 * profile_cost_estimate(P, tclk) / P.overhead_seconds());
  std::printf("%s", P.to_text().c_str());
  write_profile_json(P, "profile_dump.json");
  std::printf("   wrote profile_dump.json (schema version %d)\n", dp5::Profile::schema_version);
}

// checksum of a fixed run (used to compare builds with -DDP5_PROFILE=0/1)
static void checksum_run() {
  ProfRun r = run_profiled(true, true, 64);
  std::uint64_t h = 1469598103934665603ull;
  auto mix = [&](const std::vector<cplx>& v) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(v.data());
    for (std::size_t i = 0; i < v.size() * sizeof(cplx); ++i) { h ^= p[i]; h *= 1099511628211ull; }
  };
  mix(r.y); mix(r.frames);
  std::printf("CHECKSUM DP5_PROFILE=%d %016llx profile_attempts=%llu\n", DP5_PROFILE, (unsigned long long)h,
              (unsigned long long)r.prof.attempts);
}

// ---------------------------------------------------------------- scipy export
static void scipy_export(const std::string& dir) {
  const std::vector<cplx> lam = {cplx(-0.3, 1), cplx(-0.1, 3), cplx(-1, 0.5), cplx(0, 0.2)};
  auto rhs = [&](double, const cplx* y, cplx* dy) {
    for (int i = 0; i < 4; ++i) dy[i] = lam[i] * y[i] + cplx(0, 0.3) * std::norm(y[i]) * y[i];
  };
  for (int variant = 0; variant < 2; ++variant) {
    dp5::Options o;
    if (variant == 0) {  // scipy-faithful controller
      o.controller = dp5::Controller::Standard;
      o.landing_slack = 1.0;
      o.carry_h_across_landing = false;
    }
    o.combine = dp5::Combine::GlobalRMS;
    Integrator in({{"y", 4, 1e-9, 1e-6, 1, dp5::BlockNorm::RMS, true, true}}, rhs, o);
    const cplx y0[4] = {1, cplx(0.5, 0.5), cplx(0, -0.3), 0.8};
    std::copy(y0, y0 + 4, in.block(0));
    in.state_changed();
    const char* tag = variant == 0 ? "standard" : "pi";
    std::ofstream steps(dir + "/steps_" + tag + ".txt"), frames(dir + "/frames_" + tag + ".txt");
    steps.precision(17); frames.precision(17);
    in.set_step_observer([&](double t, double h, double err, bool acc) { steps << t << " " << h << " " << err << " " << acc << "\n"; });
    in.set_output_grid(0, 10, 201, [&](const dp5::Frame& f) {
      frames << f.t;
      for (int i = 0; i < 4; ++i) frames << " " << f[0][i].real() << " " << f[0][i].imag();
      frames << "\n";
    });
    in.integrate(10.0);
  }
  std::printf("wrote scipy cross-check data to %s\n", dir.c_str());
}

// ---------------------------------------------------------------- benchmark
// Faithful re-implementation of the old RK45.hpp register set and passes:
// Cash-Karp, serial loops, k1..k6, y_tmp, y5, y4, y_old, k1_old (full), y,
// g_interp/rho_interp, L-infinity error, three full copies per accepted step,
// next k1 evaluated after acceptance, Hermite interpolation of save and tensor
// parts at every output sample.
struct OldRK45 {
  double atol, rtol;
  using SV = std::vector<cplx>;
  double rhs_sec = 0; std::uint64_t attempts = 0, accepts = 0;
  template <class F>
  void integrate(F f, double t0, double t_end, SV& y, double dt, std::size_t n_samples, std::size_t save_size) {
    const std::size_t N = y.size(), g_size = N - save_size;
    auto call = [&](double t, const SV& a, SV& b) { double w = omp_get_wtime(); f(t, a.data(), b.data()); rhs_sec += omp_get_wtime() - w; };
    const double a21 = 1.0 / 5.0, a31 = 3.0 / 40.0, a32 = 9.0 / 40.0, a41 = 3.0 / 10.0, a42 = -9.0 / 10.0, a43 = 6.0 / 5.0;
    const double a51 = -11.0 / 54.0, a52 = 5.0 / 2.0, a53 = -70.0 / 27.0, a54 = 35.0 / 27.0;
    const double a61 = 1631.0 / 55296.0, a62 = 175.0 / 512.0, a63 = 575.0 / 13824.0, a64 = 44275.0 / 110592.0, a65 = 253.0 / 4096.0;
    const double b1 = 37.0 / 378.0, b2 = 0.0, b3 = 250.0 / 621.0, b4 = 125.0 / 594.0, b5 = 0.0, b6 = 512.0 / 1771.0;
    const double s1 = 2825.0 / 27648.0, s2 = 0.0, s3 = 18575.0 / 48384.0, s4 = 13525.0 / 55296.0, s5 = 277.0 / 14336.0, s6 = 0.25;
    std::vector<double> t_eval(n_samples);
    for (std::size_t i = 0; i < n_samples; ++i) t_eval[i] = t0 + i * (t_end - t0) / (n_samples - 1.0);
    std::vector<SV> traj; traj.reserve(n_samples);
    SV k1(N), k2(N), k3(N), k4(N), k5(N), k6(N), y_tmp(N), y5(N), y4(N), rho_interp(save_size), g_interp(g_size);
    SV final_G(g_size);
    double t = t0, t_old = t0;
    SV y_old = y, k1_old(N);
    call(t, y, k1); k1_old = k1;
    std::size_t save_idx = 0;
    auto herm = [&](double tt, std::size_t sz, std::size_t off, SV& out) {
      const double h = t - t_old, th = (tt - t_old) / h, th2 = th * th, th3 = th2 * th;
      const double h00 = 2 * th3 - 3 * th2 + 1, h10 = th3 - 2 * th2 + th, h01 = -2 * th3 + 3 * th2, h11 = th3 - th2;
      for (std::size_t i = 0; i < sz; ++i)
        out[i] = h00 * y_old[off + i] + h10 * h * k1_old[off + i] + h01 * y[off + i] + h11 * h * k1[off + i];
    };
    if (t_eval[0] <= t) { std::copy(y.begin(), y.begin() + save_size, rho_interp.begin()); traj.push_back(rho_interp); ++save_idx; }
    while (t < t_end) {
      if (t + dt > t_end) dt = t_end - t;
      bool acc = false;
      while (!acc) {
        ++attempts;
        for (std::size_t i = 0; i < N; ++i) y_tmp[i] = y[i] + dt * (a21 * k1[i]);
        call(t + 0.2 * dt, y_tmp, k2);
        for (std::size_t i = 0; i < N; ++i) y_tmp[i] = y[i] + dt * (a31 * k1[i] + a32 * k2[i]);
        call(t + 0.3 * dt, y_tmp, k3);
        for (std::size_t i = 0; i < N; ++i) y_tmp[i] = y[i] + dt * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
        call(t + 0.6 * dt, y_tmp, k4);
        for (std::size_t i = 0; i < N; ++i) y_tmp[i] = y[i] + dt * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
        call(t + dt, y_tmp, k5);
        for (std::size_t i = 0; i < N; ++i) y_tmp[i] = y[i] + dt * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
        call(t + 0.875 * dt, y_tmp, k6);
        for (std::size_t i = 0; i < N; ++i) {
          y5[i] = y[i] + dt * (b1 * k1[i] + b2 * k2[i] + b3 * k3[i] + b4 * k4[i] + b5 * k5[i] + b6 * k6[i]);
          y4[i] = y[i] + dt * (s1 * k1[i] + s2 * k2[i] + s3 * k3[i] + s4 * k4[i] + s5 * k5[i] + s6 * k6[i]);
        }
        double err = 0;
        for (std::size_t i = 0; i < N; ++i) err = std::max(err, std::abs(y5[i] - y4[i]) / (atol + rtol * std::abs(y[i])));
        if (err <= 1.0) {
          t_old = t; y_old = y; k1_old = k1; t += dt; y = y5; ++accepts;
          if (t < t_end) call(t, y, k1);
          while (save_idx < n_samples && t_eval[save_idx] <= t) {
            herm(t_eval[save_idx], save_size, 0, rho_interp); traj.push_back(rho_interp);
            herm(t_eval[save_idx], g_size, save_size, final_G);
            ++save_idx;
          }
          acc = true;
        }
        dt = dt * 0.9 * std::max(0.1, std::min(5.0, std::pow(err, -0.2)));
      }
    }
  }
};

static long maxrss_kb() { struct rusage ru; getrusage(RUSAGE_SELF, &ru); return ru.ru_maxrss; }

static void bench(const std::string& which, std::size_t n) {
  std::vector<cplx> lam(n), y0(n);
  for (std::size_t i = 0; i < n; ++i) {
    lam[i] = cplx(-0.05 - 0.3 * double(i % 101) / 101.0, 2.0 * double(i % 89) / 89.0 - 1.0);
    y0[i] = cplx(1.0, 0.01 * double(i % 7));
  }
  DiagRhs rhs{lam};
  const double T = 20.0, atol = 1e-8, rtol = 1e-6;
  const std::size_t save = std::max<std::size_t>(1, n / 100), samples = 201;
  const long rss0 = maxrss_kb();
  double total, rhs_sec; std::uint64_t att, acc; std::size_t regs;
  if (which == "old") {
    OldRK45 o{atol, rtol};
    std::vector<cplx> y = y0;
    double w = omp_get_wtime();
    o.integrate(rhs, 0.0, T, y, 0.01, samples, save);
    total = omp_get_wtime() - w; rhs_sec = o.rhs_sec; att = o.attempts; acc = o.accepts;
    regs = 12;  // + g_interp/final_G (~2 x tensor part) + trajectory
  } else {
    dp5::Options o;
    o.profile = (which != "new-noprof");
    o.profile_threads = (which == "new-thr");
    Integrator in({{"save", save, atol, rtol, 1, dp5::BlockNorm::Max, true, true},
                   {"tensor", n - save, atol, rtol, 1, dp5::BlockNorm::Max}}, rhs, o);  // L-inf like the old code
    std::copy(y0.begin(), y0.end(), in.y()); in.state_changed();
    std::size_t nfr = 0;
    std::vector<cplx> traj;
    traj.reserve(save * samples);
    in.set_output_grid(0, T, samples, [&](const dp5::Frame& f) { traj.insert(traj.end(), f[0], f[0] + save); ++nfr; });
    double w = omp_get_wtime();
    in.integrate(T);
    total = omp_get_wtime() - w; rhs_sec = in.stats().rhs_seconds;
    att = in.stats().n_accept + in.stats().n_reject; acc = in.stats().n_accept; regs = in.registers();
    if (in.profile().enabled) {
      const double tclk = clock_read_cost();
      const auto& P = in.profile();
      std::printf("PROFILE-COST n=%zu clock_read=%.1f ns estimated instrumentation %.5f%% of integrate, %.5f%% of integrator overhead\n",
                  n, 1e9 * tclk, 100 * profile_cost_estimate(P, tclk) / P.integrate_seconds,
                  100 * profile_cost_estimate(P, tclk) / P.overhead_seconds());
      if (n >= 4000000) std::printf("%s", P.to_text().c_str());
    }
  }
  const double ovh = (total - rhs_sec) / double(att);
  std::printf("BENCH %s n=%zu threads=%d attempts=%llu accepts=%llu total=%.3fs rhs=%.3fs overhead/step=%.3f ms "
              "regs=%zu reg_MB=%.1f maxrss_MB=%.1f\n",
              which.c_str(), n, omp_get_max_threads(), (unsigned long long)att, (unsigned long long)acc, total, rhs_sec,
              1e3 * ovh, regs, double(regs) * n * 16 / 1e6, (maxrss_kb() - 0 * rss0) / 1024.0);
}

int main(int argc, char** argv) {
  if (argc >= 3 && std::string(argv[1]) == "scipy") { scipy_export(argv[2]); return 0; }
  if (argc >= 2 && std::string(argv[1]) == "checksum") { checksum_run(); return 0; }
  if (argc >= 4 && std::string(argv[1]) == "bench") { bench(argv[2], std::stoul(argv[3])); return 0; }
  std::printf("dp5 integrator tests, OpenMP threads = %d\n", omp_get_max_threads());
  test_order();
  test_stiffish();
  test_landing();
  test_activation();
  test_failures();
  test_validation();
  test_restart();
  test_layout_tools();
  test_append_batch();
  test_coupled();
  test_counterterm_difference();
  test_profiling();
  test_profile_failures();
  std::printf("%s (%d failures)\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED", g_fail);
  return g_fail ? 1 : 0;
}
