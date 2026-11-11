// dp5drive -- one executable for all approximations and modes (README.md).
//
//   dp5drive --runcard RC.h5 --output OUT.h5 [--threads N] [--print-layout] [--version]
//   dp5drive --runcard RC.h5 --eval-rhs STATE.h5 RESULT.h5      (test hook: one RHS evaluation, see tests/)
//
// Exit codes: 0 Ok, 2 input error, 3 not implemented, 5 integrator failure, 6 NoSteadyState, 7 Diverged.
#include <omp.h>
#include <unistd.h>

#include <cstdlib>
#include <ctime>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <algorithm>
#include <iostream>
#include <map>

#include "output.hpp"
#include "physics.hpp"
#include "steady_state.hpp"
#include "version.hpp"

using namespace drv;
namespace fs = std::filesystem;

namespace {

struct Cli {
  std::string runcard, output;
  std::map<std::string, std::string> overrides;
  bool print_layout = false, version = false;
  std::string eval_in, eval_out;  // test hook: evaluate the RHS once on a given state
  std::string cmdline;
};

Cli parse_cli(int argc, char** argv) {
  Cli c;
  for (int i = 0; i < argc; ++i) c.cmdline += std::string(i ? " " : "") + argv[i];
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
      return argv[++i];
    };
    if (a == "--runcard") c.runcard = val();
    else if (a == "--threads") c.overrides["threads"] = val();
    else if (a == "--output") { c.output = val(); c.overrides["output"] = c.output; }
    else if (a == "--print-layout") c.print_layout = true;
    else if (a == "--version") c.version = true;
    else if (a == "--eval-rhs") { c.eval_in = val(); c.eval_out = val(); }
    else throw std::invalid_argument("unknown option " + a + " (dp5drive --runcard RC.h5 --output OUT.h5 [--threads N] "
                                     "[--print-layout] [--version])");
  }
  if (c.version) return c;
  if (c.runcard.empty()) throw std::invalid_argument("--runcard is required");
  if (!c.print_layout && c.eval_in.empty()) {
    if (c.output.empty()) throw std::invalid_argument("--output OUT.h5 is required");
    if (c.output.size() < 4 || c.output.substr(c.output.size() - 3) != ".h5")
      throw std::invalid_argument("--output must name an .h5 file (" + c.output + ")");
  }
  return c;
}

std::string make_run_id() {
  char b[64];
  std::time_t t = std::time(nullptr);
  std::strftime(b, sizeof b, "%Y%m%dT%H%M%S", std::localtime(&t));
  return std::string("run_") + b + "_" + std::to_string(getpid());
}

// initial rho [nspin,N,N] from the model and the run card's IC override
std::vector<cplx> initial_rho(const Model& m, const RunCard& rc) {
  const std::size_t n2 = std::size_t(m.N) * m.N;
  std::vector<cplx> r = m.rho0;
  if (rc.ic_mode == "full") {
    if (rc.rho0_override.size() != 2 * n2) throw std::runtime_error("run card rho0 must be [2,N,N]");
    r = rc.rho0_override;
  }
  return r;
}

LayoutConfig layout_config(const RunCard& rc, int N) {
  LayoutConfig c;
  c.approx = rc.approx; c.spin = rc.spin; c.N = N;
  for (int i = 0; i < 2; ++i) { c.tol_rho[i] = rc.tol_rho[i]; c.tol_tensor[i] = rc.tol_tensor[i]; c.tol_rtde[i] = rc.tol_rtde[i]; }
  return c;
}

std::string status_string(dp5::Status s) { return dp5::to_string(s); }

struct Run {
  const RunCard& rc;
  const Model& m;
  Output& out;
  LayoutConfig lc;
  std::chrono::steady_clock::time_point wall0 = std::chrono::steady_clock::now();
  std::vector<double> tp_eff;  // t' values in use (run card, or the detected t' with t_prime = auto)

  // rtde_tau = auto: state at the start of the current step (the step in which the stop frame falls is redone as a
  // landing on it)
  dp5::Checkpoint prev_cp;
  std::vector<cplx> prev_y;
  double T_target = 0;
  // automatic RTDE end (rtde_tau = "auto", README §1e): monitor on the G^R frames of activation k
  struct AutoEnd {
    std::unique_ptr<RtdeStopMonitor> mon;
    std::vector<int> rows;
    std::string rows_desc;
    int k = 0;
    bool fired = false, handled = false;
    double T_cut = 1e300;
  };
  AutoEnd ae;

  std::vector<int> monitor_rows(std::string& desc) const {
    std::string mode = rc.rtde_monitor;
    const bool imp = m.attrs.count("N_greater") && m.attrs.count("N_lesser");
    if (mode == "auto") mode = imp ? "impurity" : "all";
    std::vector<int> r;
    if (mode == "impurity") {
      if (!imp) throw std::runtime_error("rtde_monitor = impurity needs a model with N_greater/N_lesser attributes");
      const int nimp = m.N - int(std::lround(std::stod(m.attrs.at("N_greater")))) - int(std::lround(std::stod(m.attrs.at("N_lesser"))));
      if (nimp < 1 || nimp > m.N) throw std::runtime_error("rtde_monitor = impurity: no impurity orbitals in the model");
      for (int x = 0; x < nimp; ++x) r.push_back(x);
      desc = "impurity orbitals 0.." + std::to_string(nimp - 1);
    } else {
      for (int x = 0; x < m.N; ++x) r.push_back(x);
      desc = "all orbitals";
    }
    return r;
  }
  // m(tau) = max |G^R_{xs}| over the probe columns, spins and monitored rows x of one frame row [ncol, ns, N]
  double monitor_value(const cplx* row, int ns) const {
    double mx = 0;
    for (std::size_t c = 0; c < rc.probe_columns.size(); ++c)
      for (int sp = 0; sp < ns; ++sp)
        for (int x : ae.rows) mx = std::max(mx, std::abs(row[(c * std::size_t(ns) + std::size_t(sp)) * std::size_t(m.N) + std::size_t(x)]));
    return mx;
  }
  void setup_monitor(int k, double tp, double lam, double spread) {
    if (!rc.rtde_tau_auto) return;
    if (!(lam > 0) || !std::isfinite(lam)) throw std::runtime_error("rtde_tau = auto: lambda_ref(t') = 0 (a non-decaying one-body mode)");
    double W = rc.c_hold / lam;
    // option rtde_window_beats = n (default 8, 0: off): W >= n beat periods 2 pi/spread, so that the binned envelope (bins
    // >= one beat when W >= 8 beats) does not see the oscillation of G^R (imp_configs c3, gamma = 2, DSL: W = 1.8 ~ one
    // beat, the half-window fits succeeded once in 9e4 frames and the RTDE ran to tau = 929 instead of ~38)
    if (rc.rtde_window_beats > 0 && spread > 0) W = std::max(W, rc.rtde_window_beats * 2.0 * M_PI / spread);
    const double tmin = rc.rtde_tau_min > 0 ? rc.rtde_tau_min : rc.c_tau_min / lam;
    ae.k = k;
    ae.rows = monitor_rows(ae.rows_desc);
    ae.mon = std::make_unique<RtdeStopMonitor>(rc.tol_A, tmin, W, spread, rc.ss_fit_rtol);
    out.write_attrs("rtde_end",
                    {{"tol_A", rc.tol_A}, {"rtde_tau_max", rc.rtde_tau_max}, {"rtde_tau_min_used", tmin},
                     {"rtde_tau_min", rc.rtde_tau_min}, {"c_tau_min", rc.c_tau_min}, {"c_hold", rc.c_hold},
                     {"window", W}, {"lambda_ref_tprime", lam}, {"re_spread_tprime", spread}, {"t_prime", tp},
                     {"ss_fit_rtol", rc.ss_fit_rtol}, {"activation", double(k)}, {"rtde_window_beats", rc.rtde_window_beats}},
                    {{"monitor", ae.rows_desc}, {"rtde_monitor", rc.rtde_monitor}}, {});
  }
  // interrupt hook of the final segment: divergence stop and the automatic RTDE end
  std::function<bool()> end_hook(dp5::Integrator& in, std::function<bool()> base, int& why) {
    return [this, &in, base, &why]() {
      if (base()) { why = 1; return true; }
      if (ae.fired && !ae.handled) { why = 3; return true; }
      const double t = in.t();
      if (t < T_target - dp5::Integrator::land_tol(T_target)) {
        if (ae.mon) {  // every step once the auto-end monitor runs
          prev_cp = in.checkpoint();
          prev_y.assign(in.y(), in.y() + in.size());
        }
      }
      return false;
    };
  }
  dp5::Status integrate_to(dp5::Integrator& in, double T) {
    try {
      return in.integrate(T);
    } catch (...) {
      out.profile(in.profile());
      throw;
    }
  }
  // final segment up to T (the RTDE cap with rtde_tau = auto); returns the integrator status
  dp5::Status drive(dp5::Integrator& in, double T, const std::function<bool()>& base) {
    T_target = T;
    if (ae.mon) {
      prev_cp = in.checkpoint(); prev_y.assign(in.y(), in.y() + in.size());
    }
    int why = 0;
    in.set_interrupt(end_hook(in, base, why));
    dp5::Status st = integrate_to(in, T);
    if (st == dp5::Status::Interrupted && why == 3) {
      ae.handled = true;
      if (ae.T_cut <= in.t() + dp5::Integrator::land_tol(in.t())) {
        // the stop frame lies inside the last step: redo that step as a landing on T_cut (frames up to T_cut
        // are already written; no frame after it)
        in.restore(prev_cp, prev_y.data());
        in.set_output_grid(ae.T_cut, ae.T_cut, 1, [](const dp5::Frame&) {});
      }
      T_target = ae.T_cut;
      why = 0;
      st = integrate_to(in, ae.T_cut);
    }
    return st;
  }
  void end_report(dp5::Integrator& in, dp5::Status st) {
    if (!ae.mon) return;
    const RtdeStopMonitor& M = *ae.mon;
    out.write_series("rtde_end", "m_tau", M.taus());
    out.write_series("rtde_end", "m", M.values());
    if (st == dp5::Status::Interrupted) return;
    std::vector<std::pair<std::string, double>> A;
    std::vector<std::pair<std::string, std::string>> S;
    const double tp = tp_eff.at(std::size_t(ae.k));
    if (ae.fired) {
      A = {{"cap_reached", 0.0}, {"tau_end", M.tau_stop}, {"t_end", tp + M.tau_stop}, {"t_run_end", in.t()},
           {"lambda_GR", M.lam_gr}, {"truncation_error", M.err_est}, {"m_at_end", M.last_m()},
           {"fit_slope_half1", M.fit.s_half1}, {"fit_slope_half2", M.fit.s_half2}};
      std::cout << "RTDE end (rtde_tau = auto): tau_end = " << M.tau_stop << " (t = " << tp + M.tau_stop
                << "), lambda_GR = " << M.lam_gr << ", truncation error of A(omega) <= " << M.err_est
                << " (tol_A = " << rc.tol_A << ")\n";
    } else if (st == dp5::Status::Ok) {
      double lam = std::nan(""), err = std::nan("");
      const bool ok = M.estimate(lam, err);
      const double tau_end = M.taus().empty() ? 0.0 : M.taus().back();
      A = {{"cap_reached", 1.0}, {"tau_end", tau_end}, {"t_end", tp + tau_end}, {"t_run_end", in.t()},
           {"lambda_GR", lam}, {"truncation_error", err}, {"m_at_end", M.last_m()}};
      S = {{"warning", std::string("rtde_tau_max reached before the truncation criterion held") +
                           (ok ? "" : "; no decay rate of G^R could be fitted (truncation error unknown)")}};
      std::cout << "RTDE end (rtde_tau = auto): cap rtde_tau_max = " << rc.rtde_tau_max << " reached; lambda_GR = " << lam
                << ", estimated truncation error " << err << " (tol_A = " << rc.tol_A << ")\n";
    }
    out.write_attrs("rtde_end", A, S, {});
  }

  CheckpointMeta meta(const dp5::Integrator& in, const std::vector<long long>& activated) const {
    CheckpointMeta k;
    k.approximation = name(rc.approx); k.spin_mode = name(rc.spin); k.vertex_impl = name(rc.vertex);
    k.model_hash = m.hash_computed; k.N = m.N; k.flag_2B = rc.flag_2B; k.bare_propagators = rc.bare_propagators;
    for (std::size_t i = 0; i < in.num_blocks(); ++i) k.block_names.push_back(in.removed(i) ? std::string("-") : in.spec(i).name);
    k.activated_tp = activated; k.t_prime = tp_eff; k.probe_columns = rc.probe_columns;
    return k;
  }
  // divergence guard: an eigenvalue of rho^sigma outside [diverge_rho_lo, diverge_rho_hi] on an output frame stops the
  // run with status Diverged (exit 7) instead of letting the step size collapse
  bool diverged = false;
  std::string diverge_msg;
  std::function<bool()> interrupt() {
    return [this]() { return diverged; };
  }
  void ckpt(const std::string& group, const std::string& name, const dp5::Checkpoint& cp, const cplx* y, std::size_t n,
            const CheckpointMeta& k) {
    out.checkpoint(group, name, cp, y, n, k);
  }
  void drop_old_periodic(long long j) {  // checkpoint_keep > 0: only the newest periodic checkpoints stay
    if (rc.checkpoint_keep > 0 && j - rc.checkpoint_keep >= 1) out.drop_checkpoint("periodic_" + std::to_string(j - rc.checkpoint_keep));
  }
  int exit_code(dp5::Status st) const {
    return st == dp5::Status::Ok ? 0 : (st == dp5::Status::Interrupted && diverged ? 7 : 5);
  }
  dp5::Options options() const {
    dp5::Options o;
    o.profile = true;
    return o;
  }
  void finish(dp5::Integrator& in, dp5::Status st, const std::vector<long long>& activated) {
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
    // series, status, profile and attributes first; the final state last, then flush
    end_report(in, st);
    std::string msg;
    if (st != dp5::Status::Ok && st != dp5::Status::Interrupted) {
      const auto& d = in.diagnostics();
      msg = "worst block " + in.spec(d.worst_block).name + ", err " + std::to_string(d.err);
    }
    const bool div = diverged && st == dp5::Status::Interrupted;
    if (div) msg = diverge_msg;
    if (st == dp5::Status::Interrupted && !div) msg = "interrupted";
    out.profile(in.profile());
    out.status(div ? std::string("Diverged") : status_string(st), msg, in.t(), wall, in.stats());
    if (div) std::cerr << "dp5drive: " << msg << "\n";
    ckpt("/final", "", in.checkpoint(), in.y(), in.size(), meta(in, activated));
    out.flush();
    std::cout << in.profile().to_text();
    std::cout << "status " << (div ? std::string("Diverged") : status_string(st)) << (msg.empty() ? "" : " (" + msg + ")") << " at t = " << in.t() << " -> " << out.path() << "\n";
  }


  // ------------------------------------------------------------ g1g2 / rtde (co-propagated)
  int run_g1g2(bool with_rtde) {
    Physics ph(m, rc, make_vertex(m, rc.vertex, rc.gamma));
    const int N = m.N, ns = ph.ns;
    Physics* P = &ph;
    dp5::Integrator in(g1g2_blocks(lc), [P](double t, const cplx* y, cplx* dy, const dp5::Layout& L) { (*P)(t, y, dy, L); },
                       options(), rc.t0);
    ph.id_rho = in.block_id("rho");
    std::vector<cplx> r0 = initial_rho(m, rc);
    std::copy(r0.begin(), r0.end(), in.block(ph.id_rho));
    in.state_changed();

    // tensor blocks (g1g2 order after rho); the history-integral blocks are saved so that frames carry
    // what the observables (I, D_r, d rho/dt) need
    {
      auto blocks = g1g2_blocks(lc);
      for (std::size_t b = 1; b < blocks.size(); ++b) ph.tensor_ids.push_back(in.block_id(blocks[b].name));
      if (!ph.tensor_ids.empty()) {
        in.set_save(ph.tensor_ids[0], true);
        if (rc.approx == Approx::mGW) { in.set_save(ph.tensor_ids[1], true); in.set_save(ph.tensor_ids.back(), true); }
        if (rc.approx == Approx::DSL)  // same-spin blocks (packed, small): equal-time 2RDM observables
          for (std::size_t id : ph.tensor_ids) in.set_save(id, true);
      }
    }
    out.init_series(N, ns, m.lossy_sites, rc.rho_stride);
    std::vector<long long> activated;

    // observer: rho series, observables, RTDE frames
    std::vector<cplx> row;
    FrameObservables fo;
    std::vector<const cplx*> Xf;
    // end time: t_end, or with a numeric t' and rtde_tau = auto the cap t' + rtde_tau_max (grid t0 + j dt_out)
    const bool fixed_auto_end = with_rtde && rc.rtde_tau_auto && !rc.tprime_auto;
    const double t_end_eff = fixed_auto_end ? rc.t_prime.at(0) + rc.rtde_tau_max : rc.t_end;
    const std::size_t n_out = fixed_auto_end ? std::size_t(std::floor((t_end_eff - rc.t0) / rc.dt_out + 1e-9)) + 1
                                             : std::size_t(std::llround((rc.t_end - rc.t0) / rc.dt_out)) + 1;
    tp_eff = rc.t_prime;
    std::function<void(const dp5::Frame&)> obs = [&](const dp5::Frame& f) {
      if (ae.fired && f.t > ae.T_cut + dp5::Integrator::land_tol(ae.T_cut)) return;  // after the automatic RTDE end
      const cplx* rho = f[ph.id_rho];
      Xf.clear();
      for (std::size_t id : ph.tensor_ids) Xf.push_back(f[id]);
      ph.observables(rho, Xf, fo);
      out.frame(f.t, rho, fo);
      if (!diverged && (fo.rho_eig_min < rc.diverge_rho_lo || fo.rho_eig_max > rc.diverge_rho_hi || !std::isfinite(fo.rho_eig_min))) {
        diverged = true;
        char b[256];
        std::snprintf(b, sizeof b, "diverged: eigenvalue of rho outside [%g, %g] at t = %.6g (min %.4g, max %.4g)", rc.diverge_rho_lo,
                      rc.diverge_rho_hi, f.t, fo.rho_eig_min, fo.rho_eig_max);
        diverge_msg = b;
      }
      for (long long k : activated) {
        row.clear();
        bool have = true;
        for (const RtdeSet& s : ph.rtde)
          if (s.tp_index == k) {
            const cplx* g = f[s.idGR];
            if (!g) { have = false; break; }
            row.insert(row.end(), g, g + std::size_t(ns) * N);
          }
        if (have) out.rtde_frame(int(k), f.t, row.data());
        if (have && ae.mon && int(k) == ae.k && !ae.fired &&
            ae.mon->feed(f.t - tp_eff.at(std::size_t(k)), monitor_value(row.data(), ns))) {
          ae.fired = true;  // all activations end at the same tau (the last t' + tau_stop), within the cap
          ae.T_cut = std::min(T_target, tp_eff.back() + ae.mon->tau_stop);
        }
      }
    };
    if (fixed_auto_end) in.set_output_grid_dt(rc.t0, rc.dt_out, n_out, obs);  // frame times independent of the cap
    else in.set_output_grid(rc.t0, rc.t_end, n_out, obs);

    auto activate = [&](dp5::Integrator& it, int k) {
      // all columns' blocks in ONE append (one register reallocation; same order, ids and values as
      // appending block by block -- README §4.9)
      std::vector<dp5::BlockSpec> all;
      std::vector<std::size_t> nper;
      for (long long s : rc.probe_columns) {
        if (s < 0 || s >= N) throw std::runtime_error("probe column out of range");
        auto blocks = rtde_blocks(lc, k, s);
        nper.push_back(blocks.size());
        for (auto& b : blocks) all.push_back(b);
      }
      const std::vector<std::size_t> ids = it.append_blocks(std::move(all));
      std::size_t pos = 0;
      for (std::size_t c = 0; c < rc.probe_columns.size(); ++c) {
        RtdeSet set; set.tp_index = k; set.column = rc.probe_columns[c];
        for (std::size_t b = 0; b < nper[c]; ++b, ++pos) {
          if (b == 0) set.idGR = ids[pos]; else set.ids.push_back(ids[pos]);
        }
        ph.rtde.push_back(set);
      }
      activated.push_back(k);
    };
    auto set_gr_initial = [&](dp5::Integrator& it, int k) {
      for (const RtdeSet& s : ph.rtde)
        if (s.tp_index == k) {
          cplx* g = it.block(s.idGR);
          for (int sp = 0; sp < ns; ++sp) g[std::size_t(sp) * N + std::size_t(s.column)] = cplx(0, -1);  // G^R(t'+,t') = -i e_s
        }
    };

    const double t_start = rc.t0;
    const double tol = dp5::Integrator::land_tol(t_start);
    if (rc.tprime_auto)
      return run_auto(in, ph, activated, activate, set_gr_initial, obs, t_start);
    if (with_rtde)
      for (std::size_t k = 0; k < rc.t_prime.size(); ++k) {
        const int kk = int(k);
        in.add_stop(rc.t_prime[k], [&, kk](dp5::Integrator& it) {
          ckpt("/checkpoints", "tp" + std::to_string(kk), it.checkpoint(), it.y(), it.size(), meta(it, activated));
          activate(it, kk);
          set_gr_initial(it, kk);
          out.init_rtde(kk, rc.t_prime[std::size_t(kk)], rc.probe_columns, N, ns);
          if (kk == 0 && rc.rtde_tau_auto) {
            const Physics::RefRates r = ph.ref_rates(it.block(ph.id_rho));
            setup_monitor(0, rc.t_prime[0], r.lam_mf, r.spread);
          }
        });
      }
    if (rc.checkpoint_interval > 0)
      for (long long j = 1;; ++j) {
        const double tc = rc.t0 + double(j) * rc.checkpoint_interval;
        if (tc >= t_end_eff - tol) break;
        if (tc <= t_start + tol) continue;
        in.add_stop(tc, [&, j](dp5::Integrator& it) {
          ckpt("/checkpoints", "periodic_" + std::to_string(j), it.checkpoint(), it.y(), it.size(), meta(it, activated));
          drop_old_periodic(j);
        });
      }
    const dp5::Status st = drive(in, t_end_eff, interrupt());
    finish(in, st, activated);
    return exit_code(st);
  }

  // ------------------------------------------------------------ t_prime = auto (steady-state detection)
  // Phase 1: G1-G2 only, until the detector triggers (or the timeout c_max/lambda_ref). Phase 2: the RTDE
  // columns are opened at the detected t' and everything runs to t' + rtde_tau on the grid t' + j dt_out.
  int run_auto(dp5::Integrator& in, Physics& ph, std::vector<long long>& activated,
               const std::function<void(dp5::Integrator&, int)>& activate,
               const std::function<void(dp5::Integrator&, int)>& set_gr_initial,
               const std::function<void(const dp5::Frame&)>& obs, double t_start) {
    const int N = m.N, ns = ph.ns;
    const double dt = rc.dt_out;
    const double tau_run = rc.rtde_tau_auto ? rc.rtde_tau_max : rc.rtde_tau;  // RTDE duration (cap with auto)
    std::vector<std::size_t> gids{ph.id_rho};
    for (std::size_t id : ph.tensor_ids) gids.push_back(id);
    auto block_norms = [&](const cplx* f) {
      const dp5::Layout& L = in.layout();
      std::vector<double> nb;
      for (std::size_t id : gids) {
        const cplx* p = f + L.offset[id];
        const std::ptrdiff_t n = std::ptrdiff_t(L.size[id]);
        double mx = 0;
#pragma omp parallel for reduction(max : mx) schedule(static)
        for (std::ptrdiff_t i = 0; i < n; ++i) mx = std::max(mx, std::abs(p[i]));
        nb.push_back(mx);
      }
      return nb;
    };
    std::vector<std::string> gnames;
    for (std::size_t id : gids) gnames.push_back(in.spec(id).name);
    const Physics::RefRates r0 = ph.ref_rates(in.block(ph.id_rho));
    if (!(r0.lam_mf > 0) || !std::isfinite(r0.lam_mf))
      throw std::runtime_error("t_prime = auto: lambda_ref = min|Im eig h_o^MF| = 0 at t0 (a non-decaying one-body mode); "
                               "no steady state can be detected");
    auto tprime_activation = [&](dp5::Integrator& it, int k) {
      ckpt("/checkpoints", "tp" + std::to_string(k), it.checkpoint(), it.y(), it.size(), meta(it, activated));
      activate(it, k);
      set_gr_initial(it, k);
      it.state_changed();
      out.init_rtde(k, tp_eff.at(std::size_t(k)), rc.probe_columns, N, ns);
    };
    auto add_periodic = [&](double t_from, double t_to) {
      if (rc.checkpoint_interval <= 0) return;
      const double tol = dp5::Integrator::land_tol(t_to);
      for (long long j = 1;; ++j) {
        const double tc = rc.t0 + double(j) * rc.checkpoint_interval;
        if (tc >= t_to - tol) break;
        if (tc <= t_from + tol) continue;
        in.add_stop(tc, [&, j](dp5::Integrator& it) {
          ckpt("/checkpoints", "periodic_" + std::to_string(j), it.checkpoint(), it.y(), it.size(), meta(it, activated));
          drop_old_periodic(j);
        });
      }
    };
    const auto base = interrupt();
    std::vector<std::pair<std::string, double>> A;
    std::vector<std::pair<std::string, std::string>> S;
    std::vector<std::pair<std::string, std::vector<double>>> Vv;
    A.push_back({"tol_ss", rc.tol_ss}); A.push_back({"c_hold", rc.c_hold}); A.push_back({"c_min", rc.c_min});
    A.push_back({"c_max", rc.c_max}); A.push_back({"ss_fit_rtol", rc.ss_fit_rtol}); A.push_back({"rtde_tau", rc.rtde_tau});
    A.push_back({"rtde_tau_auto", rc.rtde_tau_auto ? 1.0 : 0.0});
    S.push_back({"ss_on_timeout", rc.ss_on_timeout});
    S.push_back({"blocks", [&] { std::string o; for (auto& n : gnames) o += (o.empty() ? "" : ",") + n; return o; }()});
    A.push_back({"lambda_ref_t0", r0.lam_mf}); A.push_back({"lambda_ref_bare_t0", r0.lam_bare});
    A.push_back({"re_spread_t0", r0.spread});

    double tp = 0;
    bool timeout = false;
    {  // phase 1
      SsConfig sc; sc.tol_ss = rc.tol_ss; sc.c_hold = rc.c_hold; sc.c_min = rc.c_min; sc.c_max = rc.c_max; sc.fit_rtol = rc.ss_fit_rtol;
      sc.c_delta = rc.ss_window_c; sc.hold_records = rc.ss_hold_records;
      std::vector<std::size_t> bsz;
      for (std::size_t id : gids) bsz.push_back(in.spec(id).size);
      const double lam_fast = 2.0 * r0.lam_max;  // pair blocks relax up to twice the fastest one-body rate
      SteadyStateDetector det(sc, t_start, r0.lam_mf, lam_fast, bsz);
      A.push_back({"t_min", det.t_min()}); A.push_back({"t_max", det.t_max()}); A.push_back({"t_detection_start", t_start});
      A.push_back({"lambda_fast_t0", lam_fast}); A.push_back({"delta_initial", det.delta_initial()});
      A.push_back({"ss_window_c", rc.ss_window_c}); A.push_back({"ss_hold_records", rc.ss_hold_records ? 1.0 : 0.0});
      std::vector<const cplx*> bptr(gids.size());
      auto drift_history = [&]() {
        std::vector<double> ht, hD, hN, hd;
        for (const auto& q : det.records()) { ht.push_back(q.t); hD.push_back(q.D); hN.push_back(q.N); hd.push_back(q.delta); }
        Vv.push_back({"drift_t", ht}); Vv.push_back({"drift_D", hD}); Vv.push_back({"drift_noise", hN}); Vv.push_back({"drift_delta", hd});
      };
      // grid t0 + j dt up to the timeout (rounded up)
      const double j0 = std::ceil((t_start - rc.t0) / dt - 1e-9), j1 = std::ceil((det.t_max() - rc.t0) / dt - 1e-9);
      const double t_cap = rc.t0 + j1 * dt;
      in.set_output_grid_dt(rc.t0, dt, std::size_t(j1) + 1, obs, std::size_t(j0));  // t0 + j dt, j >= j0
      add_periodic(t_start, t_cap);
      // state at the end of the previous accepted step: on a trigger the last step is redone as a landing on a
      // stop at t' (the arithmetic of a fixed-t' run, so that both give bit-identical results)
      dp5::Checkpoint p1_cp = in.checkpoint();
      std::vector<cplx> p1_y(in.y(), in.y() + in.size());
      int why = 0;
      double det_seconds = 0;  // wall time spent in the detector (accumulation, window closes, lambda_ref, state copy)
      in.set_interrupt([&]() {
        // the detector sees every accepted step
        const auto c0 = std::chrono::steady_clock::now();
        for (std::size_t b = 0; b < gids.size(); ++b) bptr[b] = in.block(gids[b]);
        const bool trig = det.step(in.t(), bptr, [&] { return ph.ref_rates(in.block(ph.id_rho)).lam_mf; });
        if (!trig) {
          p1_cp = in.checkpoint();
          p1_y.assign(in.y(), in.y() + in.size());
        }
        det_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
        if (trig) { why = 2; return true; }
        if (base()) { why = 1; return true; }
        return false;
      });
      dp5::Status st;
      try {
        st = in.integrate(t_cap);
      } catch (...) {
        out.profile(in.profile());
        throw;
      }
      if (st == dp5::Status::Interrupted && why == 1) {  // diverged before t': detection progress so far
        drift_history();
        A.push_back({"interrupted_before_tprime", 1.0}); A.push_back({"t_interrupt", in.t()});
        A.push_back({"delta_now", det.delta_now()}); A.push_back({"candidates", double(det.candidates)});
        out.write_attrs("steady_state", A, S, Vv);
        finish(in, st, activated);
        return exit_code(st);
      }
      if (st != dp5::Status::Ok && st != dp5::Status::Interrupted) { finish(in, st, activated); return 5; }
      tp = in.t();
      Physics::RefRates rt = ph.ref_rates(in.block(ph.id_rho));
      A.push_back({"detector_seconds", det_seconds});
      A.push_back({"seconds_to_tprime", std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count()});
      A.push_back({"phase1_accepted_steps", double(in.stats().n_accept)});
      A.push_back({"hold_record_extensions", double(det.hold_record_extensions)});
      if (st == dp5::Status::Ok) {  // reached the timeout without a trigger
        timeout = true;
        const std::vector<double> nb = block_norms(in.slope());
        A.push_back({"timeout", 1.0}); A.push_back({"t_timeout", tp}); A.push_back({"lambda_ref_t", rt.lam_mf});
        Vv.push_back({"slope_norms_at_timeout", nb});
        A.push_back({"candidates", double(det.candidates)});
        A.push_back({"delta_final", det.delta_now()}); A.push_back({"delta_doublings", double(det.delta_doublings)});
        const double lastD = det.records().empty() ? 0.0 : det.records().back().D;
        const double lastN = det.records().empty() ? 0.0 : det.records().back().N;
        A.push_back({"drift_at_timeout", lastD}); A.push_back({"noise_at_timeout", lastN});
        drift_history();
        if (rc.ss_on_timeout == "fail") {
          out.write_attrs("steady_state", A, S, Vv);
          auto g = [](double x) { char b[32]; std::snprintf(b, sizeof b, "%.3g", x); return std::string(b); };
          const std::string msg = "no stationary state found by t = c_max/lambda_ref = " + g(tp) +
                                  " (windowed drift " + g(lastD) + ", noise " + g(lastN) +
                                  ", tol_ss*lambda_ref = " + g(rc.tol_ss * rt.lam_mf) +
                                  "); the closure may be oscillating or unstable";
          ckpt("/final", "", in.checkpoint(), in.y(), in.size(), meta(in, activated));
          out.profile(in.profile());
          const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
          out.status("NoSteadyState", msg, in.t(), wall, in.stats());
          std::cerr << "dp5drive: " << msg << "\n";
          return 6;
        }
        S.push_back({"warning", "t' set at the timeout without a detected steady state (ss_on_timeout = start)"});
        std::cout << "steady state NOT detected by t = " << tp << "; RTDE opened anyway (ss_on_timeout = start)\n";
      } else {
        A.push_back({"timeout", 0.0});
        A.push_back({"t_prime", tp}); A.push_back({"lambda_ref_t", det.lam_ref_t}); A.push_back({"lambda_fit", det.fit.lam_fit});
        A.push_back({"lambda_used", det.lam_used}); A.push_back({"max_ratio_window", det.max_ratio});
        A.push_back({"fit_slope_half1", det.fit.s_half1}); A.push_back({"fit_slope_half2", det.fit.s_half2});
        A.push_back({"candidates", double(det.candidates)});
        A.push_back({"hold_window", det.t_hold}); A.push_back({"hold_extended", det.hold_extended ? 1.0 : 0.0});
        A.push_back({"delta", det.delta_used}); A.push_back({"delta_doublings", double(det.delta_doublings)});
        A.push_back({"floor_trigger", det.floor_trigger ? 1.0 : 0.0});
        if (!det.marginal_blocks.empty()) {
          std::string mb;
          for (int b : det.marginal_blocks) mb += (mb.empty() ? "" : ",") + gnames[std::size_t(b)];
          S.push_back({"marginal_blocks", mb});
          Vv.push_back({"marginal_drift", det.marginal_drift});
          S.push_back({"marginal_note", "blocks excluded from the criterion: drift flat within 20 % over two hold windows and at least 10x "
                                        "tol_ss*lambda_ref (undamped persistent oscillation or linear drift; rho is never excluded)"});
          std::cout << "note: blocks with a flat, undamped drift excluded from the criterion: " << mb << " (max drift "
                    << *std::max_element(det.marginal_drift.begin(), det.marginal_drift.end()) << ")\n";
        }
        if (det.floor_trigger) S.push_back({"floor_trigger_note", "drift flat at its floor below tol_ss*lambda_ref (" + det.fit.why +
                                                                  "); lambda_used = lambda_ref, lambda_fit not available"});
        Vv.push_back({"drift_at_tprime", det.norms_at_trigger}); Vv.push_back({"noise_at_tprime", det.noise_at_trigger});
        drift_history();
        const double nmax = *std::max_element(det.norms_at_trigger.begin(), det.norms_at_trigger.end());
        std::cout << "steady state detected: t' = " << tp << ", drift = " << nmax << " (window Delta = " << det.delta_used
                  << (det.hold_extended ? ", hold window extended to " + std::to_string(det.t_hold) : std::string()) << ")"
                  << ", lambda_used = " << det.lam_used << " (lambda_fit = " << det.fit.lam_fit << ", lambda_ref = " << det.lam_ref_t << ")"
                  << (det.floor_trigger ? " [drift floor: no fitted rate, lambda_used = lambda_ref]" : "") << "\n";
      }
      // opening of the RTDE at t' (in a stop callback after the redone landing step, or directly at the timeout)
      auto open_rtde = [&](dp5::Integrator& it) {
        tp_eff = {tp};
        tprime_activation(it, 0);
        if (rc.rtde_tau_auto) {
          const Physics::RefRates r = ph.ref_rates(it.block(ph.id_rho));
          setup_monitor(0, tp, r.lam_mf, r.spread);
        }
        out.write_attrs("steady_state", A, S, Vv);
      };
      if (timeout) {
        open_rtde(in);
        in.set_output_grid_dt(tp, dt, std::size_t(std::llround(tau_run / dt)) + 1, obs);
      } else {
        in.restore(p1_cp, p1_y.data());  // back to the start of the last step; frames up to there are written
        in.set_output_grid_dt(tp, dt, std::size_t(std::llround(tau_run / dt)) + 1, obs);
        in.add_stop(tp, open_rtde);
      }
      add_periodic(std::max(tp, t_cap), tp + tau_run);  // phase-1 stops already cover (t0, t_cap)
    }
    const dp5::Status st = drive(in, tp + tau_run, base);
    finish(in, st, activated);
    return exit_code(st);
  }

};

}  // namespace

int main(int argc, char** argv) {
  H5::Exception::dontPrint();  // errors are reported as one-line messages, not HDF5 error stacks
  Cli cli;
  RunCard rc;
  try {
    cli = parse_cli(argc, argv);
    if (cli.version) {
      std::cout << version_text();
      return 0;
    }
    rc = load_runcard(cli.runcard);
    if (cli.overrides.count("threads")) rc.threads = std::stoi(cli.overrides["threads"]);
    rc.output = cli.output;
    validate_auto(rc);
  } catch (const std::exception& e) {
    std::cerr << "dp5drive: " << e.what() << "\n";
    return 2;
  }
  // model path relative to the run card's directory
  fs::path mp(rc.model_path);
  if (mp.is_relative()) mp = fs::path(cli.runcard).parent_path() / mp;
  Model m;
  try {
    // every input file is checked up front (exists, readable, HDF5), so a bad path fails before any work
    h5::check_readable_hdf5(mp.string(), "model file (run card model_path)");
    if (!cli.eval_in.empty()) h5::check_readable_hdf5(cli.eval_in, "eval-rhs input");
    m = load_model(mp.string());
    if (rc.model_hash != m.hash_computed)
      throw std::runtime_error("run card model_hash " + rc.model_hash + " != model content hash " + m.hash_computed);
  } catch (const std::exception& e) {
    std::cerr << "dp5drive: " << e.what() << "\n";
    return 2;
  }
  if (cli.print_layout) {
    std::cout << "method " << method_label(rc.approx, rc.flag_2B, rc.bare_propagators) << "\n";
    std::cout << describe_layout(layout_config(rc, m.N));
    return 0;
  }
  if (rc.threads > 0) omp_set_num_threads(rc.threads);
  if (!cli.eval_in.empty()) {  // test hook: dy = RHS(t, y) in the G1-G2 layout, plus I and observables
    try {
      LayoutConfig lc = layout_config(rc, m.N);
      Physics ph(m, rc, make_vertex(m, rc.vertex, rc.gamma));
      Physics* P = &ph;
      auto blocks = g1g2_blocks(lc);
      const std::size_t ng = blocks.size();
      if (rc.mode == Mode::rtde)  // G1-G2 blocks followed by one RTDE block set per probe column
        for (long long s : rc.probe_columns) {
          RtdeSet set; set.column = s; set.idGR = blocks.size();
          auto rb = rtde_blocks(lc, 0, s);
          for (std::size_t b = 1; b < rb.size(); ++b) set.ids.push_back(blocks.size() + b);
          for (auto& b : rb) blocks.push_back(b);
          ph.rtde.push_back(set);
        }
      dp5::Integrator in(blocks, [P](double t, const cplx* y, cplx* dy, const dp5::Layout& L) { (*P)(t, y, dy, L); },
                         dp5::Options{}, rc.t0);
      ph.id_rho = 0;
      for (std::size_t b = 1; b < ng; ++b) ph.tensor_ids.push_back(b);
      H5::H5File fi = h5::open_ro(cli.eval_in, "eval-rhs input");
      std::vector<cplx> y = h5::read_cplx(fi.openGroup("/"), "y");
      if (y.size() != in.size()) throw std::runtime_error("eval-rhs: state size " + std::to_string(y.size()) + " != layout " + std::to_string(in.size()));
      std::vector<cplx> dy(y.size());
      ph(rc.t0, y.data(), dy.data(), in.layout());
      std::vector<const cplx*> X;
      for (std::size_t id : ph.tensor_ids) X.push_back(y.data() + in.offset(id));
      FrameObservables fo;
      ph.observables(y.data(), X, fo);
      H5::H5File fo5(cli.eval_out, H5F_ACC_TRUNC);
      H5::Group g = fo5.openGroup("/");
      h5::write_cplx(g, "dy", dy.data(), {hsize_t(dy.size())});
      h5::write_dbl(g, "D", fo.D.data(), {hsize_t(fo.D.size())});
      h5::attr_dbl(g, "loss_rate", fo.loss_rate);
      h5::attr_dbl(g, "dN_dt", fo.dN_dt);
      h5::attr_dbl(g, "energy", fo.energy);
      {
        H5::Group ge = g.createGroup("equal_time");
        for (const auto& se : fo.eq.series) {
          std::vector<hsize_t> d(se.dims.begin(), se.dims.end());
          if (d.empty()) d.push_back(1);
          h5::write_dbl(ge, se.name, se.v.data(), d);
        }
        h5::write_cplx(g, "I", ph.history_I(), {hsize_t(ph.ns), hsize_t(ph.N), hsize_t(ph.N)});
      }
      return 0;
    } catch (const std::exception& e) {
      std::cerr << "dp5drive: " << e.what() << "\n";
      return 2;
    }
  }
  const std::string run_id = make_run_id();
  const std::string path = rc.output;
  std::unique_ptr<Output> out;
  try {
    out = std::make_unique<Output>(path, run_id);  // fail fast, before any physics
    out->write_input(rc, m, cli.overrides, cli.cmdline);
    out->write_provenance(omp_get_max_threads());
  } catch (const std::exception& e) {
    std::cerr << "dp5drive: cannot create output " << path << ": " << e.what() << "\n";
    return 2;
  } catch (const H5::Exception& e) {
    std::cerr << "dp5drive: cannot create output " << path << ": " << e.getDetailMsg() << "\n";
    return 2;
  }
  Run run{rc, m, *out, layout_config(rc, m.N)};
  try {
    return run.run_g1g2(rc.mode == Mode::rtde);
  } catch (const NotImplemented& e) {
    out->status("NotImplemented", e.what(), rc.t0, 0.0, dp5::Stats{});
    std::cerr << "dp5drive: " << e.what() << "\n";
    return 3;
  } catch (const std::exception& e) {
    out->status("Error", e.what(), rc.t0, 0.0, dp5::Stats{});
    std::cerr << "dp5drive: " << e.what() << "\n";
    return 2;
  } catch (const H5::Exception& e) {
    std::cerr << "dp5drive: HDF5 error: " << e.getDetailMsg() << "\n";
    return 2;
  }
}
