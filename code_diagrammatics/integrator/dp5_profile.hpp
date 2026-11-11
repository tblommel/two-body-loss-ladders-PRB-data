// dp5_profile.hpp -- profiling data collected by dp5::Integrator (see DESIGN.md sec. 10).
//
// Plain data + text summary + flat export (no HDF5 dependency). Included by
// dp5_integrator.hpp. Compile-time switch: -DDP5_PROFILE=0 removes all timers
// (the struct still exists, counters stay zero); runtime: Options::profile.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <vector>

#ifndef DP5_PROFILE
#define DP5_PROFILE 1
#endif

namespace dp5 {

// Timed sections. Times are EXCLUSIVE (a section nested in another, e.g. a
// layout change inside a stop callback, is not counted twice), so all section
// times plus `untimed_seconds` add up to `integrate_seconds` (inside integrate).
enum Sec : int {
  SecRhsSlope = 0,   // RHS for k1 = f(t, y): start of a run and after every stop (FSAL refresh)
  SecRhsInitStep,    // RHS inside the automatic initial-step estimate
  SecRhsK2, SecRhsK3, SecRhsK4, SecRhsK5, SecRhsK6, SecRhsK7,
  SecPassStage2, SecPassStage3, SecPassStage4, SecPassStage5, SecPassStage6,  // stage inputs
  SecPassSolution,   // y5 = y + h sum b_i k_i (stage-7 input)
  SecPassError,      // fused error norm (+ non-finite check)
  SecPassInitStep,   // vector work of the initial-step estimate (norms, y0 + h0 f0)
  SecDense,          // dense-output construction for grid frames
  SecController,     // step-size control, acceptance bookkeeping, register rotation
  SecStopCallback,   // user stop callbacks          (user time)
  SecObserver,       // user output observer         (user time)
  SecStepObserver,   // user step observer           (user time)
  SecInterrupt,      // user interrupt predicate     (user time)
  SecLayout,         // append/remove/extract/set_active: reallocation + first touch + sync
  SecCheckpoint,     // checkpoint()
  SecRestore,        // restore()
  SecCount
};

inline const char* sec_name(int s) {
  static const char* n[SecCount] = {"rhs_slope",    "rhs_init_step", "rhs_k2",       "rhs_k3",       "rhs_k4",
                                    "rhs_k5",       "rhs_k6",        "rhs_k7",       "pass_stage2",  "pass_stage3",
                                    "pass_stage4",  "pass_stage5",   "pass_stage6",  "pass_solution", "pass_error",
                                    "pass_init_step", "dense",       "controller",   "stop_callback", "observer",
                                    "step_observer", "interrupt",    "layout",       "checkpoint",   "restore"};
  return (s >= 0 && s < SecCount) ? n[s] : "?";
}
inline bool sec_is_rhs(int s) { return s >= SecRhsSlope && s <= SecRhsK7; }
inline bool sec_is_user(int s) { return s >= SecStopCallback && s <= SecInterrupt; }
inline bool sec_is_pass(int s) { return s >= SecPassStage2 && s <= SecDense; }

struct TimerStat {
  double seconds = 0.0;          // exclusive wall time inside integrate()
  double seconds_outside = 0.0;  // exclusive wall time outside integrate() (e.g. layout set-up)
  std::uint64_t calls = 0;
  double imb_max = 0.0, imb_mean = 0.0;  // sums over calls of max/mean per-thread time (profile_threads)
  void add(const TimerStat& o) {
    seconds += o.seconds; seconds_outside += o.seconds_outside; calls += o.calls;
    imb_max += o.imb_max; imb_mean += o.imb_mean;
  }
};

// Statistics of one segment: the time between consecutive stops (segment 0
// starts at construction; a new segment starts after each stop's callbacks).
struct SegmentProfile {
  double t_begin = 0.0, t_end = 0.0;
  double integrate_seconds = 0.0;
  TimerStat sec[SecCount];
  std::uint64_t accepted = 0, rejected_error = 0, rejected_nonfinite = 0, clipped = 0, rhs_calls = 0;
  double h_min = std::numeric_limits<double>::infinity(), h_max = 0.0, log_h_sum = 0.0;
  std::size_t state_size = 0;  // state size (complex entries) during the segment
  // split of the segment's integrate time (same definitions as the totals)
  double rhs_seconds() const { double x = 0; for (int i = 0; i < SecCount; ++i) if (sec_is_rhs(i)) x += sec[i].seconds; return x; }
  double user_seconds() const { double x = 0; for (int i = 0; i < SecCount; ++i) if (sec_is_user(i)) x += sec[i].seconds; return x; }
  double pass_seconds() const { double x = 0; for (int i = 0; i < SecCount; ++i) if (sec_is_pass(i)) x += sec[i].seconds; return x; }
  double overhead_seconds() const { return integrate_seconds - rhs_seconds() - user_seconds(); }
};

// Return status names (Status enum values; -1 = exception thrown from the RHS or a callback).
inline const char* status_name(int s) {
  switch (s) {
    case 0: return "Ok";
    case 1: return "Interrupted";
    case 2: return "NonFinite";
    case 3: return "StepTooSmall";
    case 4: return "MaxSteps";
    case -1: return "Exception";
  }
  return "?";
}

struct BlockInfo {
  std::string name;
  std::size_t size = 0, offset = 0, rms_count = 0;
  int norm = 0;  // 0 RMS, 1 Max
  double atol = 0, rtol = 0, weight = 0;
  bool active = false, removed = false, save = false;
};

struct Profile {
  // Version of the flat export schema (visit/to_map keys). Keys are only ever
  // added; a rename or a change of meaning bumps this number (DESIGN.md 10.5).
  static constexpr int schema_version = 1;
  bool enabled = false;
  // ---- context
  int threads = 1;
  std::size_t par_threshold = 0;
  int controller = 0, combine = 0;  // enum values of Options
  std::string compiler, build_flags;
  std::vector<BlockInfo> blocks;
  std::size_t state_size = 0;
  // ---- timers
  TimerStat sec[SecCount];
  double integrate_seconds = 0.0;  // total wall time inside integrate()
  double untimed_seconds = 0.0;    // integrate_seconds - sum of exclusive sections (bookkeeping, loops)
  std::uint64_t integrate_calls = 0;
  // ---- steps
  std::uint64_t attempts = 0, accepted = 0, rejected_error = 0, rejected_nonfinite = 0;
  std::uint64_t clipped = 0;  // accepted steps shortened to land exactly on a stop or t_end
  std::uint64_t rhs_calls = 0, slope_recomputes = 0, init_step_estimates = 0;
  std::uint64_t frames_dense = 0, frames_exact = 0, stops = 0;
  std::vector<double> h0_auto_t, h0_auto_h;  // automatic initial steps (time, value)
  double h_min = std::numeric_limits<double>::infinity(), h_max = 0.0, log_h_sum = 0.0;
  // log10(h) histogram of accepted steps: bins of width 1/h_bins_per_decade from h_hist_lo
  static constexpr double h_hist_lo = -12.0;
  static constexpr int h_bins_per_decade = 10, h_hist_bins = 170;
  std::vector<std::uint64_t> h_hist = std::vector<std::uint64_t>(h_hist_bins, 0);
  // log10(err) histogram of accepted steps (err <= 1): bin 0 collects err < 1e-8
  static constexpr double err_hist_lo = -8.0;
  static constexpr int err_bins_per_decade = 5, err_hist_bins = 41;
  std::vector<std::uint64_t> err_hist = std::vector<std::uint64_t>(err_hist_bins, 0);
  double err_sum = 0.0, err_max = 0.0;
  std::vector<std::uint64_t> limiting_block;  // per block id: accepted steps where it was the worst
  std::vector<SegmentProfile> segments;
  // ---- optional trace of attempted steps (Options::profile_trace_max > 0), decimated when full
  std::size_t trace_stride = 1;
  std::vector<double> trace_t, trace_h, trace_err, trace_worst, trace_accepted;
  // ---- memory (bytes)
  std::size_t register_bytes = 0;               // current 8 registers x state size x 16 B
  std::size_t register_bytes_peak = 0;          // max over the profiled period
  std::size_t layout_transient_peak_bytes = 0;  // max during append/remove (old y + 8 new registers)
  std::size_t dense_bytes = 0;                  // dense-output buffer (saved blocks)
  long long maxrss_bytes = -1;                  // process peak RSS at the end of the last integrate() (getrusage)
  std::vector<double> layout_t, layout_register_bytes;  // history of layout changes
  // ---- end state and failures
  double t_last = 0.0, h_last = 0.0, err_old_last = 0.0;
  int last_status = 0;                          // status_name(); -1 = exception
  std::uint64_t exceptions = 0;
  std::string last_exception;
  double last_err = 0.0;                        // diagnostics() of the last attempted step
  int last_worst_block = -1;
  std::vector<double> last_block_err;

  // ---- derived quantities
  double sum_seconds(bool (*pred)(int)) const {
    double s = 0;
    for (int i = 0; i < SecCount; ++i) if (pred(i)) s += sec[i].seconds;
    return s;
  }
  double rhs_seconds() const { return sum_seconds(sec_is_rhs); }
  double user_seconds() const { return sum_seconds(sec_is_user); }
  double pass_seconds() const { return sum_seconds(sec_is_pass); }
  // integrator overhead = total - RHS - user callbacks (passes, controller, layout, bookkeeping)
  double overhead_seconds() const { return integrate_seconds - rhs_seconds() - user_seconds(); }
  double timed_seconds() const {
    double s = 0;
    for (int i = 0; i < SecCount; ++i) s += sec[i].seconds;
    return s;
  }
  static double hist_median(const std::vector<std::uint64_t>& hist, double lo, int per_decade) {
    std::uint64_t n = 0;
    for (auto c : hist) n += c;
    if (n == 0) return 0.0;
    std::uint64_t acc = 0;
    for (std::size_t b = 0; b < hist.size(); ++b) {
      acc += hist[b];
      if (2 * acc >= n) return std::pow(10.0, lo + (double(b) + 0.5) / per_decade);
    }
    return 0.0;
  }
  double h_median() const { return hist_median(h_hist, h_hist_lo, h_bins_per_decade); }  // bin-center estimate
  double h_geomean() const { return accepted ? std::exp(log_h_sum / double(accepted)) : 0.0; }

  // ---- flat numeric export: visit(name, data, n) for every field; strings separately
  void visit(const std::function<void(const std::string&, const double*, std::size_t)>& f) const {
    auto s1 = [&](const std::string& n, double v) { f(n, &v, 1); };
    auto vec = [&](const std::string& n, const std::vector<double>& v) { f(n, v.data(), v.size()); };
    auto uvec = [&](const std::string& n, const std::vector<std::uint64_t>& v) {
      std::vector<double> d(v.begin(), v.end()); f(n, d.data(), d.size());
    };
    s1("profile/schema_version", schema_version);
    s1("context/enabled", enabled); s1("context/threads", threads); s1("context/par_threshold", double(par_threshold));
    s1("context/controller", controller); s1("context/combine", combine); s1("context/state_size", double(state_size));
    {
      std::vector<double> sz, off, rc, nm, at, rt, w, ac, rm, sv;
      for (auto& b : blocks) {
        sz.push_back(double(b.size)); off.push_back(double(b.offset)); rc.push_back(double(b.rms_count));
        nm.push_back(b.norm); at.push_back(b.atol); rt.push_back(b.rtol); w.push_back(b.weight);
        ac.push_back(b.active); rm.push_back(b.removed); sv.push_back(b.save);
      }
      vec("blocks/size", sz); vec("blocks/offset", off); vec("blocks/rms_count", rc); vec("blocks/norm", nm);
      vec("blocks/atol", at); vec("blocks/rtol", rt); vec("blocks/weight", w); vec("blocks/active", ac);
      vec("blocks/removed", rm); vec("blocks/save", sv);
    }
    for (int i = 0; i < SecCount; ++i) {
      const std::string p = std::string("time/") + sec_name(i) + "/";
      s1(p + "seconds", sec[i].seconds); s1(p + "seconds_outside", sec[i].seconds_outside);
      s1(p + "calls", double(sec[i].calls)); s1(p + "imb_max", sec[i].imb_max); s1(p + "imb_mean", sec[i].imb_mean);
    }
    s1("time/integrate_seconds", integrate_seconds); s1("time/untimed_seconds", untimed_seconds);
    s1("time/integrate_calls", double(integrate_calls)); s1("time/rhs_seconds", rhs_seconds());
    s1("time/user_seconds", user_seconds()); s1("time/overhead_seconds", overhead_seconds());
    s1("steps/attempts", double(attempts)); s1("steps/accepted", double(accepted));
    s1("steps/rejected_error", double(rejected_error)); s1("steps/rejected_nonfinite", double(rejected_nonfinite));
    s1("steps/clipped", double(clipped)); s1("steps/rhs_calls", double(rhs_calls));
    s1("steps/slope_recomputes", double(slope_recomputes)); s1("steps/init_step_estimates", double(init_step_estimates));
    s1("steps/frames_dense", double(frames_dense)); s1("steps/frames_exact", double(frames_exact));
    s1("steps/stops", double(stops));
    vec("steps/h0_auto_t", h0_auto_t); vec("steps/h0_auto_h", h0_auto_h);
    s1("steps/h_min", accepted ? h_min : 0.0); s1("steps/h_max", h_max); s1("steps/h_median", h_median());
    s1("steps/h_geomean", h_geomean());
    uvec("steps/h_hist", h_hist); s1("steps/h_hist_lo_log10", h_hist_lo); s1("steps/h_bins_per_decade", h_bins_per_decade);
    uvec("steps/err_hist", err_hist); s1("steps/err_hist_lo_log10", err_hist_lo);
    s1("steps/err_bins_per_decade", err_bins_per_decade);
    s1("steps/err_mean", accepted ? err_sum / double(accepted) : 0.0); s1("steps/err_max", err_max);
    uvec("steps/limiting_block", limiting_block);
    {
      const std::size_t ns = segments.size();
      std::vector<double> tb(ns), te(ns), is(ns), acc(ns), rej(ns), nf(ns), cl(ns), rc(ns), hmin(ns), hmax(ns), hg(ns), ss(ns);
      for (std::size_t k = 0; k < ns; ++k) {
        const auto& g = segments[k];
        tb[k] = g.t_begin; te[k] = g.t_end; is[k] = g.integrate_seconds; acc[k] = double(g.accepted);
        rej[k] = double(g.rejected_error); nf[k] = double(g.rejected_nonfinite); cl[k] = double(g.clipped);
        rc[k] = double(g.rhs_calls); hmin[k] = g.accepted ? g.h_min : 0.0; hmax[k] = g.h_max;
        hg[k] = g.accepted ? std::exp(g.log_h_sum / double(g.accepted)) : 0.0; ss[k] = double(g.state_size);
      }
      vec("segments/t_begin", tb); vec("segments/t_end", te); vec("segments/integrate_seconds", is);
      vec("segments/accepted", acc); vec("segments/rejected_error", rej); vec("segments/rejected_nonfinite", nf);
      vec("segments/clipped", cl); vec("segments/rhs_calls", rc); vec("segments/h_min", hmin);
      vec("segments/h_max", hmax); vec("segments/h_geomean", hg); vec("segments/state_size", ss);
      std::vector<double> sr(ns), su(ns), so(ns), sp(ns);
      for (std::size_t k = 0; k < ns; ++k) {
        sr[k] = segments[k].rhs_seconds(); su[k] = segments[k].user_seconds();
        so[k] = segments[k].overhead_seconds(); sp[k] = segments[k].pass_seconds();
      }
      vec("segments/rhs_seconds", sr); vec("segments/user_seconds", su); vec("segments/overhead_seconds", so);
      vec("segments/pass_seconds", sp);
      for (int i = 0; i < SecCount; ++i) {
        std::vector<double> v(ns), c(ns);
        for (std::size_t k = 0; k < ns; ++k) { v[k] = segments[k].sec[i].seconds; c[k] = double(segments[k].sec[i].calls); }
        vec(std::string("segments/time/") + sec_name(i) + "/seconds", v);
        vec(std::string("segments/time/") + sec_name(i) + "/calls", c);
      }
    }
    s1("trace/stride", double(trace_stride)); vec("trace/t", trace_t); vec("trace/h", trace_h);
    vec("trace/err", trace_err); vec("trace/worst_block", trace_worst); vec("trace/accepted", trace_accepted);
    s1("end/t", t_last); s1("end/h", h_last); s1("end/err_old", err_old_last); s1("end/status", last_status);
    s1("end/exceptions", double(exceptions)); s1("end/last_err", last_err); s1("end/last_worst_block", last_worst_block);
    vec("end/last_block_err", last_block_err);
    s1("memory/register_bytes", double(register_bytes)); s1("memory/register_bytes_peak", double(register_bytes_peak));
    s1("memory/layout_transient_peak_bytes", double(layout_transient_peak_bytes));
    s1("memory/dense_bytes", double(dense_bytes)); s1("memory/maxrss_bytes", double(maxrss_bytes));
    vec("memory/layout_t", layout_t); vec("memory/layout_register_bytes", layout_register_bytes);
  }
  std::map<std::string, std::vector<double>> to_map() const {
    std::map<std::string, std::vector<double>> m;
    visit([&](const std::string& n, const double* d, std::size_t k) { m[n].assign(d, d + k); });
    return m;
  }
  std::map<std::string, std::string> strings() const {
    std::map<std::string, std::string> m;
    m["context/compiler"] = compiler;
    m["context/build_flags"] = build_flags;
    std::string names;
    for (std::size_t i = 0; i < blocks.size(); ++i) names += (i ? "," : "") + blocks[i].name;
    m["blocks/names"] = names;
    std::string secs;
    for (int i = 0; i < SecCount; ++i) secs += (i ? "," : "") + std::string(sec_name(i));
    m["time/section_names"] = secs;
    m["end/status_name"] = status_name(last_status);
    m["end/last_exception"] = last_exception;
    return m;
  }

  std::string to_text() const {
    std::string o;
    char b[512];
    auto line = [&](const char* fmt, auto... a) { std::snprintf(b, sizeof b, fmt, a...); o += b; };
    const double T = integrate_seconds > 0 ? integrate_seconds : 1.0;
    line("dp5 profile: %s, %d threads, par_threshold %zu, controller %s, combine %s\n",
         enabled ? "enabled" : "DISABLED", threads, par_threshold, controller == 0 ? "PI" : "Standard",
         combine == 0 ? "MaxOverBlocks" : "GlobalRMS");
    line("  build: %s | %s\n", compiler.c_str(), build_flags.c_str());
    line("  state: %zu complex entries;", state_size);
    for (auto& k : blocks)
      if (!k.removed)
        line(" %s[%zu %s %s atol=%.0e rtol=%.0e%s]", k.name.c_str(), k.size, k.norm ? "Max" : "RMS",
             k.active ? "on" : "off", k.atol, k.rtol, k.save ? " save" : "");
    o += "\n";
    line("  integrate: %.4f s in %llu call(s); RHS %.4f s (%.1f%%), user callbacks %.4f s (%.1f%%), "
         "integrator overhead %.4f s (%.1f%%; passes %.4f s, untimed %.4f s)\n",
         integrate_seconds, (unsigned long long)integrate_calls, rhs_seconds(), 100 * rhs_seconds() / T,
         user_seconds(), 100 * user_seconds() / T, overhead_seconds(), 100 * overhead_seconds() / T, pass_seconds(),
         untimed_seconds);
    o += "  section             seconds   %total    calls   us/call   imbalance(max/mean)\n";
    for (int i = 0; i < SecCount; ++i) {
      const auto& s = sec[i];
      if (s.calls == 0) continue;
      const double imb = s.imb_mean > 0 ? s.imb_max / s.imb_mean : 0.0;
      line("  %-17s %9.4f %7.2f%% %8llu %9.2f   %s", sec_name(i), s.seconds, 100 * s.seconds / T,
           (unsigned long long)s.calls, 1e6 * (s.seconds + s.seconds_outside) / double(s.calls),
           imb > 0 ? "" : "-");
      if (imb > 0) line("%.2f", imb);
      if (s.seconds_outside > 0) line("   (+%.4f s outside integrate)", s.seconds_outside);
      o += "\n";
    }
    line("  steps: attempts %llu = accepted %llu + rejected(err>1) %llu + rejected(non-finite) %llu; "
         "clipped to stops %llu; RHS calls %llu (slope %llu, h0 %llu); frames %llu dense + %llu exact; stops %llu\n",
         (unsigned long long)attempts, (unsigned long long)accepted, (unsigned long long)rejected_error,
         (unsigned long long)rejected_nonfinite, (unsigned long long)clipped, (unsigned long long)rhs_calls,
         (unsigned long long)slope_recomputes, (unsigned long long)init_step_estimates,
         (unsigned long long)frames_dense, (unsigned long long)frames_exact, (unsigned long long)stops);
    line("  h: min %.3e  median~ %.3e  geomean %.3e  max %.3e;  err of accepted steps: mean %.3f max %.3f; "
         "rejection rate %.2f%%\n",
         accepted ? h_min : 0.0, h_median(), h_geomean(), h_max, accepted ? err_sum / double(accepted) : 0.0,
         err_max, attempts ? 100.0 * double(rejected_error + rejected_nonfinite) / double(attempts) : 0.0);
    if (!h0_auto_h.empty()) {
      o += "  automatic h0:";
      for (std::size_t i = 0; i < h0_auto_h.size(); ++i) line(" %.3e@t=%.4g", h0_auto_h[i], h0_auto_t[i]);
      o += "\n";
    }
    if (accepted) {
      o += "  limiting block (share of accepted steps):";
      for (std::size_t i = 0; i < limiting_block.size(); ++i)
        if (limiting_block[i]) {
          const std::string nm = i < blocks.size() ? blocks[i].name : std::to_string(i);
          line(" %s %.1f%%", nm.c_str(), 100.0 * double(limiting_block[i]) / double(accepted));
        }
      o += "\n";
    }
    if (segments.size() > 1 || !segments.empty()) {
      o += "  segments (split at stops):\n";
      o += "    #   t_begin     t_end   seconds   accepted  rej  clip    h_geomean   state    RHS%  user%  ovh%  (passes%)\n";
      for (std::size_t k = 0; k < segments.size(); ++k) {
        const auto& g = segments[k];
        const double S = g.integrate_seconds > 0 ? g.integrate_seconds : 1.0;
        line("    %-3zu %9.4g %9.4g %9.4f %10llu %4llu %5llu   %.3e %8zu %6.1f %6.1f %5.1f  (%5.1f)\n", k, g.t_begin,
             g.t_end, g.integrate_seconds, (unsigned long long)g.accepted,
             (unsigned long long)(g.rejected_error + g.rejected_nonfinite), (unsigned long long)g.clipped,
             g.accepted ? std::exp(g.log_h_sum / double(g.accepted)) : 0.0, g.state_size, 100 * g.rhs_seconds() / S,
             100 * g.user_seconds() / S, 100 * g.overhead_seconds() / S, 100 * g.pass_seconds() / S);
      }
    }
    line("  memory: registers %.1f MB (peak %.1f MB, layout-change transient peak %.1f MB), dense buffer %.2f MB, "
         "process peak RSS %.1f MB\n",
         register_bytes / 1e6, register_bytes_peak / 1e6, layout_transient_peak_bytes / 1e6, dense_bytes / 1e6,
         maxrss_bytes >= 0 ? maxrss_bytes / 1e6 : -1.0);
    line("  end: t=%.6g h=%.3e err_old=%.3e status=%s; last step err=%.3e worst block %d", t_last, h_last, err_old_last,
         status_name(last_status), last_err, last_worst_block);
    if (exceptions) line("; exceptions %llu (last: %s)", (unsigned long long)exceptions, last_exception.c_str());
    o += "\n";
    return o;
  }
};

}  // namespace dp5
