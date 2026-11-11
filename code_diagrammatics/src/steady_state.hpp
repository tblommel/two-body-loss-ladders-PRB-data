// steady_state.hpp -- automatic t' (run card t_prime = "auto"; README "Steady-state detection"):
// windowed-drift steady-state detector, and the automatic RTDE end (rtde_tau = "auto").
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "common.hpp"

namespace drv {

struct SsConfig {
  double tol_ss = 1e-6, c_hold = 1.0, c_min = 1.0, c_max = 50.0, fit_rtol = 0.2, c_delta = 1.0;
  // run card ss_hold_records (default 1, 0: off): extend the hold window backwards, record by record, until each half holds
  // at least 3 drift windows. Without it a window that closes only after 4 accepted steps (>> Delta once the state is
  // stationary and the steps are long) can leave fewer than 3 windows per half of T = max(c_hold/lambda_ref, 6 Delta),
  // and then neither the fit, the Delta doubling nor the drift-floor rule can ever fire (timeouts in
  // impurity-model runs). The extension only adds windows to the hold window (stricter).
  bool hold_records = true;
};

struct SsFit {
  bool ok = false;
  double lam_fit = 0, s_half1 = 0, s_half2 = 0, se_half1 = 0, se_half2 = 0;
  int bins_half1 = 0, bins_half2 = 0;
  std::string why;  // reason if !ok
};

// Windowed drift: the accepted steps are grouped into consecutive windows of length >= Delta (and >= 4 steps);
// in each window, per component, the least-squares slope b of y(t) - y(t_window_start) and the rms residual r
// about the LS line are formed from running sums (memory: 3.5 states). Per block D_b = max|b| (the drift) and
// N_b = 3 max r / sqrt(n var t) (three standard errors for independent residuals; reported only: the integrator
// noise is oscillatory and averages out much better). D = max_b D_b. Trigger at a window end t >= t_min:
//   D_j < tol_ss * lambda_used for every window ending in the hold window [t - T, t],
//   T = max(c_hold/lambda_ref(t), 6 Delta), lambda_used = min(lambda_fit, lambda_ref(t)),
// lambda_fit from the decay of log D_j (two half-window fits, which also reject a flat noise floor).
// Delta starts at c_delta/sqrt(lambda_fast lambda_ref) clamped to [10/lambda_fast, 0.5/lambda_ref] and doubles
// (up to 0.5/lambda_ref, at most once per hold window) when, with D < 10 tol lambda_ref, the drift no longer decays
// cleanly (its noise floor).
class SteadyStateDetector {
 public:
  SteadyStateDetector(const SsConfig& c, double t0, double lam_ref0, double lam_fast0, const std::vector<std::size_t>& block_sizes);
  double t_min() const { return t_min_; }
  double t_max() const { return t_max_; }
  double delta_initial() const { return delta0_; }
  double delta_now() const { return delta_; }
  // one accepted step: G1-G2 block pointers; lam_ref_now(): lambda_ref(t), called at window ends after t_min
  bool step(double t, const std::vector<const cplx*>& blocks, const std::function<double()>& lam_ref_now);
  // results at the trigger (valid after step() returned true)
  double t_trigger = 0, lam_ref_t = 0, lam_used = 0, max_ratio = 0, t_hold = 0, delta_used = 0;
  SsFit fit;
  std::vector<double> norms_at_trigger, noise_at_trigger;
  long long candidates = 0, delta_doublings = 0;
  bool hold_extended = false;
  bool floor_trigger = false;
  long long hold_record_extensions = 0;  // window ends at which the hold window was extended (ss_hold_records)
  std::vector<int> marginal_blocks;     // blocks excluded at the trigger (flat drift above tol_ss lambda_ref)
  std::vector<double> marginal_drift;   // their maximal drift over the hold window  // triggered at the drift floor (no fitted rate; lambda_used = lambda_ref)
  struct Rec { double t = 0, delta = 0, D = 0, N = 0; long long n = 0; std::vector<double> Db, Nb; };
  const std::vector<Rec>& records() const { return recs_; }

 private:
  SsFit fit_window(double t, double T, const std::vector<char>& skip) const;
  static double deff(const Rec& r, const std::vector<char>& skip);
  void accumulate(double tau, const std::vector<const cplx*>& blocks);
  void start_window(double t, const std::vector<const cplx*>& blocks);
  Rec close_window(double t);
  SsConfig c_;
  double t0_, lam0_, lam_fast_, t_min_, t_max_, delta_ = 0, delta0_ = 0;
  std::vector<std::size_t> sizes_;
  std::vector<std::vector<cplx>> y0_, sy_, sty_;
  std::vector<std::vector<double>> syy_;
  bool started_ = false;
  double tws_ = 0, st_ = 0, stt_ = 0, t_last_double_ = -1e300;
  long long n_ = 0;
  std::vector<Rec> recs_;
};

// Automatic RTDE end time (rtde_tau = "auto"): fed with the output frames of the first RTDE activation,
// m(tau) = max |G^R_{xs}(t'+tau, t')| over the monitored entries. lambda_GR = decay rate of the binned envelope of
// log m over the window W = c_hold/lambda_ref (bins min(max(2 pi/spread, W/16), W/8); both half-window slopes
// negative, significant at 3 sigma and within fit_rtol). Stop when tau >= tau_min and
//   m(tau_i)/(pi lambda_GR) < tol_A  for every frame tau_i in [tau - W, tau]
// (bound on the truncation error of A(omega) = -Im int_0^tau_end e^{i omega tau} G^R / pi).
class RtdeStopMonitor {
 public:
  RtdeStopMonitor(double tol_A, double tau_min, double window, double spread, double fit_rtol)
      : tol_(tol_A), tau_min_(tau_min), W_(window), spread_(spread), rtol_(fit_rtol) {}
  bool feed(double tau, double m);  // true when the stop criterion holds at tau
  double tau_stop = 0, lam_gr = 0, err_est = 0;
  SsFit fit;
  double last_m() const { return ms_.empty() ? 0.0 : ms_.back(); }
  // rate and truncation-error estimate from the current history (used at the cap)
  bool estimate(double& lam, double& err) const;
  std::size_t samples() const { return ts_.size(); }
  const std::vector<double>& taus() const { return ts_; }
  const std::vector<double>& values() const { return ms_; }
  double window() const { return W_; }
  double tau_min() const { return tau_min_; }

 private:
  SsFit fit_at(double tau) const;
  double tol_, tau_min_, W_, spread_, rtol_;
  std::vector<double> ts_, ms_;
};

}  // namespace drv
