#include "steady_state.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace drv {

SteadyStateDetector::SteadyStateDetector(const SsConfig& c, double t0, double lam_ref0, double lam_fast0,
                                         const std::vector<std::size_t>& block_sizes)
    : c_(c), t0_(t0), lam0_(lam_ref0), lam_fast_(lam_fast0), sizes_(block_sizes) {
  t_min_ = t0_ + c_.c_min / lam0_;
  t_max_ = t0_ + c_.c_max / lam0_;
  // initial drift window: geometric mean of the stiff and the slow time scales, clamped to [10/lam_fast, 0.5/lam_ref]
  const double lo = 10.0 / lam_fast_, hi = 0.5 / lam0_;
  delta_ = std::min(std::max(c_.c_delta / std::sqrt(lam_fast_ * lam0_), lo), std::max(lo, hi));
  delta0_ = delta_;
  for (std::size_t n : sizes_) {
    y0_.emplace_back(n); sy_.emplace_back(n); sty_.emplace_back(n); syy_.emplace_back(n);
  }
}

// drift of a record over the blocks that are not skipped
double SteadyStateDetector::deff(const Rec& r, const std::vector<char>& skip) {
  if (skip.empty()) return r.D;
  double d = 0;
  for (std::size_t b = 0; b < r.Db.size(); ++b) if (!skip[b]) d = std::max(d, r.Db[b]);
  return d;
}

// least-squares slope and its standard error
static void ls_slope(const std::vector<double>& x, const std::vector<double>& y, double& s, double& se) {
  const std::size_t n = x.size();
  double mx = 0, my = 0;
  for (std::size_t i = 0; i < n; ++i) { mx += x[i]; my += y[i]; }
  mx /= double(n); my /= double(n);
  double sxx = 0, sxy = 0;
  for (std::size_t i = 0; i < n; ++i) { sxx += (x[i] - mx) * (x[i] - mx); sxy += (x[i] - mx) * (y[i] - my); }
  s = sxx > 0 ? sxy / sxx : 0.0;
  double rss = 0;
  for (std::size_t i = 0; i < n; ++i) { const double r = y[i] - my - s * (x[i] - mx); rss += r * r; }
  se = (n > 2 && sxx > 0) ? std::sqrt(rss / double(n - 2) / sxx) : std::numeric_limits<double>::infinity();
}

// envelope fit over the drift records with end time in [t - T, t]
SsFit SteadyStateDetector::fit_window(double t, double T, const std::vector<char>& skip) const {
  SsFit f;
  const double a = t - T, mid = a + 0.5 * T;
  std::vector<double> x1, y1, x2, y2, x, y;
  for (const Rec& r : recs_) {
    if (r.t < a) continue;
    const double tc = r.t - 0.5 * r.delta;  // the LS slope estimates the drift at the window centre
    const double ly = std::log(std::max(deff(r, skip), 1e-300));
    x.push_back(tc); y.push_back(ly);
    if (tc < mid) { x1.push_back(tc); y1.push_back(ly); } else { x2.push_back(tc); y2.push_back(ly); }
  }
  f.bins_half1 = int(x1.size()); f.bins_half2 = int(x2.size());
  if (x1.size() < 3 || x2.size() < 3) { f.why = "fewer than 3 drift windows in a half-window"; return f; }
  double s, se;
  ls_slope(x1, y1, f.s_half1, f.se_half1);
  ls_slope(x2, y2, f.s_half2, f.se_half2);
  ls_slope(x, y, s, se);
  if (!(f.s_half1 < 0 && f.s_half2 < 0)) { f.why = "drift not decaying"; return f; }
  if (!(-f.s_half1 > 3 * f.se_half1 && -f.s_half2 > 3 * f.se_half2)) { f.why = "decay not significant (slope < 3 standard errors)"; return f; }
  if (std::fabs(f.s_half1 - f.s_half2) > c_.fit_rtol * std::max(-f.s_half1, -f.s_half2)) { f.why = "half-window rates disagree"; return f; }
  f.lam_fit = -s;
  f.ok = f.lam_fit > 0;
  if (!f.ok) f.why = "fitted rate not positive";
  return f;
}

// accumulate one accepted step into the current drift window (per component, relative to the window's first state)
void SteadyStateDetector::accumulate(double tau, const std::vector<const cplx*>& blocks) {
  for (std::size_t b = 0; b < sizes_.size(); ++b) {
    const cplx* y = blocks[b];
    const cplx* y0 = y0_[b].data();
    cplx* sy = sy_[b].data(); cplx* sty = sty_[b].data(); double* syy = syy_[b].data();
    const std::ptrdiff_t n = std::ptrdiff_t(sizes_[b]);
#pragma omp parallel for schedule(static) if (n > 20000)
    for (std::ptrdiff_t i = 0; i < n; ++i) {
      const cplx d = y[i] - y0[i];
      sy[i] += d; sty[i] += tau * d; syy[i] += std::norm(d);
    }
  }
  n_ += 1; st_ += tau; stt_ += tau * tau;
}

void SteadyStateDetector::start_window(double t, const std::vector<const cplx*>& blocks) {
  tws_ = t; n_ = 0; st_ = stt_ = 0;
  for (std::size_t b = 0; b < sizes_.size(); ++b) {
    std::copy(blocks[b], blocks[b] + sizes_[b], y0_[b].begin());
    std::fill(sy_[b].begin(), sy_[b].end(), cplx(0));
    std::fill(sty_[b].begin(), sty_[b].end(), cplx(0));
    std::fill(syy_[b].begin(), syy_[b].end(), 0.0);
  }
  accumulate(0.0, blocks);
}

// close the window: per block max |LS slope| and noise 3 SE, SE = max rms residual about the LS line / sqrt(n var t)
SteadyStateDetector::Rec SteadyStateDetector::close_window(double t) {
  Rec r;
  r.t = t; r.delta = t - tws_; r.n = n_;
  const double n = double(n_), tb = st_ / n, vt = stt_ / n - tb * tb;
  for (std::size_t b = 0; b < sizes_.size(); ++b) {
    const cplx* sy = sy_[b].data(); const cplx* sty = sty_[b].data(); const double* syy = syy_[b].data();
    double dmax = 0, rmax = 0;
    const std::ptrdiff_t m = std::ptrdiff_t(sizes_[b]);
#pragma omp parallel for schedule(static) reduction(max : dmax, rmax) if (m > 20000)
    for (std::ptrdiff_t i = 0; i < m; ++i) {
      const cplx yb = sy[i] / n, cov = sty[i] / n - tb * yb;
      const cplx slope = cov / vt;
      const double rv = syy[i] / n - std::norm(yb) - std::norm(cov) / vt;
      dmax = std::max(dmax, std::abs(slope));
      rmax = std::max(rmax, rv > 0 ? std::sqrt(rv) : 0.0);
    }
    const double se = rmax / std::sqrt(n * vt);  // standard error of the LS slope (rms residual / sqrt(n var t))
    r.Db.push_back(dmax);
    r.Nb.push_back(3.0 * se);
    r.D = std::max(r.D, dmax);
    r.N = std::max(r.N, 3.0 * se);
  }
  return r;
}

bool SteadyStateDetector::step(double t, const std::vector<const cplx*>& blocks, const std::function<double()>& lam_ref_now) {
  if (!started_) { start_window(t, blocks); started_ = true; return false; }
  accumulate(t - tws_, blocks);
  if (t - tws_ < delta_ || n_ < 4) return false;  // window not complete (at least Delta and 4 samples)
  Rec r = close_window(t);
  start_window(t, blocks);
  if (t < t_min_) { recs_.push_back(r); return false; }
  const double lr = lam_ref_now();
  if (!(lr > 0)) { recs_.push_back(r); return false; }
  recs_.push_back(r);
  double T = std::max(c_.c_hold / lr, 6.0 * delta_);  // the hold window holds >= 6 drift windows (3 per half)
  if (T > c_.c_hold / lr) hold_extended = true;
  if (c_.hold_records) {  // ... which needs windows of length ~Delta: extend T until each half holds >= 3 windows
    auto halves = [&](double TT) {
      const double a = t - TT, mid = a + 0.5 * TT;
      int n1 = 0, n2 = 0;
      for (const Rec& q : recs_) if (q.t >= a) ((q.t - 0.5 * q.delta < mid) ? n1 : n2) += 1;
      return n1 >= 3 && n2 >= 3;
    };
    if (!halves(T)) {
      const double eps = 0.5e-12 * std::max(1.0, t);  // exclude the window that ends where the k-th last one starts
      bool found = false;
      for (std::size_t k = 6; k <= recs_.size(); ++k) {
        const Rec& q = recs_[recs_.size() - k];
        const double Tk = t - (q.t - q.delta) - eps;
        if (Tk > T && halves(Tk)) { T = Tk; found = true; break; }
      }
      if (!found) return false;
      ++hold_record_extensions;
    }
  }
  // the drift records must cover the hold window
  double cover = t;
  for (const Rec& q : recs_) if (q.t >= t - T) cover = std::min(cover, q.t - q.delta);
  if (cover > t - T + 1e-12 * std::max(1.0, t)) return false;
  // marginal blocks: a tensor block whose drift stays flat (within 20 %) over two hold windows while at least ten
  // times above tol_ss * lambda_ref neither relaxes nor grows: a persistent oscillation or linear drift in a sector without
  // damping (e.g. pair correlations of two formally undamped pseudomodes). Relaxation at any rate >= lambda_ref would
  // change the drift by a factor e over T >= 1/lambda_ref. Such blocks are excluded from the criterion (recorded);
  // rho (block 0) is never excluded.
  const std::size_t nb = sizes_.size();
  std::vector<double> bmax(nb, 0.0), bmin(nb, 1e300);
  double cover2 = t;
  for (const Rec& q : recs_)
    if (q.t >= t - 2 * T) {
      cover2 = std::min(cover2, q.t - q.delta);
      for (std::size_t b = 0; b < nb; ++b) { bmax[b] = std::max(bmax[b], q.Db[b]); bmin[b] = std::min(bmin[b], q.Db[b]); }
    }
  std::vector<char> skip(nb, 0);
  if (cover2 <= t - 2 * T + 1e-12 * std::max(1.0, t))  // flat over two hold windows, at least 10x above the threshold
    for (std::size_t b = 1; b < nb; ++b)
      if (bmax[b] >= 10.0 * c_.tol_ss * lr && bmin[b] > 0 && bmax[b] <= 1.2 * bmin[b]) skip[b] = 1;
  double wmax = 0;
  for (const Rec& q : recs_) if (q.t >= t - T) wmax = std::max(wmax, deff(q, skip));
  if (!(wmax < 10.0 * c_.tol_ss * lr)) return false;  // far from the threshold
  SsFit f = fit_window(t, T, skip);
  auto record_marginal = [&]() {
    marginal_blocks.clear(); marginal_drift.clear();
    for (std::size_t b = 0; b < nb; ++b) if (skip[b]) { marginal_blocks.push_back(int(b)); marginal_drift.push_back(bmax[b]); }
  };
  if (!f.ok) {
    // candidate regime but no clean decay: the windowed drift has reached its noise floor. Lower the floor by
    // doubling Delta (up to 0.5/lambda_ref), at most once per hold window.
    const double dmax_allowed = 0.5 / lr;
    if (f.bins_half1 >= 3 && f.bins_half2 >= 3 && delta_ < dmax_allowed && t - t_last_double_ >= T) {
      delta_ = std::min(2.0 * delta_, dmax_allowed);
      t_last_double_ = t;
      ++delta_doublings;
      return false;
    }
    // Delta at its maximum and the drift flat below the threshold: the floor is set by error accumulation along
    // marginal (undamped) directions, not by relaxation, so no rate can be fitted. Trigger with lambda_used =
    // lambda_ref (the slowest one-body rate, a conservative decay rate) once a full hold window, with every drift
    // window at the maximal Delta, stays below tol_ss * lambda_ref.
    if (f.bins_half1 >= 3 && f.bins_half2 >= 3 && delta_ >= dmax_allowed * (1 - 1e-12) && wmax < c_.tol_ss * lr) {
      bool all_max = true;
      for (const Rec& q : recs_)
        if (q.t >= t - T && q.delta < 0.999 * dmax_allowed) all_max = false;
      if (all_max) {
        ++candidates;
        floor_trigger = true;
        record_marginal();
        t_trigger = t; lam_ref_t = lr; lam_used = lr; max_ratio = wmax / lr; fit = f;
        norms_at_trigger = r.Db; noise_at_trigger = r.Nb; t_hold = T; delta_used = delta_;
        return true;
      }
    }
    return false;
  }
  if (!(wmax < c_.tol_ss * lr)) return false;
  ++candidates;
  const double lu = std::min(f.lam_fit, lr);
  if (!(wmax < c_.tol_ss * lu)) return false;
  record_marginal();
  t_trigger = t; lam_ref_t = lr; lam_used = lu; max_ratio = wmax / lu; fit = f;
  norms_at_trigger = r.Db; noise_at_trigger = r.Nb; t_hold = T; delta_used = delta_;
  return true;
}

// ---------------------------------------------------------------- automatic RTDE end time
SsFit RtdeStopMonitor::fit_at(double tau) const {
  SsFit f;
  const double a = tau - W_, mid = a + 0.5 * W_;
  const double beat = spread_ > 0 ? 2.0 * M_PI / spread_ : W_;
  const double L = std::min(std::max(beat, W_ / 16.0), W_ / 8.0);
  const int nb = int(std::ceil(W_ / L - 1e-12));
  std::vector<double> bmax(std::size_t(nb), -1.0);
  for (std::size_t i = 0; i < ts_.size(); ++i) {
    if (ts_[i] < a) continue;
    int k = std::min(std::max(int((ts_[i] - a) / L), 0), nb - 1);
    bmax[std::size_t(k)] = std::max(bmax[std::size_t(k)], ms_[i]);
  }
  std::vector<double> x1, y1, x2, y2, x, y;
  for (int k = 0; k < nb; ++k) {
    if (bmax[std::size_t(k)] < 0) continue;
    const double tc = std::min(a + (k + 0.5) * L, tau);
    const double ly = std::log(std::max(bmax[std::size_t(k)], 1e-300));
    x.push_back(tc); y.push_back(ly);
    if (tc < mid) { x1.push_back(tc); y1.push_back(ly); } else { x2.push_back(tc); y2.push_back(ly); }
  }
  f.bins_half1 = int(x1.size()); f.bins_half2 = int(x2.size());
  if (x1.size() < 3 || x2.size() < 3) { f.why = "fewer than 3 bins in a half-window"; return f; }
  double s, se;
  ls_slope(x1, y1, f.s_half1, f.se_half1);
  ls_slope(x2, y2, f.s_half2, f.se_half2);
  ls_slope(x, y, s, se);
  if (!(f.s_half1 < 0 && f.s_half2 < 0)) { f.why = "G^R not decaying"; return f; }
  if (!(-f.s_half1 > 3 * f.se_half1 && -f.s_half2 > 3 * f.se_half2)) { f.why = "decay not significant"; return f; }
  if (std::fabs(f.s_half1 - f.s_half2) > rtol_ * std::max(-f.s_half1, -f.s_half2)) { f.why = "half-window rates disagree"; return f; }
  f.lam_fit = -s;
  f.ok = f.lam_fit > 0;
  return f;
}

bool RtdeStopMonitor::feed(double tau, double m) {
  ts_.push_back(tau); ms_.push_back(m);
  if (tau < tau_min_ || tau < W_) return false;
  SsFit f = fit_at(tau);
  if (!f.ok) return false;
  double wmax = 0;
  for (std::size_t i = ts_.size(); i-- > 0;) {
    if (ts_[i] < tau - W_) break;
    wmax = std::max(wmax, ms_[i]);
  }
  if (!(wmax / (M_PI * f.lam_fit) < tol_)) return false;
  tau_stop = tau; lam_gr = f.lam_fit; fit = f; err_est = m / (M_PI * f.lam_fit);
  return true;
}

bool RtdeStopMonitor::estimate(double& lam, double& err) const {
  if (ts_.empty()) return false;
  SsFit f = fit_at(ts_.back());
  if (!f.ok) return false;
  lam = f.lam_fit;
  err = ms_.back() / (M_PI * lam);
  return true;
}

}  // namespace drv
