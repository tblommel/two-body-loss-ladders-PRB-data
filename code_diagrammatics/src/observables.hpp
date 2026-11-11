// observables.hpp -- equal-time observables at output times (step 3b; README "Equal-time observables").
// Never part of the integrator state: computed from rho, the saved G1-G2 tensor blocks and the history
// integral I at each output frame.
#pragma once

#include <deque>
#include <string>
#include <utility>
#include <vector>

#include "common.hpp"

namespace drv {

// One named time series: data for the current frame, fixed per-frame shape and a description.
struct ObsSeries {
  std::string name;
  std::vector<unsigned long long> dims;  // per-frame shape ({} = scalar)
  std::string desc;
  std::vector<double> v;
};
struct EqObs {
  std::deque<ObsSeries> series;  // deque: references stay valid while series are added
  std::vector<std::pair<std::string, std::string>> attrs;  // written once (flags, conventions)
  std::vector<long long> bonds;                            // [nb,2] nearest-neighbour pairs (i<j)
  ObsSeries& add(const std::string& n, std::vector<unsigned long long> d, const std::string& desc) {
    for (auto& s : series)
      if (s.name == n) return s;
    std::size_t sz = 1;
    for (auto x : d) sz *= std::size_t(x);
    series.push_back({n, std::move(d), desc, std::vector<double>(sz, 0.0)});
    return series.back();
  }
};

struct EqContext {
  Approx approx;
  int N, ns;
  const cplx* rho;                     // [ns,N,N] rho^<_{sa} = <c_a^dag c_s>
  std::vector<const cplx*> X;          // tensor blocks in g1g2_blocks order (without rho); may be empty (HF)
  const cplx* I;                       // [ns,N,N] history integral (valid if X non-empty)
  std::vector<long long> sites;        // lossy sites
  std::vector<double> gam;             // gamma_r = gamma w_r per lossy site
  const Mat* h;                        // bare one-body h per spin (ns entries)
  std::vector<long long> bonds;        // [nb,2]
  bool impurity = false;               // lossy_sites == {0}: impurity energies
  // Always false in the release (the development driver's run-card option corr_matrix is not part of it). The flag and
  // the block it guards in equal_time_observables are kept so that the compiled code of the observables is unchanged:
  // without them GCC contracts the <S_i.S_j> sums into FMAs differently at -march=x86-64-v3 (last-ulp changes of
  // S2_total and SS_nn; validation/driver gate).
  bool corr_matrix = false;
  double dN_dt = 0, dN_dt_onebody = 0; // Re tr d rho/dt (all terms) and its one-body (bath, h) part
  // Galitskii-Migdal traces of the left-sided equal-time derivative Lambda = i d/dt rho|_left (rho units,
  // d rho/dt = -i(Lambda - Lambda^dag)), summed over spins: one-body h, one-body dissipative, MF, correlated
  cplx lam_h = 0, lam_1d = 0, lam_mf = 0, lam_c = 0;
};

void equal_time_observables(const EqContext& c, EqObs& o);
// nearest-neighbour pairs: i<j with |h^sigma_ij| > 1e-12 for some spin
std::vector<long long> bonds_from_h(const Mat* h, int ns, int N);

}  // namespace drv
