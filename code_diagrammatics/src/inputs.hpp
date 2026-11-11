// inputs.hpp -- model file (immutable, hashed) and run card (README "File schemas").
#pragma once

#include <map>
#include <string>
#include <vector>

#include "common.hpp"

namespace drv {

// FNV-1a 64-bit over a byte stream; the model content hash (Python: dp5model.content_hash).
struct Fnv1a {
  unsigned long long h = 1469598103934665603ull;
  void add(const void* p, std::size_t n) {
    const unsigned char* c = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) { h ^= c[i]; h *= 1099511628211ull; }
  }
  std::string hex() const;
};

struct Model {
  std::string path;
  int N = 0;
  std::vector<cplx> h, ell_greater, ell_lesser, rho0;  // each [2,N,N], row-major, spin-resolved
  std::vector<long long> lossy_sites;                  // vertex: on-site pair loss on these orbitals
  std::vector<double> site_weight;                     // per lossy site, multiplies gamma (default 1)
  std::map<std::string, std::string> attrs;            // descriptive attributes (mu, Gamma, ordering, ...) as text
  std::string hash_stored, hash_computed;
  const cplx* h_s(int s) const { return h.data() + std::size_t(s) * N * N; }
  const cplx* gt_s(int s) const { return ell_greater.data() + std::size_t(s) * N * N; }
  const cplx* lt_s(int s) const { return ell_lesser.data() + std::size_t(s) * N * N; }
};
Model load_model(const std::string& path);
std::string model_hash(const Model& m);  // canonical content hash

// Run card: the attributes of /run (README "Run-card keys"). The key set is closed: any other /run attribute or
// dataset is an input error (exit 2), so a card can never rely on an option this build does not have.
struct RunCard {
  std::string path;
  std::string model_path, model_hash;
  double gamma = 0.0;
  Approx approx = Approx::HF;
  bool flag_2B = false;
  // bare_propagators (run-card 0/1, default 0): the correlated objects (G1-G2 two-particle correlators incl. the mGW
  // counterterm, RTDE tensors) are propagated with the bare h_o = h - i(l^> + l^<) instead of h_o^MF on all their
  // legs; rho and the G^R equation keep h_o^MF. Reporting label X_0 (e.g. TPPX_0). Not defined for HF.
  bool bare_propagators = false;
  Mode mode = Mode::g1g2;
  SpinMode spin = SpinMode::full;
  VertexImpl vertex = VertexImpl::general;
  double t0 = 0.0, t_end = 1.0, dt_out = 0.1;
  std::vector<double> t_prime;              // mode rtde: one number, or empty with t_prime = "auto"
  std::vector<long long> probe_columns{0};  // exactly one probe column j
  double tol_rho[2] = {1e-9, 1e-7}, tol_tensor[2] = {1e-8, 1e-6}, tol_rtde[2] = {1e-10, 1e-6};
  double checkpoint_interval = 0.0;
  long long checkpoint_keep = 0;            // > 0: keep only the last checkpoint_keep periodic checkpoints
  double diverge_rho_lo = -1.0, diverge_rho_hi = 2.0;  // eigenvalues of rho outside: status Diverged (exit 7)
  long long rho_stride = 1;                 // store full rho every rho_stride-th frame (observables: every frame)
  std::string ic_mode = "model";            // model | full
  std::vector<cplx> rho0_override;          // ic_mode = full: [2,N,N]
  int threads = 0;                          // 0: OMP default
  std::string output;                       // CLI --output (*.h5)
  // automatic steady-state detection (t_prime = "auto"; README "Steady-state detection")
  bool tprime_auto = false;
  double tol_ss = 1e-6, c_hold = 1.0, c_min = 1.0, c_max = 50.0, ss_fit_rtol = 0.2, ss_window_c = 1.0;
  std::string ss_on_timeout = "fail";       // fail | start
  double rtde_tau = 0.0;                    // RTDE duration after t' (with t_prime = auto: a number or "auto")
  // automatic RTDE end time (rtde_tau = "auto")
  bool rtde_tau_auto = false;
  double tol_A = 1e-4;                      // bound on the truncation error of A(omega), units 1/Gamma
  double rtde_tau_min = 0.0;                // <= 0: c_tau_min/lambda_ref(t')
  double c_tau_min = 3.0;
  double rtde_tau_max = 0.0;                // cap (required with rtde_tau = auto)
  std::string rtde_monitor = "auto";        // auto (impurity if the model ordering is d,p>,p<, else all) | impurity | all
  double rtde_window_beats = 8.0;           // rtde_tau = auto: monitor window >= this many beat periods (0: off)
  bool ss_hold_records = true;              // detector: hold window >= 3 drift windows per half (0: off; steady_state.hpp)
};
RunCard load_runcard(const std::string& path);
void validate_auto(const RunCard& r);  // t_prime checks incl. the auto options

}  // namespace drv
