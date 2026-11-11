#include "inputs.hpp"

#include <cmath>
#include <cstdio>
#include <set>

#include "h5util.hpp"

namespace drv {

std::string Fnv1a::hex() const {
  char b[17];
  std::snprintf(b, sizeof b, "%016llx", h);
  return b;
}

// Canonical byte stream: h, ell_greater, ell_lesser, rho0 as complex128 (re, im
// interleaved, C order [2,N,N]), then lossy_sites as int64, site_weight as float64.
std::string model_hash(const Model& m) {
  Fnv1a f;
  for (const auto* v : {&m.h, &m.ell_greater, &m.ell_lesser, &m.rho0}) f.add(v->data(), v->size() * sizeof(cplx));
  f.add(m.lossy_sites.data(), m.lossy_sites.size() * sizeof(long long));
  f.add(m.site_weight.data(), m.site_weight.size() * sizeof(double));
  return f.hex();
}

static Model load_model_impl(const std::string& path);
Model load_model(const std::string& path) {
  h5::check_readable_hdf5(path, "model file");
  return h5::guarded("model file '" + path + "'", [&] { return load_model_impl(path); });
}
static Model load_model_impl(const std::string& path) {
  Model m;
  m.path = path;
  H5::H5File f(path, H5F_ACC_RDONLY);
  H5::Group root = f.openGroup("/");
  const int schema = int(h5::read_attr_dbl(root, "dp5_model_schema"));
  if (schema != kModelSchema) throw std::runtime_error("model schema " + std::to_string(schema) + " not supported");
  H5::Group g = f.openGroup("/model");
  std::vector<hsize_t> d;
  m.h = h5::read_cplx(g, "h", &d);
  if (d.size() != 3 || d[0] != 2 || d[1] != d[2]) throw std::runtime_error("model: /model/h must be [2,N,N]");
  m.N = int(d[1]);
  std::vector<hsize_t> d2;
  m.ell_greater = h5::read_cplx(g, "ell_greater", &d2);
  if (d2 != d) throw std::runtime_error("model: ell_greater shape");
  m.ell_lesser = h5::read_cplx(g, "ell_lesser", &d2);
  if (d2 != d) throw std::runtime_error("model: ell_lesser shape");
  H5::Group gi = f.openGroup("/initial");
  m.rho0 = h5::read_cplx(gi, "rho", &d2);
  if (d2 != d) throw std::runtime_error("model: /initial/rho shape");
  H5::Group gv = f.openGroup("/model/vertex");
  m.lossy_sites = h5::read_i64(gv, "lossy_sites");
  m.site_weight = h5::read_dbl(gv, "site_weight");
  if (m.site_weight.size() != m.lossy_sites.size()) throw std::runtime_error("model: site_weight length");
  for (auto s : m.lossy_sites)
    if (s < 0 || s >= m.N) throw std::runtime_error("model: lossy site out of range");
  for (const char* a : {"N_greater", "N_lesser", "mu", "Gamma", "beta", "ordering", "units", "fit_info"}) {
    if (!g.attrExists(a)) continue;
    H5::Attribute at = g.openAttribute(a);
    if (at.getTypeClass() == H5T_STRING) m.attrs[a] = h5::read_attr_str(g, a);
    else m.attrs[a] = std::to_string(h5::read_attr_dbl(g, a));
  }
  m.hash_stored = h5::read_attr_str(root, "content_hash");
  m.hash_computed = model_hash(m);
  if (m.hash_stored != m.hash_computed)
    throw std::runtime_error("model content hash mismatch: stored " + m.hash_stored + ", computed " + m.hash_computed);
  return m;
}

void validate_auto(const RunCard& r) {
  if (r.mode == Mode::rtde && r.t_prime.empty() && !r.tprime_auto) throw std::runtime_error("run card: mode rtde needs t_prime");
  if (r.rtde_tau_auto) {
    auto badt = [](const std::string& w) { throw std::runtime_error("run card (rtde_tau = auto): " + w); };
    if (r.mode != Mode::rtde) badt("needs mode rtde");
    if (!r.tprime_auto && r.t_prime.size() != 1) badt("needs exactly one t_prime (or t_prime = auto)");
    if (!(r.rtde_tau_max > 0)) badt("rtde_tau_max (cap of the RTDE duration) must be > 0");
    {
      const double n = r.rtde_tau_max / r.dt_out;
      if (std::fabs(n - std::round(n)) > 1e-9 * std::max(1.0, n)) badt("rtde_tau_max/dt_out must be an integer");
    }
    if (!(r.tol_A > 0) || !(r.c_tau_min >= 0)) badt("need tol_A > 0 and c_tau_min >= 0");
    if (r.rtde_monitor != "auto" && r.rtde_monitor != "impurity" && r.rtde_monitor != "all") badt("rtde_monitor must be auto, impurity or all");
  }
  if (!r.tprime_auto) return;
  auto bad = [](const std::string& w) { throw std::runtime_error("run card (t_prime = auto): " + w); };
  if (r.mode != Mode::rtde) bad("needs mode rtde");
  if (!r.t_prime.empty()) bad("numeric t_prime given as well");
  if (!(r.tol_ss > 0) || !(r.c_hold > 0) || !(r.c_min >= 0) || !(r.c_max > r.c_min)) bad("need tol_ss > 0, c_hold > 0, c_max > c_min >= 0");
  if (!(r.ss_fit_rtol > 0) || !(r.ss_window_c > 0)) bad("need ss_fit_rtol > 0 and ss_window_c > 0");
  if (r.ss_on_timeout != "fail" && r.ss_on_timeout != "start") bad("ss_on_timeout must be fail or start");
  if (!r.rtde_tau_auto) {
    if (!(r.rtde_tau > 0)) bad("rtde_tau (RTDE duration after t') must be > 0, or \"auto\"");
    const double n = r.rtde_tau / r.dt_out;
    if (std::fabs(n - std::round(n)) > 1e-9 * std::max(1.0, n)) bad("rtde_tau/dt_out must be an integer");
  }
}

static RunCard load_runcard_impl(const std::string& path);
RunCard load_runcard(const std::string& path) {
  h5::check_readable_hdf5(path, "run card");
  return h5::guarded("run card '" + path + "'", [&] { return load_runcard_impl(path); });
}
// The closed key set of /run (README "Run-card keys"; CONVENTIONS §3 of the release). Anything else is refused.
static const std::set<std::string> kRunKeys = {
    "model_path", "model_hash", "gamma", "approximation", "flag_2B", "bare_propagators", "mode", "spin_mode",
    "vertex_impl", "ic_mode", "t0", "t_end", "dt_out", "probe_columns", "tol_rho", "tol_tensor", "tol_rtde",
    "checkpoint_interval", "checkpoint_keep", "rho_stride", "diverge_rho_lo", "diverge_rho_hi", "threads",
    // mode rtde: t', steady-state detector, automatic RTDE end
    "t_prime", "ss_on_timeout", "tol_ss", "c_hold", "c_min", "c_max", "ss_fit_rtol", "ss_window_c", "ss_hold_records",
    "rtde_tau", "tol_A", "rtde_tau_min", "c_tau_min", "rtde_tau_max", "rtde_window_beats", "rtde_monitor"};

static RunCard load_runcard_impl(const std::string& path) {
  RunCard r;
  r.path = path;
  H5::H5File f(path, H5F_ACC_RDONLY);
  H5::Group root = f.openGroup("/");
  const int schema = int(h5::read_attr_dbl(root, "dp5_runcard_schema"));
  if (schema != kRunCardSchema) throw std::runtime_error("run card schema not supported");
  H5::Group g = f.openGroup("/run");
  for (int i = 0; i < g.getNumAttrs(); ++i) {
    const std::string k = g.openAttribute(unsigned(i)).getName();
    if (!kRunKeys.count(k)) throw std::runtime_error("run card: unknown key '" + k + "' in /run (not supported by this release)");
  }
  for (hsize_t i = 0; i < g.getNumObjs(); ++i) {
    const std::string k = g.getObjnameByIdx(i);
    if (k != "rho0") throw std::runtime_error("run card: unknown object '/run/" + k + "' (only the dataset rho0)");
  }
  auto str = [&](const char* n, std::string& v) { if (g.attrExists(n)) v = h5::read_attr_str(g, n); };
  auto dbl = [&](const char* n, double& v) { if (g.attrExists(n)) v = h5::read_attr_dbl(g, n); };
  auto pair = [&](const char* n, double* v) {
    if (!g.attrExists(n)) return;
    auto x = h5::read_attr_dvec(g, n);
    if (x.size() != 2) throw std::runtime_error(std::string("run card: ") + n + " needs 2 values");
    v[0] = x[0]; v[1] = x[1];
  };
  r.model_path = h5::read_attr_str(g, "model_path");
  r.model_hash = h5::read_attr_str(g, "model_hash");
  r.gamma = h5::read_attr_dbl(g, "gamma");
  r.approx = parse_approx(h5::read_attr_str(g, "approximation"));
  r.mode = parse_mode(h5::read_attr_str(g, "mode"));
  std::string s;
  s = "full"; str("spin_mode", s); r.spin = parse_spin(s);
  s = "general"; str("vertex_impl", s); r.vertex = parse_vertex(s);
  double b = 0; dbl("flag_2B", b); r.flag_2B = b != 0;
  double bp = 0; dbl("bare_propagators", bp); r.bare_propagators = bp != 0;
  dbl("t0", r.t0);
  r.t_end = h5::read_attr_dbl(g, "t_end");
  r.dt_out = h5::read_attr_dbl(g, "dt_out");
  if (g.attrExists("t_prime")) {
    if (h5::attr_is_string(g, "t_prime")) {
      if (h5::read_attr_str(g, "t_prime") != "auto") throw std::runtime_error("run card: t_prime must be a number or \"auto\"");
      r.tprime_auto = true;
    } else {
      r.t_prime = h5::read_attr_dvec(g, "t_prime");
      if (r.t_prime.size() != 1) throw std::runtime_error("run card: t_prime must be one number (or \"auto\")");
    }
  }
  dbl("tol_ss", r.tol_ss); dbl("c_hold", r.c_hold); dbl("c_min", r.c_min); dbl("c_max", r.c_max);
  dbl("ss_fit_rtol", r.ss_fit_rtol); dbl("ss_window_c", r.ss_window_c); str("ss_on_timeout", r.ss_on_timeout); 
  if (g.attrExists("rtde_tau") && h5::attr_is_string(g, "rtde_tau")) {
    if (h5::read_attr_str(g, "rtde_tau") != "auto") throw std::runtime_error("run card: rtde_tau must be a number or \"auto\"");
    r.rtde_tau_auto = true;
  } else {
    dbl("rtde_tau", r.rtde_tau);
  }
  dbl("tol_A", r.tol_A); dbl("rtde_tau_min", r.rtde_tau_min); dbl("c_tau_min", r.c_tau_min); dbl("rtde_tau_max", r.rtde_tau_max);
  str("rtde_monitor", r.rtde_monitor);
  { double hr = r.ss_hold_records ? 1 : 0; dbl("ss_hold_records", hr); r.ss_hold_records = hr != 0; }
  dbl("rtde_window_beats", r.rtde_window_beats);
  if (r.rtde_window_beats < 0) throw std::runtime_error("run card: rtde_window_beats >= 0");
  if (g.attrExists("probe_columns")) {
    r.probe_columns.clear();
    for (double x : h5::read_attr_dvec(g, "probe_columns")) r.probe_columns.push_back((long long)x);
  }
  pair("tol_rho", r.tol_rho); pair("tol_tensor", r.tol_tensor); pair("tol_rtde", r.tol_rtde);
  dbl("checkpoint_interval", r.checkpoint_interval);
  { double k = 0; dbl("checkpoint_keep", k); r.checkpoint_keep = (long long)k; }
  dbl("diverge_rho_lo", r.diverge_rho_lo); dbl("diverge_rho_hi", r.diverge_rho_hi);
  { double st = 1; dbl("rho_stride", st); r.rho_stride = (long long)st; if (r.rho_stride < 1) throw std::runtime_error("run card: rho_stride >= 1"); }
  str("ic_mode", r.ic_mode);
  if (r.ic_mode == "full") r.rho0_override = h5::read_cplx(g, "rho0");
  double th = 0; dbl("threads", th); r.threads = int(th);
  // validation (D4 of the review)
  if (!(r.gamma >= 0)) throw std::runtime_error("run card: gamma must be >= 0");
  if (r.bare_propagators && r.approx == Approx::HF)
    throw std::runtime_error("run card: bare_propagators = 1 is not defined for approximation HF (HF has no correlator; "
                             "the flag only removes l^{>,MF} from the drifts of the correlated objects)");
  if (!(r.dt_out > 0) || !(r.t_end > r.t0)) throw std::runtime_error("run card: need dt_out > 0, t_end > t0");
  const double nint = (r.t_end - r.t0) / r.dt_out;
  if (std::fabs(nint - std::round(nint)) > 1e-9 * std::max(1.0, nint))
    throw std::runtime_error("run card: (t_end - t0)/dt_out must be an integer");
  for (double tp : r.t_prime)  // (with rtde_tau = auto the run ends at t' + tau_end <= t' + rtde_tau_max; t_end unused)
    if (!(tp >= r.t0 && (tp < r.t_end || r.rtde_tau_auto))) throw std::runtime_error("run card: t_prime outside [t0, t_end)");

  if (r.probe_columns.size() != 1) throw std::runtime_error("run card: probe_columns must hold exactly one column");
  if (r.ic_mode != "model" && r.ic_mode != "full")
    throw std::runtime_error("run card: ic_mode must be model or full");
  if (r.ic_mode != "full" && g.exists("rho0")) throw std::runtime_error("run card: dataset rho0 given but ic_mode is not full");
  return r;
}

}  // namespace drv
