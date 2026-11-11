#include "output.hpp"

#include <unistd.h>

#include <algorithm>
#include <ctime>
#include <sstream>

#include "version.hpp"

namespace drv {

static std::string join(const std::vector<std::string>& v, char sep = ',') {
  std::string o;
  for (std::size_t i = 0; i < v.size(); ++i) { if (i) o += sep; o += v[i]; }
  return o;
}
static std::string now_iso() {
  char b[64];
  std::time_t t = std::time(nullptr);
  std::strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
  return b;
}

Output::Output(const std::string& path, const std::string& run_id) : path_(path) {
  // fail fast: exclusive create (never overwrite an existing run)
  f_ = H5::H5File(path, H5F_ACC_EXCL);
  H5::Group root = f_.openGroup("/");
  h5::attr_int(root, "dp5_driver_output_schema", kOutputSchema);
  h5::attr_str(root, "run_id", run_id);
  h5::attr_str(root, "created", now_iso());
  h5::attr_str(root, "release_version", release_version());
  for (const char* g : {"input", "time", "rho", "observables", "rtde", "checkpoints", "final", "status", "profile", "provenance"})
    f_.createGroup(g);
  f_.flush(H5F_SCOPE_GLOBAL);
}

void Output::write_input(const RunCard& rc, const Model& m, const std::map<std::string, std::string>& ov,
                         const std::string& cmdline) {
  H5::Group g = f_.openGroup("/input");
  {  // exact copy of the run card's /run group
    H5::H5File rf(rc.path, H5F_ACC_RDONLY);
    H5Ocopy(rf.getId(), "/run", g.getId(), "run", H5P_DEFAULT, H5P_DEFAULT);
  }
  h5::attr_str(g, "runcard_path", rc.path);
  h5::attr_str(g, "model_path", m.path);
  h5::attr_str(g, "model_hash", m.hash_computed);
  h5::attr_int(g, "N", m.N);
  for (auto& kv : m.attrs) h5::attr_str(g, "model_" + kv.first, kv.second);
  std::string o;
  for (auto& kv : ov) o += kv.first + "=" + kv.second + ";";
  h5::attr_str(g, "cli_overrides", o);
  h5::attr_str(g, "command_line", cmdline);
  // self-describing method (the run card copy /input/run holds bare_propagators only if the card sets it)
  const std::string label = method_label(rc.approx, rc.flag_2B, rc.bare_propagators);
  h5::attr_str(g, "approximation", name(rc.approx));
  h5::attr_int(g, "flag_2B", rc.flag_2B);
  h5::attr_int(g, "bare_propagators", rc.bare_propagators);
  h5::attr_str(g, "method", label);
  H5::Group gp = f_.openGroup("/provenance");
  h5::attr_str(gp, "method", label);
  h5::attr_int(gp, "bare_propagators", rc.bare_propagators);
  h5::attr_str(gp, "bare_propagators_note", "1: correlated objects (G1-G2 correlators, RTDE tensors) propagated with the "
                                            "bare h_o = h - i(l^> + l^<); rho and G^R keep h_o^MF (label X_0; TPPX_0 = former BARE)");
  // effective settings after overrides
  h5::attr_int(g, "threads_effective", rc.threads);
}

void Output::write_provenance(int threads) {
  H5::Group g = f_.openGroup("/provenance");
  h5::attr_str(g, "release_version", release_version());
  h5::attr_str(g, "git_describe", git_describe());
  h5::attr_str(g, "git_hash", git_describe());  // name kept from the development driver's output schema
  h5::attr_str(g, "build_flags", build_flags());
  h5::attr_str(g, "compiler", compiler());
  h5::attr_str(g, "eigen_version", std::to_string(EIGEN_WORLD_VERSION) + "." + std::to_string(EIGEN_MAJOR_VERSION) +
                                       "." + std::to_string(EIGEN_MINOR_VERSION));
  unsigned maj, min, rel;
  H5get_libversion(&maj, &min, &rel);
  h5::attr_str(g, "hdf5_version", std::to_string(maj) + "." + std::to_string(min) + "." + std::to_string(rel));
  char host[256] = {0};
  gethostname(host, sizeof host - 1);
  h5::attr_str(g, "host", host);
  h5::attr_int(g, "threads", threads);
  h5::attr_int(g, "pid", getpid());
  h5::attr_int(g, "integrator_profile_schema", dp5::Profile::schema_version);
}

void Output::init_series(int N, int ns, const std::vector<long long>& sites, long long rho_stride) {
  N_ = N; ns_ = ns; rho_stride_ = rho_stride; nsites_ = sites.size();
  H5::Group gt = f_.openGroup("/time"), gr = f_.openGroup("/rho"), go = f_.openGroup("/observables");
  t_ = h5::Appender(gt, "t", h5::Appender::Double, {});
  rho_ = h5::Appender(gr, "rho", h5::Appender::Complex, {hsize_t(ns), hsize_t(N), hsize_t(N)}, 16);
  rho_t_ = h5::Appender(gr, "t", h5::Appender::Double, {});
  h5::attr_int(gr, "rho_stride", rho_stride);
  nd_ = h5::Appender(go, "n_d", h5::Appender::Double, {hsize_t(ns)});
  npart_ = h5::Appender(go, "N", h5::Appender::Double, {hsize_t(ns)});
  rhodot_ = h5::Appender(go, "rho_dot_max", h5::Appender::Double, {});
  D_ = h5::Appender(go, "D", h5::Appender::Double, {hsize_t(sites.size())});
  loss_ = h5::Appender(go, "loss_rate", h5::Appender::Double, {});
  dndt_ = h5::Appender(go, "dN_dt", h5::Appender::Double, {});
  energy_ = h5::Appender(go, "energy", h5::Appender::Double, {});
  eigmin_ = h5::Appender(go, "rho_eig_min", h5::Appender::Double, {});
  eigmax_ = h5::Appender(go, "rho_eig_max", h5::Appender::Double, {});
  h5::attr_str(go, "rho_eig_min", "min over spins of the smallest eigenvalue of rho^sigma (physical: >= 0) [T]");
  h5::attr_str(go, "rho_eig_max", "max over spins of the largest eigenvalue of rho^sigma (physical: <= 1) [T]");
  h5::attr_ivec(go, "lossy_sites", sites);
  h5::attr_str(go, "n_d", "Re rho^sigma_00 [T, nspin]");
  h5::attr_str(go, "N", "Re tr rho^sigma [T, nspin]");
  h5::attr_str(go, "rho_dot_max", "max_ij |d rho/dt| including I (convergence monitor) [T]");
  h5::attr_str(go, "D", "double occupancy <n_r,up n_r,dn> = rho^u_rr rho^d_rr - Re I^u_rr / (gamma w_r) per lossy site [T, M]");
  h5::attr_str(go, "loss_rate", "-4 gamma sum_r w_r D_r (two-body part of dN/dt, both spins) [T]");
  h5::attr_str(go, "dN_dt", "Re tr d rho/dt summed over spins (includes one-body bath terms) [T]");
  h5::attr_str(go, "energy", "sum_sigma Re tr h^sigma rho^sigma (bare one-body h; no two-body Hamiltonian) [T]");
}

void Output::frame(double t, const cplx* rho, const FrameObservables& o) {
  const std::size_t n2 = std::size_t(N_) * N_;
  std::vector<double> nd(ns_), np(ns_);
  for (int s = 0; s < ns_; ++s) {
    nd[s] = rho[s * n2].real();
    double tr = 0;
    for (int x = 0; x < N_; ++x) tr += rho[s * n2 + std::size_t(x) * N_ + x].real();
    np[s] = tr;
  }
  t_.append(&t);
  if (frame_count_ % rho_stride_ == 0) { rho_.append(rho); rho_t_.append(&t); }
  ++frame_count_;
  nd_.append(nd.data());
  npart_.append(np.data());
  rhodot_.append(&o.rho_dot_max);
  if (nsites_) D_.append(o.D.data());
  loss_.append(&o.loss_rate);
  dndt_.append(&o.dN_dt);
  energy_.append(&o.energy);
  eigmin_.append(&o.rho_eig_min);
  eigmax_.append(&o.rho_eig_max);
  if (!o.eq.series.empty()) {
    if (eq_.empty()) {
      H5::Group go = f_.openGroup("/observables");
      H5::Group ge = h5::group(go, "equal_time");
      for (const auto& se : o.eq.series) {
        std::vector<hsize_t> d(se.dims.begin(), se.dims.end());
        eq_.emplace_back(ge, se.name, h5::Appender::Double, d);
        h5::attr_str(ge, se.name, se.desc);
      }
      for (const auto& a : o.eq.attrs) h5::attr_str(ge, "_" + a.first, a.second);
      h5::attr_ivec(ge, "_bonds", o.eq.bonds);
    }
    if (eq_.size() != o.eq.series.size()) throw std::runtime_error("equal-time observables changed shape");
    for (std::size_t i = 0; i < eq_.size(); ++i) eq_[i].append(o.eq.series[i].v.data());
  }
  have_frames_ = true;
  diag_max_dN_dt = std::max(diag_max_dN_dt, o.dN_dt);
  if (o.loss_rate > diag_max_loss_rate) { diag_max_loss_rate = o.loss_rate; diag_t_max_loss = t; }
  for (double d : o.D)
    if (d < diag_min_D) { diag_min_D = d; diag_t_min_D = t; }
  diag_min_eig = std::min(diag_min_eig, o.rho_eig_min);
  diag_max_eig = std::max(diag_max_eig, o.rho_eig_max);
  if (++frames_since_flush_ >= 100) flush();
}

void Output::init_rtde(int k, double tp, const std::vector<long long>& cols, int N, int ns, const std::string& note) {
  H5::Group r = f_.openGroup("/rtde");
  H5::Group g = h5::group(r, "tp=" + std::to_string(k));
  h5::attr_dbl(g, "t_prime", tp);
  h5::attr_ivec(g, "probe_columns", cols);
  h5::attr_str(g, "GR_layout", "[T, column, spin, x] = G^R_{x s}(t, t'), s = probe_columns[column]");
  if (!note.empty()) h5::attr_str(g, "note", note);
  Rt rt;
  rt.ncol = cols.size();
  rt.t = h5::Appender(g, "t", h5::Appender::Double, {});
  rt.gr = h5::Appender(g, "GR", h5::Appender::Complex, {hsize_t(cols.size()), hsize_t(ns), hsize_t(N)});
  rt_[k] = std::move(rt);
}

void Output::rtde_frame(int k, double t, const cplx* row) {
  Rt& r = rt_.at(k);
  r.t.append(&t);
  r.gr.append(row);
}

void Output::checkpoint(const std::string& group, const std::string& name, const dp5::Checkpoint& cp, const cplx* y,
                        std::size_t n, const CheckpointMeta& m) {
  H5::Group base = f_.openGroup(group);
  H5::Group g = name.empty() ? base : h5::group(base, name);
  h5::write_cplx(g, "y", y, {hsize_t(n)});
  h5::attr_dbl(g, "t", cp.t);
  h5::attr_dbl(g, "h", cp.h);
  h5::attr_dbl(g, "err_old", cp.err_old);
  h5::attr_int(g, "grid_next", (long long)cp.grid_next);
  std::vector<long long> sizes, act, rem;
  for (std::size_t i = 0; i < cp.sizes.size(); ++i) {
    sizes.push_back((long long)cp.sizes[i]); act.push_back(cp.active[i]); rem.push_back(cp.removed[i]);
  }
  h5::attr_ivec(g, "block_sizes", sizes);
  h5::attr_ivec(g, "block_active", act);
  h5::attr_ivec(g, "block_removed", rem);
  h5::attr_str(g, "block_names", join(m.block_names));
  h5::attr_ivec(g, "stats", {(long long)cp.stats.n_accept, (long long)cp.stats.n_reject, (long long)cp.stats.n_rhs,
                             (long long)cp.stats.n_nonfinite, (long long)cp.stats.n_frames, (long long)cp.stats.n_stops});
  h5::attr_str(g, "approximation", m.approximation);
  h5::attr_str(g, "spin_mode", m.spin_mode);
  h5::attr_str(g, "vertex_impl", m.vertex_impl);
  h5::attr_str(g, "model_hash", m.model_hash);
  h5::attr_int(g, "N", m.N);
  h5::attr_int(g, "flag_2B", m.flag_2B);
  h5::attr_int(g, "bare_propagators", m.bare_propagators);
  h5::attr_ivec(g, "activated_tp", m.activated_tp);
  h5::attr_dvec(g, "t_prime", m.t_prime);
  h5::attr_ivec(g, "probe_columns", m.probe_columns);
  h5::attr_int(g, "layout_entries", (long long)n);
  flush();
}

void Output::status(const std::string& status, const std::string& message, double t, double wall,
                    const dp5::Stats& st) {
  H5::Group g = f_.openGroup("/status");
  h5::attr_str(g, "status", status);
  h5::attr_str(g, "message", message);
  h5::attr_dbl(g, "t_reached", t);
  h5::attr_dbl(g, "wall_seconds", wall);
  h5::attr_int(g, "accepted_steps", (long long)st.n_accept);
  h5::attr_int(g, "rejected_steps", (long long)st.n_reject);
  h5::attr_int(g, "rhs_calls", (long long)st.n_rhs);
  h5::attr_str(g, "finished", now_iso());
  if (have_frames_) {  // diagnostics over all frames (reported, never enforced)
    h5::attr_dbl(g, "diag_max_dN_dt", diag_max_dN_dt);
    h5::attr_dbl(g, "diag_max_two_body_dN_dt", diag_max_loss_rate);
    h5::attr_dbl(g, "diag_t_max_two_body_dN_dt", diag_t_max_loss);
    h5::attr_dbl(g, "diag_min_D", nsites_ ? diag_min_D : 0.0);
    h5::attr_dbl(g, "diag_t_min_D", diag_t_min_D);
    h5::attr_dbl(g, "diag_min_rho_eig", diag_min_eig);
    h5::attr_dbl(g, "diag_max_rho_eig", diag_max_eig);
    std::string w;
    if (diag_max_loss_rate > 1e-12) w += "two-body dN/dt > 0 at some frames (D < 0); ";
    if (diag_min_eig < -1e-10) w += "rho has negative eigenvalues; ";
    if (diag_max_eig > 1 + 1e-10) w += "rho has eigenvalues > 1; ";
    h5::attr_str(g, "diag_warnings", w.empty() ? "none" : w);
  }
  flush();
}

void Output::profile(const dp5::Profile& p) {
  H5::Group g = f_.openGroup("/profile");
  p.visit([&](const std::string& name, const double* d, std::size_t n) {
    // "a/b/c" -> group /profile/a/b, dataset c
    H5::Group cur = g;
    std::size_t pos = 0, nx;
    while ((nx = name.find('/', pos)) != std::string::npos) {
      cur = h5::group(cur, name.substr(pos, nx - pos));
      pos = nx + 1;
    }
    std::vector<double> buf(d, d + n);
    if (n == 0) { h5::write_dbl(cur, name.substr(pos), buf.data(), {0}); return; }
    h5::write_dbl(cur, name.substr(pos), buf.data(), {hsize_t(n)});
  });
  for (auto& kv : p.strings()) {
    std::string k = kv.first;
    for (auto& c : k) if (c == '/') c = '.';
    h5::attr_str(g, k, kv.second);
  }
  h5::attr_int(g, "schema_version", dp5::Profile::schema_version);
  h5::attr_str(g, "text", p.to_text());
  flush();
}

void Output::flush() {
  f_.flush(H5F_SCOPE_GLOBAL);
  frames_since_flush_ = 0;
}

}  // namespace drv

namespace drv {
void Output::write_attrs(const std::string& group, const std::vector<std::pair<std::string, double>>& d,
                         const std::vector<std::pair<std::string, std::string>>& s,
                         const std::vector<std::pair<std::string, std::vector<double>>>& v) {
  H5::Group g = h5::group(f_.openGroup("/"), group);
  for (auto& x : d) h5::attr_dbl(g, x.first, x.second);
  for (auto& x : s) h5::attr_str(g, x.first, x.second);
  for (auto& x : v) h5::attr_dvec(g, x.first, x.second);
  f_.flush(H5F_SCOPE_GLOBAL);
}
}  // namespace drv

namespace drv {
void Output::write_series(const std::string& group, const std::string& name, const std::vector<double>& v) {
  H5::Group g = h5::group(f_.openGroup("/"), group);
  h5::write_dbl(g, name, v.data(), {hsize_t(v.size())});
  f_.flush(H5F_SCOPE_GLOBAL);
}
}  // namespace drv

namespace drv {
void Output::drop_checkpoint(const std::string& name) {
  H5::Group g = f_.openGroup("/checkpoints");
  if (h5::exists(g, name)) g.unlink(name);
}
}  // namespace drv
