// output.hpp -- one HDF5 output file per run (README "Output file, schema 1").
#pragma once

#include <map>
#include <functional>
#include <string>
#include <vector>

#include "common.hpp"
#include "h5util.hpp"
#include "inputs.hpp"
#include "physics.hpp"

namespace drv {

// Metadata stored with every checkpoint (block layout and run identity; readers use block_names/block_sizes).
struct CheckpointMeta {
  std::string approximation, spin_mode, vertex_impl, model_hash;
  long long N = 0, flag_2B = 0, bare_propagators = 0;
  std::vector<std::string> block_names;
  std::vector<long long> activated_tp;   // indices k of t'_k already activated
  std::vector<double> t_prime;
  std::vector<long long> probe_columns;
};

class Output {
 public:
  Output(const std::string& path, const std::string& run_id);
  const std::string& path() const { return path_; }
  void write_input(const RunCard& rc, const Model& m, const std::map<std::string, std::string>& overrides,
                   const std::string& cmdline);
  void write_provenance(int threads);
  void init_series(int N, int ns, const std::vector<long long>& lossy_sites, long long rho_stride);
  void frame(double t, const cplx* rho, const FrameObservables& o);
  void init_rtde(int k, double tp, const std::vector<long long>& cols, int N, int ns, const std::string& note = "");
  void rtde_frame(int k, double t, const cplx* row);
  void checkpoint(const std::string& group, const std::string& name, const dp5::Checkpoint& cp, const cplx* y,
                  std::size_t n, const CheckpointMeta& meta);
  void status(const std::string& status, const std::string& message, double t, double wall_seconds,
              const dp5::Stats& st);
  void profile(const dp5::Profile& p);
  // attributes on a group (created if missing)
  void write_attrs(const std::string& group, const std::vector<std::pair<std::string, double>>& d,
                   const std::vector<std::pair<std::string, std::string>>& s,
                   const std::vector<std::pair<std::string, std::vector<double>>>& v);
  void flush();
  // remove /checkpoints/<name> if present (its file space is reused by later checkpoints of the same run)
  void drop_checkpoint(const std::string& name);
  // 1-d double dataset (replaced if present) in a group created if missing
  void write_series(const std::string& group, const std::string& name, const std::vector<double>& v);

 private:
  std::string path_;
  H5::H5File f_;
  int N_ = 0, ns_ = 2;
  h5::Appender t_, rho_, rho_t_, nd_, npart_, rhodot_, D_, loss_, dndt_, energy_, eigmin_, eigmax_;
  // run-wide diagnostics, written to /status (production monitors; see README "Diagnostics")
  double diag_max_dN_dt = -1e300, diag_max_loss_rate = -1e300, diag_min_D = 1e300, diag_min_eig = 1e300,
         diag_max_eig = -1e300, diag_t_min_D = 0, diag_t_max_loss = 0;
  bool have_frames_ = false;
  long long rho_stride_ = 1, frame_count_ = 0;
  std::size_t nsites_ = 0;
  struct Rt { h5::Appender t, gr; std::size_t ncol = 0; };
  std::map<int, Rt> rt_;
  std::size_t frames_since_flush_ = 0;
  std::vector<h5::Appender> eq_;  // /observables/equal_time series, created at the first frame
};

}  // namespace drv
