// physics.hpp -- block layouts per approximation, the vertex-term interface, and the shared
// right-hand side. Step 2: G1-G2 for HF, TPPX (+2B), mGW (with counterterm), DSL (each optionally with bare
// propagators, run card bare_propagators); lattice
// (general lossy-site set) and impurity fast path. RTDE tensor terms: NotImplemented (step 3).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common.hpp"
#include "inputs.hpp"
#include "observables.hpp"

namespace drv {

// ------------------------------------------------------------------ layouts
struct LayoutConfig {
  Approx approx = Approx::HF;
  SpinMode spin = SpinMode::full;
  int N = 0;
  double tol_rho[2], tol_tensor[2], tol_rtde[2];
};
int nspin(SpinMode s);
std::vector<dp5::BlockSpec> g1g2_blocks(const LayoutConfig& c);
std::vector<dp5::BlockSpec> rtde_blocks(const LayoutConfig& c, int k, long long s);
std::string describe_layout(const LayoutConfig& c);
// tensor blocks the history integral I reads (indices into the tensor list, i.e. g1g2 order minus rho)
std::vector<int> history_blocks(Approx a);

// ------------------------------------------------------------------ vertex terms
struct TensorContext {
  Approx approx; bool flag_2B; int N, ns;
  const cplx* rho;                 // [ns,N,N] rho^<
  const Mat* ho_pair;              // h_o in the drifts of the correlators (bare_propagators: without l^{>,MF})
  std::vector<const cplx*> X;      // tensor blocks in g1g2_blocks order (without rho)
  std::vector<cplx*> dX;
  cplx* tmp;                       // N^4 scratch (paramagnetic same-spin images)
};
// One RTDE block set (probe column j). Tensors stored in the slot order F^{pp}_{ablj} with the probe
// j fixed (column reduction, R7): each block is an N^3 array [a][b][l].
struct RtdeContext {
  Approx approx; bool flag_2B; int N, ns;
  const cplx* rho;                 // [ns,N,N] rho^<
  const Mat* ho_pair;              // h_o on every leg at time t of the RTDE tensors (bare_propagators: no l^{>,MF})
  const cplx* Kslice;              // [n_lossy][N*N]: G^{pp,udud}_{a b r r} per lossy site r (null for 2B)
  const cplx* Wslice = nullptr;    // mGW: [6][n_lossy][N][N] G1-G2 slices (see Physics::mgw_slices)
  cplx* tmp3 = nullptr;            // mGW paramagnetic: N^3 scratch (same-spin block = A + P)
  const cplx* GR; cplx* dGR;       // [ns][N] = G^{R sigma}_{x j}; the vertex adds -i I to dGR
  long long column;
  std::vector<const cplx*> F;      // RTDE tensor blocks in rtde_blocks order (after GR)
  std::vector<cplx*> dF;
};

class VertexTerms {
 public:
  virtual ~VertexTerms() = default;
  virtual const char* name() const = 0;
  // h_o^sigma += -i l^{>,MF,sigma}, l^{>,MF,sigma}_{rr} = gamma w_r Re rho^{sigma-bar}_{rr} on lossy sites (A1)
  virtual void add_mf_loss(const cplx* rho_opp, int N, cplx* ho) const = 0;
  // G1-G2 tensor right-hand sides (assign dX), all approximations
  virtual void g1g2_tensor_terms(const TensorContext&) const = 0;
  // history integral I^sigma [ns,N,N] (assigned); tensors in g1g2 order
  virtual void history_integral(Approx a, int N, int ns, const std::vector<const cplx*>& X, cplx* I) const = 0;
  // ---- step 3
  virtual void rtde_tensor_terms(const RtdeContext&) const {
    throw NotImplemented(std::string(name()) + ": RTDE tensor terms (step 3)");
  }
  virtual void history_integral_R(const RtdeContext&, cplx*) const {
    throw NotImplemented(std::string(name()) + ": retarded history integral I^R (step 3)");
  }
};
std::unique_ptr<VertexTerms> make_vertex(const Model& m, VertexImpl v, double gamma);

// ------------------------------------------------------------------ physics
struct RtdeSet {
  int tp_index = 0;
  long long column = 0;
  std::size_t idGR = 0;
  std::vector<std::size_t> ids;
};

struct FrameObservables {
  std::vector<double> D;      // double occupancy per lossy site
  double loss_rate = 0;       // -4 gamma sum_r w_r D_r  (two-body part of dN/dt)
  double dN_dt = 0;           // Re tr d rho/dt (both spins; includes one-body bath terms)
  double energy = 0;          // sum_sigma Re tr h^sigma rho^sigma (bare one-body h)
  double rho_dot_max = 0;
  double rho_eig_min = 0, rho_eig_max = 0;  // extremes of the spectra of rho^sigma (Hermitian part)
  EqObs eq;                   // equal-time observables (step 3b)
};

class Physics {
 public:
  Physics(const Model& m, const RunCard& rc, std::unique_ptr<VertexTerms> v);
  int N = 0, ns = 2;
  double gamma = 0;
  Approx approx = Approx::HF;
  bool flag_2B = false;
  bool bare_propagators = false;  // drifts of the correlated objects with the bare h_o (rho, G^R keep h_o^MF)
  static constexpr std::size_t npos = std::size_t(-1);
  std::size_t id_rho = npos;
  // mGW RTDE inputs from the G1-G2 tensors (g1g2 order; full: P, A, UU, DD, P2; para: P, A, P2), per lossy
  // site q = r_k:  0: A[p,q,q,o]  1: P[p,q,q,o]  2: DD[q,p,o,q]  3: A[q,p,o,q]  4: UU[q,p,o,q]  5: P[q,p,o,q],
  // stored [slice][k][p][o]  (paramagnetic: UU = DD = A + P)
  static void mgw_slices(const std::vector<const cplx*>& X, bool full, int N, const std::vector<long long>& sites,
                         std::vector<cplx>& out);
  // DSL RTDE inputs (tensors: full G_opp (P), G_UU, G_DD packed (N > 1); para P), per lossy site v:
  //  0: P[x,y,v,v]  1: P[x,v,v,y]  2: P[v,x,y,v]  3: DD[v,x,y,v]  4: UU[v,x,y,v]  5: P[x,v,y,v]  6: P[v,x,v,y]
  // stored [slice][k][x][y]   (paramagnetic: UU = DD = P - P_slba)
  static void dsl_slices(const std::vector<const cplx*>& X, bool full, int N, const std::vector<long long>& sites,
                         std::vector<cplx>& out);
  const std::vector<long long>& lossy_sites() const { return sites_; }
  std::vector<std::size_t> tensor_ids;  // g1g2 tensor blocks (order of g1g2_blocks without rho)
  std::vector<RtdeSet> rtde;
  const VertexTerms& vertex() const { return *vertex_; }
  const cplx* history_I() const { return I_.data(); }  // valid after rho_dot / observables

  void build_ho(const cplx* rho, Mat* ho) const;
  // Reference rates of the one-body generator (steady-state detection): eigenvalues e_k of h_o^MF(rho) and of the
  // bare h_o (no MF), all spins; lam = min_k |Im e_k|, spread = max Re e_k - min Re e_k (MF-dressed).
  struct RefRates { double lam_mf = 0, lam_bare = 0, spread = 0, lam_max = 0; };  // lam_max = max_k |Im e_k| (MF)
  RefRates ref_rates(const cplx* rho) const;
  // full d rho/dt: HF part + I + I^dag (I from tensors X, may be empty for HF)
  void rho_dot(const cplx* rho, const Mat* ho, const std::vector<const cplx*>& X, cplx* drho) const;
  void observables(const cplx* rho, const std::vector<const cplx*>& X, FrameObservables& o) const;
  void operator()(double t, const cplx* y, cplx* dy, const dp5::Layout& L);

 private:
  Mat hbase_o_[2], two_lt_[2], h_[2];
  std::vector<long long> sites_;
  std::vector<double> w_;
  std::vector<long long> bonds_;
  std::unique_ptr<VertexTerms> vertex_;
  Mat ho_[2];
  mutable std::vector<cplx> I_, tmp_, kslice_, wslice_, tmp3_;
};

}  // namespace drv
