#include "physics.hpp"

#include <iostream>

#include <limits>
#include <sstream>

#include "tensor.hpp"

namespace drv {

int nspin(SpinMode s) { return s == SpinMode::full ? 2 : 1; }

static dp5::BlockSpec obs(const std::string& n, std::size_t sz, const double* tol) {
  return dp5::observable_block(n, sz, tol[0], tol[1]);
}
static dp5::BlockSpec ten(const std::string& n, std::size_t sz, const double* tol) {
  return dp5::tensor_block(n, sz, tol[0], tol[1]);
}

std::vector<dp5::BlockSpec> g1g2_blocks(const LayoutConfig& c) {
  const std::size_t N = std::size_t(c.N), N2 = N * N, N4 = N2 * N2;
  const bool full = c.spin == SpinMode::full;
  std::vector<dp5::BlockSpec> b{obs("rho", std::size_t(nspin(c.spin)) * N2, c.tol_rho)};
  switch (c.approx) {
    case Approx::HF: break;
    case Approx::TPPX: b.push_back(ten("Kpp", N4, c.tol_tensor)); break;  // G^{pp,udud}_{slab}
    case Approx::mGW:
      b.push_back(ten("GW_P", N4, c.tol_tensor));   // G^{W,udud}_{slab}
      b.push_back(ten("GW_A", N4, c.tol_tensor));   // G^{W,uddu}
      if (full) { b.push_back(ten("GW_UU", N4, c.tol_tensor)); b.push_back(ten("GW_DD", N4, c.tol_tensor)); }
      b.push_back(ten("GW2_P", N4, c.tol_tensor));  // counterterm G^{W,2,udud}
      break;
    case Approx::DSL:
      b.push_back(ten("G_opp", N4, c.tol_tensor));  // G^{udud}_{slab}
      if (full && N > 1) {  // same-spin blocks, packed: independent entries s<l, a<b only (A^2 each)
        const std::size_t A = N * (N - 1) / 2;
        b.push_back(ten("G_UU", A * A, c.tol_tensor));
        b.push_back(ten("G_DD", A * A, c.tol_tensor));
      }
      break;
  }
  return b;
}

std::vector<int> history_blocks(Approx a) {
  switch (a) {
    case Approx::HF: return {};
    case Approx::TPPX:
    case Approx::DSL: return {0};
    case Approx::mGW: return {0, 1, -1};  // P, A, P2 (P2 = last block; resolved by caller)
  }
  return {};
}

std::vector<dp5::BlockSpec> rtde_blocks(const LayoutConfig& c, int k, long long s) {
  const std::size_t N = std::size_t(c.N), N3 = N * N * N, A = N * (N - 1) / 2;
  const bool full = c.spin == SpinMode::full;
  const std::string sfx = "_tp" + std::to_string(k) + "_c" + std::to_string(s);
  std::vector<dp5::BlockSpec> b{obs("GR" + sfx, std::size_t(nspin(c.spin)) * N, c.tol_rtde)};
  switch (c.approx) {
    case Approx::HF: break;
    case Approx::TPPX:
      b.push_back(ten("Dpp" + sfx, N3, c.tol_rtde));
      if (full) b.push_back(ten("Upp" + sfx, N3, c.tol_rtde));
      break;
    case Approx::mGW:
      if (full) b.push_back(ten("FW_UU" + sfx, N3, c.tol_rtde));
      b.push_back(ten("FW_UA" + sfx, N3, c.tol_rtde));
      b.push_back(ten("FW_UP" + sfx, N3, c.tol_rtde));
      b.push_back(ten("FW2_UP" + sfx, N3, c.tol_rtde));
      if (full) {
        b.push_back(ten("FW_DD" + sfx, N3, c.tol_rtde)); b.push_back(ten("FW_DA" + sfx, N3, c.tol_rtde));
        b.push_back(ten("FW_DP" + sfx, N3, c.tol_rtde)); b.push_back(ten("FW2_DP" + sfx, N3, c.tol_rtde));
      }
      break;
    case Approx::DSL:
      b.push_back(ten("F_opp_u" + sfx, N3, c.tol_rtde));
      if (full) {
        if (A > 0) b.push_back(ten("F_UU" + sfx, N * A, c.tol_rtde));
        b.push_back(ten("F_opp_d" + sfx, N3, c.tol_rtde));
        if (A > 0) b.push_back(ten("F_DD" + sfx, N * A, c.tol_rtde));
      }
      break;
  }
  return b;
}

std::string describe_layout(const LayoutConfig& c) {
  std::ostringstream o;
  std::size_t tot = 0;
  o << "approximation " << name(c.approx) << ", spin " << name(c.spin) << ", N = " << c.N << "\n  G1-G2:";
  for (auto& b : g1g2_blocks(c)) { o << " " << b.name << "[" << b.size << "]"; tot += b.size; }
  o << "  (" << tot << " entries, " << 8.0 * 16.0 * double(tot) / 1e6 << " MB in 8 registers)\n  RTDE per column:";
  std::size_t tr = 0;
  for (auto& b : rtde_blocks(c, 0, 0)) { o << " " << b.name << "[" << b.size << "]"; tr += b.size; }
  o << "  (" << tr << " entries)\n";
  return o.str();
}

// ------------------------------------------------------------------ DSL RTDE (shared by both site sets)
  // DSL RTDE, slot order F_{ablj} (sec:DSLRTDElat eqs. DSLRlat-D..GRd, impurity DSLRimp-D..Sd). Column j fixed;
// per probe spin sigma: O = opposite-spin block (sigma bar bar sigma) [a][b][l] (N^3), S = same-spin block,
// xi-antisymmetric in (a,b), stored packed S[pair(a<b)][l] (N A). The loop index k below runs over slot l. Paramagnetic: O only, S = O - O_bak.
// Lattice: each term carries gamma of the site its vertex is pinned to (see README). A free function (one
// machine code for both site sets) so that the impurity and general paths agree bitwise.
static __attribute__((noinline)) void dsl_rtde_impl(const RtdeContext& c, const std::vector<int>& sites,
                                                 const std::vector<double>& gams, std::vector<cplx>& tl3_) {
  using namespace tensor;
  const int N = c.N;
  const std::size_t Nz = std::size_t(N), n2 = Nz * Nz, n3 = n2 * Nz, M = sites.size(), Ap = Nz * (Nz - 1) / 2;
  const cplx I(0, 1);
  const bool full = c.ns == 2;
  const cplx* rho[2] = {c.rho, full ? c.rho + n2 : c.rho};
  const Mat* h[2] = {&c.ho_pair[0], &c.ho_pair[full ? 1 : 0]};
  auto slice = [&](int w, std::size_t k) { return c.Wslice + (std::size_t(w) * M + k) * n2; };
  cplx* Sfull = c.tmp3;        // unpacked same-spin block
  cplx* dSfull = c.tmp3 + n3;  // its RHS before packing
  const std::size_t nb = full ? (Ap > 0 ? 2 : 1) : 1;  // blocks per probe spin
  for (int sg = 0; sg < (full ? 2 : 1); ++sg) {
    const int bs = 1 - sg;
    const cplx* GRs = c.GR + std::size_t(full ? sg : 0) * Nz;
    cplx* dGRs = c.dGR + std::size_t(full ? sg : 0) * Nz;
    const cplx* O = c.F[std::size_t(sg) * nb];
    cplx* dO = c.dF[std::size_t(sg) * nb];
    const bool haveS = full && Ap > 0;
    // same-spin block, unpacked
    if (!full) {
      for (std::size_t a = 0; a < Nz; ++a)
        for (std::size_t b = 0; b < Nz; ++b)
          for (std::size_t k = 0; k < Nz; ++k) Sfull[(a * Nz + b) * Nz + k] = O[(a * Nz + b) * Nz + k] - O[(b * Nz + a) * Nz + k];
    } else if (haveS) {
      const cplx* Sp = c.F[std::size_t(sg) * nb + 1];
      for (std::size_t a = 0; a < Nz; ++a)
        for (std::size_t b = 0; b < Nz; ++b)
          for (std::size_t k = 0; k < Nz; ++k) {
            cplx v = 0.0;
            if (a < b) v = Sp[pair_index(Nz, a, b) * Nz + k];
            else if (a > b) v = -Sp[pair_index(Nz, b, a) * Nz + k];
            Sfull[(a * Nz + b) * Nz + k] = v;
          }
    } else {
      std::fill(Sfull, Sfull + n3, cplx(0));
    }
    const cplx *rs = rho[sg], *rb = rho[bs];
    // ---- opposite-spin block: TPPX part + Z^F + Z^G
    drift3(*h[sg], *h[bs], *h[bs], O, dO, N);
    const int wK = 0, wZGd = sg == 0 ? 1 : 2, wZGc = sg == 0 ? 3 : 4, wSG = sg == 0 ? 5 : 6;
    for (std::size_t q = 0; q < M; ++q) {
      const std::size_t r = std::size_t(sites[q]);
      const double g = gams[q];
      const cplx Gr = GRs[r];
      const cplx* Ks = slice(wK, q);
      const cplx* Orr = O + (r * Nz + r) * Nz;
      for (std::size_t a = 0; a < Nz; ++a)
        for (std::size_t b = 0; b < Nz; ++b) {
          const cplx ql = rs[a * Nz + r] * rb[b * Nz + r];
          const cplx qg = (rs[a * Nz + r] - (a == r ? 1.0 : 0.0)) * (rb[b * Nz + r] - (b == r ? 1.0 : 0.0));
          const cplx V = qg - ql;
          cplx* row = dO + (a * Nz + b) * Nz;
          const cplx mv = -g * V, sv = 0.5 * g * Gr * V;
          for (std::size_t k = 0; k < Nz; ++k) row[k] += mv * Orr[k] + sv * rb[r * Nz + k];
          row[r] += -0.5 * g * Gr * ql;                                   // -(g/2) q^< delta_rl G_r
          const cplx Kab = sg == 0 ? Ks[a * Nz + b] : Ks[b * Nz + a];    // G^{s sb s sb}_{ab rr}
          row[r] += I * g * Kab * Gr;
        }
      // Z^F direct: g_l rho^s_{al} O_{lbl} (l = r) + g_a rho^sb_{al} O_{aba} (a = r)
      for (std::size_t a = 0; a < Nz; ++a) {
        const cplx c1 = g * rs[a * Nz + r];
        for (std::size_t b = 0; b < Nz; ++b) dO[(a * Nz + b) * Nz + r] += c1 * O[(r * Nz + b) * Nz + r];
      }
      for (std::size_t b = 0; b < Nz; ++b) {
        const cplx Ob = g * O[(r * Nz + b) * Nz + r];
        for (std::size_t k = 0; k < Nz; ++k) dO[(r * Nz + b) * Nz + k] += Ob * rb[r * Nz + k];
      }
      // Z^F crossed: g_l rho^sb_{bl} S_{lal} (l = r) + g_b rho^sb_{bl} S_{bab} (b = r)
      for (std::size_t b = 0; b < Nz; ++b) {
        const cplx c1 = g * rb[b * Nz + r];
        for (std::size_t a = 0; a < Nz; ++a) dO[(a * Nz + b) * Nz + r] += c1 * Sfull[(r * Nz + a) * Nz + r];
      }
      for (std::size_t a = 0; a < Nz; ++a) {
        const cplx Sa = g * Sfull[(r * Nz + a) * Nz + r];
        for (std::size_t k = 0; k < Nz; ++k) dO[(a * Nz + r) * Nz + k] += Sa * rb[r * Nz + k];
      }
      // Z^G direct: i g_b G^{s sb s sb}_{abbl} G_b (b = r); crossed: i g_a G^{sb sb sb sb}_{abla} G_a (a = r)
      const cplx* Zd = slice(wZGd, q);
      const cplx* Zc = slice(wZGc, q);
      for (std::size_t a = 0; a < Nz; ++a)
        for (std::size_t k = 0; k < Nz; ++k) dO[(a * Nz + r) * Nz + k] += I * g * Zd[a * Nz + k] * Gr;
      for (std::size_t b = 0; b < Nz; ++b)
        for (std::size_t k = 0; k < Nz; ++k) dO[(r * Nz + b) * Nz + k] += I * g * Zc[b * Nz + k] * Gr;
      // G^R: + 2 g_s O_sss
      dGRs[r] += 2.0 * g * O[(r * Nz + r) * Nz + r];
    }
    if (!haveS) continue;
    // ---- same-spin block: drift + g (1 + xi P_ab)[ rho^s_{al}(g_l O_{bll} + g_a O_{baa}) + i g_b G^{s sb s sb}_{ablb} G_b ]
    drift3(*h[sg], *h[sg], *h[sg], Sfull, dSfull, N);
    for (std::size_t x = 0; x < n3; ++x) tl3_[x] = 0.0;
    for (std::size_t q = 0; q < M; ++q) {
      const std::size_t r = std::size_t(sites[q]);
      const double g = gams[q];
      const cplx Gr = GRs[r];
      for (std::size_t a = 0; a < Nz; ++a) {      // l = r
        const cplx c1 = g * rs[a * Nz + r];
        for (std::size_t b = 0; b < Nz; ++b) tl3_[(a * Nz + b) * Nz + r] += c1 * O[(b * Nz + r) * Nz + r];
      }
      for (std::size_t b = 0; b < Nz; ++b) {      // a = r
        const cplx Ob = g * O[(b * Nz + r) * Nz + r];
        for (std::size_t k = 0; k < Nz; ++k) tl3_[(r * Nz + b) * Nz + k] += Ob * rs[r * Nz + k];
      }
      const cplx* Zs = slice(wSG, q);              // b = r
      for (std::size_t a = 0; a < Nz; ++a)
        for (std::size_t k = 0; k < Nz; ++k) tl3_[(a * Nz + r) * Nz + k] += I * g * Zs[a * Nz + k] * Gr;
    }
    for (std::size_t a = 0; a < Nz; ++a)
      for (std::size_t b = 0; b < Nz; ++b)
        for (std::size_t k = 0; k < Nz; ++k) dSfull[(a * Nz + b) * Nz + k] += tl3_[(a * Nz + b) * Nz + k] - tl3_[(b * Nz + a) * Nz + k];
    cplx* dSp = c.dF[std::size_t(sg) * nb + 1];
    for (std::size_t a = 0; a < Nz; ++a)
      for (std::size_t b = a + 1; b < Nz; ++b)
        for (std::size_t k = 0; k < Nz; ++k) dSp[pair_index(Nz, a, b) * Nz + k] = dSfull[(a * Nz + b) * Nz + k];
  }
}


// ------------------------------------------------------------------ site-set vertex implementation
template <class S>
class SiteVertex : public VertexTerms {
 public:
  SiteVertex(const char* nm, S st) : name_(nm), st_(std::move(st)) {}
  const char* name() const override { return name_; }

  void add_mf_loss(const cplx* rho_opp, int N, cplx* ho) const override {
    for (int k = 0; k < st_.size(); ++k) {
      const std::size_t r = std::size_t(st_.site(k));
      ho[r * std::size_t(N) + r] += cplx(0, -1) * (st_.gam(k) * rho_opp[r * std::size_t(N) + r].real());
    }
  }

  void history_integral(Approx a, int Ni, int ns, const std::vector<const cplx*>& X, cplx* I) const override {
    const std::size_t N = std::size_t(Ni), n2 = N * N;
    std::fill(I, I + std::size_t(ns) * n2, cplx(0, 0));
    using tensor::I4;
    for (int k = 0; k < st_.size(); ++k) {
      const std::size_t r = std::size_t(st_.site(k));
      const double g = st_.gam(k);
      for (std::size_t s = 0; s < N; ++s) {
        cplx up, dn;
        if (a == Approx::mGW) {  // I = i g (G^W - G^{W,2})_{.. r r r}, P - A - P2 (mGWlat-I / mGWimp-I)
          const cplx *P = X[0], *A = X[1], *P2 = X.back();
          up = cplx(0, g) * (P[I4(N, s, r, r, r)] - A[I4(N, s, r, r, r)] - P2[I4(N, s, r, r, r)]);
          dn = cplx(0, g) * (P[I4(N, r, s, r, r)] - A[I4(N, r, s, r, r)] - P2[I4(N, r, s, r, r)]);
        } else {  // TPPX, DSL: I = 2 i g G^{udud}_{s r r r} (DSLimp-I, R5)
          const cplx* P = X[0];
          up = cplx(0, 2 * g) * P[I4(N, s, r, r, r)];
          dn = cplx(0, 2 * g) * P[I4(N, r, s, r, r)];
        }
        I[s * N + r] += up;
        if (ns == 2) I[n2 + s * N + r] += dn;
      }
    }
  }

  // TPPX / 2B RTDE (bare_propagators: bare h_o on the legs a, b, l), slot order F^{pp}_{ablj} (sec:tppxrtde): impurity Eq. (UDfinal); lattice = the same with
  // the vertex site summed over the lossy set (derived from Eq. Fppgen, see README). Blocks: D = F^{udd u}
  // (feeds G^{R,up}), U = F^{udud} (feeds G^{R,dn}); paramagnetic: D only (R6).
  // mGW RTDE with counterterm, slot order F^W_{abjl} (sec:mGWRTDElat, eqs. mGWRlat-U..GRd; impurity
  // eqs. mGWRimp-U..UP2). Column j fixed: each block is T[a][b][l] (the loops below call slots a, b, l p, q, o).
  // Lattice: vertex sites summed with gamma_r (kernel term 1 pinned to the site of slot l, term 2 to slot a,
  // G^W coupling to slot b, I to row s).
  // Per probe spin sigma (bar = opposite): S = (s s . s), A = (bar s . bar), P = (s bar . bar), P2 = counterterm P.
  void mgw_rtde(const RtdeContext& c) const {
    using namespace tensor;
    const int N = c.N;
    const std::size_t Nz = std::size_t(N), n2 = Nz * Nz, n3 = n2 * Nz, M = std::size_t(st_.size());
    const cplx I(0, 1);
    const cplx* rho[2] = {c.rho, c.ns == 2 ? c.rho + n2 : c.rho};
    const Mat* h[2] = {&c.ho_pair[0], &c.ho_pair[c.ns == 2 ? 1 : 0]};
    gx_.assign(Nz, 0.0);
    for (std::size_t k = 0; k < M; ++k) gx_[std::size_t(st_.site(int(k)))] = st_.gam(int(k));
    auto slice = [&](int w, std::size_t k) { return c.Wslice + (std::size_t(w) * M + k) * n2; };
    // t^{sl sa sb}_{labr} G_{rj} gamma_r summed over lossy r, added with factor f into dT[a][b][l] (so, sp, sq = sl, sa, sb)
    auto tsrc = [&](cplx* dT, int so, int sp, int sq, const cplx* GRs, double f) {
      for (std::size_t k = 0; k < M; ++k) {
        const std::size_t r = std::size_t(st_.site(int(k)));
        const cplx g = f * st_.gam(int(k)) * GRs[r];
        if (g == cplx(0)) continue;
        tl_.resize(6 * Nz);
        cplx *pl = tl_.data(), *pg = pl + Nz, *ol = pg + Nz, *og = ol + Nz, *ql = og + Nz, *qg = ql + Nz;
        for (std::size_t x = 0; x < Nz; ++x) {
          pl[x] = rho[sp][x * Nz + r]; pg[x] = pl[x] - (x == r ? 1.0 : 0.0);
          ol[x] = rho[so][r * Nz + x]; og[x] = ol[x] - (x == r ? 1.0 : 0.0);
          ql[x] = rho[sq][x * Nz + r]; qg[x] = ql[x] - (x == r ? 1.0 : 0.0);
        }
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t pi = 0; pi < std::ptrdiff_t(Nz); ++pi) {
          const std::size_t p = std::size_t(pi);
          for (std::size_t q = 0; q < Nz; ++q) {
            const cplx a1 = g * pg[p] * qg[q], a2 = g * pl[p] * ql[q], a3 = -2.0 * g * pl[p] * ql[q];
            cplx* row = dT + (p * Nz + q) * Nz;
            for (std::size_t o = 0; o < Nz; ++o) row[o] += a1 * ol[o] + a2 * og[o] + a3 * ol[o];
          }
        }
      }
    };
    // G^W coupling: dT[p][q][o] += f gamma_q slice[k(q)][p][o] G_{qj}
    auto gcoup = [&](cplx* dT, int w, const cplx* GRs, cplx f) {
      for (std::size_t k = 0; k < M; ++k) {
        const std::size_t q = std::size_t(st_.site(int(k)));
        const cplx g = f * st_.gam(int(k)) * GRs[q];
        const cplx* Sl = slice(w, k);
        for (std::size_t p = 0; p < Nz; ++p)
          for (std::size_t o = 0; o < Nz; ++o) dT[(p * Nz + q) * Nz + o] += g * Sl[p * Nz + o];
      }
    };
    // kernel: dT[p][q][o] += f [ gamma_o rho1_{po} F[o][q][o] + gamma_p rho2_{po} F[p][q][p] ]
    auto kern = [&](cplx* dT, const cplx* F, const cplx* r1, const cplx* r2, double f) {
      for (std::size_t k = 0; k < M; ++k) {
        const std::size_t v = std::size_t(st_.site(int(k)));
        const double g = f * st_.gam(int(k));
        for (std::size_t p = 0; p < Nz; ++p) {      // term 1, o = v
          const cplx a = g * r1[p * Nz + v];
          for (std::size_t q = 0; q < Nz; ++q) dT[(p * Nz + q) * Nz + v] += a * F[(v * Nz + q) * Nz + v];
        }
        for (std::size_t q = 0; q < Nz; ++q) {      // term 2, p = v
          const cplx b = g * F[(v * Nz + q) * Nz + v];
          for (std::size_t o = 0; o < Nz; ++o) dT[(v * Nz + q) * Nz + o] += b * r2[v * Nz + o];
        }
      }
    };
    const bool full = c.ns == 2;
    for (int sg = 0; sg < (full ? 2 : 1); ++sg) {
      const int bs = 1 - sg;
      const cplx* GRs = c.GR + std::size_t(full ? sg : 0) * Nz;
      cplx* dGRs = c.dGR + std::size_t(full ? sg : 0) * Nz;
      const std::size_t o0 = full ? std::size_t(4 * sg) : 0;
      const cplx *Sb, *A, *P, *P2;
      cplx *dSb = nullptr, *dA, *dP, *dP2;
      if (full) {
        Sb = c.F[o0]; A = c.F[o0 + 1]; P = c.F[o0 + 2]; P2 = c.F[o0 + 3];
        dSb = c.dF[o0]; dA = c.dF[o0 + 1]; dP = c.dF[o0 + 2]; dP2 = c.dF[o0 + 3];
      } else {  // paramagnetic: Sb = A + P (SU(2))
        A = c.F[0]; P = c.F[1]; P2 = c.F[2]; dA = c.dF[0]; dP = c.dF[1]; dP2 = c.dF[2];
        for (std::size_t x = 0; x < n3; ++x) c.tmp3[x] = A[x] + P[x];
        Sb = c.tmp3;
      }
      const int wS = sg == 0 ? 0 : 3, wA = sg == 0 ? 2 : 4, wP = sg == 0 ? 1 : 5;
      if (full) {  // Sb = (s s . s)
        drift3(*h[sg], *h[sg], *h[sg], Sb, dSb, N);
        gcoup(dSb, wS, GRs, I);
        kern(dSb, A, rho[sg], rho[sg], -1.0);
      }
      // A = (bar s . bar)
      drift3(*h[bs], *h[sg], *h[bs], A, dA, N);
      tsrc(dA, bs, bs, sg, GRs, +1.0);
      gcoup(dA, wA, GRs, I);
      kern(dA, Sb, rho[bs], rho[bs], -1.0);
      // P = (s bar . bar)
      drift3(*h[sg], *h[bs], *h[bs], P, dP, N);
      tsrc(dP, bs, sg, bs, GRs, -1.0);
      gcoup(dP, wP, GRs, -I);
      kern(dP, P, rho[sg], rho[bs], +1.0);
      // counterterm: drift + t source only
      drift3(*h[sg], *h[bs], *h[bs], P2, dP2, N);
      tsrc(dP2, bs, sg, bs, GRs, -1.0);
      // G^R: + gamma_s [A_sss - P_sss + P2_sss]
      for (std::size_t k = 0; k < M; ++k) {
        const std::size_t i = std::size_t(st_.site(int(k)));
        const std::size_t ii = (i * Nz + i) * Nz + i;
        dGRs[i] += st_.gam(int(k)) * (A[ii] - P[ii] + P2[ii]);
      }
    }
  }

  void rtde_tensor_terms(const RtdeContext& c) const override {
    if (c.approx == Approx::mGW) { mgw_rtde(c); return; }
    if (c.approx == Approx::DSL) {
      tl3_.resize(std::size_t(c.N) * c.N * c.N);
      std::vector<int> sites;
      std::vector<double> gams;
      for (int q = 0; q < st_.size(); ++q) { sites.push_back(st_.site(q)); gams.push_back(st_.gam(q)); }
      dsl_rtde_impl(c, sites, gams, tl3_);
      return;
    }
    if (c.approx != Approx::TPPX)
      throw NotImplemented(std::string(name()) + ": RTDE tensor terms for " + drv::name(c.approx) + " (being rewritten)");
    using namespace tensor;
    const int N = c.N;
    const std::size_t Nz = std::size_t(N), n2 = Nz * Nz;
    const cplx* ru = c.rho;
    const cplx* rd = c.ns == 2 ? c.rho + n2 : c.rho;
    const Mat& hu = c.ho_pair[0];
    const Mat& hd = c.ho_pair[c.ns == 2 ? 1 : 0];
    const cplx* GRu = c.GR;
    const cplx* GRd = c.ns == 2 ? c.GR + Nz : c.GR;
    pair_vectors(st_, ru, rd, N, ql_, qg_);
    // one block: X_{abl} with closing-leg (slot l) spin sk (rho_l, h_l; loop index k = slot l), coupled column GRs,
    // sign (+1 U, -1 D)
    auto block = [&](const cplx* X, cplx* dX, const Mat& hk, const cplx* rk, const cplx* GRs, double sign) {
      drift3(hu, hd, hk, X, dX, N);
      for (int q = 0; q < st_.size(); ++q) {
        const std::size_t r = std::size_t(st_.site(q));
        const double g = st_.gam(q);
        const cplx* L = ql_.data() + std::size_t(q) * n2;
        const cplx* G = qg_.data() + std::size_t(q) * n2;
        const cplx* K = c.flag_2B ? nullptr : c.Kslice + std::size_t(q) * n2;
        const cplx gs = sign * 0.5 * g * GRs[r];
        for (std::size_t ab = 0; ab < n2; ++ab) {
          const cplx V = G[ab] - L[ab];
          cplx* row = dX + ab * Nz;
          if (!c.flag_2B) {  // memory -g V_{ab r} X_{r r l}
            const cplx* Xrr = X + (r * Nz + r) * Nz;
            const cplx mv = -g * V;
            for (std::size_t k = 0; k < Nz; ++k) row[k] += mv * Xrr[k];
          }
          // source (delta form): sign g/2 [ (Q^<_{ab r} - K_{ab r r}) delta_{l r} - V_{ab r} rho_{r l} ] G^R_{r j}
          const cplx Kab = K ? cplx(0, 2) * K[ab] : cplx(0, 0);  // K^{pp} = 2i G^{pp}
          row[r] += gs * (L[ab] - Kab);
          const cplx gv = -gs * V;
          for (std::size_t k = 0; k < Nz; ++k) row[k] += gv * rk[r * Nz + k];
        }
      }
    };
    block(c.F[0], c.dF[0], hd, rd, GRu, -1.0);                 // D
    if (c.ns == 2) block(c.F[1], c.dF[1], hu, ru, GRd, +1.0);  // U
    // G^R: -i I, I^{up}_{r j} = 2i g D_{r r r}, I^{dn}_{r j} = -2i g U_{r r r}
    for (int q = 0; q < st_.size(); ++q) {
      const std::size_t r = std::size_t(st_.site(q));
      const double g = st_.gam(q);
      c.dGR[r] += 2.0 * g * c.F[0][(r * Nz + r) * Nz + r];
      if (c.ns == 2) c.dGR[Nz + r] += -2.0 * g * c.F[1][(r * Nz + r) * Nz + r];
    }
  }

  void g1g2_tensor_terms(const TensorContext& c) const override {
    using namespace tensor;
    const int N = c.N;
    const std::size_t Nz = std::size_t(N), n2 = Nz * Nz, n4 = n2 * n2;
    const cplx* ru = c.rho;
    const cplx* rd = c.ns == 2 ? c.rho + n2 : c.rho;
    const Mat& hu = c.ho_pair[0];
    const Mat& hd = c.ho_pair[c.ns == 2 ? 1 : 0];
    pair_vectors(st_, ru, rd, N, ql_, qg_);
    const cplx I(0, 1);
    switch (c.approx) {
      case Approx::HF: return;
      case Approx::TPPX: {  // sec:tppxg1g2, Eq. final with K = 2iG undone; 2B: no memory
        drift4(hu, hd, hu, hd, c.X[0], c.dX[0], N);
        pp_source(st_, ql_, qg_, N, -0.5 * I, false, c.dX[0]);
        if (!c.flag_2B) pp_memory(st_, ql_, qg_, N, c.X[0], c.dX[0]);
        return;
      }
      case Approx::DSL: {  // sec:DSLG1G2lat / DSLimp: TPPX part + particle-hole rung
        const cplx* P = c.X[0];
        const cplx *U, *D;
        const bool same = c.ns == 2 && c.X.size() == 3;  // N = 1: same-spin blocks vanish identically
        cplx *bU = c.tmp, *bD = c.tmp + n4, *bT = c.tmp + 2 * n4;
        if (c.ns == 2) {
          if (same) { unpack_anti(c.X[1], bU, N); unpack_anti(c.X[2], bD, N); }
          else std::fill(bU, bU + 2 * n4, cplx(0, 0));
          U = bU; D = bD;
        } else {  // paramagnetic: G^{uuuu} = G^{dddd} = P + xi P_slba
#pragma omp parallel for schedule(static)
          for (std::ptrdiff_t x = 0; x < std::ptrdiff_t(n2); ++x)
            for (std::size_t a = 0; a < Nz; ++a)
              for (std::size_t b = 0; b < Nz; ++b)
                c.tmp[std::size_t(x) * n2 + a * Nz + b] = P[std::size_t(x) * n2 + a * Nz + b] - P[std::size_t(x) * n2 + b * Nz + a];
          U = D = c.tmp;
        }
        drift4(hu, hd, hu, hd, P, c.dX[0], N);
        pp_source(st_, ql_, qg_, N, -0.5 * I, false, c.dX[0]);
        pp_memory(st_, ql_, qg_, N, P, c.dX[0]);
        dsl_rung_P(st_, ru, rd, N, P, U, D, c.dX[0]);
        if (same) {  // full-array drift (GEMMs) + rung on scratch, then pack
          drift4(hu, hu, hu, hu, U, bT, N);
          dsl_rung_same(st_, ru, N, P, false, bT);
          pack_anti(bT, c.dX[1], N);
          drift4(hd, hd, hd, hd, D, bT, N);
          dsl_rung_same(st_, rd, N, P, true, bT);
          pack_anti(bT, c.dX[2], N);
        }
        return;
      }
      case Approx::mGW: {  // sec:mGWG1G2lat Eqs. mGWlat-P..P2 / mGWimp-P..P2
        const bool full = c.ns == 2;
        const cplx *P = c.X[0], *A = c.X[1], *U, *D, *P2 = c.X.back();
        cplx *dP = c.dX[0], *dA = c.dX[1], *dP2 = c.dX.back();
        if (full) { U = c.X[2]; D = c.X[3]; }
        else {  // paramagnetic: G^{W,uuuu} = G^{W,dddd} = A + P
#pragma omp parallel for schedule(static)
          for (std::ptrdiff_t x = 0; x < std::ptrdiff_t(n4); ++x) c.tmp[x] = A[x] + P[x];
          U = D = c.tmp;
        }
        auto at = [Nz](const cplx* T) { return [T, Nz](std::size_t a, std::size_t b, std::size_t cc, std::size_t d) { return T[I4(Nz, a, b, cc, d)]; }; };
        auto tP = at(P), tA = at(A), tU = at(U), tD = at(D);
        drift4(hu, hd, hu, hd, P, dP, N);
        pp_source(st_, ql_, qg_, N, -I, false, dP);
        mgw_kernel(st_, N, +1.0, ru, rd, rd, ru,
                   [&](std::size_t o, std::size_t p, std::size_t j) { return tP(j, o, p, j); },
                   [&](std::size_t s, std::size_t o, std::size_t p) { return tP(s, o, p, s); },
                   [&](std::size_t s, std::size_t p, std::size_t j) { return tP(s, p, p, j); },
                   [&](std::size_t s, std::size_t o, std::size_t j) { return tP(s, o, o, j); }, dP);
        drift4(hu, hd, hd, hu, A, dA, N);
        pp_source(st_, ql_, qg_, N, I, true, dA);
        mgw_kernel(st_, N, -1.0, ru, ru, rd, rd,
                   [&](std::size_t o, std::size_t p, std::size_t j) { return tD(j, o, p, j); },
                   [&](std::size_t s, std::size_t o, std::size_t p) { return tD(s, o, p, s); },
                   [&](std::size_t s, std::size_t p, std::size_t j) { return tU(s, p, p, j); },
                   [&](std::size_t s, std::size_t o, std::size_t j) { return tU(s, o, o, j); }, dA);
        if (full) {
          drift4(hu, hu, hu, hu, U, c.dX[2], N);
          mgw_kernel(st_, N, -1.0, ru, ru, ru, ru,
                     [&](std::size_t o, std::size_t p, std::size_t j) { return tA(o, j, j, p); },
                     [&](std::size_t s, std::size_t o, std::size_t p) { return tA(o, s, s, p); },
                     [&](std::size_t s, std::size_t p, std::size_t j) { return tA(s, p, p, j); },
                     [&](std::size_t s, std::size_t o, std::size_t j) { return tA(s, o, o, j); }, c.dX[2]);
          drift4(hd, hd, hd, hd, D, c.dX[3], N);
          mgw_kernel(st_, N, -1.0, rd, rd, rd, rd,
                     [&](std::size_t o, std::size_t p, std::size_t j) { return tA(j, o, p, j); },
                     [&](std::size_t s, std::size_t o, std::size_t p) { return tA(s, o, p, s); },
                     [&](std::size_t s, std::size_t p, std::size_t j) { return tA(p, s, j, p); },
                     [&](std::size_t s, std::size_t o, std::size_t j) { return tA(o, s, j, o); }, c.dX[3]);
        }
        drift4(hu, hd, hu, hd, P2, dP2, N);  // counterterm: drift + source, no kernel (symm-G2pp)
        pp_source(st_, ql_, qg_, N, -I, false, dP2);
        return;
      }
    }
  }

 private:
  const char* name_;
  S st_;
  mutable std::vector<cplx> ql_, qg_, tl_, tl3_;
  mutable std::vector<double> gx_;
};

std::unique_ptr<VertexTerms> make_vertex(const Model& m, VertexImpl v, double gamma) {
  if (v == VertexImpl::general) {
    tensor::ListSites s;
    for (std::size_t k = 0; k < m.lossy_sites.size(); ++k) {
      s.r.push_back(int(m.lossy_sites[k]));
      s.g.push_back(gamma * m.site_weight[k]);
    }
    return std::make_unique<SiteVertex<tensor::ListSites>>("general", s);
  }
  if (m.lossy_sites.size() != 1 || m.lossy_sites[0] != 0)
    throw std::runtime_error("vertex_impl impurity requires lossy_sites == {0} in the model");
  tensor::Site0 s;
  s.g0 = gamma * m.site_weight[0];
  return std::make_unique<SiteVertex<tensor::Site0>>("impurity", s);
}

// ------------------------------------------------------------------ physics
Physics::Physics(const Model& m, const RunCard& rc, std::unique_ptr<VertexTerms> v)
    : N(m.N), ns(nspin(rc.spin)), gamma(rc.gamma), approx(rc.approx), flag_2B(rc.flag_2B), bare_propagators(rc.bare_propagators),
      sites_(m.lossy_sites), w_(m.site_weight), vertex_(std::move(v)) {
  if (flag_2B && approx != Approx::TPPX) throw std::runtime_error("flag_2B is defined for TPPX only");
  if (bare_propagators && approx == Approx::HF)
    throw std::runtime_error("bare_propagators is not defined for HF (no correlator)");
  for (int s = 0; s < ns; ++s) {
    CMatMap h(m.h_s(s), N, N), gt(m.gt_s(s), N, N), lt(m.lt_s(s), N, N);
    h_[s] = h;
    hbase_o_[s] = h - cplx(0, 1) * (gt + lt);  // h_o = h - i[l^> - xi l^<], xi = -1
    two_lt_[s] = 2.0 * lt;
    ho_[s].resize(N, N);
  }
  I_.assign(std::size_t(ns) * N * N, 0.0);
  bonds_ = bonds_from_h(h_, ns, N);
  const std::size_t n4 = std::size_t(N) * N * N * N;
  if (approx == Approx::DSL) tmp_.assign((ns == 2 ? 3 : 1) * n4, 0.0);  // unpacked U, D and a dU/dD buffer
  if (approx == Approx::mGW && ns == 1) tmp_.assign(n4, 0.0);
}

void Physics::build_ho(const cplx* rho, Mat* ho) const {
  const std::size_t n2 = std::size_t(N) * N;
  for (int s = 0; s < ns; ++s) {
    ho[s] = hbase_o_[s];
    const cplx* rho_opp = rho + std::size_t(ns == 2 ? 1 - s : 0) * n2;
    vertex_->add_mf_loss(rho_opp, N, ho[s].data());
  }
}

Physics::RefRates Physics::ref_rates(const cplx* rho) const {
  Mat ho[2];
  build_ho(rho, ho);
  RefRates r;
  r.lam_mf = r.lam_bare = std::numeric_limits<double>::infinity();
  double remin = std::numeric_limits<double>::infinity(), remax = -remin;
  for (int s = 0; s < ns; ++s) {
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es(Eigen::MatrixXcd(ho[s]), false);
    for (Eigen::Index k = 0; k < es.eigenvalues().size(); ++k) {
      const cplx e = es.eigenvalues()[k];
      r.lam_mf = std::min(r.lam_mf, std::abs(e.imag()));
      r.lam_max = std::max(r.lam_max, std::abs(e.imag()));
      remin = std::min(remin, e.real()); remax = std::max(remax, e.real());
    }
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> eb(Eigen::MatrixXcd(hbase_o_[s]), false);
    for (Eigen::Index k = 0; k < eb.eigenvalues().size(); ++k) r.lam_bare = std::min(r.lam_bare, std::abs(eb.eigenvalues()[k].imag()));
  }
  r.spread = remax - remin;
  return r;
}

void Physics::rho_dot(const cplx* rho, const Mat* ho, const std::vector<const cplx*>& X, cplx* drho) const {
  const std::size_t n2 = std::size_t(N) * N;
  const bool corr = approx != Approx::HF && !X.empty();
  if (corr) vertex_->history_integral(approx, N, ns, X, I_.data());
  for (int s = 0; s < ns; ++s) {
    CMatMap r(rho + s * n2, N, N);
    MatMap d(drho + s * n2, N, N);
    d.noalias() = cplx(0, -1) * (ho[s] * r);
    d.noalias() += cplx(0, 1) * (r * ho[s].adjoint());
    d += two_lt_[s];
    if (corr) {  // -xi [I + I^dag] (main Eq. KBErho)
      CMatMap Is(I_.data() + s * n2, N, N);
      d += Is + Is.adjoint();
    }
    // keep rho exactly Hermitian: the anti-Hermitian direction can be linearly unstable in unphysical states
    // (seen for TPPX_0 on the plaquette from a 3up-1down start), and round-off would seed it. Zero for Hermitian rho.
    d = (cplx(0.5) * (d + d.adjoint())).eval();
  }
}

void Physics::observables(const cplx* rho, const std::vector<const cplx*>& X, FrameObservables& o) const {
  const std::size_t n2 = std::size_t(N) * N, Nz = std::size_t(N);
  Mat ho[2];
  build_ho(rho, ho);
  std::vector<cplx> d(std::size_t(ns) * n2);
  rho_dot(rho, ho, X, d.data());  // also fills I_ for correlated schemes
  o.rho_dot_max = 0;
  for (auto z : d) o.rho_dot_max = std::max(o.rho_dot_max, std::abs(z));
  const double mult = ns == 2 ? 1.0 : 2.0;
  o.dN_dt = 0; o.energy = 0;
  for (int s = 0; s < ns; ++s)
    for (std::size_t x = 0; x < Nz; ++x) {
      o.dN_dt += mult * d[s * n2 + x * Nz + x].real();
      for (std::size_t y = 0; y < Nz; ++y) o.energy += mult * (h_[s](Eigen::Index(x), Eigen::Index(y)) * rho[s * n2 + y * Nz + x]).real();
    }
  o.rho_eig_min = 1e300; o.rho_eig_max = -1e300;
  for (int s = 0; s < ns; ++s) {
    CMatMap r(rho + s * n2, N, N);
    Mat hr = 0.5 * (r + r.adjoint());
    Eigen::SelfAdjointEigenSolver<Mat> es(hr, Eigen::EigenvaluesOnly);
    o.rho_eig_min = std::min(o.rho_eig_min, es.eigenvalues().minCoeff());
    o.rho_eig_max = std::max(o.rho_eig_max, es.eigenvalues().maxCoeff());
  }
  o.D.assign(sites_.size(), 0.0);
  o.loss_rate = 0;
  const bool corr = approx != Approx::HF && !X.empty();
  for (std::size_t k = 0; k < sites_.size(); ++k) {
    const std::size_t r = std::size_t(sites_[k]);
    const double up = rho[r * Nz + r].real(), dn = rho[(ns == 2 ? n2 : 0) + r * Nz + r].real();
    double Dr = up * dn;
    if (corr) {  // D_r = rho^u_rr rho^d_rr - Re I^u_rr / (gamma w_r), I from the unit-rate tensors
      using tensor::I4;
      if (approx == Approx::mGW)
        Dr -= (cplx(0, 1) * (X[0][I4(Nz, r, r, r, r)] - X[1][I4(Nz, r, r, r, r)] - X.back()[I4(Nz, r, r, r, r)])).real();
      else
        Dr -= (cplx(0, 2) * X[0][I4(Nz, r, r, r, r)]).real();
    }
    o.D[k] = Dr;
    o.loss_rate += -4.0 * gamma * w_[k] * Dr;
  }
  // equal-time observables (step 3b): one-body part of dN/dt from the bare h_o and l^<
  double dn1 = 0;
  for (int s = 0; s < ns; ++s) {
    CMatMap r(rho + s * n2, N, N);
    Mat d1 = cplx(0, -1) * (hbase_o_[s] * r) + cplx(0, 1) * (r * hbase_o_[s].adjoint()) + two_lt_[s];
    dn1 += mult * d1.trace().real();
  }
  EqContext ec{approx, N, ns, rho, X, I_.data(), sites_, {}, h_, bonds_, sites_.size() == 1 && sites_[0] == 0,
               false, o.dN_dt, dn1};
  for (double w : w_) ec.gam.push_back(gamma * w);
  for (int s = 0; s < ns; ++s) {  // Lambda = h_o rho + i l^< + i I (left-sided limit; the right-sided one is Lambda^dag)
    CMatMap r(rho + s * n2, N, N);
    ec.lam_h += mult * (h_[s] * r).trace();
    ec.lam_1d += mult * (((hbase_o_[s] - h_[s]) * r).trace() + cplx(0, 0.5) * two_lt_[s].trace());
    ec.lam_mf += mult * ((ho[s] - hbase_o_[s]) * r).trace();
    if (approx != Approx::HF && !X.empty()) ec.lam_c += mult * cplx(0, 1) * CMatMap(I_.data() + s * n2, N, N).trace();
  }
  equal_time_observables(ec, o.eq);
}

void Physics::mgw_slices(const std::vector<const cplx*>& X, bool full, int Ni, const std::vector<long long>& sites,
                         std::vector<cplx>& out) {
  using tensor::I4;
  const std::size_t N = std::size_t(Ni), n2 = N * N, M = sites.size();
  out.assign(6 * M * n2, 0.0);
  const cplx *P = X[0], *A = X[1];
  const cplx *UU = full ? X[2] : nullptr, *DD = full ? X[3] : nullptr;
  for (std::size_t k = 0; k < M; ++k) {
    const std::size_t q = std::size_t(sites[k]);
    auto at = [&](int w) { return out.data() + (std::size_t(w) * M + k) * n2; };
    for (std::size_t p = 0; p < N; ++p)
      for (std::size_t o = 0; o < N; ++o) {
        const std::size_t in = I4(N, p, q, q, o), ou = I4(N, q, p, o, q);
        at(0)[p * N + o] = A[in];
        at(1)[p * N + o] = P[in];
        at(2)[p * N + o] = full ? DD[ou] : A[ou] + P[ou];
        at(3)[p * N + o] = A[ou];
        at(4)[p * N + o] = full ? UU[ou] : A[ou] + P[ou];
        at(5)[p * N + o] = P[ou];
      }
  }
}

void Physics::dsl_slices(const std::vector<const cplx*>& X, bool full, int Ni, const std::vector<long long>& sites,
                         std::vector<cplx>& out) {
  using tensor::I4;
  const std::size_t N = std::size_t(Ni), n2 = N * N, M = sites.size(), Ap = N * (N - 1) / 2;
  out.assign(7 * M * n2, 0.0);
  const cplx* P = X[0];
  const bool packed = full && X.size() >= 3;
  // same-spin element (packed, antisymmetric in both pairs); paramagnetic: P - P_slba
  auto same = [&](int sp, std::size_t s, std::size_t l, std::size_t a, std::size_t b) -> cplx {
    if (!full) return P[I4(N, s, l, a, b)] - P[I4(N, s, l, b, a)];
    if (!packed || s == l || a == b) return 0.0;
    const double sg = ((s < l) == (a < b)) ? 1.0 : -1.0;
    return sg * X[1 + sp][tensor::pair_index(N, std::min(s, l), std::max(s, l)) * Ap +
                          tensor::pair_index(N, std::min(a, b), std::max(a, b))];
  };
  for (std::size_t k = 0; k < M; ++k) {
    const std::size_t v = std::size_t(sites[k]);
    auto at = [&](int w) { return out.data() + (std::size_t(w) * M + k) * n2; };
    for (std::size_t x = 0; x < N; ++x)
      for (std::size_t y = 0; y < N; ++y) {
        const std::size_t j = x * N + y;
        at(0)[j] = P[I4(N, x, y, v, v)];
        at(1)[j] = P[I4(N, x, v, v, y)];
        at(2)[j] = P[I4(N, v, x, y, v)];
        at(3)[j] = same(1, v, x, y, v);
        at(4)[j] = same(0, v, x, y, v);
        at(5)[j] = P[I4(N, x, v, y, v)];
        at(6)[j] = P[I4(N, v, x, v, y)];
      }
  }
}

void Physics::operator()(double, const cplx* y, cplx* dy, const dp5::Layout& L) {
  const bool rho_in_state = id_rho != npos && !L.removed[id_rho];
  if (!rho_in_state) throw std::runtime_error("RHS: no rho block in the state");
  const cplx* rho = y + L.offset[id_rho];
  build_ho(rho, ho_);
  std::vector<const cplx*> X;
  std::vector<cplx*> dX;
  for (std::size_t id : tensor_ids) { X.push_back(y + L.offset[id]); dX.push_back(dy + L.offset[id]); }
  if (rho_in_state && L.active[id_rho]) rho_dot(rho, ho_, X, dy + L.offset[id_rho]);
  if (!tensor_ids.empty() && L.active[tensor_ids[0]]) {
    TensorContext c{approx, flag_2B, N, ns, rho, bare_propagators ? hbase_o_ : ho_, X, dX, tmp_.data()};
    vertex_->g1g2_tensor_terms(c);
  }
  bool wslice_ready = false;
  for (const RtdeSet& r : rtde) {
    if (L.removed[r.idGR] || !L.active[r.idGR]) continue;
    const cplx* g = y + L.offset[r.idGR];
    cplx* dg = dy + L.offset[r.idGR];
    for (int s = 0; s < ns; ++s) {
      Eigen::Map<const Vec> gv(g + std::size_t(s) * N, N);
      Eigen::Map<Vec> dv(dg + std::size_t(s) * N, N);
      dv.noalias() = cplx(0, -1) * (ho_[s] * gv);  // HF part of the G^R equation
    }
    if (!r.ids.empty()) {
      const cplx* ks = nullptr;
      if (!flag_2B && approx == Approx::TPPX) {
        if (!X.empty()) {  // co-propagated G1-G2: slices G^{pp}_{ab r r} of the current K block
          const std::size_t Nz = std::size_t(N), n2 = Nz * Nz;
          kslice_.resize(sites_.size() * n2);
          for (std::size_t q = 0; q < sites_.size(); ++q) {
            const std::size_t rr = std::size_t(sites_[q]);
            for (std::size_t ab = 0; ab < n2; ++ab) kslice_[q * n2 + ab] = X[0][ab * n2 + rr * Nz + rr];
          }
          ks = kslice_.data();
        } else {
          throw std::runtime_error("RTDE: no G1-G2 input (K slices) available");
        }
      }
      const cplx* ws = nullptr;
      if (approx == Approx::mGW) {
        if (!X.empty()) {
          if (!wslice_ready) { mgw_slices(X, ns == 2, N, sites_, wslice_); wslice_ready = true; }
          ws = wslice_.data();
        } else {
          throw std::runtime_error("RTDE: no mGW G1-G2 input slices available");
        }
        tmp3_.resize(std::size_t(N) * N * N);
      } else if (approx == Approx::DSL) {
        if (!X.empty()) {
          if (!wslice_ready) { dsl_slices(X, ns == 2, N, sites_, wslice_); wslice_ready = true; }
          ws = wslice_.data();
        } else {
          throw std::runtime_error("RTDE: no DSL G1-G2 input slices available");
        }
        tmp3_.resize(2 * std::size_t(N) * N * N);
      }
      RtdeContext rc{approx, flag_2B, N, ns, rho, bare_propagators ? hbase_o_ : ho_, ks, ws, tmp3_.data(),
                     g, dg, r.column, {}, {}};
      for (std::size_t id : r.ids) { rc.F.push_back(y + L.offset[id]); rc.dF.push_back(dy + L.offset[id]); }
      vertex_->rtde_tensor_terms(rc);
    }
  }
}

}  // namespace drv
