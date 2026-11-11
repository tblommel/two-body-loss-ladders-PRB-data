// tensor.hpp -- G1-G2 tensor terms (step 2): drifts (shared) and the vertex terms, written once as
// templates over a site set S. GeneralSites instantiates them with a runtime list of lossy sites
// (lattice reference); ImpurityFastPath with the compile-time single site 0. Same code, same
// operation order -> the two agree bitwise when the list is {0}.
//
// Conventions (Implementation_verification, xi = -1, wedge = 0, vee = (gamma_r/2) eps eps on lossy
// site r, gamma_r = gamma w_r): rho = rho^<, rho^> = rho - 1; blocks row-major [N]^4.
//   TPPX / DSL (slot order slab): P = G^{ud ud}; DSL same-spin U = G^{uuuu}, D = G^{dddd}, stored as
//     full N^4 arrays (antisymmetric in both pairs; rms_count = N^2 (N-1)^2).
//   mGW (slot order slab): P = G^{W,udud}, A = G^{W,uddu}, U = G^{W,uuuu}, D = G^{W,dddd},
//     P2 = G^{W,2,udud} (counterterm).
// Every vertex term below cites the reduced lattice equations (sec:tppxg1g2 Eq. final undone,
// sec:DSLG1G2lat Eqs. DSLlat-P/U/D, sec:mGWG1G2lat Eqs. mGWlat-P..P2) with the vertex site restricted
// to the lossy set; on the impurity they are Eqs. DSLimp-P/U/D and mGWimp-P..P2.
#pragma once

#include <omp.h>

#include <vector>

#include "common.hpp"

namespace drv {
namespace tensor {

inline std::size_t I4(std::size_t N, std::size_t a, std::size_t b, std::size_t c, std::size_t d) {
  return ((a * N + b) * N + c) * N + d;
}

struct ListSites {
  std::vector<int> r;
  std::vector<double> g;  // gamma_r = gamma w_r
  int size() const { return int(r.size()); }
  int site(int k) const { return r[std::size_t(k)]; }
  double gam(int k) const { return g[std::size_t(k)]; }
};
struct Site0 {
  double g0 = 0;
  static constexpr int size() { return 1; }
  static constexpr int site(int) { return 0; }
  double gam(int) const { return g0; }
};

// ------------------------------------------------------------------ drift (vertex independent)
// dX = -i h1_{sx} X_{xlab} - i h2_{lx} X_{sxab} + i X_{slxb} h3^dag_{xa} + i X_{slax} h4^dag_{xb}
// (assigns dX). Two large GEMMs (slots 1, 4) and two batched GEMMs (slots 2, 3).
inline void drift4(const Mat& h1, const Mat& h2, const Mat& h3, const Mat& h4, const cplx* X, cplx* dX, int Ni) {
  const Eigen::Index N = Ni, N2 = N * N, N3 = N2 * N;
  const cplx I(0, 1);
  const Mat m1 = -I * h1, m2 = -I * h2, m3 = I * h3.conjugate(), m4 = I * h4.adjoint();
  MatMap(dX, N, N3).noalias() = m1 * CMatMap(X, N, N3);
  MatMap(dX, N3, N).noalias() += CMatMap(X, N3, N) * m4;
#pragma omp parallel for schedule(static)
  for (Eigen::Index s = 0; s < N; ++s)
    MatMap(dX + s * N3, N, N2).noalias() += m2 * CMatMap(X + s * N3, N, N2);
#pragma omp parallel for schedule(static)
  for (Eigen::Index so = 0; so < N2; ++so)
    MatMap(dX + so * N2, N, N).noalias() += m3 * CMatMap(X + so * N2, N, N);
}

// RTDE drift on X_{abl} (probe j fixed; mGW blocks [a][b][l] of F^W_{abjl}): dX = -i h1_{ax} X_{xbl} - i h2_{bx} X_{axl}
// + i X_{abx} h3^dag_{xl} (assigns dX); O(N^4): one GEMM per slot (slot b batched over a).
inline void drift3(const Mat& h1, const Mat& h2, const Mat& h3, const cplx* X, cplx* dX, int Ni) {
  const Eigen::Index N = Ni, N2 = N * N;
  const cplx I(0, 1);
  const Mat m1 = -I * h1, m2 = -I * h2, m3 = I * h3.adjoint();
  MatMap(dX, N, N2).noalias() = m1 * CMatMap(X, N, N2);
  MatMap(dX, N2, N).noalias() += CMatMap(X, N2, N) * m3;
  for (Eigen::Index a = 0; a < N; ++a) MatMap(dX + a * N2, N, N).noalias() += m2 * CMatMap(X + a * N2, N, N);
}

// ------------------------------------------------------------------ packed antisymmetric storage
// Same-spin DSL blocks are antisymmetric in both pairs (xi = -1): store only s<l, a<b as an A x A array,
// A = N(N-1)/2, pair index p(s,l) = s N - s(s+1)/2 + (l - s - 1). Unpack -> GEMM drift -> pack keeps the
// drift a plain GEMM; both conversions are O(N^4).
inline std::size_t pair_index(std::size_t N, std::size_t s, std::size_t l) { return s * N - s * (s + 1) / 2 + (l - s - 1); }
inline void unpack_anti(const cplx* Xp, cplx* X, int Ni) {
  const std::size_t N = std::size_t(Ni), A = N * (N - 1) / 2;
#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t si = 0; si < std::ptrdiff_t(N); ++si) {
    const std::size_t s = std::size_t(si);
    for (std::size_t l = 0; l < N; ++l)
      for (std::size_t a = 0; a < N; ++a)
        for (std::size_t b = 0; b < N; ++b) {
          cplx v = 0.0;
          if (s != l && a != b) {
            const double sg = ((s < l) == (a < b)) ? 1.0 : -1.0;
            v = sg * Xp[pair_index(N, std::min(s, l), std::max(s, l)) * A + pair_index(N, std::min(a, b), std::max(a, b))];
          }
          X[I4(N, s, l, a, b)] = v;
        }
  }
}
inline void pack_anti(const cplx* X, cplx* Xp, int Ni) {
  const std::size_t N = std::size_t(Ni), A = N * (N - 1) / 2;
#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t si = 0; si < std::ptrdiff_t(N); ++si) {
    const std::size_t s = std::size_t(si);
    for (std::size_t l = s + 1; l < N; ++l)
      for (std::size_t a = 0; a < N; ++a)
        for (std::size_t b = a + 1; b < N; ++b) Xp[pair_index(N, s, l) * A + pair_index(N, a, b)] = X[I4(N, s, l, a, b)];
  }
}

// ------------------------------------------------------------------ pair vectors
// q^x_{s l r} = rho^{up x}_{s r} rho^{dn x}_{l r} for the lossy sites (index k of the site list)
template <class S>
void pair_vectors(const S& st, const cplx* ru, const cplx* rd, int Ni, std::vector<cplx>& ql, std::vector<cplx>& qg) {
  const std::size_t N = std::size_t(Ni), n2 = N * N;
  ql.assign(std::size_t(st.size()) * n2, 0.0);
  qg.assign(std::size_t(st.size()) * n2, 0.0);
  for (int k = 0; k < st.size(); ++k) {
    const std::size_t r = std::size_t(st.site(k));
    for (std::size_t s = 0; s < N; ++s)
      for (std::size_t l = 0; l < N; ++l) {
        const cplx us = ru[s * N + r], dl = rd[l * N + r];
        ql[k * n2 + s * N + l] = us * dl;
        qg[k * n2 + s * N + l] = (us - (s == r ? 1.0 : 0.0)) * (dl - (l == r ? 1.0 : 0.0));
      }
  }
}

// pp source: dP_{(sl)(ab)} += c * sum_r g_r [qg_sl ql*_ab - 2 ql_sl ql*_ab + ql_sl qg*_ab],
// transposed second pair if swap34 (mGW A block: (ab) -> (ba)). TPPX/DSL: c = -i/2; mGW P: -i; mGW A: +i (swap).
template <class S>
void pp_source(const S& st, const std::vector<cplx>& ql, const std::vector<cplx>& qg, int Ni, cplx c, bool swap34,
               cplx* dP) {
  const std::size_t N = std::size_t(Ni), n2 = N * N;
  for (int k = 0; k < st.size(); ++k) {
    const cplx ck = c * st.gam(k);
    const cplx* L = ql.data() + std::size_t(k) * n2;
    const cplx* G = qg.data() + std::size_t(k) * n2;
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t x = 0; x < std::ptrdiff_t(n2); ++x) {
      const cplx gx = ck * G[x], lx = ck * L[x];
      cplx* row = dP + std::size_t(x) * n2;
      for (std::size_t a = 0; a < N; ++a)
        for (std::size_t b = 0; b < N; ++b) {
          const std::size_t y = swap34 ? b * N + a : a * N + b;
          const cplx Ly = std::conj(L[y]), Gy = std::conj(G[y]);
          row[a * N + b] += gx * Ly - 2.0 * lx * Ly + lx * Gy;
        }
    }
  }
}

// TPPX memory (h^pp kernel): dP_slab += -sum_f g_f V_slf P_ffab - sum_d g_d P_sldd V*_abd, V = qg - ql
template <class S>
void pp_memory(const S& st, const std::vector<cplx>& ql, const std::vector<cplx>& qg, int Ni, const cplx* P, cplx* dP) {
  const std::size_t N = std::size_t(Ni), n2 = N * N;
  for (int k = 0; k < st.size(); ++k) {
    const std::size_t r = std::size_t(st.site(k));
    const double g = st.gam(k);
    const cplx* L = ql.data() + std::size_t(k) * n2;
    const cplx* G = qg.data() + std::size_t(k) * n2;
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t x = 0; x < std::ptrdiff_t(n2); ++x) {
      const cplx vx = g * (G[x] - L[x]);
      const cplx pxr = g * P[std::size_t(x) * n2 + r * N + r];
      const cplx* Prr = P + (r * N + r) * n2;
      cplx* row = dP + std::size_t(x) * n2;
      for (std::size_t y = 0; y < n2; ++y) row[y] += -vx * Prr[y] - pxr * std::conj(G[y] - L[y]);
    }
  }
}

// ------------------------------------------------------------------ DSL particle-hole rung
// P block (direct + crossed), sec:DSLG1G2lat (DSLlat-P), pinned label = lossy site:
//  direct : g_b ru_sb P_blab + g_s rd_sb P_slas + g_a rd_la P_saab + g_l ru_la P_sllb
//  crossed: g_a ru_sa D_alba + g_s ru_sa D_slbs + g_b rd_lb U_sbba + g_l rd_lb U_slla
template <class S>
void dsl_rung_P(const S& st, const cplx* ru, const cplx* rd, int Ni, const cplx* P, const cplx* U, const cplx* D,
                cplx* dP) {
  const std::size_t N = std::size_t(Ni);
  for (int k = 0; k < st.size(); ++k) {
    const std::size_t r = std::size_t(st.site(k));
    const double g = st.gam(k);
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t si = 0; si < std::ptrdiff_t(N); ++si) {
      const std::size_t s = std::size_t(si);
      for (std::size_t l = 0; l < N; ++l)
        for (std::size_t a = 0; a < N; ++a) {
          // pinned b = r and pinned a = r terms write column (.., ., r) / (., ., r, .) -- loop over the free index
          dP[I4(N, s, l, a, r)] += g * (ru[s * N + r] * P[I4(N, r, l, a, r)] + rd[l * N + r] * U[I4(N, s, r, r, a)]);
          dP[I4(N, s, l, r, a)] += g * (rd[l * N + r] * P[I4(N, s, r, r, a)] + ru[s * N + r] * D[I4(N, r, l, a, r)]);
        }
    }
    // pinned s = r and pinned l = r
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t xi = 0; xi < std::ptrdiff_t(N); ++xi) {
      const std::size_t x = std::size_t(xi);
      for (std::size_t a = 0; a < N; ++a)
        for (std::size_t b = 0; b < N; ++b) {
          // s = r: g rd_rb P_rxar + g ru_ra D_rxbr      (l -> x)
          dP[I4(N, r, x, a, b)] += g * (rd[r * N + b] * P[I4(N, r, x, a, r)] + ru[r * N + a] * D[I4(N, r, x, b, r)]);
        }
    }
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t xi = 0; xi < std::ptrdiff_t(N); ++xi) {
      const std::size_t x = std::size_t(xi);
      for (std::size_t a = 0; a < N; ++a)
        for (std::size_t b = 0; b < N; ++b) {
          // l = r: g ru_ra P_xrrb + g rd_rb U_xrra     (s -> x)
          dP[I4(N, x, r, a, b)] += g * (ru[r * N + a] * P[I4(N, x, r, r, b)] + rd[r * N + b] * U[I4(N, x, r, r, a)]);
        }
    }
  }
}

// same-spin block (DSLlat-U / DSLlat-D): dU += g (X_slab - X_slba),
//  X_slab = ru_sb P_lbab [b] + ru_sb P_lsas [s] + ru_la P_saba [a] + ru_la P_slbl [l]      (U, rho = ru)
//  X_slab = rd_sb P_blba [b] + rd_sb P_slsa [s] + rd_la P_asab [a] + rd_la P_lslb [l]      (D, rho = rd)
template <class S>
void dsl_rung_same(const S& st, const cplx* rho, int Ni, const cplx* P, bool down, cplx* dX) {
  const std::size_t N = std::size_t(Ni);
  // P index helpers for the up (U) and down (D) variants
  auto Pb = [&](std::size_t l, std::size_t a, std::size_t b) {  // pinned b
    return down ? P[I4(N, b, l, b, a)] : P[I4(N, l, b, a, b)];
  };
  auto Ps = [&](std::size_t s, std::size_t l, std::size_t a) {  // pinned s
    return down ? P[I4(N, s, l, s, a)] : P[I4(N, l, s, a, s)];
  };
  auto Pa = [&](std::size_t s, std::size_t a, std::size_t b) {  // pinned a
    return down ? P[I4(N, a, s, a, b)] : P[I4(N, s, a, b, a)];
  };
  auto Pl = [&](std::size_t s, std::size_t l, std::size_t b) {  // pinned l
    return down ? P[I4(N, l, s, l, b)] : P[I4(N, s, l, b, l)];
  };
  for (int k = 0; k < st.size(); ++k) {
    const std::size_t r = std::size_t(st.site(k));
    const double g = st.gam(k);
    // X contributions, each added with + at (s,l,a,b) and - at (s,l,b,a)
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t si = 0; si < std::ptrdiff_t(N); ++si) {
      const std::size_t s = std::size_t(si);
      for (std::size_t l = 0; l < N; ++l)
        for (std::size_t a = 0; a < N; ++a) {
          // [b = r]: rho_sr Pb(l,a,r) at (s,l,a,r); antisym copy at (s,l,r,a)
          const cplx v1 = g * rho[s * N + r] * Pb(l, a, r);
          // [a = r] with free b -> rename b -> a: rho_lr Pa(s, r, a) at (s,l,r,a); copy at (s,l,a,r)
          const cplx v3 = g * rho[l * N + r] * Pa(s, r, a);
          dX[I4(N, s, l, a, r)] += v1 - v3;
          dX[I4(N, s, l, r, a)] += v3 - v1;
        }
    }
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t xi = 0; xi < std::ptrdiff_t(N); ++xi) {
      const std::size_t x = std::size_t(xi);
      for (std::size_t a = 0; a < N; ++a)
        for (std::size_t b = 0; b < N; ++b) {
          // [s = r]: rho_rb Ps(r, x, a) at (r,x,a,b) ; antisym: -rho_ra Ps(r,x,b) at (r,x,a,b)
          dX[I4(N, r, x, a, b)] += g * (rho[r * N + b] * Ps(r, x, a) - rho[r * N + a] * Ps(r, x, b));
          // [l = r]: rho_ra Pl(x, r, b) at (x,r,a,b) ; antisym: -rho_rb Pl(x,r,a)
          dX[I4(N, x, r, a, b)] += g * (rho[r * N + a] * Pl(x, r, b) - rho[r * N + b] * Pl(x, r, a));
        }
    }
  }
}

// ------------------------------------------------------------------ mGW kernels (mGWlat-P..D)
// sign = +1 (P) or -1 (A, U, D). Terms with pinned label j, s, p, o:
//  rho1_{sj} T1_{j o p j} [j] + rho2_{sj} T2_{s o p s} [s] + rho3_{op} T3_{s p p j} [p] + rho4_{op} T4_{s o o j} [o]
// with a per-term index permutation of the read tensor given by the caller (lambdas).
template <class S, class R1, class R2, class R3, class R4>
void mgw_kernel(const S& st, int Ni, double sign, const cplx* rho1, const cplx* rho2, const cplx* rho3,
                const cplx* rho4, R1 t1, R2 t2, R3 t3, R4 t4, cplx* dX) {
  const std::size_t N = std::size_t(Ni);
  for (int k = 0; k < st.size(); ++k) {
    const std::size_t r = std::size_t(st.site(k));
    const double g = sign * st.gam(k);
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t si = 0; si < std::ptrdiff_t(N); ++si) {
      const std::size_t s = std::size_t(si);
      for (std::size_t o = 0; o < N; ++o)
        for (std::size_t p = 0; p < N; ++p) {
          dX[I4(N, s, o, p, r)] += g * rho1[s * N + r] * t1(o, p, r);   // [j = r]
          dX[I4(N, s, o, r, p)] += g * rho3[o * N + r] * t3(s, r, p);   // [p = r], free j renamed p
        }
    }
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t xi = 0; xi < std::ptrdiff_t(N); ++xi) {
      const std::size_t x = std::size_t(xi);
      for (std::size_t a = 0; a < N; ++a)
        for (std::size_t b = 0; b < N; ++b) {
          dX[I4(N, r, x, a, b)] += g * rho2[r * N + b] * t2(r, x, a);   // [s = r]: (r, o=x, p=a, j=b)
          dX[I4(N, x, r, a, b)] += g * rho4[r * N + a] * t4(x, r, b);   // [o = r]: (s=x, r, p=a, j=b)
        }
    }
  }
}

}  // namespace tensor
}  // namespace drv
