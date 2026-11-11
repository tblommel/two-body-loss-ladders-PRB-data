// observables.cpp -- equal-time observables (README "Equal-time observables"). Conventions (xi = -1):
//   rho^sigma_{sa} = <c^dag_{a sigma} c_{s sigma}>,  Gamma_{slab} = <c^dag_a c^dag_b c_l c_s>
//                  = rho_sa rho_lb - rho_la rho_sb + C_{slab},   C_2 = 2 i xi G = -2i G   (the cumulant of
//   the two-particle correlator in the index order of G_{slab}).
// Effective pp-ordered correlator of each scheme (spin string up,dn,up,dn):
//   TPPX(_0)/DSL(_0): G_eff = G^{pp/DSL, udud};  mGW: G_eff = G^{W,udud} - (1/2) G^{W,2,udud}
//   (this is the identification that reproduces the history integral; for mGW it is NOT antisymmetric,
//   so the mGW 2RDM is ambiguous -- see README).
#include "observables.hpp"

#include <cmath>
#include <limits>

#include "tensor.hpp"

namespace drv {

std::vector<long long> bonds_from_h(const Mat* h, int ns, int N) {
  std::vector<long long> b;
  for (int i = 0; i < N; ++i)
    for (int j = i + 1; j < N; ++j) {
      bool on = false;
      for (int s = 0; s < ns; ++s) on = on || std::abs(h[s](i, j)) > 1e-12 || std::abs(h[s](j, i)) > 1e-12;
      if (on) { b.push_back(i); b.push_back(j); }
    }
  return b;
}

void equal_time_observables(const EqContext& c, EqObs& o) {
  using tensor::I4;
  const std::size_t N = std::size_t(c.N), n2 = N * N, M = c.sites.size();
  const cplx* ru = c.rho;
  const cplx* rd = c.ns == 2 ? c.rho + n2 : c.rho;
  const cplx* rs[2] = {ru, rd};
  const bool corr = c.approx != Approx::HF && !c.X.empty() && c.X[0] != nullptr;
  const bool mgw = c.approx == Approx::mGW;
  const bool dsl = c.approx == Approx::DSL;
  // cumulant used in the 2RDM observables (S^2, correlations): pp-type schemes only
  const bool cum2 = corr && !mgw;
  const cplx mi2(0, -2);
  const cplx* P = corr ? c.X[0] : nullptr;
  const cplx* P2 = corr && mgw ? c.X.back() : nullptr;
  const cplx* A = corr && mgw ? c.X[1] : nullptr;
  auto Geff = [&](std::size_t s, std::size_t l, std::size_t a, std::size_t b) -> cplx {
    if (!corr) return 0.0;
    const std::size_t k = I4(N, s, l, a, b);
    return mgw ? P[k] - 0.5 * P2[k] : P[k];
  };
  auto Cud = [&](std::size_t s, std::size_t l, std::size_t a, std::size_t b) { return mi2 * Geff(s, l, a, b); };
  auto Cdu = [&](std::size_t s, std::size_t l, std::size_t a, std::size_t b) { return mi2 * Geff(l, s, b, a); };
  const std::size_t Ap = N * (N - 1) / 2;
  auto Csame = [&](int sp, std::size_t i, std::size_t j) -> cplx {  // C_{i s, j s, i s, j s}
    if (!cum2 || !dsl || i == j) return 0.0;
    if (c.ns == 1) return mi2 * (P[I4(N, i, j, i, j)] - P[I4(N, i, j, j, i)]);
    if (c.X.size() < 3) return 0.0;
    const std::size_t p = tensor::pair_index(N, std::min(i, j), std::max(i, j));
    return mi2 * c.X[1 + sp][p * Ap + p];
  };
  const double nan = std::numeric_limits<double>::quiet_NaN();
  o.bonds = c.bonds;

  // ---- 1. densities
  auto& nsite = o.add("n_site", {2, N}, "Re rho^sigma_rr [T, spin, N]").v;
  auto& Ns = o.add("N_sigma", {2}, "Re tr rho^sigma [T, spin]").v;
  double Nt = 0;
  for (int s = 0; s < 2; ++s) {
    Ns[s] = 0;
    for (std::size_t r = 0; r < N; ++r) { nsite[s * N + r] = rs[s][r * N + r].real(); Ns[s] += nsite[s * N + r]; }
    Nt += Ns[s];
  }
  o.add("N_tot", {}, "N_up + N_dn [T]").v[0] = Nt;
  o.add("Mz", {}, "(N_up - N_dn)/2 [T]").v[0] = 0.5 * (Ns[0] - Ns[1]);

  // ---- 2. double occupancy, routes (a) and (b)
  auto& Db = o.add("D_b", {N}, "route (b): <n_r,up n_r,dn> = rho^u_rr rho^d_rr + Re C_{r u r d r u r d}, every site [T, N]").v;
  for (std::size_t r = 0; r < N; ++r) Db[r] = (ru[r * N + r] * rd[r * N + r] + Cud(r, r, r, r)).real();
  double maxdiff = 0, loss_a = 0, loss_b = 0;
  if (M) {
    auto& Da = o.add("D_a", {M, 2}, "route (a): rho^u_rr rho^d_rr - Re I^sigma_rr / gamma_r per lossy site and spin "
                                    "(NaN where gamma_r = 0) [T, M, spin]").v;
    for (std::size_t q = 0; q < M; ++q) {
      const std::size_t r = std::size_t(c.sites[q]);
      const double g = c.gam[q];
      for (int s = 0; s < 2; ++s) {
        if (g == 0.0) { Da[q * 2 + s] = nan; continue; }
        double d = (ru[r * N + r] * rd[r * N + r]).real();
        if (corr) d -= c.I[std::size_t(c.ns == 2 ? s : 0) * n2 + r * N + r].real() / g;
        Da[q * 2 + s] = d;
        maxdiff = std::max(maxdiff, std::abs(d - Db[r]));
      }
      if (g != 0.0) loss_a += g * 0.5 * (Da[q * 2] + Da[q * 2 + 1]);
      loss_b += g * Db[r];
    }
    if (mgw) {
      auto& Dx = o.add("D_b_exchange", {M}, "mGW only: <n_up n_dn> read from the up,dn,dn,up ordering, rho rho - Re C_{r u r d r d r u} "
                                            "(D_b uses up,dn,up,dn; D_a is their mean) [T, M]").v;
      for (std::size_t q = 0; q < M; ++q) {
        const std::size_t r = std::size_t(c.sites[q]);
        cplx ge = 0.0;
        if (corr) ge = A[I4(N, r, r, r, r)] + 0.5 * P2[I4(N, r, r, r, r)];  // G_eff^{uddu}_{rrrr}, G^{W,2,uddu} = xi G^{W,2,udud}_{slba}
        Dx[q] = (ru[r * N + r] * rd[r * N + r] - mi2 * ge).real();
      }
    }
    o.add("D_ab_maxdiff", {}, "max over lossy sites (gamma_r > 0) and spins of |D_a - D_b| [T]").v[0] = maxdiff;
    // ---- 3. loss rate
    o.add("loss_rate_sum", {}, "sum_r gamma_r D_r (route a, spin mean); dN/dt two-body part = -4 x this [T]").v[0] = loss_a;
    o.add("loss_rate_sum_b", {}, "sum_r gamma_r D_r with route (b) [T]").v[0] = loss_b;
    o.add("dN_dt_check", {}, "dN/dt - (one-body part) + 4 sum_r gamma_r D_a,r (identity: 0 up to round-off) [T]").v[0] =
        c.dN_dt - c.dN_dt_onebody + 4.0 * loss_a;
  }

  // ---- 4/5/6. 2RDM observables
  auto nn = [&](int s1, int s2, std::size_t i, std::size_t j) -> double {
    if (s1 == s2 && i == j) return rs[s1][i * N + i].real();
    cplx v = rs[s1][i * N + i] * rs[s2][j * N + j];
    if (s1 == s2) v -= rs[s1][j * N + i] * rs[s1][i * N + j];
    if (cum2) {
      if (s1 == 0 && s2 == 1) v += Cud(i, j, i, j);
      else if (s1 == 1 && s2 == 0) v += Cdu(i, j, i, j);
      else v += Csame(s1, i, j);
    }
    return v.real();
  };
  std::vector<double> SS(n2), NC(n2);
  double S2 = 0, Sz2 = 0;
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = 0; j < N; ++j) {
      const double uu = nn(0, 0, i, j), dd = nn(1, 1, i, j), ud = nn(0, 1, i, j), du = nn(1, 0, i, j);
      cplx F = (i == j ? ru[i * N + i] : cplx(0)) - ru[j * N + i] * rd[i * N + j];   // <S+_i S-_j>
      cplx Fb = (i == j ? rd[i * N + i] : cplx(0)) - rd[j * N + i] * ru[i * N + j];  // <S-_i S+_j>
      if (cum2) { F -= Cud(j, i, i, j); Fb -= Cdu(j, i, i, j); }
      const double szsz = 0.25 * (uu + dd - ud - du);
      SS[i * N + j] = szsz + 0.5 * (F + Fb).real();
      NC[i * N + j] = uu + dd + ud + du - (nsite[i] + nsite[N + i]) * (nsite[j] + nsite[N + j]);
      S2 += SS[i * N + j];
      Sz2 += szsz;
    }
  auto& lmb = o.add("local_moment_b", {N}, "<S_r^2> = (3/4)(n_r - 2 D_b,r) (equals the 2RDM <S_r.S_r>) [T, N]").v;
  for (std::size_t r = 0; r < N; ++r) lmb[r] = 0.75 * (nsite[r] + nsite[N + r] - 2.0 * Db[r]);
  if (M) {
    auto& lma = o.add("local_moment_a", {M}, "(3/4)(n_r - 2 D_a,r), D_a spin mean (NaN where gamma_r = 0) [T, M]").v;
    const auto& Da = o.add("D_a", {M, 2}, "").v;
    for (std::size_t q = 0; q < M; ++q) {
      const std::size_t r = std::size_t(c.sites[q]);
      lma[q] = 0.75 * (nsite[r] + nsite[N + r] - (Da[q * 2] + Da[q * 2 + 1]));
    }
  }
  auto& lm2 = o.add("local_moment_2rdm", {N}, "<S_r.S_r> from the 2RDM contraction (check: = local_moment_b) [T, N]").v;
  for (std::size_t r = 0; r < N; ++r) lm2[r] = SS[r * N + r];
  o.add("S2_total", {}, "<S^2> = sum_ij <S_i.S_j> [T]").v[0] = S2;
  o.add("Sz2_total", {}, "<S_z^2> = sum_ij <S^z_i S^z_j> [T]").v[0] = Sz2;
  const std::size_t nb = c.bonds.size() / 2;
  if (nb) {
    auto& ssb = o.add("SS_nn", {nb}, "<S_i.S_j> on the bonds (attribute bonds) [T, nb]").v;
    auto& ncb = o.add("nn_conn_nn", {nb}, "<n_i n_j> - <n_i><n_j> on the bonds [T, nb]").v;
    for (std::size_t b = 0; b < nb; ++b) {
      const std::size_t i = std::size_t(c.bonds[2 * b]), j = std::size_t(c.bonds[2 * b + 1]);
      ssb[b] = SS[i * N + j];
      ncb[b] = NC[i * N + j];
    }
  }
  if (c.corr_matrix) {  // never true in the release (see EqContext::corr_matrix)
    o.add("SS_matrix", {N, N}, "<S_i.S_j> [T, N, N] (run card corr_matrix)").v = SS;
    o.add("nn_conn_matrix", {N, N}, "<n_i n_j> - <n_i><n_j> (i = j: includes the on-site term) [T, N, N]").v = NC;
  }

  // ---- 7. energies
  auto& E1 = o.add("E1", {2}, "one-body energy Re tr h^sigma rho^sigma (bare h) [T, spin]").v;
  for (int s = 0; s < 2; ++s) {
    const Mat& h = c.h[c.ns == 2 ? s : 0];
    cplx e = 0;
    for (std::size_t x = 0; x < N; ++x)
      for (std::size_t y = 0; y < N; ++y) e += h(Eigen::Index(x), Eigen::Index(y)) * rs[s][y * N + x];
    E1[s] = e.real();
  }
  if (c.impurity) {
    auto& Ei = o.add("E_imp", {2}, "impurity: h_00 rho_00 [T, spin]").v;
    auto& Eh = o.add("E_hyb", {2}, "impurity: sum_k (h_0k <d^dag c_k> + c.c.) = 2 Re sum_{k>0} h_0k rho_k0 [T, spin]").v;
    auto& Eb = o.add("E_bath", {2}, "impurity: E1 - E_imp - E_hyb [T, spin]").v;
    for (int s = 0; s < 2; ++s) {
      const Mat& h = c.h[c.ns == 2 ? s : 0];
      Ei[s] = (h(0, 0) * rs[s][0]).real();
      cplx e = 0;
      for (std::size_t k = 1; k < N; ++k) e += h(0, Eigen::Index(k)) * rs[s][k * N + 0] + h(Eigen::Index(k), 0) * rs[s][0 * N + k];
      Eh[s] = e.real();
      Eb[s] = E1[s] - Ei[s] - Eh[s];
    }
  }
  // ---- 8. dissipative Galitskii-Migdal trace (complex; [re, im] = [coherent, dissipative])
  {
    auto put = [&](const char* n, cplx v, const char* d) { auto& x = o.add(n, {2}, d).v; x[0] = v.real(); x[1] = v.imag(); };
    put("GM_E_MF", 0.5 * c.lam_mf, "(1/2) tr[-i l^{>,MF} rho] = (1/2) Tr[Sigma^MF G]^< in rho units [T, (re, im)]");
    put("GM_E_corr", 0.5 * c.lam_c, "(1/2) tr[i I] = -(i/2) Tr[Sigma^c * G]^<(t,t), left-sided limit [T, (re, im)]");
    const cplx Eint = 0.5 * (c.lam_mf + c.lam_c);
    put("GM_E_int", Eint, "GM_E_MF + GM_E_corr: re = coherent part (= <H_int> in the Hermitian GM formula; exact Lindblad "
                          "with on-site loss: 0), im = dissipative part (exact: -sum_r gamma_r D_r) [T, (re, im)]");
    put("GM_E_onebody_diss", 0.5 * c.lam_1d, "(1/2) tr[-i(l^> + l^<) rho + i l^<] (one-body bath dissipator) [T, (re, im)]");
    put("GM_E_total", c.lam_h + 0.5 * (c.lam_1d + c.lam_mf + c.lam_c), "tr[h rho] + (1/2)(Lambda_1d + Lambda_MF + Lambda_c) [T, (re, im)]");
    if (M) o.add("GM_check", {}, "Im GM_E_int + sum_r gamma_r D_a,r (identity: 0) [T]").v[0] =
        Eint.imag() + o.add("loss_rate_sum", {}, "").v[0];
  }
  if (o.attrs.empty()) {
    o.attrs.push_back({"cumulant", cum2 ? (dsl ? "C_2 = -2i G (opposite-spin block and same-spin blocks)"
                                                : "C_2 = -2i G^{pp} (opposite-spin block; same-spin blocks are exactly zero in this scheme)")
                                        : (mgw ? "mGW: 2RDM ambiguous (G^W - G^{W,2}/2 is not antisymmetric). D_b and D_b_exchange use the "
                                                 "two orderings; S2, SS, nn_conn, local_moment_2rdm use the HF 2RDM (cumulant set to 0)"
                                               : "none (HF 2RDM)")});
    o.attrs.push_back({"conventions", "rho_sa = <c_a^dag c_s>; Gamma_slab = <c_a^dag c_b^dag c_l c_s> = rho_sa rho_lb - rho_la rho_sb + C_slab; "
                                      "spin-diagonal rho (A1); paramagnetic layout: down = up"});
  }
}

}  // namespace drv
