"""Step-3 tests: TPPX RTDE (slot order F^{pp}_{ablj}, probe j last) with 2B and bare-propagator variants
(run card bare_propagators = 1: TPPX_0, TPPX2B_0).

    python3 -I tests/test_step3.py [test names]

References (vendored in tests/ref/):
  impurity_blocks_check.py: full_rhs (Eq. Fppgen, spin-orbital, any vee), reduced_rhs (Eq. UDfinal, impurity), Ffull;
  item0_check.py: G1-G2 full RHS (via test_step2's full_system).
Uses at most 4 threads.
"""
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv_saved = sys.argv
import test_step2 as T2  # noqa: E402  (models, full G1-G2 references, driver helpers)

AG = {}
exec(open(os.path.join(HERE, "ref", "impurity_blocks_check.py")).read().split("def static_checks")[0], AG)
es, xi = T2.es, -1
EXE = T2.EXE
WORK = os.path.join(HERE, "work3")
T2.WORK = WORK
FAIL = []
rng = np.random.default_rng(77)


def paths_agree(a, b, tol=1e-13):
    """general vs impurity fast path: bitwise, or (when the compiler contracts a*b+c into FMAs differently in the two
    template instantiations, e.g. GCC 16.2 at -march=x86-64-v3) max|a-b| <= tol max|a|. Returns (ok, label)."""
    if np.array_equal(a, b):
        return True, "bitwise"
    d = np.abs(a - b).max() / max(np.abs(a).max(), 1e-300)
    return d <= tol, f"not bitwise, max|d|/max|RHS| = {d:.1e} (tol {tol:.0e}; FMA contraction)"


def check(cond, msg):
    print(("   ok   " if cond else "   FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


rel = T2.rel


# ---------------------------------------------------------------- full spin-orbital RTDE reference (Eq. Fppgen)
def F_full(U, D, s0, N):
    """column j = s0 of F^{pp}_{ablj} from the reduced blocks (all other probe columns zero)."""
    M = 2 * N
    F = np.zeros((M,) * 4, complex)
    for a in range(N):
        for b in range(N):
            for k in range(N):
                F[2 * a, 2 * b + 1, 2 * k, 2 * s0 + 1] = U[a, b, k]
                F[2 * a, 2 * b + 1, 2 * k + 1, 2 * s0] = D[a, b, k]
                F[2 * a + 1, 2 * b, 2 * k, 2 * s0 + 1] = xi * U[b, a, k]
                F[2 * a + 1, 2 * b, 2 * k + 1, 2 * s0] = xi * D[b, a, k]
    return F


def full_rtde_rhs(V, rl, h_pair, GR, G, F, two_b=False):
    """Eq. (Fppgen) with wedge = 0; two_b: h-kernel and G^pp term dropped (Eq. symm-F2pp).
    einsum letters vs IV: F_{abks} here = F^{pp}_{ablj} (k = closing leg l, s = probe j); IR[i, s] = I_{sj}
    (row i -> s, probe s -> j) = 2 [X^F + Xbar^B]_{lsab} F^{pp}_{ablj} = 2i vee_{lsab} F^{pp}_{ablj}."""
    rg = rl + xi * np.eye(rl.shape[0])
    Lg, Ll = es("ac,bd->abdc", rg, rg), es("ac,bd->abdc", rl, rl)
    LL = Lg - Ll
    hd = h_pair.conj().T
    r = -1j * es("ax,xbks->abks", h_pair, F) - 1j * es("bx,axks->abks", h_pair, F) + 1j * es("abxs,xk->abks", F, hd)
    S = es("abdc,dckj->abkj", Ll, V) + xi * es("abdc,dclj,lk->abkj", LL, V, rl)
    if not two_b:
        r = r + es("abcd,cdks->abks", -es("abmn,mncd->abcd", LL, V), F)
        S = S + 1j * es("abef,efkj->abkj", G, V)
    r = r + es("abkj,js->abks", S, GR)
    IR = 2j * es("kiab,abks->is", V, F)
    return r, IR


def reference(m, gamma, ap, ru, rd, P, U, D, GRu, GRd, s0, para=False):
    N = m["h"].shape[1]
    V = T2.full_V(N, m["lossy_sites"], m["site_weight"], gamma)
    hb = [m["h"][s] - 1j * (m["ell_greater"][s] + m["ell_lesser"][s]) for s in range(2)]
    rl = T2.sd(ru, rd)
    hbare = T2.sd(*hb)
    ho = hbare - 1j * 2 * es("iklj,lk->ij", V, rl)
    a, two_b, bare = T2.split_label(ap)
    hp = hbare if bare else ho  # legs a, b, k of F; G^R keeps h_o^MF
    G = T2.assemble_pp(P, 0 * P, 0 * P)
    F = F_full(U, D, s0, N)
    M = 2 * N
    GR = np.zeros((M, M), complex)
    GR[0::2, 2 * s0] = GRu
    GR[1::2, 2 * s0 + 1] = GRd
    r, IR = full_rtde_rhs(V, rl, hp, GR, G, F, two_b=two_b)
    if not two_b:  # cross-check: the agent's full RHS (same slot order) gives the same
        r2, IR2 = AG["full_rhs"](dict(vee=V), F, (rl, rl + xi * np.eye(M), hp, GR, G))
        assert rel(r2, r) < 1e-13 and rel(IR2, IR) < 1e-13 if np.abs(r).max() > 0 else True
    dGR = -1j * ho @ GR - 1j * IR
    dU = np.array([[[r[2 * x, 2 * y + 1, 2 * k, 2 * s0 + 1] for k in range(N)] for y in range(N)] for x in range(N)])
    dD = np.array([[[r[2 * x, 2 * y + 1, 2 * k + 1, 2 * s0] for k in range(N)] for y in range(N)] for x in range(N)])
    return dU, dD, dGR[0::2, 2 * s0], dGR[1::2, 2 * s0 + 1]


def eval_rtde(rc, ru, rd, P, GRu, GRd, D, U, para=False):
    N = ru.shape[0]
    parts = [ru.ravel()] + ([] if para else [rd.ravel()]) + [P.ravel(), GRu] + ([] if para else [GRd]) + [D.ravel()] \
        + ([] if para else [U.ravel()])
    y = np.concatenate(parts).astype(complex)
    fi, fo = os.path.join(WORK, "ein.h5"), os.path.join(WORK, "eout.h5")
    with h5py.File(fi, "w") as f:
        f["y"] = y
    p = subprocess.run([EXE, "--runcard", rc, "--eval-rhs", fi, fo], env=T2.ENV, capture_output=True, text=True)
    if p.returncode:
        print(p.stdout, p.stderr)
    with h5py.File(fo, "r") as f:
        dy = f["dy"][()]
    n2, n3, n4 = N * N, N ** 3, N ** 4
    ns = 1 if para else 2
    o = ns * n2 + n4
    dGR = dy[o:o + ns * N]
    o += ns * N
    dD = dy[o:o + n3].reshape(N, N, N)
    dU = None if para else dy[o + n3:o + 2 * n3].reshape(N, N, N)
    return dGR[:N], (dGR[:N] if para else dGR[N:]), dD, dU, dy


def rand_inputs(N, para=False):
    ru = T2.herm(N)
    rd = ru.copy() if para else T2.herm(N)
    P = T2.rand_blocks("TPPX", N, para=para)[0]
    U = T2.crand(N, N, N)
    D = xi * U.transpose(1, 0, 2) if para else T2.crand(N, N, N)  # R6: D_ablj = xi U_balj
    GRu = T2.crand(N)
    GRd = GRu.copy() if para else T2.crand(N)
    return ru, rd, P, U, D, GRu, GRd


# ---------------------------------------------------------------- tests
VARIANTS = ["TPPX", "TPPX2B", "TPPX_0", "TPPX2B_0"]


def test_rhs_random():
    print("[1] RTDE RHS on random inputs vs full spin-orbital Eq. (Fppgen) (slot order F^pp_ablj)")
    imp = T2.impurity_model(os.path.join(WORK, "imp5.h5"), 5)
    ring = T2.ring_model(os.path.join(WORK, "ring.h5"), 4, [0, 2], [1.0, 0.5])
    for gname, mp, m, vimpls, s0 in (("impurity N=5", "imp5.h5", imp, ("general", "impurity"), 0),
                                      ("impurity N=5, probe column 2", "imp5.h5", imp, ("general", "impurity"), 2),
                                      ("ring 4, loss {0,2}, column 1", "ring.h5", ring, ("general",), 1)):
        N = m["h"].shape[1]
        for ap in VARIANTS:
            ru, rd, P, U, D, GRu, GRd = rand_inputs(N)
            ref = reference(m, 0.8, ap, ru, rd, P, U, D, GRu, GRd, s0)
            res = {}
            for vi in vimpls:
                rc = T2.write_rc(f"rc3_{ap}_{vi}_{s0}_{mp}", os.path.join(WORK, mp), 0.8, ap, vertex_impl=vi, mode="rtde",
                                 t_prime=[0.5], probe_columns=[s0])
                gu, gd, dD, dU, dy = eval_rtde(rc, ru, rd, P, GRu, GRd, D, U)
                res[vi] = dy
                e = max(rel(dU, ref[0]), rel(dD, ref[1]), rel(gu, ref[2]), rel(gd, ref[3]))
                check(e < 1e-13, f"{gname:30s} {ap:8s} {vi:8s}: U, D, G^R vs full Fppgen {e:.1e}")
            if len(vimpls) == 2:
                ok, how = paths_agree(res["general"], res["impurity"])
                check(ok, f"{gname:30s} {ap:8s}: general {{0}} == impurity fast path, {how}")
    # impurity: against the agent's reduced Eq. (UDfinal) and the three-rho form (UDsrc3rho)
    m = imp
    N = 5
    ru, rd, P, U, D, GRu, GRd = rand_inputs(N)
    hb = [m["h"][s] - 1j * (m["ell_greater"][s] + m["ell_lesser"][s]) for s in range(2)]
    ho = [hb[0].copy(), hb[1].copy()]
    ho[0][0, 0] -= 1j * 0.8 * rd[0, 0].real
    ho[1][0, 0] -= 1j * 0.8 * ru[0, 0].real
    mag = dict(N=N, gam=0.8, rl={0: ru, 1: rd}, h={0: ho[0], 1: ho[1]}, Gred=P)
    dUa, dDa, dGu, dGd, _, _ = AG["reduced_rhs"](mag, U[..., None], D[..., None], {0: GRu[:, None], 1: GRd[:, None]})
    rc = T2.write_rc("rc3_ag.h5", os.path.join(WORK, "imp5.h5"), 0.8, "TPPX", vertex_impl="impurity", mode="rtde",
                     t_prime=[0.5], probe_columns=[0])
    gu, gd, dD, dU, _ = eval_rtde(rc, ru, rd, P, GRu, GRd, D, U)
    e = max(rel(dU, dUa[..., 0]), rel(dD, dDa[..., 0]), rel(gu, dGu[:, 0]), rel(gd, dGd[:, 0]))
    check(e < 1e-13, f"impurity: driver == agent's reduced Eq. (UDfinal) (impurity_blocks_check.reduced_rhs): {e:.1e}")
    K = 2j * P[:, :, 0, 0]
    Ql = np.outer(ru[:, 0], rd[:, 0])
    Qg = np.outer(ru[:, 0] - np.eye(N)[:, 0], rd[:, 0] - np.eye(N)[:, 0])
    Vv = Qg - Ql
    for sg, rs in (("up", ru), ("dn", rd)):
        rsg = rs - np.eye(N)
        delta = es("ab,k->abk", Ql - K, np.eye(N)[0]) - es("ab,k->abk", Vv, rs[0])
        three = -(es("ab,k->abk", K, np.eye(N)[0]) + es("ab,k->abk", Qg, rs[0]) + es("ab,k->abk", Ql, rsg[0])
                  - 2 * es("ab,k->abk", Ql, rs[0]))
        check(rel(delta, three) < 1e-14, f"source: delta form (UDfinal) == three-rho form (UDsrc3rho), sigma {sg}: "
                                         f"{rel(delta, three):.1e}")
    # HF limit: with U = D = 0, d G^R equals the HF G^R equation
    ru, rd, P, U, D, GRu, GRd = rand_inputs(5)
    rcT = T2.write_rc("rc3_hfl_T.h5", os.path.join(WORK, "imp5.h5"), 0.8, "TPPX", mode="rtde", t_prime=[0.5], probe_columns=[0])
    rcH = T2.write_rc("rc3_hfl_H.h5", os.path.join(WORK, "imp5.h5"), 0.8, "HF", mode="rtde", t_prime=[0.5], probe_columns=[0])
    gu, gd, _, _, _ = eval_rtde(rcT, ru, rd, P, GRu, GRd, 0 * D, 0 * U)
    yH = np.concatenate([ru.ravel(), rd.ravel(), GRu, GRd]).astype(complex)
    with h5py.File(os.path.join(WORK, "ein.h5"), "w") as f:
        f["y"] = yH
    subprocess.run([EXE, "--runcard", rcH, "--eval-rhs", os.path.join(WORK, "ein.h5"), os.path.join(WORK, "eout.h5")], env=T2.ENV)
    with h5py.File(os.path.join(WORK, "eout.h5"), "r") as f:
        dH = f["dy"][()][2 * 25:]
    check(np.array_equal(np.concatenate([gu, gd]), dH), "HF limit: with U = D = 0 the TPPX d G^R equals the HF d G^R bitwise")


def augmented_reference(m, gamma, ap, tp, t_end, s0, t_eval):
    """full spin-orbital G1-G2 (item0 rhs_pp) to t', then G1-G2 + F_ablj (column j = s0) + G^R to t_end."""
    N = m["h"].shape[1]
    M = 2 * N
    a, two_b, bare = T2.split_label(ap)
    full_rhs, _, V = T2.full_system(m, gamma, ap)
    hb = [m["h"][s] - 1j * (m["ell_greater"][s] + m["ell_lesser"][s]) for s in range(2)]
    hbare = T2.sd(*hb)
    def f1(t, y):
        rl = y[:M * M].reshape(M, M); G = y[M * M:].reshape((M,) * 4)
        drho, dT = full_rhs(rl, [G])
        return np.concatenate([drho.ravel(), dT[0].ravel()])
    y0 = np.concatenate([T2.sd(m["rho"][0], m["rho"][1]).ravel(), np.zeros(M ** 4, complex)])
    s1 = solve_ivp(f1, (0, tp), y0, method="DOP853", rtol=1e-11, atol=1e-13)
    def f2(t, y):
        rl = y[:M * M].reshape(M, M); G = y[M * M:M * M + M ** 4].reshape((M,) * 4)
        F = y[M * M + M ** 4:M * M + 2 * M ** 4].reshape((M,) * 4); GR = y[M * M + 2 * M ** 4:].reshape(M, M)
        drho, dT = full_rhs(rl, [G])
        ho = hbare - 1j * 2 * es("iklj,lk->ij", V, rl)
        dF, IR = full_rtde_rhs(V, rl, hbare if bare else ho, GR, G, F, two_b=two_b)
        dGR = -1j * ho @ GR - 1j * IR
        return np.concatenate([drho.ravel(), dT[0].ravel(), dF.ravel(), dGR.ravel()])
    GR0 = np.zeros((M, M), complex)
    GR0[2 * s0, 2 * s0] = GR0[2 * s0 + 1, 2 * s0 + 1] = -1j
    y1 = np.concatenate([s1.y[:, -1], np.zeros(M ** 4, complex), GR0.ravel()])
    s2 = solve_ivp(f2, (tp, t_end), y1, method="DOP853", rtol=1e-11, atol=1e-13, t_eval=t_eval)
    GR = s2.y[M * M + 2 * M ** 4:].T.reshape(len(t_eval), M, M)
    return np.stack([GR[:, 0::2, 2 * s0], GR[:, 1::2, 2 * s0 + 1]], axis=1)  # [T, spin, x]


def test_trajectories():
    print("[2] coupled G1-G2 + RTDE trajectories vs full spin-orbital solve_ivp (rtol 1e-11)")
    imp = T2.impurity_model(os.path.join(WORK, "imp3.h5"), 3)
    ring = T2.ring_model(os.path.join(WORK, "ring3.h5"), 3, [0, 2], [1.0, 0.5])
    for gname, mp, m, s0 in (("impurity N=3", "imp3.h5", imp, 0), ("ring 3, loss {0,2}, column 1", "ring3.h5", ring, 1)):
        for ap in VARIANTS:
            rc = T2.write_rc(f"rc3t_{ap}_{mp}", os.path.join(WORK, mp), 1.0, ap, mode="rtde", t_prime=[1.0],
                             probe_columns=[s0], t_end=3.0, dt_out=0.1)
            T2.run(rc, f"t3_{ap}_{mp}")
            with h5py.File(os.path.join(WORK, f"t3_{ap}_{mp}"), "r") as f:
                tt, GR = f["rtde/tp=0/t"][()], f["rtde/tp=0/GR"][()][:, 0]
            ref = augmented_reference(m, 1.0, ap, 1.0, 3.0, s0, tt)
            e = np.abs(GR - ref).max()
            rch = T2.write_rc(f"rc3t_HF_{mp}", os.path.join(WORK, mp), 1.0, "HF", mode="rtde", t_prime=[1.0],
                              probe_columns=[s0], t_end=3.0, dt_out=0.1)
            if not os.path.exists(os.path.join(WORK, f"t3_HF_{mp}")):
                T2.run(rch, f"t3_HF_{mp}")
            with h5py.File(os.path.join(WORK, f"t3_HF_{mp}"), "r") as f:
                dHF = np.abs(f["rtde/tp=0/GR"][()][:, 0] - GR).max()
            check(e < 2e-6, f"{gname:28s} {ap:8s}: max|G^R - ref| over {len(tt)} frames = {e:.1e} "
                            f"(|G^R - G^R_HF| up to {dHF:.1e})")


def test_gamma0():
    print("[3] gamma = 0: G^R = -i exp(-i h_o tau) e_j for HF and every TPPX variant")
    imp = os.path.join(WORK, "imp3.h5")
    m = T2.dp5model.load_model(imp)
    ho = m["h"] - 1j * (m["ell_greater"] + m["ell_lesser"])
    for ap in ("HF",) + tuple(VARIANTS):
        e = 0.0
        for s in (0, 2):
            rc = T2.write_rc(f"rc3g0_{ap}_{s}.h5", imp, 0.0, ap, mode="rtde", t_prime=[1.0], probe_columns=[s], t_end=4.0,
                             dt_out=0.1)
            T2.run(rc, f"g0_{ap}_{s}.h5")
            with h5py.File(os.path.join(WORK, f"g0_{ap}_{s}.h5"), "r") as f:
                tt, GR = f["rtde/tp=0/t"][()], f["rtde/tp=0/GR"][()]
            e = max(e, max(np.abs(GR[k, 0, sp] - (-1j * expm(-1j * ho[sp] * (t - 1.0))[:, s])).max()
                           for k, t in enumerate(tt) for sp in range(2)))
        check(e < 1e-7, f"gamma = 0, {ap:6s}: G^R = -i exp(-i h_o tau) e_s (columns 0, 2): {e:.1e}")


def exact_single_site_GR(eps0, gam, nu, nd, tp, taus):
    """4-state Lindblad (main Eq. LIND, L = sqrt(gamma) d_dn d_up), quantum regression for
    G^R_up(tp + tau, tp) = -i <{d_up(tp+tau), d_up^dag(tp)}>."""
    c = np.array([[0, 1], [0, 0]], complex)
    Z = np.diag([1.0, -1.0]).astype(complex)
    I2 = np.eye(2)
    du, dd = np.kron(c, I2), np.kron(Z, c)                 # Jordan-Wigner, mode order (up, dn)
    nu_, nd_ = du.conj().T @ du, dd.conj().T @ dd
    H = eps0 * (nu_ + nd_)
    L = np.sqrt(gam) * dd @ du
    LdL = L.conj().T @ L
    I4 = np.eye(4)
    # vec(rho) column-major: vec(A X B) = (B^T kron A) vec X
    Lv = (-1j * (np.kron(I4, H) - np.kron(H.T, I4)) + 2 * np.kron(L.conj(), L) - np.kron(I4, LdL) - np.kron(LdL.T, I4))
    vec = lambda X: X.reshape(-1, order="F")
    mat = lambda v: v.reshape(4, 4, order="F")
    r0 = np.kron(np.diag([1 - nu, nu]), np.diag([1 - nd, nd])).astype(complex)  # product state, basis (up, dn) occupation
    # occupation basis order of kron(c-space): |0>=empty? c = [[0,1],[0,0]] annihilates |1>(index 1) -> index 0
    rho_tp = mat(expm(Lv * tp) @ vec(r0))
    out = []
    for tau in taus:
        E = expm(Lv * tau)
        a = np.trace(du @ mat(E @ vec(du.conj().T @ rho_tp)))
        b = np.trace(du @ mat(E @ vec(rho_tp @ du.conj().T)))
        out.append(-1j * (a + b))
    return np.array(out)


def test_single_site():
    print("[4] single site: G^R vs exact 4-state Lindblad (quantum regression)")
    eps0, gam, nu, nd, tp = -0.4, 1.3, 0.8, 0.6, 0.7
    h = np.full((2, 1, 1), eps0, complex)
    z = np.zeros_like(h)
    T2.dp5model.write_model(os.path.join(WORK, "site.h5"), h, z, z, np.array([[[nu]], [[nd]]], complex), [0], [1.0], {},
                            dict(generator="single site"))
    res = {}
    m = T2.dp5model.load_model(os.path.join(WORK, "site.h5"))
    for ap in ("HF",) + tuple(VARIANTS):
        rc = T2.write_rc(f"rc3ss_{ap}.h5", os.path.join(WORK, "site.h5"), gam, ap, mode="rtde", t_prime=[tp], t_end=tp + 3.0,
                         dt_out=0.1)
        if os.path.exists(os.path.join(WORK, f"site3_{ap}.h5")):
            os.remove(os.path.join(WORK, f"site3_{ap}.h5"))
        T2.run(rc, f"site3_{ap}.h5")
        with h5py.File(os.path.join(WORK, f"site3_{ap}.h5"), "r") as f:
            tt, GR = f["rtde/tp=0/t"][()], f["rtde/tp=0/GR"][()][:, 0, 0, 0]
            trho, nup = f["rho/t"][()], f["rho/rho"][()][:, 0, 0, 0].real
        ex = exact_single_site_GR(eps0, gam, nu, nd, tp, tt - tp)
        if ap == "HF":  # quantum regression == closed form -i e^{-i eps tau}[1 - (1 - e^{-gamma tau}) n_dn(t')]
            ndt = exact_single_site_n(eps0, gam, nu, nd, tp)[1]
            an = -1j * np.exp(-1j * eps0 * (tt - tp)) * (1 - (1 - np.exp(-gam * (tt - tp))) * ndt)
            check(rel(ex, an) < 1e-13, f"exact reference: quantum regression == closed form {rel(ex, an):.1e}")
        res[ap] = np.abs(GR - ex).max()
        en = max(abs(nup[k] - exact_single_site_n(eps0, gam, nu, nd, t)[0]) for k, t in enumerate(trho))
        print(f"      {ap:6s}: max|G^R_up - exact| = {res[ap]:.2e}   max|n_up - exact| = {en:.2e}")
        if ap == "TPPX_0":
            check(en < 1e-7, f"TPPX_0: n_up(t) exact on a pure-loss single site ({en:.1e})")
        if ap != "HF":
            ref = augmented_reference(m, gam, ap, tp, tp + 3.0, 0, tt)[:, 0, 0]
            check(np.abs(GR - ref).max() < 1e-6, f"{ap:6s}: single-site G^R == full spin-orbital reference "
                                                 f"{np.abs(GR - ref).max():.1e}")
    # finding (not asserted as exactness): the TPPX_0 RTDE G^R is NOT exact on one site although its density is
    check(res["TPPX_0"] < res["HF"] < res["TPPX"], "single-site G^R error ordering TPPX_0 < HF < TPPX (recorded finding)")


def exact_single_site_n(eps0, gam, nu, nd, t):
    c = np.array([[0, 1], [0, 0]], complex)
    Z = np.diag([1.0, -1.0]).astype(complex)
    du, dd = np.kron(c, np.eye(2)), np.kron(Z, c)
    L = np.sqrt(gam) * dd @ du
    LdL = L.conj().T @ L
    H = eps0 * (du.conj().T @ du + dd.conj().T @ dd)
    I4 = np.eye(4)
    Lv = (-1j * (np.kron(I4, H) - np.kron(H.T, I4)) + 2 * np.kron(L.conj(), L) - np.kron(I4, LdL) - np.kron(LdL.T, I4))
    r0 = np.kron(np.diag([1 - nu, nu]), np.diag([1 - nd, nd])).astype(complex)
    r = (expm(Lv * t) @ r0.reshape(-1, order="F")).reshape(4, 4, order="F")
    return np.trace(du.conj().T @ du @ r).real, np.trace(dd.conj().T @ dd @ r).real

if __name__ == "__main__":
    tests = [test_rhs_random, test_trajectories, test_gamma0, test_single_site]
    sel = sys.argv[1:]
    if not sel:
        shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK, exist_ok=True)
    if sel and not os.path.exists(os.path.join(WORK, "imp5.h5")):
        sel = ["test_rhs_random"] + sel
    for t in tests:
        if not sel or t.__name__ in sel:
            t()
    print(f"\n{'ALL STEP-3 TESTS PASSED' if not FAIL else 'FAILURES: ' + str(len(FAIL))}")
    sys.exit(1 if FAIL else 0)
