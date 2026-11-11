"""Step-3b tests: equal-time observables (/observables/equal_time) and the spectral post-processing.

    python3 -I tests/test_step3b.py [test names]

References: an exact Lindblad density-matrix solver (exact diagonalization, Jordan-Wigner, written here);
a generic spin-orbital 2RDM contraction (Gamma = HF + C_2, C_2 = 2 i xi G, keldysh-rules 19.4) written
independently of the driver's closed formulas. Uses at most 4 threads.
"""
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_step2 as T2  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))
import read_output  # noqa: E402

es, xi = T2.es, -1
EXE = T2.EXE
WORK = os.path.join(HERE, "work3b")
T2.WORK = WORK
FAIL = []
rng = np.random.default_rng(31)
EQ = "observables/equal_time/"


def check(cond, msg):
    print(("   ok   " if cond else "   FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def run(rc, out):
    p = os.path.join(WORK, out)
    if os.path.exists(p):
        os.remove(p)
    r = T2.run(rc, out)
    if r.returncode:
        raise RuntimeError(f"driver failed on {rc}")
    return p


def load_eq(path):
    with h5py.File(path, "r") as f:
        g = f[EQ]
        d = {k: g[k][()] for k in g.keys()}
        d["t"] = f["time/t"][()]
        d["_attrs"] = {k: g.attrs[k] for k in g.attrs.keys()}
        d["dN_dt"] = f["observables/dN_dt"][()]
    return d


# ---------------------------------------------------------------- generic 2RDM observables (spin-orbital 2r+sigma)
def pauli():
    return [np.array([[0, 1], [1, 0]], complex) / 2, np.array([[0, -1j], [1j, 0]]) / 2, np.diag([0.5, -0.5]).astype(complex)]


def obs_from_rdm(rho, Gam, N):
    """rho_{sa} = <c_a^dag c_s>, Gam_{slab} = <c_a^dag c_b^dag c_l c_s> (spin-orbital 2r+sigma)."""
    o = {}
    n = np.array([[rho[2 * r + s, 2 * r + s].real for r in range(N)] for s in range(2)])
    o["n_site"] = n
    o["D_b"] = np.array([Gam[2 * r, 2 * r + 1, 2 * r, 2 * r + 1].real for r in range(N)])
    # <S_i.S_j> = 1/4 sum_a tau^a_{st} tau^a_{s't'} [delta_ij delta_{t s'} rho_{j t', i s} + Gam_{i t, j t', i s, j s'}]
    SS = np.zeros((N, N))
    NN = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            v = 0
            for tau in pauli():
                for s in range(2):
                    for t in range(2):
                        for s2 in range(2):
                            for t2 in range(2):
                                c = tau[s, t] * tau[s2, t2]
                                if c == 0:
                                    continue
                                x = Gam[2 * i + t, 2 * j + t2, 2 * i + s, 2 * j + s2]
                                if i == j and t == s2:
                                    x += rho[2 * j + t2, 2 * i + s]
                                v += c * x
            SS[i, j] = v.real
            w = 0
            for s in range(2):
                for s2 in range(2):
                    x = Gam[2 * i + s, 2 * j + s2, 2 * i + s, 2 * j + s2]
                    if i == j and s == s2:
                        x += rho[2 * i + s, 2 * i + s]
                    w += x
            NN[i, j] = w.real - n[:, i].sum() * n[:, j].sum()
    o["SS_matrix"], o["nn_conn_matrix"], o["S2_total"] = SS, NN, SS.sum()
    o["local_moment_2rdm"] = np.diag(SS).copy()
    return o


def gam_hf(rho):
    return es("sa,lb->slab", rho, rho) - es("la,sb->slab", rho, rho)


# ---------------------------------------------------------------- exact Lindblad (ED)
def fermion_ops(M):
    a = sp.csr_matrix(np.array([[0, 1], [0, 0]], complex))
    Z = sp.csr_matrix(np.diag([1.0, -1.0]).astype(complex))
    I2 = sp.identity(2, dtype=complex, format="csr")
    ops = []
    for k in range(M):
        o = sp.identity(1, dtype=complex, format="csr")
        for q in range(M):
            o = sp.kron(o, Z if q < k else (a if q == k else I2), format="csr")
        ops.append(o)
    return ops


class Exact:
    """rho_ED(t) for main Eq. (LIND): d rho = -i[H, rho] + sum_L (2 L rho L^dag - {L^dag L, rho}),
    H = sum h^s_xy c^dag_xs c_ys; one-body jumps from l^> (loss) and l^< (gain); L_r = sqrt(gamma w_r) c_r,dn c_r,up."""

    def __init__(self, m, gamma):
        self.N = N = m["h"].shape[1]
        M = 2 * N
        self.c = c = fermion_ops(M)
        cd = [x.conj().T.tocsr() for x in c]
        self.cd = cd
        H = sum(m["h"][s][x, y] * cd[2 * x + s] @ c[2 * y + s] for s in range(2) for x in range(N) for y in range(N)
                if m["h"][s][x, y] != 0)
        Ls = []
        for s in range(2):
            for which, mat in (("g", m["ell_greater"][s]), ("l", m["ell_lesser"][s])):
                lam, U = np.linalg.eigh(mat)
                for k in range(N):
                    if lam[k] < 1e-14:
                        continue
                    if which == "g":  # l^>_xy = sum alpha*_x alpha_y, L = sum_y alpha_y c_y
                        L = sum(np.sqrt(lam[k]) * U[y, k].conj() * c[2 * y + s] for y in range(N))
                    else:  # l^<_xy = beta_x beta*_y, L = sum_y beta_y c^dag_y
                        L = sum(np.sqrt(lam[k]) * U[y, k] * cd[2 * y + s] for y in range(N))
                    Ls.append(sp.csr_matrix(L))
        self.n_odd = len(Ls)  # one-body jumps are fermion-odd
        for r, w in zip(m["lossy_sites"], m["site_weight"]):
            if gamma * w > 0:
                Ls.append(sp.csr_matrix(np.sqrt(gamma * w) * c[2 * r + 1] @ c[2 * r]))
        K = sum((L.conj().T @ L for L in Ls), sp.csr_matrix(H.shape, dtype=complex))
        self.Heff = sp.csr_matrix(H - 1j * K)
        self.Ls, self.LsH = Ls, [L.conj().T.tocsr() for L in Ls]
        self.dim = H.shape[0]
        # Gaussian initial state with 1RDM rho0 (natural orbitals)
        rho1 = T2.sd(m["rho"][0], m["rho"][1])
        nk, V = np.linalg.eigh(rho1)
        R = np.eye(self.dim, dtype=complex)
        for k in range(M):
            b = sum(V[s, k].conj() * c[s] for s in range(M))
            nb = (b.conj().T @ b).toarray()
            R = R @ ((1 - nk[k]) * np.eye(self.dim) + (2 * nk[k] - 1) * nb)
        self.R0 = R
        self.h = m["h"]
        self.m, self.gamma = m, gamma

    def L_on(self, Y, odd):
        """Lindbladian on an operator Y; for fermion-odd Y the odd jumps get a sign (2 L Y L^dag -> -2 L Y L^dag)."""
        d = -1j * (self.Heff @ Y) + 1j * (self.Heff.conj() @ Y.T).T
        for k, L in enumerate(self.Ls):
            sg = -1.0 if (odd and k < self.n_odd) else 1.0
            d = d + 2 * sg * (L.conj() @ (L @ Y).T).T
        return d

    def gm(self, R):
        """exact left/right equal-time limits by quantum regression:
        Lambda_left = sum_s i d/dtau <c_s^dag(t) c_s(t+tau)>|_0 = sum_s i Tr[c_s L(R c_s^dag)],
        Lambda_right = sum_s -i d/dtau <c_s^dag(t+tau) c_s(t)>|_0 = sum_s -i Tr[c_s^dag L(c_s R)]."""
        M = 2 * self.N
        lf = sum(1j * np.trace(self.c[s] @ self.L_on(R @ self.cd[s].toarray(), True)) for s in range(M))
        rt = sum(-1j * np.trace(self.cd[s] @ self.L_on(self.c[s] @ R, True)) for s in range(M))
        dN = sum(np.trace((self.cd[s] @ self.c[s]) @ self.L_on(R, False)) for s in range(M))
        return lf, rt, dN

    def rhs(self, t, y):
        R = y.reshape(self.dim, self.dim)
        HR = self.Heff @ R
        d = -1j * HR + 1j * (self.Heff @ R.conj().T).conj().T  # -i Heff R + i R Heff^dag
        for L, LH in zip(self.Ls, self.LsH):
            X = L @ R
            d += 2 * (L.conj() @ X.T).T  # 2 L R L^dag = 2 (L^* (L R)^T)^T
        return d.ravel()

    def evolve(self, ts):
        s = solve_ivp(self.rhs, (ts[0], ts[-1]), self.R0.ravel(), t_eval=ts, method="DOP853", rtol=1e-10, atol=1e-12)
        return [s.y[:, k].reshape(self.dim, self.dim) for k in range(len(ts))]

    def observables(self, R):
        N, c, cd = self.N, self.c, self.cd
        M = 2 * N
        ev = lambda O: np.trace(O @ R) if not sp.issparse(O) else (O.multiply(R.T)).sum()
        rho = np.array([[ev(cd[a] @ c[s]) for a in range(M)] for s in range(M)])
        nop = [cd[k] @ c[k] for k in range(M)]
        o = {"n_site": np.array([[rho[2 * r + s, 2 * r + s].real for r in range(N)] for s in range(2)])}
        o["D_b"] = np.array([ev(nop[2 * r] @ nop[2 * r + 1]).real for r in range(N)])
        Sops = [[sum(tau[s, t] * cd[2 * i + s] @ c[2 * i + t] for s in range(2) for t in range(2) if tau[s, t] != 0)
                 for tau in pauli()] for i in range(N)]
        SS = np.array([[sum(ev(Sops[i][a] @ Sops[j][a]) for a in range(3)).real for j in range(N)] for i in range(N)])
        ntot = [nop[2 * i] + nop[2 * i + 1] for i in range(N)]
        nmean = [ev(x).real for x in ntot]
        NN = np.array([[ev(ntot[i] @ ntot[j]).real - nmean[i] * nmean[j] for j in range(N)] for i in range(N)])
        o.update(SS_matrix=SS, nn_conn_matrix=NN, S2_total=SS.sum(), local_moment_2rdm=np.diag(SS).copy())
        o["E1"] = np.array([sum(self.h[s][x, y] * rho[2 * y + s, 2 * x + s] for x in range(N) for y in range(N)).real
                            for s in range(2)])
        o["E_hyb"] = np.array([2 * sum(self.h[s][0, k] * rho[2 * k + s, 2 * 0 + s] for k in range(1, N)).real for s in range(2)])
        return o


# ---------------------------------------------------------------- models
def lattice2(path, sites=(0, 1), weights=(1.0, 0.7), rho=None):
    h = np.zeros((2, 2, 2), complex)
    for s, dz in enumerate((0.25, -0.25)):
        h[s] = np.array([[0.3 + dz, -1.0], [-1.0, -0.2 + dz]])
    if rho is None:
        rho = np.stack([np.array([[0.9, 0.1 + 0.05j], [0.1 - 0.05j, 0.6]]), np.array([[0.5, -0.1], [-0.1, 0.8]])])
    z = np.zeros_like(h)
    T2.dp5model.write_model(path, h, z, z, rho, list(sites), list(weights), dict(ordering="2-site"), dict(generator="test_step3b"))
    return T2.dp5model.load_model(path)


def swap_spin(m, path):
    T2.dp5model.write_model(path, m["h"][::-1], m["ell_greater"][::-1], m["ell_lesser"][::-1], m["rho"][::-1],
                            list(m["lossy_sites"]), list(m["site_weight"]), {}, dict(generator="spin swapped"))
    return T2.dp5model.load_model(path)


APPROX = ["HF", "TPPX", "TPPX2B", "TPPX_0", "mGW", "DSL", "mGW_0", "DSL_0"]  # X_0: bare propagators


def bonds(h):
    """the driver's bond list (observables.cpp bonds_from_h): pairs i<j with |h^sigma_ij| > 1e-12 for some spin"""
    N = h.shape[1]
    return [(i, j) for i in range(N) for j in range(i + 1, N)
            if any(abs(h[s, i, j]) > 1e-12 or abs(h[s, j, i]) > 1e-12 for s in range(h.shape[0]))]


def on_bonds(X, bl):
    """[..., N, N] matrix -> [..., nb] values on the bonds"""
    return np.stack([X[..., i, j] for i, j in bl], axis=-1)


# ---------------------------------------------------------------- tests
def test_formulas():
    print("[1] closed formulas vs generic spin-orbital 2RDM contraction (random inputs, --eval-rhs)")
    m = T2.ring_model(os.path.join(WORK, "ring4.h5"), 4, [0, 2], [1.0, 0.5])
    bl = bonds(m["h"])
    N = 4
    for para in (False,):
        mp = "ring4.h5"
        for ap in ("HF", "TPPX", "DSL"):
            ru = T2.herm(N)
            rd = T2.herm(N)
            blocks = [] if ap == "HF" else T2.rand_blocks(ap, N)
            rc = T2.write_rc(f"rcf_{ap}_{para}.h5", os.path.join(WORK, mp), 0.9, ap)
            T2.eval_rhs(rc, ru, rd, T2.driver_blocks(ap, blocks))
            with h5py.File(os.path.join(WORK, "eval_out.h5"), "r") as f:
                drv = {k: f["equal_time"][k][()] for k in f["equal_time"].keys()}
            rho = T2.sd(ru, rd)
            if ap == "HF":
                G = np.zeros((2 * N,) * 4, complex)
            elif ap == "TPPX":
                G = T2.assemble_pp(blocks[0], 0 * blocks[0], 0 * blocks[0])
            else:
                G = T2.assemble_pp(*blocks)
            ref = obs_from_rdm(rho, gam_hf(rho) - 2j * G, N)
            ref["SS_nn"], ref["nn_conn_nn"] = on_bonds(ref["SS_matrix"], bl), on_bonds(ref["nn_conn_matrix"], bl)
            e = max(rel(drv[k], ref[k]) for k in ("SS_nn", "nn_conn_nn", "D_b", "local_moment_2rdm")) \
                + abs(drv["S2_total"][0] - ref["S2_total"]) / max(1, abs(ref["S2_total"]))
            e2 = rel(drv["local_moment_b"], drv["local_moment_2rdm"])
            check(e < 1e-13 and e2 < 1e-13, f"{ap:4s} {'paramagnetic' if para else 'full':12s}: S_i.S_j and <n_i n_j>_c on the bonds, D_b, "
                                            f"<S_r^2>, S^2 vs generic contraction {e:.1e}; (3/4)(n-2D) == 2RDM <S_r.S_r> {e2:.1e}")
            if ap != "HF":
                e3 = np.nanmax(np.abs(drv["D_a"] - drv["D_b"][[0, 2] if not para else [0]][:, None]))
                check(e3 < 1e-13, f"{ap:4s} {'paramagnetic' if para else 'full':12s}: route (a) == route (b) on lossy sites {e3:.1e}")


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)


def test_mgw_cumulant():
    print("[2] mGW: C = -2i (G^W - G^{W,2}/2) reproduces I; antisymmetry violation (= the 2RDM ambiguity)")
    m = T2.dp5model.load_model(os.path.join(WORK, "ring4.h5"))
    N, gamma = 4, 0.9
    ru, rd = T2.herm(N), T2.herm(N)
    P, A, U, D, P2 = T2.rand_blocks("mGW", N)
    rc = T2.write_rc("rcm.h5", os.path.join(WORK, "ring4.h5"), gamma, "mGW")
    T2.eval_rhs(rc, ru, rd, [P, A, U, D, P2])
    with h5py.File(os.path.join(WORK, "eval_out.h5"), "r") as f:
        Idrv = f["I"][()]
        Db, Dx, Da = f["equal_time/D_b"][()], f["equal_time/D_b_exchange"][()], f["equal_time/D_a"][()]
    GW, G2 = T2.assemble_W(P, A, U, D, P2)
    Geff = GW - 0.5 * G2
    C = -2j * Geff
    V = T2.full_V(N, m["lossy_sites"], m["site_weight"], gamma)
    I = -es("ablj,slab->sj", V, C)  # exact rho-equation identity I = -vee C (main Eq. KBErhoI, SM DSL_IcG)
    e = max(np.abs(I[0::2, 0::2] - Idrv[0]).max(), np.abs(I[1::2, 1::2] - Idrv[1]).max()) / np.abs(Idrv).max()
    check(e < 1e-13, f"I = -vee . C_eff, C_eff = -2i(G^W - G^W2/2) (full spin-orbital) == driver's history integral {e:.1e}")
    rho = T2.sd(ru, rd)
    Gam = gam_hf(rho) + C
    eb = max(abs(Db[r] - Gam[2 * r, 2 * r + 1, 2 * r, 2 * r + 1].real) for r in (0, 2))
    ex = max(abs(Dx[q] - (-Gam[2 * r, 2 * r + 1, 2 * r + 1, 2 * r]).real) for q, r in enumerate((0, 2)))
    em = np.abs(Da - 0.5 * (Db[[0, 2]] + Dx)[:, None]).max()
    check(max(eb, ex, em) < 1e-13, f"D_b = Gamma_udud, D_b_exchange = -Gamma_uddu, D_a = their mean ({max(eb, ex, em):.1e})")
    asym = np.abs(C + C.transpose(1, 0, 2, 3)).max() / np.abs(C).max()
    print(f"      antisymmetry violation |C_slab + C_lsab| / |C| = {asym:.2f} (pp schemes: 0) -> mGW 2RDM ambiguous")


def compare_exact(mname, m, gamma, aps, t_end, dt_out, tol_exact, label, info_only=False):
    ex = Exact(m, gamma)
    ts = np.round(np.arange(0, t_end + 1e-9, dt_out), 12)
    E = [ex.observables(R) for R in ex.evolve(ts)]
    res = {}
    for ap in aps:
        rc = T2.write_rc(f"rce_{label}_{ap}.h5", os.path.join(WORK, mname), gamma, ap, t_end=t_end, dt_out=dt_out)
        d = load_eq(run(rc, f"e_{label}_{ap}.h5"))
        errs = {}
        bl = bonds(m["h"])
        for k in ("n_site", "D_b", "SS_nn", "nn_conn_nn", "S2_total", "local_moment_b", "E1"):
            if k in ("SS_nn", "nn_conn_nn"):
                if not bl:
                    errs[k] = 0.0
                    continue
                ref = np.array([on_bonds(E[i][k.replace("_nn", "_matrix") if k == "SS_nn" else "nn_conn_matrix"], bl)
                                for i in range(len(ts))])
            else:
                ref = np.array([E[i]["local_moment_2rdm" if k == "local_moment_b" else k] for i in range(len(ts))])
            errs[k] = np.abs(d[k].reshape(ref.shape) - ref).max()
        if "D_a" in d:
            ref = np.array([E[i]["D_b"][list(m["lossy_sites"])] for i in range(len(ts))])
            Da = d["D_a"]
            ok = ~np.isnan(Da[:, :, 0])
            errs["D_a"] = np.abs(Da[:, :, 0][ok] - ref[ok]).max() if ok.any() else 0.0
        if "E_hyb" in d:
            errs["E_hyb"] = np.abs(d["E_hyb"] - np.array([E[i]["E_hyb"] for i in range(len(ts))])).max()
        res[ap] = errs
    return res


def test_exact():
    print("[3] exact Lindblad (ED): gamma = 0 (all observables exact), single site, 2-site lattice, N = 5 impurity")
    lattice2(os.path.join(WORK, "l2.h5"))
    T2.impurity_model(os.path.join(WORK, "imp5.h5"), 5)
    for mname in ("l2.h5", "imp5.h5"):
        m = T2.dp5model.load_model(os.path.join(WORK, mname))
        res = compare_exact(mname, m, 0.0, APPROX, 2.0, 0.2, 1e-6, "g0" + mname)
        e = max(max(v.values()) for v in res.values())
        check(e < 2e-6, f"{mname:8s} gamma = 0: every observable of every scheme == exact Lindblad, max error {e:.1e}")
    # single site, pure loss: DSL and TPPX_0 exact for D and <S^2>; DSL_0 not (recorded)
    h = np.full((2, 1, 1), -0.4, complex)
    z = np.zeros_like(h)
    T2.dp5model.write_model(os.path.join(WORK, "site.h5"), h, z, z, np.array([[[0.8]], [[0.6]]], complex), [0], [1.0], {},
                            dict(generator="site"))
    m = T2.dp5model.load_model(os.path.join(WORK, "site.h5"))
    res = compare_exact("site.h5", m, 1.3, APPROX, 3.0, 0.1, 0, "site")
    for ap in APPROX:
        r = res[ap]
        print(f"      single site {ap:8s}: D_b {r['D_b']:.1e}  D_a {r['D_a']:.1e}  <S_r^2> {r['local_moment_b']:.1e}  "
              f"S^2 {r['S2_total']:.1e}  n {r['n_site']:.1e}")
    for ap in ("DSL", "TPPX_0"):
        e = max(res[ap][k] for k in ("D_b", "D_a", "local_moment_b", "S2_total"))
        check(e < 1e-7, f"single site, pure loss: {ap} D (both routes), <S_r^2>, <S^2> exact ({e:.1e})")
    check(min(res["HF"]["D_b"], res["TPPX"]["D_b"], res["DSL_0"]["D_b"]) > 1e-3,
          "single site: HF, TPPX and DSL_0 D not exact (recorded)")
    for mname, g in (("l2.h5", 1.0), ("imp5.h5", 1.0)):
        m = T2.dp5model.load_model(os.path.join(WORK, mname))
        res = compare_exact(mname, m, g, APPROX, 2.0, 0.2, 0, "g1" + mname)
        print(f"      {mname} gamma = {g}: max errors vs exact (information)")
        for ap in APPROX:
            r = res[ap]
            print(f"        {ap:8s} n {r['n_site']:.1e}  D_b {r['D_b']:.1e}  D_a {r.get('D_a', 0):.1e}  S^2 {r['S2_total']:.1e}  "
                  f"S_i.S_j (bonds) {r['SS_nn']:.1e}  <nn>_c (bonds) {r['nn_conn_nn']:.1e}  E1 {r['E1']:.1e}")


def test_gm():
    print("[3b] dissipative Galitskii-Migdal trace: exact regression vs approximations")
    for mname, g in (("site.h5", 1.3), ("l2.h5", 1.0), ("imp5.h5", 1.0)):
        m = T2.dp5model.load_model(os.path.join(WORK, mname))
        ex = Exact(m, g)
        ts = np.round(np.arange(0, 2.0 + 1e-9, 0.25), 12)
        Rs = ex.evolve(ts)
        N = ex.N
        out = []
        for R in Rs:
            lf, rt, dN = ex.gm(R)
            o = ex.observables(R)
            M = 2 * N
            rho = np.array([[np.trace(R @ (ex.cd[a] @ ex.c[s]).toarray()) for a in range(M)] for s in range(M)])
            l1 = 0
            for sp_ in range(2):
                r = rho[sp_::2, sp_::2]
                hb = m["h"][sp_] - 1j * (m["ell_greater"][sp_] + m["ell_lesser"][sp_])
                l1 += np.trace(hb @ r)  # one-sided limits carry no gain contact term (see below)
                ctc = 2 * np.trace(m["ell_lesser"][sp_]).real if sp_ == 0 else ctc + 2 * np.trace(m["ell_lesser"][sp_]).real
            lam2 = lf - l1
            gD = sum(g * w * o["D_b"][r] for r, w in zip(m["lossy_sites"], m["site_weight"]))
            out.append((lf, rt, dN, lam2, gD, (-1j * (lf - rt)).real + ctc))
        out = np.array(out)
        e_conj = np.abs(out[:, 1] - out[:, 0].conj()).max()
        e_dn = np.abs(out[:, 2] - out[:, 5]).max()
        e_re = np.abs(out[:, 3].real).max()
        e_im = np.abs(out[:, 3].imag + 2 * out[:, 4].real).max()
        check(max(e_conj, e_dn) < 1e-9, f"{mname:8s} exact: Lambda_right = conj(Lambda_left) ({e_conj:.1e}); dN/dt = -i(Lambda_left - Lambda_right) "
                                          f"+ 2 tr l^< (gain contact term) ({e_dn:.1e})")
        check(max(e_re, e_im) < 1e-9, f"{mname:8s} exact: two-body part of Lambda_left: Re = {e_re:.1e} (claim: 0), "
                                      f"Im + 2 sum gamma_r D_r = {e_im:.1e} (claim: 0)")
        for ap in APPROX:
            rc = T2.write_rc(f"rcgm_{mname}_{ap}.h5", os.path.join(WORK, mname), g, ap, t_end=2.0, dt_out=0.25)
            d = load_eq(run(rc, f"gm_{mname}_{ap}.h5"))
            Ei = d["GM_E_int"]
            ref_im = -out[:, 4].real
            print(f"      {mname:8s} {ap:8s}: max|Re E_int| (dissipation-induced correlation energy) = {np.abs(Ei[:, 0]).max():.2e}, "
                  f"max|Im E_int - exact| = {np.abs(Ei[:, 1] - ref_im).max():.2e}, GM_check {np.abs(d['GM_check']).max():.1e}")
            check(np.abs(d["GM_check"]).max() < 1e-12, f"{mname:8s} {ap:8s}: Im E_int = -sum gamma_r D_a,r (identity)")
        if mname == "site.h5":  # gamma = 0 check of the odd-jump sign rule with the bath model
            pass
    m = T2.dp5model.load_model(os.path.join(WORK, "imp5.h5"))
    ex = Exact(m, 0.0)
    R = ex.evolve(np.array([0.0, 0.5]))[-1]
    lf, rt, dN = ex.gm(R)
    M = 2 * ex.N
    rho = np.array([[np.trace(R @ (ex.cd[a] @ ex.c[s]).toarray()) for a in range(M)] for s in range(M)])
    l1 = sum(np.trace((m["h"][q] - 1j * (m["ell_greater"][q] + m["ell_lesser"][q])) @ rho[q::2, q::2]) for q in range(2))
    lt = sum(np.trace(m["ell_lesser"][q]).real for q in range(2))
    check(abs(lf - l1) < 1e-9, f"impurity with bath, gamma = 0: exact one-sided Lambda_left == tr[h_o rho] ({abs(lf - l1):.1e}); "
                               f"the driver's Lambda adds i tr l^< = {lt:.3f}i (half of the gain contact term 2 l^<)")


def test_trajectory_checks():
    print("[4] along trajectories: route (a) vs (b), dN/dt identity, spin-flip invariance")
    imp = T2.dp5model.load_model(os.path.join(WORK, "imp5.h5"))
    swp = swap_spin(imp, os.path.join(WORK, "imp5_swap.h5"))
    for ap in APPROX:
        rc = T2.write_rc(f"rct_{ap}.h5", os.path.join(WORK, "imp5.h5"), 1.5, ap, t_end=3.0, dt_out=0.1)
        rcs = T2.write_rc(f"rcts_{ap}.h5", os.path.join(WORK, "imp5_swap.h5"), 1.5, ap, t_end=3.0, dt_out=0.1)
        a, b = load_eq(run(rc, f"tr_{ap}.h5")), load_eq(run(rcs, f"trs_{ap}.h5"))
        dab = np.nanmax(a["D_ab_maxdiff"])
        chk = np.abs(a["dN_dt_check"]).max()
        if ap.startswith("mGW"):
            print(f"      {ap}: max|D_a - D_b| = {dab:.2e} (ordering ambiguity of the mGW 2RDM; information); "
                  f"|D_b - D_b_exchange| up to {np.abs(a['D_b'][:, 0] - a['D_b_exchange'][:, 0]).max():.2e}")
        else:
            check(dab < 1e-12, f"{ap:8s}: route (a) == route (b) along the trajectory, max {dab:.1e}")
        check(chk < 1e-12, f"{ap:8s}: dN/dt - one-body part + 4 sum gamma_r D_r = {chk:.1e}")
        e = max(np.abs(a["S2_total"] - b["S2_total"]).max(), np.abs(a["Mz"] + b["Mz"]).max(),
                np.abs(a["SS_nn"] - b["SS_nn"]).max(), np.abs(a["D_b"] - b["D_b"]).max(),
                np.abs(a["n_site"] - b["n_site"][:, ::-1]).max())
        check(e < 1e-7, f"{ap:8s}: spin flip (spin-broken start, Mz(0) = {a['Mz'][0]:.2f}): S^2, SS_nn, D invariant, "
                        f"Mz -> -Mz, n_up <-> n_dn: {e:.1e}")
        print(f"             S^2(t=3) = {a['S2_total'][-1]:.4f}, Mz(t=3) = {a['Mz'][-1]:.4f}")


def test_spectral():
    print("[5] read_output: spectral function A(w)")
    m = T2.dp5model.load_model(os.path.join(WORK, "l2.h5"))
    rc = T2.write_rc("rcsp.h5", os.path.join(WORK, "l2.h5"), 0.0, "HF", mode="rtde", t_prime=[1.0], probe_columns=[0],
                     t_end=81.0, dt_out=0.05)
    out = run(rc, "sp.h5")
    w, Aw = read_output.spectral_function(out, k=0, column=0, spin=0, eta=0.05)
    dw = w[1] - w[0]
    ev = np.linalg.eigvalsh(m["h"][0])
    check(abs(Aw.sum() * dw - 1) < 2e-2, f"gamma = 0: int A(w) dw = {Aw.sum() * dw:.4f} (1)")
    pk = [w[i] for i in range(1, len(w) - 1) if Aw[i] > Aw[i - 1] and Aw[i] > Aw[i + 1] and Aw[i] > 0.1 * Aw.max()]
    check(len(pk) == 2 and np.abs(np.sort(pk) - ev).max() < 2 * dw + 0.01, f"peaks {np.round(pk, 3)} at eig h = {np.round(ev, 3)}")


if __name__ == "__main__":
    tests = [test_formulas, test_mgw_cumulant, test_exact, test_gm, test_trajectory_checks, test_spectral]
    sel = sys.argv[1:]
    if not sel:
        shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK, exist_ok=True)
    for t in tests:
        if not sel or t.__name__ in sel:
            t()
    print(f"\n{'ALL STEP-3b TESTS PASSED' if not FAIL else 'FAILURES: ' + str(len(FAIL))}")
    sys.exit(1 if FAIL else 0)
