"""Step-7a tests: mGW RTDE with counterterm, slot order F^W_{abjl} (probe j in slot 3).

    python3 -I tests/test_step7a.py [test names]

Reference: an independent full spin-orbital implementation of Implementation_verification eq. mGWRlat-general
(main PHIR / PHRTDE), written here; cross-checked against the vendored full RHS
(tests/ref/item0_check.py: rhsF_W), whose slot order is F_{jlab} = F^W_{abjl} (slots 3,4,1,2).
The einsum strings below use the letters p, q, s, o for the slots a, b, j, l of F^W_{abjl};
IR[i, s] = I_{sj} (row i -> s, probe s -> j).
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
import test_step2 as T2  # noqa: E402
import test_step3 as T3  # noqa: E402  (TPPX RTDE full reference, exact single-site G^R)

es, xi = T2.es, -1
EXE = T2.EXE
WORK = os.path.join(HERE, "work7a")
T2.WORK = WORK
T3.WORK = WORK
FAIL = []
rel = T2.rel


def check(cond, msg):
    print(("   ok   " if cond else "   FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def run(rc, out, *args):
    p = os.path.join(WORK, out)
    if os.path.exists(p):
        os.remove(p)
    r = subprocess.run([EXE, "--runcard", rc, "--output", out, *args], cwd=WORK, env=T2.ENV, capture_output=True, text=True)
    return r


# ---------------------------------------------------------------- full spin-orbital reference
def hW_full(V, rl):
    return -es("icdj,si->scdj", V, rl) - es("scdl,lj->scdj", V, rl)


def Rsrc_full(V, rl):
    M = rl.shape[0]
    rg = rl + xi * np.eye(M)
    P = lambda x, y: es("bd,ca->bcda", x, y)
    Pg, Pl, Pll = P(rg, rl), P(rl, rg), P(rl, rl)
    return 2 * (es("pabo,ql,blja->pqjo", Pg, rg, V) + es("pabo,ql,blja->pqjo", Pl, rl, V)
                - 2 * es("pabo,ql,blja->pqjo", Pll, rl, V))


def drift_full(h, F):
    return -1j * es("px,xqso->pqso", h, F) - 1j * es("qx,pxso->pqso", h, F) + 1j * es("pqsx,xo->pqso", F, h.conj().T)


def mgw_rtde_full(V, rl, ho, GR, GW, F, F2, hp=None):
    """hp: h_o on the legs of F, F2 (bare propagators: bare h_o); the G^R equation keeps ho (= h_o^MF)."""
    hp = ho if hp is None else hp
    R = Rsrc_full(V, rl)
    dF = drift_full(hp, F) + 2 * es("pcdo,dqsc->pqso", hW_full(V, rl), F) \
        + es("pqjo,js->pqso", R + 2j * es("pabo,bqja->pqjo", GW, V), GR)
    dF2 = drift_full(hp, F2) + es("pqjo,js->pqso", R, GR)
    IR = 2j * es("iopq,pqso->is", V, F - 0.5 * F2)
    return dF, dF2, -1j * ho @ GR - 1j * IR


# block placement (spin-orbital index 2x + sigma), column s0
UPB = {"U": (0, 0, 0, 0), "UA": (1, 0, 0, 1), "UP": (0, 1, 0, 1), "D": (1, 1, 1, 1), "DA": (0, 1, 1, 0), "DP": (1, 0, 1, 0)}


def assemble_F(blocks, s0, N):
    """blocks: dict name -> [p][q][o] (U, UA, UP, UP2, D, DA, DP, DP2)"""
    M = 2 * N
    F = np.zeros((M,) * 4, complex)
    F2 = np.zeros((M,) * 4, complex)
    for n, (a, b, c, d) in UPB.items():
        F[a::2, b::2, 2 * s0 + c, d::2] = blocks[n]
    F2[0::2, 1::2, 2 * s0, 1::2] = blocks["UP2"]
    F2[1::2, 0::2, 2 * s0, 1::2] = xi * blocks["UP2"].transpose(1, 0, 2)   # P12 image
    F2[1::2, 0::2, 2 * s0 + 1, 0::2] = blocks["DP2"]
    F2[0::2, 1::2, 2 * s0 + 1, 0::2] = xi * blocks["DP2"].transpose(1, 0, 2)
    return F, F2


def extract(dF, dF2, s0):
    out = {n: dF[a::2, b::2, 2 * s0 + c, d::2] for n, (a, b, c, d) in UPB.items()}
    out["UP2"] = dF2[0::2, 1::2, 2 * s0, 1::2]
    out["DP2"] = dF2[1::2, 0::2, 2 * s0 + 1, 0::2]
    return out


def reference(m, gamma, ru, rd, G1, blocks, GRu, GRd, s0, check_old=True, bare=False):
    """G1 = (P, A, UU, DD, P2) G1-G2 blocks; returns reduced d-blocks and dG^R columns."""
    N = m["h"].shape[1]
    V = T2.full_V(N, m["lossy_sites"], m["site_weight"], gamma)
    rl = T2.sd(ru, rd)
    hb = T2.sd(*[m["h"][s] - 1j * (m["ell_greater"][s] + m["ell_lesser"][s]) for s in range(2)])
    ho = hb - 1j * 2 * es("iklj,lk->ij", V, rl)
    GW, _ = T2.assemble_W(*G1)
    F, F2 = assemble_F(blocks, s0, N)
    M = 2 * N
    GR = np.zeros((M, M), complex)
    GR[0::2, 2 * s0] = GRu
    GR[1::2, 2 * s0 + 1] = GRd
    dF, dF2, dGR = mgw_rtde_full(V, rl, ho, GR, GW, F, F2, hp=hb if bare else ho)
    if check_old and not bare:  # item0's RHS through the slot permutation (independent derivation path)
        o = T2.R["objects"](rl, xi, V, 0 * V)
        T_ = lambda X: X.transpose(2, 3, 0, 1)  # F^W_{abjl} -> item0's F_{jlab}
        old = T2.R["rhsF_W"](o, ho, GR, T_(F), GW)
        old2 = T2.R["rhsF_W"](o, ho, GR, T_(F2), GW, full=False)
        e = max(rel(old.transpose(2, 3, 0, 1), dF), rel(old2.transpose(2, 3, 0, 1), dF2))
        assert e < 1e-12, f"full RHS != item0 rhsF_W through the slot permutation F^W_abjl -> F_jlab: {e}"
    return extract(dF, dF2, s0), dGR[0::2, 2 * s0], dGR[1::2, 2 * s0 + 1]


ORDER_FULL = ["U", "UA", "UP", "UP2", "D", "DA", "DP", "DP2"]
ORDER_PARA = ["UA", "UP", "UP2"]


def eval_rtde(rc, ru, rd, G1, blocks, GRu, GRd, para=False):
    N = ru.shape[0]
    g1 = [G1[0], G1[1], G1[4]] if para else list(G1)
    parts = [ru.ravel()] + ([] if para else [rd.ravel()]) + [b.ravel() for b in g1] + [GRu] + ([] if para else [GRd])
    parts += [blocks[n].ravel() for n in (ORDER_PARA if para else ORDER_FULL)]
    y = np.concatenate(parts).astype(complex)
    fi, fo = os.path.join(WORK, "ein.h5"), os.path.join(WORK, "eout.h5")
    with h5py.File(fi, "w") as f:
        f["y"] = y
    p = subprocess.run([EXE, "--runcard", rc, "--eval-rhs", fi, fo], env=T2.ENV, capture_output=True, text=True)
    if p.returncode:
        print(p.stdout, p.stderr)
    with h5py.File(fo, "r") as f:
        dy = f["dy"][()]
    o = sum(x.size for x in parts[: (1 if para else 2) + len(g1)])
    ns = 1 if para else 2
    gu = dy[o:o + N]
    gd = gu if para else dy[o + N:o + 2 * N]
    o += ns * N
    out = {}
    for n in (ORDER_PARA if para else ORDER_FULL):
        out[n] = dy[o:o + N ** 3].reshape(N, N, N)
        o += N ** 3
    return out, gu, gd, dy


def rand_inputs(N, para=False):
    ru = T2.herm(N)
    rd = ru.copy() if para else T2.herm(N)
    if para:
        P, A, P2 = T2.rand_blocks("mGW", N, para=True)
        G1 = (P, A, A + P, A + P, P2)
    else:
        G1 = tuple(T2.rand_blocks("mGW", N))
    c = lambda: T2.crand(N, N, N)
    if para:
        UA, UP, UP2 = c(), c(), c()
        UP2 = 0.5 * (UP2 + UP2.transpose(1, 0, 2))   # paramagnetic counterterm is (p,q)-symmetric
        b = dict(UA=UA, UP=UP, UP2=UP2, U=UA + UP)
        b.update(D=b["U"], DA=UA, DP=UP, DP2=UP2)
        GRu = T2.crand(N)
        return ru, rd, G1, b, GRu, GRu.copy()
    b = {n: c() for n in ORDER_FULL}
    return ru, rd, G1, b, T2.crand(N), T2.crand(N)


# ---------------------------------------------------------------- tests
def test_rhs_random():
    print("[1] mGW RTDE RHS on random inputs vs full spin-orbital eq. mGWRlat-general (slot order F^W_abjl)")
    imp = T2.impurity_model(os.path.join(WORK, "imp5.h5"), 5)
    ring = T2.ring_model(os.path.join(WORK, "ring.h5"), 4, [0, 2], [1.0, 0.5])
    for gname, mp, m, vimpls, s0 in (("impurity N=5, column 0", "imp5.h5", imp, ("general", "impurity"), 0),
                                      ("impurity N=5, column 2", "imp5.h5", imp, ("general", "impurity"), 2),
                                      ("ring 4, loss {0,2}, column 1", "ring.h5", ring, ("general",), 1)):
        N = m["h"].shape[1]
        for lab in ("mGW", "mGW_0"):
            ru, rd, G1, b, GRu, GRd = rand_inputs(N)
            ref, gu0, gd0 = reference(m, 0.8, ru, rd, G1, b, GRu, GRd, s0, bare=(lab == "mGW_0"))
            res = {}
            for vi in vimpls:
                rc = T2.write_rc(f"rc7_{lab}_{vi}_{s0}_{mp}", os.path.join(WORK, mp), 0.8, lab, vertex_impl=vi, mode="rtde",
                                 t_prime=[0.5], probe_columns=[s0])
                out, gu, gd, dy = eval_rtde(rc, ru, rd, G1, b, GRu, GRd)
                res[vi] = dy
                e = max([rel(out[n], ref[n]) for n in ORDER_FULL] + [rel(gu, gu0), rel(gd, gd0)])
                eb = {n: rel(out[n], ref[n]) for n in ("UP2", "DP2")}
                check(e < 1e-13, f"{gname:30s} {lab:5s} {vi:8s}: 8 blocks + G^R (both spins) vs full {e:.1e} (counterterm "
                                 f"{max(eb.values()):.1e})")
            if len(vimpls) == 2:
                ok, how = T3.paths_agree(res["general"], res["impurity"])
                check(ok, f"{gname:30s} {lab:5s}: general {{0}} == impurity fast path, {how}")
    print("      (the full reference equals item0 rhsF_W through the slot permutation F^W_abjl -> F_jlab, asserted)")


def test_ct_identity_rhs():
    print("[2] counterterm identity at RHS level, driver vs driver: dF^{W,2,udud} = 2 xi dD^{pp}|_2B  (UP2 = 2 xi D)")
    m = T2.dp5model.load_model(os.path.join(WORK, "ring.h5"))
    N = 4
    ru, rd, G1, b, GRu, GRd = rand_inputs(N)
    D = T2.crand(N, N, N)
    b["UP2"] = 2 * xi * D
    rc = T2.write_rc("rc7ct.h5", os.path.join(WORK, "ring.h5"), 0.8, "mGW", mode="rtde", t_prime=[0.5], probe_columns=[1])
    out, _, _, _ = eval_rtde(rc, ru, rd, G1, b, GRu, GRd)
    rct = T2.write_rc("rc7ct_t.h5", os.path.join(WORK, "ring.h5"), 0.8, "TPPX2B", mode="rtde", t_prime=[0.5], probe_columns=[1])
    P = T2.rand_blocks("TPPX", N)[0]
    _, _, dD, _, _ = T3.eval_rtde(rct, ru, rd, P, GRu, GRd, D, T2.crand(N, N, N))
    e = rel(out["UP2"], 2 * xi * dD)
    check(e < 1e-14, f"ring, lattice path: mGW counterterm RHS == 2 xi x TPPX-2B D RHS (same rho, h_o, G^R) {e:.1e}")
    # bare propagators: the counterterm stays the second-order part of the mGW_0 RTDE (= 2 xi x TPPX2B_0)
    rc0 = T2.write_rc("rc7ct0.h5", os.path.join(WORK, "ring.h5"), 0.8, "mGW_0", mode="rtde", t_prime=[0.5], probe_columns=[1])
    out0, _, _, _ = eval_rtde(rc0, ru, rd, G1, b, GRu, GRd)
    rct0 = T2.write_rc("rc7ct_t0.h5", os.path.join(WORK, "ring.h5"), 0.8, "TPPX2B_0", mode="rtde", t_prime=[0.5], probe_columns=[1])
    _, _, dD0, _, _ = T3.eval_rtde(rct0, ru, rd, P, GRu, GRd, D, T2.crand(N, N, N))
    e0, d0 = rel(out0["UP2"], 2 * xi * dD0), rel(out0["UP2"], out["UP2"])
    check(e0 < 1e-14 and d0 > 1e-3, f"ring, bare propagators: mGW_0 counterterm RHS == 2 xi x TPPX2B_0 D RHS {e0:.1e} "
                                    f"(flag changes it by {d0:.1e} rel.)")


def augmented(m, gamma, tp, t_end, s0, t_eval, with_pp=False, bare=False):
    """full spin-orbital mGW G1-G2 to t', then G1-G2 + F^W, F^{W,2nd} (column j = s0) + G^R; optional TPPX-2B F^pp
    driven by the same rho(t), h_o(t) (counterterm identity)."""
    N = m["h"].shape[1]
    M = 2 * N
    full_rhs, _, V = T2.full_system(m, gamma, "mGW_0" if bare else "mGW")
    hb = T2.sd(*[m["h"][s] - 1j * (m["ell_greater"][s] + m["ell_lesser"][s]) for s in range(2)])
    M4 = M ** 4

    def f1(t, y):
        rl = y[:M * M].reshape(M, M)
        GW = y[M * M:M * M + M4].reshape((M,) * 4)
        G2 = y[M * M + M4:M * M + 2 * M4].reshape((M,) * 4)
        drho, dT = full_rhs(rl, [GW, G2])
        return np.concatenate([drho.ravel(), dT[0].ravel(), dT[1].ravel()])
    y0 = np.concatenate([T2.sd(m["rho"][0], m["rho"][1]).ravel(), np.zeros(2 * M4, complex)])
    s1 = solve_ivp(f1, (0, tp), y0, method="DOP853", rtol=1e-11, atol=1e-13)
    n1 = M * M + 2 * M4

    def f2(t, y):
        dy1 = f1(t, y[:n1])
        rl = y[:M * M].reshape(M, M)
        GW = y[M * M:M * M + M4].reshape((M,) * 4)
        F = y[n1:n1 + M4].reshape((M,) * 4)
        F2 = y[n1 + M4:n1 + 2 * M4].reshape((M,) * 4)
        GR = y[n1 + 2 * M4:n1 + 2 * M4 + M * M].reshape(M, M)
        ho = hb - 1j * 2 * es("iklj,lk->ij", V, rl)
        hp = hb if bare else ho
        dF, dF2, dGR = mgw_rtde_full(V, rl, ho, GR, GW, F, F2, hp=hp)
        parts = [dy1, dF.ravel(), dF2.ravel(), dGR.ravel()]
        if with_pp:
            Fpp = y[n1 + 2 * M4 + M * M:].reshape((M,) * 4)
            dpp, _ = T3.full_rtde_rhs(V, rl, hp, GR, 0 * GW, Fpp, two_b=True)
            parts.append(dpp.ravel())
        return np.concatenate(parts)
    GR0 = np.zeros((M, M), complex)
    GR0[2 * s0, 2 * s0] = GR0[2 * s0 + 1, 2 * s0 + 1] = -1j
    y1 = [s1.y[:, -1], np.zeros(2 * M4, complex), GR0.ravel()] + ([np.zeros(M4, complex)] if with_pp else [])
    s2 = solve_ivp(f2, (tp, t_end), np.concatenate(y1), method="DOP853", rtol=1e-11, atol=1e-13, t_eval=t_eval)
    Y = s2.y.T
    GR = Y[:, n1 + 2 * M4:n1 + 2 * M4 + M * M].reshape(len(t_eval), M, M)
    out = dict(GR=np.stack([GR[:, 0::2, 2 * s0], GR[:, 1::2, 2 * s0 + 1]], axis=1),
               F2=Y[:, n1 + M4:n1 + 2 * M4].reshape((len(t_eval),) + (M,) * 4))
    if with_pp:
        out["Fpp"] = Y[:, n1 + 2 * M4 + M * M:].reshape((len(t_eval),) + (M,) * 4)
    return out


def test_trajectories():
    print("[3] coupled G1-G2 + RTDE vs full spin-orbital solve_ivp (rtol 1e-11); counterterm identity along it")
    imp = T2.impurity_model(os.path.join(WORK, "imp3.h5"), 3)
    ring = T2.ring_model(os.path.join(WORK, "ring3.h5"), 3, [0, 2], [1.0, 0.5])
    for gname, mp, m, s0 in (("impurity N=3", "imp3.h5", imp, 0), ("ring 3, loss {0,2}, column 1", "ring3.h5", ring, 1)):
      for lab in ("mGW", "mGW_0"):
        rc = T2.write_rc(f"rc7t_{lab}_{mp}", os.path.join(WORK, mp), 1.0, lab, mode="rtde", t_prime=[1.0], probe_columns=[s0],
                         t_end=3.0, dt_out=0.1)
        run(rc, f"t7_{lab}_{mp}")
        rch = T2.write_rc(f"rc7h_{mp}", os.path.join(WORK, mp), 1.0, "HF", mode="rtde", t_prime=[1.0], probe_columns=[s0],
                          t_end=3.0, dt_out=0.1)
        run(rch, f"t7h_{mp}")
        with h5py.File(os.path.join(WORK, f"t7_{lab}_{mp}"), "r") as f, h5py.File(os.path.join(WORK, f"t7h_{mp}"), "r") as g:
            tt, GR = f["rtde/tp=0/t"][()], f["rtde/tp=0/GR"][()][:, 0]
            dHF = np.abs(GR - g["rtde/tp=0/GR"][()][:, 0]).max()
        ref = augmented(m, 1.0, 1.0, 3.0, s0, tt, with_pp=True, bare=(lab == "mGW_0"))
        e = np.abs(GR - ref["GR"]).max()
        check(e < 2e-6, f"{gname:28s} {lab:5s}: max|G^R - ref| over {len(tt)} frames = {e:.1e} (|G^R - G^R_HF| up to {dHF:.1e})")
        F2, Fpp = ref["F2"], ref["Fpp"]
        eid = np.abs(F2 - 2 * xi * Fpp.transpose(0, 1, 2, 4, 3)).max() / max(np.abs(F2).max(), 1e-300)
        check(eid < 1e-8, f"{gname:28s} {lab:5s}: F^{{W,2nd}}_abjl = 2 xi F^pp_ablj|_(h=0, G^pp=0) along the trajectory (rel {eid:.1e}, "
                          f"|F2| <= {np.abs(F2).max():.1e})")


def test_gamma0():
    print("[4] gamma = 0: G^R = -i exp(-i h_o tau) e_j")
    imp = os.path.join(WORK, "imp3.h5")
    m = T2.dp5model.load_model(imp)
    ho = m["h"] - 1j * (m["ell_greater"] + m["ell_lesser"])
    e = 0.0
    for s in (0, 2):
        rc0 = T2.write_rc(f"rc7g0_{s}.h5", imp, 0.0, "mGW", mode="rtde", t_prime=[1.0], probe_columns=[s], t_end=4.0, dt_out=0.1)
        run(rc0, f"g07_{s}.h5")
        with h5py.File(os.path.join(WORK, f"g07_{s}.h5"), "r") as f:
            tt, GR = f["rtde/tp=0/t"][()], f["rtde/tp=0/GR"][()]
        e = max(e, max(np.abs(GR[k, 0, sp] - (-1j * expm(-1j * ho[sp] * (t - 1.0))[:, s])).max()
                       for k, t in enumerate(tt) for sp in range(2)))
    check(e < 1e-7, f"gamma = 0: G^R = -i exp(-i h_o tau) e_j (columns 0, 2): {e:.1e}")


def test_single_site_and_observables():
    print("[5] single site vs exact regression G^R (recorded); equal-time observables in mGW RTDE runs")
    eps0, gam, nu, nd, tp = -0.4, 1.3, 0.8, 0.6, 0.7
    h = np.full((2, 1, 1), eps0, complex)
    z = np.zeros_like(h)
    T2.dp5model.write_model(os.path.join(WORK, "site.h5"), h, z, z, np.array([[[nu]], [[nd]]], complex), [0], [1.0], {},
                            dict(generator="single site"))
    m = T2.dp5model.load_model(os.path.join(WORK, "site.h5"))
    rc = T2.write_rc("rc7ss.h5", os.path.join(WORK, "site.h5"), gam, "mGW", mode="rtde", t_prime=[tp], t_end=tp + 3.0, dt_out=0.1)
    run(rc, "site7.h5")
    with h5py.File(os.path.join(WORK, "site7.h5"), "r") as f:
        tt, GR = f["rtde/tp=0/t"][()], f["rtde/tp=0/GR"][()][:, 0, 0, 0]
    ex = T3.exact_single_site_GR(eps0, gam, nu, nd, tp, tt - tp)
    ref = augmented(m, gam, tp, tp + 3.0, 0, tt)["GR"][:, 0, 0]
    print(f"      mGW single site: max|G^R_up - exact| = {np.abs(GR - ex).max():.2e} (TPPX 9.4e-2, TPPX_0 3.6e-2, HF 7.3e-2; step 3a)")
    check(np.abs(GR - ref).max() < 1e-6, f"single-site mGW G^R == full spin-orbital reference {np.abs(GR - ref).max():.1e}")
    imp = os.path.join(WORK, "imp3.h5")
    rcg = T2.write_rc("rc7og.h5", imp, 1.0, "mGW", t_end=3.0, dt_out=0.1)
    rcr = T2.write_rc("rc7or.h5", imp, 1.0, "mGW", mode="rtde", t_prime=[1.0], probe_columns=[1], t_end=3.0, dt_out=0.1)
    run(rcg, "og7.h5"); run(rcr, "or7.h5")
    with h5py.File(os.path.join(WORK, "og7.h5"), "r") as a, h5py.File(os.path.join(WORK, "or7.h5"), "r") as b:
        ga, gb = a["observables/equal_time"], b["observables/equal_time"]
        keys = sorted(ga.keys())
        e = max(np.nanmax(np.abs(ga[k][()] - gb[k][()])) for k in keys)
        same = keys == sorted(gb.keys())
    check(same and e < 1e-7, f"equal-time observables of an mGW RTDE run == those of the G1-G2 run ({len(keys)} series, {e:.1e})")


if __name__ == "__main__":
    tests = [test_rhs_random, test_ct_identity_rhs, test_trajectories, test_gamma0, test_single_site_and_observables]
    sel = sys.argv[1:]
    if not sel:
        shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK, exist_ok=True)
    for t in tests:
        if not sel or t.__name__ in sel:
            t()
    print(f"\n{'ALL STEP-7a TESTS PASSED' if not FAIL else 'FAILURES: ' + str(len(FAIL))}")
    sys.exit(1 if FAIL else 0)
