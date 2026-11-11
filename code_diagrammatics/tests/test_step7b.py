"""Step-7b tests: DSL RTDE, slot order F_{ablj} (pair (a,b), closing leg l, probe j last).

    python3 -I tests/test_step7b.py [test names]

Reference: an independent full spin-orbital implementation of Implementation_verification eq. DSLRlat-general /
SM eq. DSLR_final written here (TPPX part from test_step3.full_rtde_rhs with the DSL correlator, plus Z^F, Z^G),
cross-checked against the vendored item0 rhsF_dsl (tests/ref/item0_check.py), whose slot order is F_{ljab} = F_{ablj} (slots 3,4,1,2).
The einsum strings below use k, s for the slots l, j of F_{ablj}.
Single-site exactness of the DSL G^R against the exact 4-state Lindblad regression (asserted).
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
import test_step3 as T3  # noqa: E402
import test_step3b as T3b  # noqa: E402  (exact Lindblad solver)

es, xi = T2.es, -1
EXE = T2.EXE
WORK = os.path.join(HERE, "work7b")
for mod in (T2, T3, T3b):
    mod.WORK = WORK
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
    return subprocess.run([EXE, "--runcard", rc, "--output", out, *args], cwd=WORK, env=T2.ENV, capture_output=True, text=True)


# ---------------------------------------------------------------- full spin-orbital reference
def hW_full(V, rl):
    return -es("icdj,si->scdj", V, rl) - es("scdl,lj->scdj", V, rl)


def dsl_rtde_full(V, rl, ho, GR, G, F, hp=None):
    """hp: h_o on the legs of F (bare propagators: bare h_o); the G^R equation keeps ho (= h_o^MF)."""
    tp, IR = T3.full_rtde_rhs(V, rl, ho if hp is None else hp, GR, G, F)          # TPPX RTDE with the DSL correlator (memory, source, coupling)
    X = 2 * es("acdk,dbcs->abks", hW_full(V, rl), F)
    ZF = X + xi * X.transpose(1, 0, 2, 3)
    Y = 2 * xi * 1j * es("ayxk,xbzy->abkz", G, V)
    ZG = es("abkz,zs->abks", Y + xi * Y.transpose(1, 0, 2, 3), GR)
    return tp + ZF + ZG, ZF + ZG, -1j * ho @ GR - 1j * IR


def pairs(N):
    return [(a, b) for a in range(N) for b in range(a + 1, N)]


def unpackS(Sp, N):
    S = np.zeros((N, N, N), complex)
    for p, (a, b) in enumerate(pairs(N)):
        S[a, b] = Sp[p]
        S[b, a] = -Sp[p]
    return S


def packS(S, N):
    return np.array([S[a, b] for (a, b) in pairs(N)]).reshape(-1, N) if N > 1 else np.zeros((0, N), complex)


def assemble_F(b, s0, N):
    M = 2 * N
    F = np.zeros((M,) * 4, complex)
    F[0::2, 1::2, 1::2, 2 * s0] = b["Ou"]
    F[1::2, 0::2, 1::2, 2 * s0] = xi * b["Ou"].transpose(1, 0, 2)
    F[0::2, 0::2, 0::2, 2 * s0] = b["Su"]
    F[1::2, 0::2, 0::2, 2 * s0 + 1] = b["Od"]
    F[0::2, 1::2, 0::2, 2 * s0 + 1] = xi * b["Od"].transpose(1, 0, 2)
    F[1::2, 1::2, 1::2, 2 * s0 + 1] = b["Sd"]
    return F


def extract(dF, s0):
    return dict(Ou=dF[0::2, 1::2, 1::2, 2 * s0], Su=dF[0::2, 0::2, 0::2, 2 * s0], Od=dF[1::2, 0::2, 0::2, 2 * s0 + 1],
                Sd=dF[1::2, 1::2, 1::2, 2 * s0 + 1])


def g1_full(P, UU, DD):
    return T2.assemble_pp(P, UU, DD)


def reference(m, gamma, ru, rd, G1, b, GRu, GRd, s0, check_old=True, bare=False):
    """G1 = (P, UU, DD) full arrays; b: Ou, Su, Od, Sd full arrays."""
    N = m["h"].shape[1]
    V = T2.full_V(N, m["lossy_sites"], m["site_weight"], gamma)
    rl = T2.sd(ru, rd)
    hb = T2.sd(*[m["h"][s] - 1j * (m["ell_greater"][s] + m["ell_lesser"][s]) for s in range(2)])
    ho = hb - 1j * 2 * es("iklj,lk->ij", V, rl)
    G = g1_full(*G1)
    F = assemble_F(b, s0, N)
    M = 2 * N
    GR = np.zeros((M, M), complex)
    GR[0::2, 2 * s0] = GRu
    GR[1::2, 2 * s0 + 1] = GRd
    dF, Z, dGR = dsl_rtde_full(V, rl, ho, GR, G, F, hp=hb if bare else ho)
    if check_old and not bare:
        o = T2.R["objects"](rl, xi, V, 0 * V)
        old = T2.R["rhsF_dsl"](o, ho, GR, F.transpose(2, 3, 0, 1), G)
        e = rel(old.transpose(2, 3, 0, 1), dF)
        assert e < 1e-12, f"full DSL RTDE RHS != item0 rhsF_dsl through the slot permutation F_ablj -> F_ljab: {e}"
    return extract(dF, s0), extract(Z, s0), dGR[0::2, 2 * s0], dGR[1::2, 2 * s0 + 1]


def rand_inputs(N, para=False):
    ru = T2.herm(N)
    rd = ru.copy() if para else T2.herm(N)
    blocks = T2.rand_blocks("DSL", N, para=para)
    if para:
        P = blocks[0]
        Ps = P + xi * P.transpose(0, 1, 3, 2)
        G1 = (P, Ps, Ps)
    else:
        G1 = tuple(blocks) if N > 1 else (blocks[0], 0 * blocks[0], 0 * blocks[0])
    c = lambda: T2.crand(N, N, N)
    anti = lambda X: 0.5 * (X - X.transpose(1, 0, 2))
    if para:
        O = c()
        b = dict(Ou=O, Su=O + xi * O.transpose(1, 0, 2), Od=O.copy(), Sd=O + xi * O.transpose(1, 0, 2))
        g = T2.crand(N)
        return ru, rd, G1, b, g, g.copy()
    b = dict(Ou=c(), Su=anti(c()), Od=c(), Sd=anti(c()))
    return ru, rd, G1, b, T2.crand(N), T2.crand(N)


def eval_rtde(rc, ru, rd, G1, b, GRu, GRd, para=False):
    N = ru.shape[0]
    if para:
        g1 = [G1[0]]
    else:
        g1 = [G1[0]] + ([T2.pack(G1[1]), T2.pack(G1[2])] if N > 1 else [])
    parts = [ru.ravel()] + ([] if para else [rd.ravel()]) + [x.ravel() for x in g1] + [GRu] + ([] if para else [GRd])
    order = [("Ou", False)] if para else [("Ou", False), ("Su", True), ("Od", False), ("Sd", True)]
    if N == 1:
        order = [x for x in order if not x[1]]
    parts += [packS(b[n], N).ravel() if pk else b[n].ravel() for n, pk in order]
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
    for n, pk in order:
        sz = N * len(pairs(N)) if pk else N ** 3
        v = dy[o:o + sz]
        out[n] = unpackS(v.reshape(-1, N), N) if pk else v.reshape(N, N, N)
        o += sz
    return out, gu, gd, dy


# ---------------------------------------------------------------- tests
def test_rhs_random():
    print("[1] DSL RTDE RHS on random inputs vs full spin-orbital eq. DSLRlat-general (slot order F_ablj)")
    imp = T2.impurity_model(os.path.join(WORK, "imp5.h5"), 5)
    ring = T2.ring_model(os.path.join(WORK, "ring.h5"), 4, [0, 2], [1.0, 0.5])
    for gname, mp, m, vimpls, s0 in (("impurity N=5, column 0", "imp5.h5", imp, ("general", "impurity"), 0),
                                      ("impurity N=5, column 2", "imp5.h5", imp, ("general", "impurity"), 2),
                                      ("ring 4, loss {0,2}, column 1", "ring.h5", ring, ("general",), 1)):
        N = m["h"].shape[1]
        for lab in ("DSL", "DSL_0"):
            ru, rd, G1, b, GRu, GRd = rand_inputs(N)
            ref, _, gu0, gd0 = reference(m, 0.8, ru, rd, G1, b, GRu, GRd, s0, bare=(lab == "DSL_0"))
            res = {}
            for vi in vimpls:
                rc = T2.write_rc(f"rc7b_{lab}_{vi}_{s0}_{mp}", os.path.join(WORK, mp), 0.8, lab, vertex_impl=vi, mode="rtde",
                                 t_prime=[0.5], probe_columns=[s0])
                out, gu, gd, dy = eval_rtde(rc, ru, rd, G1, b, GRu, GRd)
                res[vi] = dy
                e = max([rel(out[n], ref[n]) for n in ("Ou", "Su", "Od", "Sd")] + [rel(gu, gu0), rel(gd, gd0)])
                es_ = max(rel(out[n], ref[n]) for n in ("Su", "Sd"))
                check(e < 1e-13, f"{gname:30s} {lab:5s} {vi:8s}: opposite + same-spin blocks, G^R (both probe spins) vs full "
                                 f"{e:.1e} (same-spin {es_:.1e}, |dS|/|dO| = {np.abs(ref['Su']).max() / np.abs(ref['Ou']).max():.2f})")
            if len(vimpls) == 2:
                check(np.array_equal(res["general"], res["impurity"]), f"{gname:30s} {lab:5s}: general {{0}} == impurity fast path, bitwise")
    print("      (the full reference equals item0 rhsF_dsl through the slot permutation F_ablj -> F_ljab, asserted)")


def test_tppx_part():
    print("[2] DSL minus its particle-hole terms == the driver's TPPX RTDE RHS (same inputs, driver vs driver)")
    for (mp, s0), sfx in [(x, y) for x in (("imp5.h5", 0), ("ring.h5", 1)) for y in ("", "_0")]:
        m = T2.dp5model.load_model(os.path.join(WORK, mp))
        N = m["h"].shape[1]
        ru, rd, G1, b, GRu, GRd = rand_inputs(N)
        _, Z, _, _ = reference(m, 0.8, ru, rd, G1, b, GRu, GRd, s0, check_old=False)
        rc = T2.write_rc(f"rc7bt{sfx}_{mp}", os.path.join(WORK, mp), 0.8, "DSL" + sfx, mode="rtde", t_prime=[0.5], probe_columns=[s0])
        out, gu, gd, _ = eval_rtde(rc, ru, rd, G1, b, GRu, GRd)
        rct = T2.write_rc(f"rc7btt{sfx}_{mp}", os.path.join(WORK, mp), 0.8, "TPPX" + sfx, mode="rtde", t_prime=[0.5], probe_columns=[s0])
        U = xi * b["Od"].transpose(1, 0, 2)   # TPPX U^pp_{ablj} = xi F^{dn up up dn}_{balj}
        tu, td, dD, dU, _ = T3.eval_rtde(rct, ru, rd, G1[0], GRu, GRd, b["Ou"], U)
        e = max(rel(out["Ou"] - Z["Ou"], dD), rel(out["Od"] - Z["Od"], xi * dU.transpose(1, 0, 2)), rel(gu, tu), rel(gd, td))
        check(e < 1e-13, f"{mp:8s} column {s0}: dO_DSL{sfx} - (Z^F + Z^G) == TPPX{sfx} dD (up), xi dU_bak (down); G^R equal: {e:.1e}")


def augmented(m, gamma, tp, t_end, s0, t_eval, bare=False):
    N = m["h"].shape[1]
    M = 2 * N
    full_rhs, _, V = T2.full_system(m, gamma, "DSL_0" if bare else "DSL")
    hb = T2.sd(*[m["h"][s] - 1j * (m["ell_greater"][s] + m["ell_lesser"][s]) for s in range(2)])
    M4 = M ** 4

    def f1(t, y):
        rl = y[:M * M].reshape(M, M)
        G = y[M * M:M * M + M4].reshape((M,) * 4)
        drho, dT = full_rhs(rl, [G])
        return np.concatenate([drho.ravel(), dT[0].ravel()])
    y0 = np.concatenate([T2.sd(m["rho"][0], m["rho"][1]).ravel(), np.zeros(M4, complex)])
    s1 = solve_ivp(f1, (0, tp), y0, method="DOP853", rtol=1e-11, atol=1e-13)
    n1 = M * M + M4

    def f2(t, y):
        rl = y[:M * M].reshape(M, M)
        G = y[M * M:n1].reshape((M,) * 4)
        F = y[n1:n1 + M4].reshape((M,) * 4)
        GR = y[n1 + M4:].reshape(M, M)
        ho = hb - 1j * 2 * es("iklj,lk->ij", V, rl)
        dF, _, dGR = dsl_rtde_full(V, rl, ho, GR, G, F, hp=hb if bare else ho)
        return np.concatenate([f1(t, y[:n1]), dF.ravel(), dGR.ravel()])
    GR0 = np.zeros((M, M), complex)
    GR0[2 * s0, 2 * s0] = GR0[2 * s0 + 1, 2 * s0 + 1] = -1j
    s2 = solve_ivp(f2, (tp, t_end), np.concatenate([s1.y[:, -1], np.zeros(M4, complex), GR0.ravel()]), method="DOP853",
                   rtol=1e-11, atol=1e-13, t_eval=t_eval)
    GR = s2.y[n1 + M4:].T.reshape(len(t_eval), M, M)
    return np.stack([GR[:, 0::2, 2 * s0], GR[:, 1::2, 2 * s0 + 1]], axis=1)


def test_trajectories():
    print("[3] coupled DSL G1-G2 + RTDE vs full spin-orbital solve_ivp (rtol 1e-11)")
    imp = T2.impurity_model(os.path.join(WORK, "imp3.h5"), 3)
    ring = T2.ring_model(os.path.join(WORK, "ring3.h5"), 3, [0, 2], [1.0, 0.5])
    for (gname, mp, m, s0), sfx in [(x, y) for x in (("impurity N=3", "imp3.h5", imp, 0),
                                                      ("ring 3, loss {0,2}, column 1", "ring3.h5", ring, 1)) for y in ("", "_0")]:
        rc = T2.write_rc(f"rc7bt{sfx}_{mp}", os.path.join(WORK, mp), 1.0, "DSL" + sfx, mode="rtde", t_prime=[1.0], probe_columns=[s0],
                         t_end=3.0, dt_out=0.1)
        run(rc, f"t7b{sfx}_{mp}")
        rct = T2.write_rc(f"rc7bp{sfx}_{mp}", os.path.join(WORK, mp), 1.0, "TPPX" + sfx, mode="rtde", t_prime=[1.0], probe_columns=[s0],
                          t_end=3.0, dt_out=0.1)
        run(rct, f"t7bp{sfx}_{mp}")
        with h5py.File(os.path.join(WORK, f"t7b{sfx}_{mp}"), "r") as f, h5py.File(os.path.join(WORK, f"t7bp{sfx}_{mp}"), "r") as g:
            tt, GR = f["rtde/tp=0/t"][()], f["rtde/tp=0/GR"][()][:, 0]
            dT = np.abs(GR - g["rtde/tp=0/GR"][()][:, 0]).max()
        ref = augmented(m, 1.0, 1.0, 3.0, s0, tt, bare=bool(sfx))
        e = np.abs(GR - ref).max()
        check(e < 2e-6, f"{gname:28s} DSL{sfx:2s}: max|G^R - ref| over {len(tt)} frames = {e:.1e} (|G^R_DSL - G^R_TPPX| up to {dT:.1e})")


def exact_GR_site(m, gamma, tp, taus):
    """single site (2 modes): G^R_up(t'+tau, t') = -i <{d_up(t'+tau), d_up^dag(t')}> by regression with the
    parity-twisted Lindbladian for odd operators (one-body jumps odd)."""
    ex = T3b.Exact(m, gamma)
    R = ex.evolve(np.array([0.0, tp]))[-1]
    d = ex.c[0].toarray()
    dim = ex.dim
    Lodd = np.zeros((dim * dim, dim * dim), complex)
    for j in range(dim * dim):
        E = np.zeros(dim * dim, complex)
        E[j] = 1.0
        Lodd[:, j] = ex.L_on(E.reshape(dim, dim), True).ravel()
    Y0 = (d.conj().T @ R + R @ d.conj().T).ravel()
    return np.array([-1j * np.trace(d @ (expm(Lodd * tau) @ Y0).reshape(dim, dim)) for tau in taus])


def test_single_site():
    print("[4] single site: DSL G^R EXACT vs exact Lindblad regression (two-body loss; + one-body gain; + one-body loss and gain)")
    eps0, gam, nu, nd, tp = -0.4, 1.3, 0.8, 0.6, 0.7
    cases = (("two-body loss", 0.0, 0.0), ("+ one-body gain", 0.0, 0.35), ("+ one-body loss and gain", 0.25, 0.35))
    for label, lg, ll in cases:
        h = np.full((2, 1, 1), eps0, complex)
        gt = np.full((2, 1, 1), lg, complex)
        lt = np.full((2, 1, 1), ll, complex)
        name = f"site_{lg}_{ll}.h5"
        T2.dp5model.write_model(os.path.join(WORK, name), h, gt, lt, np.array([[[nu]], [[nd]]], complex), [0], [1.0], {},
                                dict(generator="single site"))
        m = T2.dp5model.load_model(os.path.join(WORK, name))
        res = {}
        for ap in ("DSL", "DSL_0", "TPPX", "TPPX_0", "mGW", "mGW_0", "HF"):
            rc = T2.write_rc(f"rc7bs_{ap}_{name}", os.path.join(WORK, name), gam, ap, mode="rtde", t_prime=[tp],
                             t_end=tp + 3.0, dt_out=0.1)
            run(rc, f"s7b_{ap}_{name}")
            with h5py.File(os.path.join(WORK, f"s7b_{ap}_{name}"), "r") as f:
                tt, GR = f["rtde/tp=0/t"][()], f["rtde/tp=0/GR"][()][:, 0, 0, 0]
            if ap == "DSL":
                ex = exact_GR_site(m, gam, tp, tt - tp)
                ex0 = exact_GR_site(m, 0.0, tp, tt - tp)
                ho = eps0 - 1j * (lg + ll)
                e0 = np.abs(ex0 - (-1j * np.exp(-1j * ho * (tt - tp)))).max()
            res[ap] = np.abs(GR - ex).max()
            if ap == "DSL_0":  # DSL_0 against its own full spin-orbital solution (augmented reference, bare legs)
                res["DSL_0_ref"] = np.abs(GR - augmented(m, gam, tp, tp + 3.0, 0, tt, bare=True)[:, 0, 0]).max()
        check(e0 < 1e-9, f"{label:26s}: regression check, gamma = 0 exact G^R = -i e^(-i h_o tau) ({e0:.1e})")
        check(res["DSL"] < 1e-6, f"{label:26s}: DSL G^R exact on one site: max error {res['DSL']:.1e} "
                                 f"(TPPX {res['TPPX']:.1e}, TPPX_0 {res['TPPX_0']:.1e}, mGW {res['mGW']:.1e}, "
                                 f"mGW_0 {res['mGW_0']:.1e}, HF {res['HF']:.1e})")
        check(res["DSL_0"] > 1e-3 and res["DSL_0_ref"] < 1e-6,
              f"{label:26s}: DSL_0 G^R NOT exact on one site (max error {res['DSL_0']:.1e}; DSL is already exact, the "
              f"flag over-corrects); driver == full spin-orbital DSL_0 ({res['DSL_0_ref']:.1e})")


def test_gamma0_obs():
    print("[5] gamma = 0: G^R = -i exp(-i h_o tau) e_j; equal-time observables of an RTDE run")
    imp = os.path.join(WORK, "imp3.h5")
    m = T2.dp5model.load_model(imp)
    ho = m["h"] - 1j * (m["ell_greater"] + m["ell_lesser"])
    e = 0.0
    for s in (0, 2):
        rc0 = T2.write_rc(f"rc7bg0_{s}.h5", imp, 0.0, "DSL", mode="rtde", t_prime=[1.0], probe_columns=[s], t_end=4.0, dt_out=0.1)
        run(rc0, f"g07b_{s}.h5")
        with h5py.File(os.path.join(WORK, f"g07b_{s}.h5"), "r") as f:
            tt, GR = f["rtde/tp=0/t"][()], f["rtde/tp=0/GR"][()]
        e = max(e, max(np.abs(GR[k, 0, sp] - (-1j * expm(-1j * ho[sp] * (t - 1.0))[:, s])).max()
                       for k, t in enumerate(tt) for sp in range(2)))
    check(e < 1e-7, f"gamma = 0: G^R = -i exp(-i h_o tau) e_j (columns 0, 2): {e:.1e}")
    rcg = T2.write_rc("rc7bog.h5", imp, 1.0, "DSL", t_end=3.0, dt_out=0.1)
    rco = T2.write_rc("rc7bor.h5", imp, 1.0, "DSL", mode="rtde", t_prime=[1.0], probe_columns=[1], t_end=3.0, dt_out=0.1)
    run(rcg, "og7b.h5"); run(rco, "or7b.h5")
    with h5py.File(os.path.join(WORK, "og7b.h5"), "r") as a, h5py.File(os.path.join(WORK, "or7b.h5"), "r") as b:
        ga, gb = a["observables/equal_time"], b["observables/equal_time"]
        keys = sorted(ga.keys())
        e = max(np.nanmax(np.abs(ga[k][()] - gb[k][()])) for k in keys)
        same = keys == sorted(gb.keys())
    check(same and e < 1e-7, f"equal-time observables of a DSL RTDE run == those of the G1-G2 run ({len(keys)} series, {e:.1e})")


if __name__ == "__main__":
    tests = [test_rhs_random, test_tppx_part, test_trajectories, test_single_site, test_gamma0_obs]
    sel = sys.argv[1:]
    if not sel:
        shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK, exist_ok=True)
    for t in tests:
        if not sel or t.__name__ in sel:
            t()
    print(f"\n{'ALL STEP-7b TESTS PASSED' if not FAIL else 'FAILURES: ' + str(len(FAIL))}")
    sys.exit(1 if FAIL else 0)
