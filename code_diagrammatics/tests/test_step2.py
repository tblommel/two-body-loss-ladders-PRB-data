"""Step-2 tests: G1-G2 right-hand sides for TPPX (+2B), mGW (with counterterm), DSL, each also with bare propagators
(run card bare_propagators = 1; labels X_0, e.g. TPPX_0).

    python3 -I tests/test_step2.py

References: the full spin-orbital right-hand sides of tests/ref/item0_check.py (main Eqs. TPPG2EOM, PHG1G2;
SM DSL_final; an independent implementation of the equations in spin-orbital form), loaded by exec.
Spin-orbital index 2r + sigma (sigma = 0 up) inside the references only.
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
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))
import dp5model  # noqa: E402
import runcard  # noqa: E402

REF = os.path.join(HERE, "ref", "item0_check.py")  # vendored reference (full spin-orbital right-hand sides)
R = {}
exec(open(REF).read().split("# ================================================================ PART A")[0], R)
_eo = lambda *a, **k: np.einsum(*a, optimize=True, **k)
R["es"] = _eo  # same formulas, contraction order optimized (the reference functions look up `es` at call time)
es = _eo
xi = -1
EXE = os.environ.get("DP5_EXE", os.path.join(ROOT, "build", "dp5drive"))
WORK = os.path.join(HERE, "work2")
ENV = dict(os.environ, OMP_NUM_THREADS="4")
FAIL = []
rng = np.random.default_rng(2026)
eps = np.array([[0, 1], [-1, 0]], float)


def check(cond, msg):
    print(("   ok   " if cond else "   FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def crand(*s):
    return rng.normal(size=s) + 1j * rng.normal(size=s)


def herm(n):
    X = crand(n, n)
    return (X + X.conj().T) / 2


def rand_dm(n):
    Z = crand(n, n)
    _, U = np.linalg.eigh((Z + Z.conj().T) / 2)
    return U @ np.diag(rng.uniform(0.1, 0.9, n)) @ U.conj().T


# ---------------------------------------------------------------- models
def impurity_model(path, N, seed=1):
    g = np.random.default_rng(seed)
    n = (N - 1) // 2
    h = np.zeros((2, N, N), complex)
    for s, dz in enumerate((0.15, -0.15)):  # spin-dependent levels (Zeeman-like) to break spin symmetry
        H = np.zeros((N, N), complex)
        H[0, 0] = -1.0 + dz
        H[0, 1:] = 0.5 * g.normal(size=N - 1)
        H[1:, 0] = H[0, 1:]
        for k in range(1, N):
            H[k, k] = g.normal() + dz
        h[s] = H
    gt = np.zeros((2, N, N), complex)
    lt = np.zeros((2, N, N), complex)
    for s in range(2):
        gt[s, 1:1 + n, 1:1 + n] = np.diag(0.3 + 0.3 * g.random(n))
        lt[s, 1 + n:, 1 + n:] = np.diag(0.3 + 0.3 * g.random(n))
    rho = np.zeros((2, N, N), complex)
    for s in range(2):
        rho[s, 1 + n:, 1 + n:] = np.eye(N - 1 - n)
    rho[0, 0, 0], rho[1, 0, 0] = 0.9, 0.3
    dp5model.write_model(path, h, gt, lt, rho, [0], [1.0], dict(N_greater=n, N_lesser=N - 1 - n, ordering="d,p>,p<"),
                         dict(generator="test_step2 impurity"))
    return dp5model.load_model(path)


def symmetric_model(path, N, seed=4):  # spin-independent impurity model with symmetric start (paramagnetic tests)
    g = np.random.default_rng(seed)
    n = (N - 1) // 2
    H = np.zeros((N, N), complex)
    H[0, 0] = -1.0
    H[0, 1:] = 0.5 * g.normal(size=N - 1)
    H[1:, 0] = H[0, 1:]
    for k in range(1, N):
        H[k, k] = g.normal()
    G = np.zeros((N, N), complex)
    L = np.zeros((N, N), complex)
    G[1:1 + n, 1:1 + n] = np.diag(0.3 + 0.3 * g.random(n))
    L[1 + n:, 1 + n:] = np.diag(0.3 + 0.3 * g.random(n))
    rho = np.zeros((2, N, N), complex)
    for s in range(2):
        rho[s, 1 + n:, 1 + n:] = np.eye(N - 1 - n)
        rho[s, 0, 0] = 0.7
    dp5model.write_model(path, np.stack([H, H]), np.stack([G, G]), np.stack([L, L]), rho, [0], [1.0],
                         dict(ordering="d,p>,p<", N_greater=n, N_lesser=N - 1 - n), dict(generator="test_step2 sym"))
    return dp5model.load_model(path)


def ring_model(path, Ns, sites, weights, zeeman=0.2, seed=3, rho=None):
    g = np.random.default_rng(seed)
    h = np.zeros((2, Ns, Ns), complex)
    for s, dz in enumerate((zeeman, -zeeman)):
        for r in range(Ns):
            h[s, r, (r + 1) % Ns] = h[s, (r + 1) % Ns, r] = -1.0
            h[s, r, r] = 0.1 * g.normal() + dz
    if rho is None:
        rho = np.stack([rand_dm(Ns), rand_dm(Ns)])
    z = np.zeros_like(h)
    dp5model.write_model(path, h, z, z, rho, sites, weights, dict(ordering="ring"), dict(generator="test_step2 ring"))
    return dp5model.load_model(path)


# ---------------------------------------------------------------- full spin-orbital reference
def full_V(N, sites, weights, gamma):
    M = 2 * N
    V = np.zeros((M,) * 4, complex)
    for r, w in zip(sites, weights):
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for d in range(2):
                        V[2 * r + a, 2 * r + b, 2 * r + c, 2 * r + d] = gamma * w / 2 * eps[b, a] * eps[c, d]
    return V


def sd(u, d):
    N = u.shape[0]
    A = np.zeros((2 * N, 2 * N), complex)
    A[0::2, 0::2], A[1::2, 1::2] = u, d
    return A


def blk(T, s):
    a, b, c, d = s
    return T[a::2, b::2, c::2, d::2]


def put(T, s, X):
    a, b, c, d = s
    T[a::2, b::2, c::2, d::2] = X


PB, AB, MP, MA, UB, DB = (0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0), (0, 0, 0, 0), (1, 1, 1, 1)
mirP, mirA = (1, 0, 1, 0), (1, 0, 0, 1)


def assemble_pp(P, U, D):
    M = 2 * P.shape[0]
    G = np.zeros((M,) * 4, complex)
    put(G, PB, P); put(G, AB, xi * P.transpose(0, 1, 3, 2)); put(G, MP, xi * P.transpose(1, 0, 2, 3))
    put(G, MA, P.transpose(1, 0, 3, 2)); put(G, UB, U); put(G, DB, D)
    return G


def assemble_W(P, A, U, D, P2):
    M = 2 * P.shape[0]
    GW = np.zeros((M,) * 4, complex); G2 = np.zeros((M,) * 4, complex)
    put(GW, PB, P); put(GW, AB, A); put(GW, UB, U); put(GW, DB, D)
    put(GW, mirP, P.transpose(1, 0, 3, 2)); put(GW, mirA, A.transpose(1, 0, 3, 2))
    put(G2, PB, P2); put(G2, AB, xi * P2.transpose(0, 1, 3, 2))
    put(G2, mirA, xi * P2.transpose(1, 0, 2, 3)); put(G2, mirP, P2.transpose(1, 0, 3, 2))
    return GW, G2


def split_label(ap):
    """method label -> (approximation, flag_2B, bare_propagators): "TPPX2B" = TPPX + flag_2B, suffix "_0" = bare
    propagators (e.g. "TPPX_0", "DSL_0", "mGW_0", "TPPX2B_0")."""
    bare = ap.endswith("_0")
    b = ap[:-2] if bare else ap
    return ("TPPX", True, bare) if b == "TPPX2B" else (b, False, bare)


def full_system(m, gamma, approx, flag2B=False):
    """returns f(rho_u, rho_d, blocks) -> (drho_u, drho_d, dblocks) from the full spin-orbital equations,
    and the full-state pack/unpack for trajectories. approx may be a label (X_0: bare h_o in the drifts of the
    correlators, h_o^MF in the rho equation)."""
    approx, f2, bare = split_label(approx)
    flag2B = flag2B or f2
    N = m["h"].shape[1]
    V = full_V(N, m["lossy_sites"], m["site_weight"], gamma)
    hb = [m["h"][s] - 1j * (m["ell_greater"][s] + m["ell_lesser"][s]) for s in range(2)]
    lt2 = sd(2 * m["ell_lesser"][0], 2 * m["ell_lesser"][1])
    hbare = sd(hb[0], hb[1])

    def full_rhs(rl, T):
        ho = hbare - 1j * 2 * es("iklj,lk->ij", V, rl)
        hp = hbare if bare else ho  # drifts of the correlators
        o = R["objects"](rl, xi, V, 0 * V)
        if approx == "TPPX":
            G = T[0]
            dT = [R["rhs_pp"](o, hp, G, kern=not flag2B)]
            I = 2 * es("ablj,slab->sj", 1j * V, G)
        elif approx == "DSL":
            G = T[0]
            dT = [R["rhs_dsl"](o, hp, G)]
            I = 2 * es("ablj,slab->sj", 1j * V, G)
        elif approx == "mGW":
            GW, G2 = T
            dT = [R["rhs_W"](o, hp, GW), R["rhs_W"](o, hp, G2, kern=False)]
            # = 2 es("ablj,slab->sj", 1j V, GW - 0.5 G2) in the main/IV letters G^W_{slab} (X_{abjl} = xi X_{ablj})
            I = 2 * xi * es("pjko,sopj->sk", 1j * V, GW - 0.5 * G2)
        else:
            dT, I = [], 0 * rl
        drho = -1j * ho @ rl + 1j * rl @ ho.conj().T + lt2 - xi * (I + I.conj().T)
        return drho, dT

    def reduced(ru, rd, blocks):
        if approx == "TPPX":
            T = [assemble_pp(blocks[0], 0 * blocks[0], 0 * blocks[0])]
        elif approx == "DSL":
            T = [assemble_pp(*blocks)] if len(blocks) == 3 else [assemble_pp(blocks[0], *(2 * [blocks[0] + xi * blocks[0].transpose(0, 1, 3, 2)]))]
        elif approx == "mGW":
            if len(blocks) == 5:
                T = list(assemble_W(*blocks))
            else:
                P, A, P2 = blocks
                T = list(assemble_W(P, A, A + P, A + P, P2))
        else:
            T = []
        drho, dT = full_rhs(sd(ru, rd), T)
        out = [drho[0::2, 0::2], drho[1::2, 1::2]]
        if approx == "TPPX":
            out.append([blk(dT[0], PB)])
        elif approx == "DSL":
            out.append([blk(dT[0], PB), blk(dT[0], UB), blk(dT[0], DB)])
        elif approx == "mGW":
            out.append([blk(dT[0], PB), blk(dT[0], AB), blk(dT[0], UB), blk(dT[0], DB), blk(dT[1], PB)])
        else:
            out.append([])
        return out
    return full_rhs, reduced, V


def rand_blocks(approx, N, para=False):
    approx = split_label(approx)[0]
    c = lambda: crand(N, N, N, N)
    if approx in ("TPPX", "DSL"):
        P = c(); P = (P - P.conj().transpose(2, 3, 0, 1)) / 2
        if para:
            P = (P + P.transpose(1, 0, 3, 2)) / 2
            return [P]
        if approx != "DSL":
            return [P]
        def same(X):
            X = (X + xi * X.transpose(1, 0, 2, 3)) / 2; X = (X + xi * X.transpose(0, 1, 3, 2)) / 2
            return (X - X.conj().transpose(2, 3, 0, 1)) / 2
        return [P, same(c()), same(c())]
    P = c(); P = (P - P.conj().transpose(2, 3, 0, 1)) / 2
    A = c(); A = (A - A.conj().transpose(3, 2, 1, 0)) / 2
    U = c(); U = (U + U.transpose(1, 0, 3, 2)) / 2; U = (U - U.conj().transpose(3, 2, 1, 0)) / 2
    D = c(); D = (D + D.transpose(1, 0, 3, 2)) / 2; D = (D - D.conj().transpose(3, 2, 1, 0)) / 2
    P2 = c(); P2 = (P2 - P2.conj().transpose(2, 3, 0, 1)) / 2
    if para:
        P = (P + P.transpose(1, 0, 3, 2)) / 2; P = (P - P.conj().transpose(2, 3, 0, 1)) / 2
        A = (A + A.transpose(1, 0, 3, 2)) / 2; A = (A - A.conj().transpose(3, 2, 1, 0)) / 2
        P2 = (P2 + P2.transpose(1, 0, 3, 2)) / 2; P2 = (P2 - P2.conj().transpose(2, 3, 0, 1)) / 2
        P2 = (P2 + P2.transpose(0, 1, 3, 2)) / 2
        return [P, A, P2]
    return [P, A, U, D, P2]


# ---------------------------------------------------------------- packed antisymmetric storage (DSL same-spin)
def pairs(N):
    return [(s, l) for s in range(N) for l in range(s + 1, N)]  # = drv::tensor::pair_index order


def pack(X):
    N = X.shape[0]
    pr = pairs(N)
    return np.array([[X[s, l, a, b] for (a, b) in pr] for (s, l) in pr], complex)


def unpack(Xp, N):
    X = np.zeros((N,) * 4, complex)
    for i, (s, l) in enumerate(pairs(N)):
        for j, (a, b) in enumerate(pairs(N)):
            v = Xp[i, j]
            X[s, l, a, b], X[l, s, a, b], X[s, l, b, a], X[l, s, b, a] = v, -v, -v, v
    return X


def driver_blocks(ap, blocks, para=False):
    """full-array blocks -> the driver's storage (DSL same-spin blocks packed)."""
    if ap == "DSL" and not para and len(blocks) == 3:
        return [blocks[0], pack(blocks[1]), pack(blocks[2])]
    return blocks


# ---------------------------------------------------------------- driver helpers
def write_rc(name, model, gamma, approx, **kw):
    p = os.path.join(WORK, name)
    a, f2b, bare = split_label(approx)
    kw2 = dict(kw)
    if os.environ.get("DP5_RC_EXTRA"):  # test hook: extra run-card attributes for every card (e.g. '{"ss_hold_records": 1}')
        import json
        kw2.update(json.loads(os.environ["DP5_RC_EXTRA"]))
    if f2b:
        kw2["flag_2B"] = 1
    if bare:
        kw2["bare_propagators"] = 1
    runcard.write_runcard(p, model, gamma, kw2.pop("t_end", 1.0), kw2.pop("dt_out", 0.1), approximation=a, **kw2)
    return p


def eval_rhs(rc, ru, rd, blocks, para=False):
    y = np.concatenate([ru.ravel()] + ([] if para else [rd.ravel()]) + [b.ravel() for b in blocks]).astype(complex)
    fi, fo = os.path.join(WORK, "eval_in.h5"), os.path.join(WORK, "eval_out.h5")
    with h5py.File(fi, "w") as f:
        f["y"] = y
    p = subprocess.run([EXE, "--runcard", rc, "--eval-rhs", fi, fo], env=ENV, capture_output=True, text=True)
    if p.returncode:
        print(p.stdout, p.stderr)
    with h5py.File(fo, "r") as f:
        dy, D = f["dy"][()], f["D"][()]
        attrs = dict(f.attrs)
    N = ru.shape[0]
    n2 = N * N
    k = n2 if para else 2 * n2
    out = [dy[:n2].reshape(N, N), (dy[:n2] if para else dy[n2:2 * n2]).reshape(N, N)]
    bl = []
    for b in blocks:
        bl.append(dy[k:k + b.size].reshape(b.shape))
        k += b.size
    out.append(bl)
    return out, D, attrs


def run(rc, out, expect=0):
    p = subprocess.run([EXE, "--runcard", rc, "--output", out], cwd=WORK, env=ENV, capture_output=True, text=True)
    if p.returncode != expect:
        print(p.stdout[-3000:], p.stderr[-2000:])
    return p


def rel(a, b):
    n = max(np.abs(a).max(), np.abs(b).max(), 1e-300)
    return np.abs(a - b).max() / n


# ---------------------------------------------------------------- tests
APPROX = ["TPPX", "TPPX2B", "mGW", "DSL"]
APPROX0 = ["TPPX_0", "TPPX2B_0", "mGW_0", "DSL_0"]  # bare propagators
APPROX_ALL = APPROX + APPROX0


def test_rhs_random():
    print("[1] RHS on random inputs vs full spin-orbital references (main TPPG2EOM, PHG1G2; SM DSL_final)")
    imp = impurity_model(os.path.join(WORK, "imp4.h5"), 5)
    ring = ring_model(os.path.join(WORK, "ring_sub.h5"), 4, [0, 2], [1.0, 0.5])
    for gname, m, vimpls in (("impurity N=5", imp, ("general", "impurity")), ("ring 4, loss on {0,2} w=(1,.5)", ring, ("general",))):
        N = m["h"].shape[1]
        mp = os.path.join(WORK, "imp4.h5" if "impurity" in gname else "ring_sub.h5")
        for ap in APPROX_ALL:
            a = split_label(ap)[0]
            ru, rd = herm(N), herm(N)
            blocks = rand_blocks(a, N)
            _, reduced, _ = full_system(m, 0.8, ap)
            ref = reduced(ru, rd, blocks)
            res = {}
            for vi in vimpls:
                rc = write_rc(f"rc_eval_{ap}_{vi}.h5", mp, 0.8, ap, vertex_impl=vi)
                got, D, at = eval_rhs(rc, ru, rd, driver_blocks(a, blocks))
                res[vi] = got
                refb = driver_blocks(a, ref[2])
                e = max([rel(got[0], ref[0]), rel(got[1], ref[1])] + [rel(x, y) for x, y in zip(got[2], refb)])
                check(e < 1e-12, f"{gname:30s} {ap:8s} vertex {vi:8s}: max rel. dev. from full reference {e:.1e} "
                                 f"({len(blocks)} blocks + rho)")
            if len(vimpls) == 2:
                g, i = res["general"], res["impurity"]
                same = all(np.array_equal(x, y) for x, y in zip(g[:2] + g[2], i[:2] + i[2]))
                check(same, f"{gname:30s} {ap:8s} general {{0}} == impurity fast path: bitwise {same}")


def test_counterterm_identity():
    print("[2] counterterm identity G^{W,2} = 2 G^pp|_{h=0} at equal inputs (symm-G2pp)")
    imp = os.path.join(WORK, "imp4.h5")
    m = dp5model.load_model(imp)
    N = m["h"].shape[1]
    ru, rd = herm(N), herm(N)
    B = rand_blocks("mGW", N)
    K = B[4] / 2  # TPPX-2B tensor chosen as half the counterterm
    rcw = write_rc("rc_ct_w.h5", imp, 0.8, "mGW", vertex_impl="impurity")
    rcp = write_rc("rc_ct_p.h5", imp, 0.8, "TPPX2B", vertex_impl="impurity")
    gw, _, _ = eval_rhs(rcw, ru, rd, B)
    gp, _, _ = eval_rhs(rcp, ru, rd, [K])
    e = rel(gw[2][4], 2 * gp[2][0])
    check(e < 1e-14, f"RHS level: d G^{{W,2}} = 2 d G^pp_2B (same rho, h_o) to {e:.1e}")
    # bare propagators: the counterterm stays the second-order part of the mGW_0 equations (= 2 x TPPX2B_0)
    gw0, _, _ = eval_rhs(write_rc("rc_ct_w0.h5", imp, 0.8, "mGW_0", vertex_impl="impurity"), ru, rd, B)
    gp0, _, _ = eval_rhs(write_rc("rc_ct_p0.h5", imp, 0.8, "TPPX2B_0", vertex_impl="impurity"), ru, rd, [K])
    e0, d0 = rel(gw0[2][4], 2 * gp0[2][0]), rel(gw0[2][4], gw[2][4])
    check(e0 < 1e-14 and d0 > 1e-3, f"RHS level, bare propagators: d G^{{W,2}}(mGW_0) = 2 d G^pp_2B(TPPX2B_0) to {e0:.1e} "
                                    f"(the flag changes the counterterm RHS by {d0:.1e} rel.)")
    # trajectory: full spin-orbital mGW system augmented with a TPPX-2B tensor driven by the same rho(t)
    full_rhs, _, V = full_system(m, 0.8, "mGW")
    M = 2 * N
    hb = [m["h"][s] - 1j * (m["ell_greater"][s] + m["ell_lesser"][s]) for s in range(2)]
    hbare = sd(*hb)

    def f(t, y):
        rl = y[:M * M].reshape(M, M)
        GW, G2, Kf = (y[M * M + i * M ** 4:M * M + (i + 1) * M ** 4].reshape((M,) * 4) for i in range(3))
        drho, dT = full_rhs(rl, [GW, G2])
        ho = hbare - 1j * 2 * es("iklj,lk->ij", V, rl)
        dK = R["rhs_pp"](R["objects"](rl, xi, V, 0 * V), ho, Kf, kern=False)
        return np.concatenate([drho.ravel(), dT[0].ravel(), dT[1].ravel(), dK.ravel()])
    y0 = np.concatenate([sd(m["rho"][0], m["rho"][1]).ravel(), np.zeros(3 * M ** 4, complex)])
    sol = solve_ivp(f, (0, 1.0), y0, method="DOP853", rtol=1e-10, atol=1e-13, t_eval=[0.5, 1.0])
    G2t = sol.y[M * M + M ** 4:M * M + 2 * M ** 4, -1].reshape((M,) * 4)
    Kt = sol.y[M * M + 2 * M ** 4:, -1].reshape((M,) * 4)
    e2 = np.abs(G2t - 2 * Kt).max() / np.abs(G2t).max()
    check(e2 < 1e-12, f"trajectory (same solve, same rho(t), h_o(t)): G^{{W,2}}(t=1) = 2 G^pp_2B(t=1) to {e2:.1e} "
                      f"(|G^{{W,2}}| = {np.abs(G2t).max():.2e})")
    # and the driver's counterterm block at t=1 against this reference
    rc = write_rc("rc_ct_run.h5", imp, 0.8, "mGW", vertex_impl="impurity", t_end=1.0, dt_out=0.1)
    run(rc, "ct_run.h5")
    with h5py.File(os.path.join(WORK, "ct_run.h5"), "r") as fo:
        y = fo["final/y"][()]
    n2, n4 = N * N, N ** 4
    P2d = y[2 * n2 + 4 * n4:2 * n2 + 5 * n4].reshape(N, N, N, N)
    e3 = np.abs(P2d - blk(G2t, PB)).max() / np.abs(blk(G2t, PB)).max()
    check(e3 < 1e-5, f"driver counterterm block at t=1 vs reference: rel. dev. {e3:.1e}")


def traj_reference(m, gamma, approx, t_eval, flag2B=False):
    N = m["h"].shape[1]
    M = 2 * N
    full_rhs, _, _ = full_system(m, gamma, approx, flag2B)
    nt = {"TPPX": 1, "DSL": 1, "mGW": 2, "HF": 0}[split_label(approx)[0]]

    def f(t, y):
        rl = y[:M * M].reshape(M, M)
        T = [y[M * M + i * M ** 4:M * M + (i + 1) * M ** 4].reshape((M,) * 4) for i in range(nt)]
        drho, dT = full_rhs(rl, T)
        return np.concatenate([drho.ravel()] + [d.ravel() for d in dT])
    y0 = np.concatenate([sd(m["rho"][0], m["rho"][1]).ravel(), np.zeros(nt * M ** 4, complex)])
    sol = solve_ivp(f, (t_eval[0], t_eval[-1]), y0, method="DOP853", rtol=1e-11, atol=1e-13, t_eval=t_eval)
    rl = sol.y[:M * M].T.reshape(len(t_eval), M, M)
    return np.stack([rl[:, 0::2, 0::2], rl[:, 1::2, 1::2]], axis=1), sol.y[M * M:, -1]


def test_trajectories():
    print("[3] trajectories vs full spin-orbital solve_ivp (rtol 1e-11)")
    imp = impurity_model(os.path.join(WORK, "imp3.h5"), 3)
    ring = ring_model(os.path.join(WORK, "ring4.h5"), 4, [0, 2], [1.0, 0.5])
    t = np.linspace(0, 2, 21)
    for gname, mp, m in (("impurity N=3", "imp3.h5", imp), ("ring 4, loss on {0,2}", "ring4.h5", ring)):
        for ap in APPROX_ALL:
            ref, _ = traj_reference(m, 1.0, ap, t)
            rc = write_rc(f"rc_tr_{ap}_{mp}", os.path.join(WORK, mp), 1.0, ap, t_end=2.0, dt_out=0.1)
            run(rc, f"tr_{ap}_{mp}")
            with h5py.File(os.path.join(WORK, f"tr_{ap}_{mp}"), "r") as f:
                rho = f["rho/rho"][()]
            e = np.abs(rho - ref).max()
            check(e < 2e-6, f"{gname:22s} {ap:8s}: max|rho - rho_ref| over 21 frames = {e:.1e}")


def test_exactness():
    print("[4] single site (N=1 per spin): exact 4-state Lindblad (main Eq. LIND: 2 L rho L^dag - {L^dag L, rho})")
    eps0, gam = -0.4, 1.3
    nu, nd = 0.8, 0.6
    h = np.full((2, 1, 1), eps0, complex)
    z = np.zeros_like(h)
    rho = np.array([[[nu]], [[nd]]], complex)
    dp5model.write_model(os.path.join(WORK, "site.h5"), h, z, z, rho, [0], [1.0], {}, dict(generator="single site"))
    t = np.linspace(0, 3, 31)
    D_ex = nu * nd * np.exp(-2 * gam * t)  # L = sqrt(gamma) d_dn d_up: jump rate 2 gamma <n_up n_dn>
    n_ex = nu * (1 - nd) + D_ex
    res = {}
    msite = dp5model.load_model(os.path.join(WORK, "site.h5"))
    for ap in ("HF", "TPPX", "TPPX_0", "TPPX2B", "TPPX2B_0", "DSL", "DSL_0", "mGW", "mGW_0"):
        rc = write_rc(f"rc_site_{ap}.h5", os.path.join(WORK, "site.h5"), gam, ap, t_end=3.0, dt_out=0.1)
        run(rc, f"site_{ap}.h5")
        with h5py.File(os.path.join(WORK, f"site_{ap}.h5"), "r") as f:
            n = f["observables/n_d"][()][:, 0]
            D = f["observables/D"][()][:, 0]
            rho = f["rho/rho"][()]
        res[ap] = (np.abs(n - n_ex).max(), np.abs(D - D_ex).max())
        line = f"      {ap:8s}: max|n_up - exact| = {res[ap][0]:.2e}, max|D - exact| = {res[ap][1]:.2e}"
        if ap.endswith("_0"):  # the bare-propagator variants against their own full spin-orbital solution
            ref, _ = traj_reference(msite, gam, ap, t)
            line += f"   (driver vs full spin-orbital {ap}: {np.abs(rho - ref).max():.1e})"
            res[ap + "_ref"] = np.abs(rho - ref).max()
        print(line)
    check(res["DSL"][0] < 1e-6 and res["DSL"][1] < 1e-6, "DSL is exact on one site (C3 = 0 for two spin-orbitals)")
    check(res["TPPX_0"][0] < 1e-6 and res["TPPX_0"][1] < 1e-6, "TPPX_0 is exact on one site for pure loss")
    check(res["TPPX"][0] > 1e-3, f"TPPX is NOT exact (self-damping 2 gamma S k): deviation {res['TPPX'][0]:.2e}")
    # DSL_0: DSL is already exact here, so removing l^MF from its pair drifts (the same drift change that makes TPPX
    # exact) over-corrects: NOT exact -- an approximation property, confirmed by the full spin-orbital DSL_0 solution
    check(res["DSL_0"][0] > 1e-3 and res["DSL_0_ref"] < 1e-6,
          f"DSL_0 is NOT exact on one site (n {res['DSL_0'][0]:.2e}, D {res['DSL_0'][1]:.2e}); driver == full "
          f"spin-orbital DSL_0 ({res['DSL_0_ref']:.1e})")
    check(all(res[a + "_0_ref"] < 1e-6 for a in ("TPPX", "TPPX2B", "mGW")), "TPPX_0, TPPX2B_0, mGW_0 single site == "
          "their full spin-orbital solutions (" + ", ".join(f"{res[a + '_0_ref']:.1e}" for a in ("TPPX", "TPPX2B", "mGW")) + ")")
    # RHS mechanism on one site: the flag adds the same drift change to every scheme's P block, so
    # d(DSL_0) - d(DSL) == d(TPPX_0) - d(TPPX) for the same state (and DSL - TPPX = the exact - TPPX rung difference)
    ru, rd = np.array([[0.7]], complex), np.array([[0.45]], complex)
    P = np.array([[[[0.03j]]]])
    d = {}
    for ap in ("TPPX", "TPPX_0", "DSL", "DSL_0"):
        got, _, _ = eval_rhs(write_rc(f"rc_site_rhs_{ap}.h5", os.path.join(WORK, "site.h5"), gam, ap), ru, rd, [P])
        d[ap] = got[2][0]
    e = abs((d["DSL_0"] - d["DSL"]) - (d["TPPX_0"] - d["TPPX"])).max()
    check(e < 1e-15, f"one site, RHS: d(DSL_0) - d(DSL) == d(TPPX_0) - d(TPPX) = {complex((d['TPPX_0'] - d['TPPX']).ravel()[0]):.4f} "
                     f"({e:.1e}): DSL_0 = exact RHS + this term")
    # dN/dt identity on a single site: loss_rate = -4 gamma D = dN/dt
    with h5py.File(os.path.join(WORK, "site_DSL.h5"), "r") as f:
        e = np.abs(f["observables/loss_rate"][()] - f["observables/dN_dt"][()]).max()
    check(e < 1e-12, f"loss_rate (-4 gamma D) == Re tr d rho/dt without bath: {e:.1e}")


def test_symmetry():
    print("[5] symmetries along trajectories")
    imp = os.path.join(WORK, "imp4.h5")
    m = dp5model.load_model(imp)
    N = m["h"].shape[1]
    n2, n4 = N * N, N ** 4
    for ap in ("TPPX", "DSL", "mGW", "TPPX_0", "DSL_0", "mGW_0"):
        base = split_label(ap)[0]
        rc = write_rc(f"rc_sym_{ap}.h5", imp, 1.0, ap, t_end=2.0, dt_out=0.1)
        run(rc, f"sym_{ap}.h5")
        with h5py.File(os.path.join(WORK, f"sym_{ap}.h5"), "r") as f:
            y = f["final/y"][()]
        if base == "DSL":
            A2 = (N * (N - 1) // 2) ** 2
            o = 2 * n2
            B = [y[o:o + n4].reshape(N, N, N, N), unpack(y[o + n4:o + n4 + A2].reshape(-1, int(A2 ** .5)), N),
                 unpack(y[o + n4 + A2:o + n4 + 2 * A2].reshape(-1, int(A2 ** .5)), N)]
        else:
            B = [y[2 * n2 + i * n4:2 * n2 + (i + 1) * n4].reshape(N, N, N, N) for i in range((len(y) - 2 * n2) // n4)]
        if base == "TPPX":
            P = B[0]
            r = dict(conj=rel(P, -P.conj().transpose(2, 3, 0, 1)))
        elif base == "DSL":
            P, U, D = B
            r = dict(conjP=rel(P, -P.conj().transpose(2, 3, 0, 1)), U_P12=rel(U, xi * U.transpose(1, 0, 2, 3)),
                     U_P34=rel(U, xi * U.transpose(0, 1, 3, 2)), D_P34=rel(D, xi * D.transpose(0, 1, 3, 2)),
                     U_conj=rel(U, -U.conj().transpose(2, 3, 0, 1)), Pauli=np.abs(np.einsum('ssab->sab', U)).max())
        else:
            P, A, U, D, P2 = B
            r = dict(P_conj=rel(P, -P.conj().transpose(2, 3, 0, 1)), A_conj=rel(A, -A.conj().transpose(3, 2, 1, 0)),
                     U_dbl=rel(U, U.transpose(1, 0, 3, 2)), U_conj=rel(U, -U.conj().transpose(3, 2, 1, 0)),
                     P2_conj=rel(P2, -P2.conj().transpose(2, 3, 0, 1)))
        check(max(r.values()) < 1e-10, f"{ap:6s} symmetries at t=2: " + ", ".join(f"{k} {v:.1e}" for k, v in r.items()))


def test_plaquette():
    print("[6] plaquette smoke test: 4-site ring, loss on all sites, no bath")
    rho = np.stack([np.diag([0.9, 0.8, 0.7, 0.6]).astype(complex), np.diag([0.6, 0.5, 0.6, 0.5]).astype(complex)])
    ring_model(os.path.join(WORK, "plaq.h5"), 4, [0, 1, 2, 3], [1, 1, 1, 1], zeeman=0.0, rho=rho)
    for ap in ["HF"] + APPROX_ALL:
        rc = write_rc(f"rc_pl_{ap}.h5", os.path.join(WORK, "plaq.h5"), 1.0, ap, t_end=6.0, dt_out=0.1)
        run(rc, f"pl_{ap}.h5", expect=7 if ap == "mGW_0" else 0)
        with h5py.File(os.path.join(WORK, f"pl_{ap}.h5"), "r") as f:
            Ntot = f["observables/N"][()].sum(axis=1)
            rhos = f["rho/rho"][()]
            trho = f["rho/t"][()]
            status = f["status"].attrs["status"]
            lr, dn = f["observables/loss_rate"][()], f["observables/dN_dt"][()]
            D = f["observables/D"][()]
        ev = np.concatenate([np.linalg.eigvalsh(rhos[k, s]) for k in range(len(rhos)) for s in range(2)])
        mono = np.all(np.diff(Ntot) <= 1e-12)
        print(f"      {ap:8s}: N {Ntot[0]:.3f} -> {Ntot[-1]:.4f}, monotone {mono}, eig(rho) in [{ev.min():.2e}, {ev.max():.4f}], "
              f"min D {D.min():.2e}, |loss_rate - dN/dt| {np.abs(lr - dn).max():.1e}")
        check(np.abs(lr - dn).max() < 1e-12, f"{ap:6s}: dN/dt == -4 gamma sum_r D_r at every frame")
        if ap in ("HF", "TPPX", "TPPX2B", "DSL"):
            check(mono and ev.min() > -1e-8 and ev.max() < 1 + 1e-8, f"{ap:6s}: N(t) decreasing, rho eigenvalues in [0,1]")
        else:  # TPPX_0, mGW and the other bare-propagator variants: N(t) need not be monotone here
            # (D < 0 at times) -- a property of the approximation, confirmed by the independent full spin-orbital
            # solution of the same equations
            # (a run may stop early with status Diverged, exit 7: compared on the frames it wrote)
            ref, _ = traj_reference(dp5model.load_model(os.path.join(WORK, "plaq.h5")), 1.0, ap, trho)
            Nref = np.real(np.einsum("tsii->t", ref))
            e = np.abs(rhos - ref).max() / max(1.0, np.abs(ref).max())
            note = f"; driver status {status} at t = {trho[-1]:.2f} (reference max|rho| there {np.abs(ref[-1]).max():.2f})" \
                if status != "Ok" else ""
            check(e < 1e-6 and (np.all(np.diff(Nref) <= 1e-12) == mono),
                  f"{ap:6s}: N(t) monotone = {mono} (min D = {D.min():.2e}); the full spin-orbital reference agrees "
                  f"(max|rho - ref| = {e:.1e} rel., ref monotone = {np.all(np.diff(Nref) <= 1e-12)}){note}: approximation property")


def layout_sizes(rc):
    """[(name, size)] of the eval-rhs state (G1-G2 blocks, then one RTDE block set per probe column)."""
    p = subprocess.run([EXE, "--runcard", rc, "--print-layout"], env=ENV, capture_output=True, text=True)
    g1 = p.stdout.split("G1-G2:")[1].split("(")[0].split()
    rt = p.stdout.split("RTDE per column:")[1].split("(")[0].split()
    parse = lambda L: [(x.split("[")[0], int(x.split("[")[1].rstrip("]"))) for x in L]
    with h5py.File(rc, "r") as f:
        mode = f["run"].attrs["mode"]
        ncol = len(f["run"].attrs.get("probe_columns", [0]))
    return parse(g1) + (parse(rt) * ncol if mode == "rtde" else [])


def eval_raw(rc, y):
    fi, fo = os.path.join(WORK, "eval_in.h5"), os.path.join(WORK, "eval_out.h5")
    with h5py.File(fi, "w") as f:
        f["y"] = y
    p = subprocess.run([EXE, "--runcard", rc, "--eval-rhs", fi, fo], env=ENV, capture_output=True, text=True)
    if p.returncode:
        print(p.stdout, p.stderr)
    with h5py.File(fo, "r") as f:
        return f["dy"][()]


def split_blocks(sizes, v):
    out, k = {}, 0
    for n, sz in sizes:
        out[n] = v[k:k + sz]
        k += sz
    return out


def test_bare_flag():
    print("[7] bare-propagator flag (bare_propagators = 1, labels X_0): validation, drift-only effect, self-description")
    imp = os.path.join(WORK, "imp4.h5")
    m = dp5model.load_model(imp)
    N = m["h"].shape[1]
    # (a) rejected for HF; (b) approximation BARE removed with a pointer to TPPX + flag
    for ap, needles in (("HF_0", ("bare_propagators", "HF")), ("BARE", ("TPPX", "bare_propagators = 1"))):
        rc = write_rc(f"rc_rej_{ap}.h5", imp, 0.8, "HF" if ap == "HF_0" else ap,
                      **(dict(bare_propagators=1) if ap == "HF_0" else {}))
        p = run(rc, f"rej_{ap}.h5", expect=2)
        msg = (p.stdout + p.stderr).strip().splitlines()[-1] if (p.stdout + p.stderr).strip() else ""
        check(p.returncode == 2 and all(x in msg for x in needles) and not os.path.exists(os.path.join(WORK, f"rej_{ap}.h5")),
              f"{'HF + bare_propagators' if ap == 'HF_0' else 'approximation = BARE'} refused (exit {p.returncode}): {msg[:150]}")
    # (c) no-op when l^MF = 0 (gamma = 0, or rho^sigma_rr = 0 on the lossy site): bitwise identical RHS, all blocks,
    #     G1-G2 and RTDE (mode rtde, probe column 2)
    for case in ("gamma = 0", "rho_00 = 0"):
        gam = 0.0 if case == "gamma = 0" else 0.8
        for ap in APPROX:
            kw = dict(mode="rtde", t_prime=[0.5], probe_columns=[2])
            rc0 = write_rc(f"rc_noop_{ap}.h5", imp, gam, ap, **kw)
            rc1 = write_rc(f"rc_noop_{ap}_0.h5", imp, gam, ap + "_0", **kw)
            sz = layout_sizes(rc0)
            y = crand(sum(x for _, x in sz))
            b = split_blocks(sz, y)
            r = b["rho"].reshape(2, N, N)
            r[:] = r + r.conj().transpose(0, 2, 1)
            if case == "rho_00 = 0":
                r[:, 0, 0] = 0.0
            same = np.array_equal(eval_raw(rc0, y), eval_raw(rc1, y))
            check(same, f"l^MF = 0 ({case:10s}) {ap:6s}: flag on == flag off bitwise (all G1-G2 and RTDE blocks)")
    # (d) l^MF != 0: the rho block and the G^R blocks (own drift -i h_o^MF G^R + I^R) are bitwise unchanged; every
    #     correlator block changes; the change of the opposite-spin pp-ordered block is the same drift term in every
    #     scheme (TPPX Kpp, DSL G_opp, mGW GW_P and the counterterm GW2_P)
    ru, rd = herm(N), herm(N)
    Pin = rand_blocks("TPPX", N)[0]
    diffP = {}
    for ap in APPROX:
        kw = dict(mode="rtde", t_prime=[0.5], probe_columns=[2])
        rc0 = write_rc(f"rc_dr_{ap}.h5", imp, 0.8, ap, **kw)
        rc1 = write_rc(f"rc_dr_{ap}_0.h5", imp, 0.8, ap + "_0", **kw)
        sz = layout_sizes(rc0)
        y = crand(sum(x for _, x in sz))
        b = split_blocks(sz, y)
        b["rho"][:] = np.concatenate([ru.ravel(), rd.ravel()])
        for n in ("Kpp", "G_opp", "GW_P", "GW2_P"):
            if n in b:
                b[n][:] = Pin.ravel()
        y = np.concatenate([b[n] for n, _ in sz])
        d0, d1 = split_blocks(sz, eval_raw(rc0, y)), split_blocks(sz, eval_raw(rc1, y))
        unchanged = [n for n, _ in sz if n == "rho" or n.startswith("GR_")]
        changed = [n for n, _ in sz if n not in unchanged]
        ok_u = all(np.array_equal(d0[n], d1[n]) for n in unchanged)
        dmin = min(rel(d0[n], d1[n]) for n in changed)
        check(ok_u and dmin > 1e-4, f"l^MF != 0, {ap:6s}: rho and G^R blocks bitwise unchanged ({', '.join(unchanged)}); "
                                    f"all {len(changed)} correlator blocks change (min rel. change {dmin:.1e})")
        for n in ("Kpp", "G_opp", "GW_P", "GW2_P"):
            if n in d0:
                diffP[f"{ap}:{n}"] = d1[n] - d0[n]
    ref = diffP["TPPX:Kpp"]
    e = max(rel(v, ref) for v in diffP.values())
    check(e < 1e-13, f"flag-induced change of the opposite-spin pp block is the same drift term in all schemes "
                     f"({', '.join(diffP)}): {e:.1e}")
    # (e) self-describing output and checkpoint
    rc = write_rc("rc_sd.h5", imp, 0.8, "DSL_0", t_end=0.4, dt_out=0.1, checkpoint_interval=0.2)
    run(rc, "sd.h5")
    with h5py.File(os.path.join(WORK, "sd.h5"), "r") as f:
        a, pv, ck = dict(f["input"].attrs), dict(f["provenance"].attrs), dict(f["final"].attrs)
        ok = (a["method"] == "DSL_0" and a["bare_propagators"] == 1 and a["approximation"] == "DSL" and
              f["input/run"].attrs["bare_propagators"] == 1 and pv["method"] == "DSL_0" and pv["bare_propagators"] == 1
              and ck["bare_propagators"] == 1)
    check(ok, f"output self-describing: /input method {a['method']}, bare_propagators {a['bare_propagators']}; /provenance "
              f"method {pv['method']}; /final bare_propagators {ck['bare_propagators']}")
    for ap in ("TPPX", "TPPX2B", "mGW", "DSL", "HF"):
        rcl = write_rc(f"rc_lbl_{ap}.h5", imp, 0.8, ap)
        p = subprocess.run([EXE, "--runcard", rcl, "--print-layout"], env=ENV, capture_output=True, text=True)
        lab = p.stdout.splitlines()[0]
        rcl0 = write_rc(f"rc_lbl_{ap}_0.h5", imp, 0.8, ap + "_0") if ap != "HF" else None
        lab0 = subprocess.run([EXE, "--runcard", rcl0, "--print-layout"], env=ENV, capture_output=True, text=True).stdout.splitlines()[0] if rcl0 else "-"
        print(f"      labels: {ap:6s} -> '{lab}', with flag -> '{lab0}'")
    # (f) optional: against a reference binary (DP5_REF_EXE, e.g. the development driver built with the same flags):
    #     bitwise equal final state, rho series and G^R frames for every scheme, g1g2 and rtde
    ref_exe = os.environ.get("DP5_REF_EXE")
    if not ref_exe:
        print("      (DP5_REF_EXE not set: comparison with a reference binary skipped)")
        return
    imp3 = os.path.join(WORK, "imp3.h5")
    if not os.path.exists(imp3):
        impurity_model(imp3, 3)
    def run_with(exe, rc, out, *a):
        if os.path.exists(os.path.join(WORK, out)):
            os.remove(os.path.join(WORK, out))
        return subprocess.run([exe, "--runcard", rc, "--output", out, *a], cwd=WORK, env=ENV, capture_output=True, text=True)
    def same_out(o1, o2):
        with h5py.File(os.path.join(WORK, o1), "r") as a, h5py.File(os.path.join(WORK, o2), "r") as b:
            ok = np.array_equal(a["final/y"][()], b["final/y"][()]) and np.array_equal(a["rho/rho"][()], b["rho/rho"][()])
            if "rtde" in a and "tp=0" in a["rtde"]:
                ok = ok and np.array_equal(a["rtde/tp=0/GR"][()], b["rtde/tp=0/GR"][()])
        return ok
    for ap in APPROX_ALL + ["HF"]:
        for mode in ("g1g2", "rtde"):
            kw = dict(t_end=3.0, dt_out=0.1, mode=mode, **(dict(t_prime=[1.0], probe_columns=[1]) if mode == "rtde" else {}))
            rc = write_rc(f"rc_ref_{ap}_{mode}.h5", imp3, 1.0, ap, **kw)
            run_with(ref_exe, rc, f"ref_old_{ap}_{mode}.h5"); run_with(EXE, rc, f"ref_new_{ap}_{mode}.h5")
            check(same_out(f"ref_old_{ap}_{mode}.h5", f"ref_new_{ap}_{mode}.h5"),
                  f"{ap:8s} {mode:5s}: this binary == reference binary bitwise (final state, rho, G^R)")


def test_inputs():
    print("[8] closed run-card key set and CLI: refused inputs exit with code 2 and a message")
    imp = os.path.join(WORK, "imp4.h5")
    rc_ok = write_rc("rc_in_ok.h5", imp, 0.8, "TPPX", t_end=0.2, dt_out=0.1)
    def with_attr(name, key, val):
        p = os.path.join(WORK, name)
        shutil.copy(rc_ok, p)
        with h5py.File(p, "a") as f:
            f["run"].attrs[key] = val
        return p
    cases = [("unknown key", with_attr("rc_in_k.h5", "walltime_hours", 1.0), [], "walltime_hours"),
             ("unknown key", with_attr("rc_in_c.h5", "corr_matrix", 1), [], "corr_matrix"),
             ("two probe columns", with_attr("rc_in_p.h5", "probe_columns", np.array([0.0, 1.0])), [], "probe_columns"),
             ("two t'", with_attr("rc_in_t.h5", "t_prime", np.array([0.05, 0.1])), [], "t_prime"),
             ("spin_mode paramagnetic", with_attr("rc_in_s.h5", "spin_mode", "paramagnetic"), [], "spin_mode"),
             ("mode rtde_frozen", with_attr("rc_in_m.h5", "mode", "rtde_frozen"), [], "mode"),
             ("approximation BARE", with_attr("rc_in_b.h5", "approximation", "BARE"), [], "bare_propagators = 1"),
             ("CLI --restart", rc_ok, ["--restart", "x.h5"], "unknown option"),
             ("CLI --walltime-hours", rc_ok, ["--walltime-hours", "1"], "unknown option"),
             ("output not .h5", rc_ok, ["--output", "outdir"], ".h5")]
    for label, rc, extra, needle in cases:
        args = [EXE, "--runcard", rc] + (extra if "--output" in extra else ["--output", "in_bad.h5"] + extra)
        p = subprocess.run(args, cwd=WORK, env=ENV, capture_output=True, text=True)
        msg = p.stderr.strip()
        check(p.returncode == 2 and needle in msg and not os.path.exists(os.path.join(WORK, "in_bad.h5")),
              f"{label:24s}: exit {p.returncode}, '...{msg[-90:]}'")
    p = subprocess.run([EXE, "--runcard", rc_ok], cwd=WORK, env=ENV, capture_output=True, text=True)
    check(p.returncode == 2 and "--output" in p.stderr, "missing --output: exit 2")
    p = subprocess.run([EXE, "--version"], env=ENV, capture_output=True, text=True)
    ok_v = p.returncode == 0 and "git_describe" in p.stdout and "build_flags" in p.stdout
    run(rc_ok, "in_ok.h5")
    with h5py.File(os.path.join(WORK, "in_ok.h5"), "r") as f:
        pv = dict(f["provenance"].attrs)
        rv = f.attrs["release_version"]
    ok_s = all(k in pv for k in ("release_version", "git_describe", "build_flags", "compiler", "eigen_version", "hdf5_version"))
    check(ok_v and ok_s and rv == pv["release_version"] and rv in p.stdout,
          f"--version and output stamp agree: release_version {rv}")


if __name__ == "__main__":
    tests = [test_rhs_random, test_counterterm_identity, test_trajectories, test_exactness, test_symmetry,
             test_plaquette, test_bare_flag, test_inputs]
    sel = sys.argv[1:]
    if not sel:
        shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK, exist_ok=True)
    if sel and not os.path.exists(os.path.join(WORK, "imp4.h5")):
        sel = ["test_rhs_random"] + sel
    for t in tests:
        if not sel or t.__name__ in sel:
            t()
    print(f"\n{'ALL STEP-2 TESTS PASSED' if not FAIL else 'FAILURES: ' + str(len(FAIL))}")
    sys.exit(1 if FAIL else 0)
