"""Item 0 (symmetry section): numerical checks of every symmetry claimed in item0_symm_v1.tex.

Part A: general statements, generic graded vee/wedge (several jump operators), both statistics.
Part B: model statements (on-site singlet loss, lattice and impurity), xi = -1, wedge = 0.

All right-hand sides are written directly from main.tex / SM.tex:
  TPPX G1-G2       main (TPPG2EOM)
  TPPX RTDE        SM   (TPP_RTDE_FINALEOM), slot order F_ksab
  mGW G1-G2        main (PHG1G2), counterterm = without the two hW terms
  mGW RTDE         main (PHRTDE), counterterm = without kernel and G^W terms
  GW (unmodified)  same with factors 2 removed, no counterterm
  DSL G1-G2        SM   (DSL_final), (DSL_Yform)
  DSL RTDE         SM   (DSLR_final), (DSLR_ZF), (DSLR_ZG)
Run: python3 item0_check.py > item0_check.txt
"""
import numpy as np
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

es = np.einsum
rng = np.random.default_rng(20261027)


def crand(*shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def relF(a, b):
    """Frobenius relative distance, used for the 'no symmetry expected' lines (0 = symmetric, ~1 = generic)."""
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300)


def rel(a, b):
    n = max(np.abs(a).max(), np.abs(b).max(), 1e-300)
    return np.abs(a - b).max() / n


# ---------------------------------------------------------------- objects
def objects(rl, xi, V, W):
    M = rl.shape[0]
    I = np.eye(M)
    rg = rl + xi * I
    Lg = es('si,lk->slki', rg, rg)
    Ll = es('si,lk->slki', rl, rl)
    Lf = Lg - Ll
    hpp = es('slki,kief->slef', Lf, W - V)
    P = lambda x, y: es('bd,ca->bcda', x, y)
    Pg, Pl, Pgg, Pll = P(rg, rl), P(rl, rg), P(rg, rg), P(rl, rl)
    hW = (es('icdj,si->scdj', W, rg) + es('scdl,lj->scdj', W, rg)
          - es('icdj,si->scdj', V, rl) - es('scdl,lj->scdj', V, rl))
    return dict(I=I, rg=rg, rl=rl, Lg=Lg, Ll=Ll, Lf=Lf, hpp=hpp,
                Pg=Pg, Pl=Pl, Pgg=Pgg, Pll=Pll, hW=hW, V=V, W=W, xi=xi)


def drift4(h, G):  # -i h_s G_x... -i h_l G_sx.. + i G_..x. h^dag_xa + i G_...x h^dag_xb
    hd = h.conj().T
    return (-1j * es('sx,xlab->slab', h, G) - 1j * es('lx,sxab->slab', h, G)
            + 1j * es('slxb,xa->slab', G, hd) + 1j * es('slax,xb->slab', G, hd))


# ---- TPPX
def src_pp(o):
    V, W = o['V'], o['W']
    return (1j * es('kicd,slki,abcd->slab', V, o['Lf'], o['Ll'].conj())
            + 1j * es('kicd,slki,abcd->slab', V, o['Ll'], o['Lf'].conj())
            - 1j * es('kicd,slki,abcd->slab', W, o['Lf'], o['Lg'].conj())
            - 1j * es('kicd,slki,abcd->slab', W, o['Lg'], o['Lf'].conj()))


def mem_pp(o, G):
    return es('slef,efab->slab', o['hpp'], G) + es('slcd,abcd->slab', G, o['hpp'].conj())


def rhs_pp(o, h, G, kern=True):
    r = drift4(h, G) + src_pp(o)
    return r + mem_pp(o, G) if kern else r


def driftF_pp(h, F):  # F_ksab
    hd = h.conj().T
    return (1j * es('xsab,xk->ksab', F, hd) - 1j * es('ax,ksxb->ksab', h, F)
            - 1j * es('bx,ksax->ksab', h, F))


def srcF_pp(o, GR, Gpp, withG=True):
    V, W, xi, rl, rg = o['V'], o['W'], o['xi'], o['rl'], o['rg']
    br = (xi * (es('lk,cdjl,abdc->kabj', rl, V, o['Lf']) - es('lk,cdjl,abdc->kabj', rg, W, o['Lf']))
          + es('cdjk,abdc->kabj', V, o['Ll']) - es('cdjk,abdc->kabj', W, o['Lg']))
    if withG:
        br = br - 1j * es('efkj,abef->kabj', W - V, Gpp)
    return es('kabj,js->ksab', br, GR)


def rhsF_pp(o, h, GR, F, Gpp, full=True):
    r = driftF_pp(h, F) + srcF_pp(o, GR, Gpp, withG=full)
    if full:
        r = r + es('ksmn,abmn->ksab', F, o['hpp'])
    return r


# ---- mGW / GW G1-G2   (G_sopj)
def src_W(o, fac=2.0):
    V, W = o['V'], o['W']
    s = (1j * es('icdl,slij,dopc->sopj', W + V, o['Pg'], o['Pl'])
         + 1j * es('icdl,slij,dopc->sopj', W + V, o['Pl'], o['Pg'])
         - 2j * es('icdl,slij,dopc->sopj', W, o['Pgg'], o['Pgg'])
         - 2j * es('icdl,slij,dopc->sopj', V, o['Pll'], o['Pll']))
    return fac * s


def mem_W(o, G, fac=2.0):
    hW = o['hW']
    return fac * (es('scdj,dopc->sopj', hW, G) + es('scdj,odcp->sopj', G, hW))


def rhs_W(o, h, G, fac=2.0, kern=True):
    r = drift4(h, G) + src_W(o, fac)
    return r + mem_W(o, G, fac) if kern else r


# ---- mGW / GW RTDE   (F_sopq)
def driftF_W(h, F):
    hd = h.conj().T
    return (-1j * es('qx,sopx->sopq', h, F) - 1j * es('px,soxq->sopq', h, F)
            + 1j * es('sxpq,xo->sopq', F, hd))


def srcF_W(o, GR, GW, fac=2.0, withG=True):
    V, W, rg, rl = o['V'], o['W'], o['rg'], o['rl']
    br = (es('lbaj,pabo,ql->opqj', W + V, o['Pg'], rg) + es('lbaj,pabo,ql->opqj', W + V, o['Pl'], rl)
          - 2 * es('lbaj,pabo,ql->opqj', W, o['Pgg'], rg) - 2 * es('lbaj,pabo,ql->opqj', V, o['Pll'], rl))
    br = fac * br
    if withG:
        br = br + fac * 1j * es('qbaj,apob->opqj', V - W, GW)
    return es('opqj,js->sopq', br, GR)


def rhsF_W(o, h, GR, F, GW, fac=2.0, full=True):
    r = driftF_W(h, F) + srcF_W(o, GR, GW, fac, withG=full)
    if full:
        r = r + fac * es('scdq,odcp->sopq', F, o['hW'].conj())
    return r


# ---- DSL
def Pab(T):  # exchange labels of slots 3,4
    return T.transpose(0, 1, 3, 2)


def Psl(T):
    return T.transpose(1, 0, 2, 3)


def B_dsl(o, G):
    hW = o['hW']
    return es('scdb,dlac->slab', hW, G) + es('scdb,ldca->slab', G, hW)


def rhs_dsl(o, h, G):
    xi = o['xi']
    B = B_dsl(o, G)
    return rhs_pp(o, h, G) + 2 * (B + xi * Pab(B))


def Y_dsl(o, G):
    xi, V, W, rl, rg = o['xi'], o['V'], o['W'], o['rl'], o['rg']
    core = es('qpma,sq,lmpb->slab', V, rl, G) - es('msqp,qa,plbm->slab', W, rg, G)
    t = core + xi * Psl(core)
    t = t + xi * Pab(t)
    return -2 * xi * t


def ZF(o, F):
    xi = o['xi']
    t = es('csdb,kdca->ksab', F, o['hW'].conj())
    return 2 * (t + xi * Pab(t))


def ZG(o, GR, G):
    xi, V, W = o['xi'], o['V'], o['W']
    t = 1j * es('bxyz,yakx,zs->ksab', V - W, G, GR)
    return 2 * xi * (t + xi * Pab(t))


def rhsF_dsl(o, h, GR, F, G):
    return rhsF_pp(o, h, GR, F, G) + ZF(o, F) + ZG(o, GR, G)


# ---------------------------------------------------------------- symmetry projectors
def sym_pp(X, xi):  # G_slab = xi G_lsab = xi G_slba, G_slab = -G*_absl
    X = (X + xi * Psl(X)) / 2
    X = (X + xi * Pab(X)) / 2
    X = (X - X.conj().transpose(2, 3, 0, 1)) / 2
    return X


def sym_W(X):  # G_sopj = G_osjp, G*_sopj = -G_jpos
    X = (X + X.transpose(1, 0, 3, 2)) / 2
    X = (X - X.conj().transpose(3, 2, 1, 0)) / 2
    return X


def sym_F2(X, xi):  # xi-exchange of slots 3,4
    return (X + xi * Pab(X)) / 2


# residuals of symmetry relations
def res_pp(T, xi):
    return dict(P12=rel(T, xi * Psl(T)), P34=rel(T, xi * Pab(T)),
                conj=rel(T, -T.conj().transpose(2, 3, 0, 1)))


def res_W(T):
    return dict(dbl=rel(T, T.transpose(1, 0, 3, 2)), conj=rel(T, -T.conj().transpose(3, 2, 1, 0)))


# ---------------------------------------------------------------- generic inputs
def generic(M, xi, nj=3):
    V = np.zeros((M,) * 4, complex)
    W = np.zeros((M,) * 4, complex)
    for _ in range(nj):
        a = crand(M, M); a = a + xi * a.T
        b = crand(M, M); b = b + xi * b.T
        V += 2 * es('ki,lj->iklj', a.conj(), a)
        W += 2 * es('ki,lj->iklj', b.conj(), b)
    A = crand(M, M); rl = (A + A.conj().T) / 2
    h = crand(M, M)
    GR = crand(M, M)
    return V, W, rl, h, GR


def out(s):
    print(s, flush=True)


def fmt(d):
    return ", ".join(f"{k}={v:.1e}" for k, v in d.items())


# ================================================================ PART A
out("=" * 78)
out("PART A: general (generic graded vee and wedge, 3 jump ops each, M=5)")
out("=" * 78)
M = 5
for xi in (-1, +1):
    out(f"\n------------------------------ xi = {xi:+d}")
    V, W, rl, h, GR = generic(M, xi)
    o = objects(rl, xi, V, W)
    I = o['I']
    # interaction tensors
    out("vee/wedge: xi-swap 1st pair %.1e, 2nd pair %.1e, conj vee*_iklj=vee_jlki %.1e" % (
        rel(V, xi * V.transpose(1, 0, 2, 3)), rel(V, xi * V.transpose(0, 1, 3, 2)),
        rel(V.conj(), V.transpose(3, 2, 1, 0))))
    # L, h
    Lg, Ll, hpp = o['Lg'], o['Ll'], o['hpp']
    out("Lfrak: double swap L_slki=L_lsik %.1e ; conj L*_slki=L_ikls %.1e ; single swap L_slki vs xi L_lski %.1e (expected O(1))" % (
        rel(Lg, Lg.transpose(1, 0, 3, 2)), rel(Lg.conj(), Lg.transpose(3, 2, 1, 0)),
        relF(Lg, xi * Lg.transpose(1, 0, 2, 3))))
    out("hpp: h_slef=xi h_lsef %.1e, =xi h_slfe %.1e, =h_lsfe %.1e" % (
        rel(hpp, xi * Psl(hpp)), rel(hpp, xi * Pab(hpp)), rel(hpp, hpp.transpose(1, 0, 3, 2))))
    # P frak
    for nm, (x, y) in dict(gl=(o['rg'], o['rl']), gg=(o['rg'], o['rg']), ll=(o['rl'], o['rl'])).items():
        Pxy = es('bd,ca->bcda', x, y); Pyx = es('bd,ca->bcda', y, x)
        out(f"Pfrak^{nm}: conj P*_bcda=P_dabc {rel(Pxy.conj(), Pxy.transpose(2,3,0,1)):.1e}; "
            f"double swap P^xy_bcda=P^yx_cbad {rel(Pxy, Pyx.transpose(1,0,3,2)):.1e}")
    out("Lfrak^x_slki = Pfrak^xx_slik: %.1e" % rel(Lg, o['Pgg'].transpose(0, 1, 3, 2)))
    hW = o['hW']
    out("hW: conj hW*_scdj = hW_jdcs %.1e ; double swap hW_scdj vs hW_csjd %.2f, vs hW_sdcj %.2f (no symmetry expected)" % (
        rel(hW.conj(), hW.transpose(3, 2, 1, 0)), relF(hW, hW.transpose(1, 0, 3, 2)), relF(hW, hW.transpose(0, 2, 1, 3))))
    # four-row form of hW from SM eq GWG1G2_hW second line
    hW2 = xi * (es('icdl,si,lj->scdj', W, o['rg'], o['rg'] - o['rl'])
                + es('icdl,si,lj->scdj', W, o['rg'] - o['rl'], o['rg'])
                - es('icdl,si,lj->scdj', V, o['rl'], o['rg'] - o['rl'])
                - es('icdl,si,lj->scdj', V, o['rg'] - o['rl'], o['rl']))
    out("hW delta form vs pre-delta form: %.1e" % rel(hW, hW2))

    # ---- TPPX G1-G2
    G = sym_pp(crand(M, M, M, M), xi)
    out("\n[TPPX G1-G2] input: " + fmt(res_pp(G, xi)))
    out("  RHS: " + fmt(res_pp(rhs_pp(o, h, G), xi)))
    # ---- TPPX RTDE
    F = sym_F2(crand(M, M, M, M), xi)
    R = rhsF_pp(o, h, GR, F, G)
    out("[TPPX RTDE] RHS xi-swap (a,b): %.1e ; (k,s): %.2f (none expected)" % (rel(R, xi * Pab(R)), relF(R, xi * Psl(R))))

    # ---- mGW G1-G2
    GW = sym_W(crand(M, M, M, M))
    R = rhs_W(o, h, GW)
    out("\n[mGW G1-G2] input: " + fmt(res_W(GW)))
    out("  RHS(G^W): " + fmt(res_W(R)))
    Rgw = rhs_W(o, h, GW, fac=1.0)
    out("  RHS(G^GW, unmodified GW): " + fmt(res_W(Rgw)))
    # single swaps are NOT preserved by the resummed equation
    Gpp_in = sym_pp(crand(M, M, M, M), xi)   # an input that has all pp symmetries
    R = rhs_W(o, h, Gpp_in)
    out("  RHS(G^W) on input with single-swap symmetry: P12 %.2f, P34 %.2f (O(1) => kernel breaks single swaps)" % (
        relF(R, xi * Psl(R)), relF(R, xi * Pab(R))))
    Rk = mem_W(o, Gpp_in)
    out("    kernel term alone: P12 %.2f, P34 %.2f" % (relF(Rk, xi * Psl(Rk)), relF(Rk, xi * Pab(Rk))))
    # counterterm
    Rct = rhs_W(o, h, Gpp_in, kern=False)
    out("  counterterm RHS on pp-symmetric input: " + fmt(res_pp(Rct, xi)) + " ; PHGsym: " + fmt(res_W(Rct)))
    Sw = src_W(o); Spp = src_pp(o)
    out("  source(mGW)_sopj = 2 source(TPPX)_slab with (s,l,a,b)=(s,o,p,j): %.1e" % rel(Sw, 2 * Spp))
    X = crand(M, M, M, M)
    out("  drift(mGW) == drift(TPPX) as operators on the same array: identical function (same formula)")
    # time propagation: G^{W,2}(t) vs 2 G^{pp,2B}(t), with time-dependent rho, h
    def prop(rhs_fun, T=0.5, nt=200):
        G = np.zeros((M,) * 4, complex); dt = T / nt
        for n in range(nt):
            def f(t, G):
                th = t
                rl_t = rl + 0.3 * np.sin(th) * (rl @ rl); rl_t = (rl_t + rl_t.conj().T) / 2
                o_t = objects(rl_t, xi, V, W)
                return rhs_fun(o_t, h * (1 + 0.2 * np.cos(th)), G)
            t = n * dt
            k1 = f(t, G); k2 = f(t + dt / 2, G + dt / 2 * k1); k3 = f(t + dt / 2, G + dt / 2 * k2); k4 = f(t + dt, G + dt * k3)
            G = G + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return G
    Gct = prop(lambda o_, h_, G_: rhs_W(o_, h_, G_, kern=False))
    G2B = prop(lambda o_, h_, G_: rhs_pp(o_, h_, G_, kern=False))
    out("  propagated (t=0.5, time-dependent rho,h): G^{W,2} vs 2 G^{pp}_[no kernel]: %.1e ; G^{W,2} pp-symmetries: %s" % (
        rel(Gct, 2 * G2B), fmt(res_pp(Gct, xi))))
    GWr = prop(lambda o_, h_, G_: rhs_W(o_, h_, G_), T=0.05, nt=100)
    out("  propagated G^W (resummed, t=0.05): PHGsym %s ; single swaps P12 %.2e, P34 %.2e (nonzero expected)" % (
        fmt(res_W(GWr)), relF(GWr, xi * Psl(GWr)), relF(GWr, xi * Pab(GWr))))

    # ---- mGW RTDE
    F = crand(M, M, M, M)
    R = rhsF_W(o, h, GR, sym_F2(F, xi), GW)
    out("\n[mGW RTDE] RHS(F^W) with xi-symmetric (p,q) input: xi-swap(p,q) %.2f (O(1): no symmetry)" % relF(R, xi * Pab(R)))
    Rk = 2 * es('scdq,odcp->sopq', sym_F2(F, xi), o['hW'].conj())
    Rg = srcF_W(o, GR, GW) - srcF_W(o, GR, GW, withG=False)
    out("  kernel term alone %.2f ; G^W term alone %.2f" % (relF(Rk, xi * Pab(Rk)), relF(Rg, xi * Pab(Rg))))
    Rct = rhsF_W(o, h, GR, sym_F2(F, xi), GW, full=False)
    out("  counterterm RHS on xi-symmetric input: xi-swap(p,q) %.1e" % rel(Rct, xi * Pab(Rct)))
    S2 = srcF_W(o, GR, GW, withG=False)
    Spp = srcF_pp(o, GR, None, withG=False)
    out("  source F^{W,2}_skab = 2 xi * source F^pp[no G]_ksab: %.1e" % rel(S2, 2 * xi * Spp.transpose(1, 0, 2, 3)))
    Xr = crand(M, M, M, M)
    out("  drift(mGW)[Y]_skab with Y_skab=X_ksab equals drift(TPPX)[X]_ksab: %.1e" % rel(
        driftF_W(h, Xr.transpose(1, 0, 2, 3)), driftF_pp(h, Xr).transpose(1, 0, 2, 3)))
    Rgw = rhsF_W(o, h, GR, sym_F2(F, xi), GW, fac=1.0)
    out("  unmodified GW F^GW RHS: xi-swap(p,q) %.2f (no symmetry)" % relF(Rgw, xi * Pab(Rgw)))

    # projection statement: the history integrals see only the xi-symmetric part in the pair contracted with the vertex
    XFB = 1j * (V - W)
    Gr = crand(M, M, M, M); Fr = crand(M, M, M, M)
    I1 = 2 * xi * es('pjko,sopj->sk', XFB, Gr); I2 = 2 * xi * es('pjko,sopj->sk', XFB, sym_F2(Gr, xi))
    J1 = 2 * es('iopq,sopq->is', XFB, Fr); J2 = 2 * es('iopq,sopq->is', XFB, sym_F2(Fr, xi))
    out("  history integrals see only the xi-symmetric projection: I (mGW G1-G2) %.1e, I^R (mGW RTDE) %.1e" % (rel(I1, I2), rel(J1, J2)))
    # ---- DSL G1-G2
    G = sym_pp(crand(M, M, M, M), xi)
    R = rhs_dsl(o, h, G)
    out("\n[DSL G1-G2] RHS: " + fmt(res_pp(R, xi)))
    B = B_dsl(o, G)
    out("  B: double swap B_slab=B_lsba %.1e ; conj B*_absl=-B_slab %.1e ; single swap P12 %.2f (none)" % (
        rel(B, B.transpose(1, 0, 3, 2)), rel(B.conj().transpose(2, 3, 0, 1), -B), relF(B, xi * Psl(B))))
    out("  B = (1/2) mGW memory term with (s,o,p,j)=(s,l,a,b): %.1e" % rel(B, 0.5 * mem_W(o, G)))
    T = 2 * (B + xi * Pab(B))
    out("  T=2(1+xiP_ab)B: " + fmt(res_pp(T, xi)))
    Y = Y_dsl(o, G)
    out("  Y: P_sl %.1e, P_ab %.1e ; Y - Y*_absl == T: %.1e ; Y alone pair-anti-Hermitian? %.2f (no)" % (
        rel(Y, xi * Psl(Y)), rel(Y, xi * Pab(Y)), rel(Y - Y.conj().transpose(2, 3, 0, 1), T),
        relF(Y, -Y.conj().transpose(2, 3, 0, 1))))
    # ---- DSL RTDE
    F = sym_F2(crand(M, M, M, M), xi)
    R = rhsF_dsl(o, h, GR, F, G)
    zf, zg = ZF(o, F), ZG(o, GR, G)
    out("[DSL RTDE] RHS xi-swap(a,b) %.1e ; ZF %.1e ; ZG %.1e ; (k,s) swap %.2f (none)" % (
        rel(R, xi * Pab(R)), rel(zf, xi * Pab(zf)), rel(zg, xi * Pab(zg)), relF(R, xi * Psl(R))))
    # ZF direct half = mGW kernel term with map Ftilde_ksab <-> xi F^W_skab
    FW = xi * F.transpose(1, 0, 2, 3)          # F^W_skab = xi F_ksab
    kerW = 2 * es('scdq,odcp->sopq', FW, o['hW'].conj())   # indices s,k,a,b
    direct = 2 * es('csdb,kdca->ksab', F, o['hW'].conj())
    out("  ZF direct attachment = xi * (mGW kernel on F^W=xi F^T)_skab: %.1e" % rel(direct, xi * kerW.transpose(1, 0, 2, 3)))


# ================================================================ PART B: model
out("\n" + "=" * 78)
out("PART B: model (on-site singlet loss, xi=-1, wedge=0), N=3 sites, M=6, index i=2r+sigma")
out("=" * 78)
xi = -1
N = 3; M = 2 * N
eps = np.array([[0, 1], [-1, 0]], float)   # eps_{up,dn}=1


def vee_model(sites):
    V = np.zeros((M,) * 4)
    for r in sites:
        for si in range(2):
            for sk in range(2):
                for sl in range(2):
                    for sj in range(2):
                        V[2*r+si, 2*r+sk, 2*r+sl, 2*r+sj] = 0.5 * eps[sk, si] * eps[sl, sj]  # gamma=1
    return V.astype(complex)


spin = np.array([i % 2 for i in range(M)])
site = np.array([i // 2 for i in range(M)])
flipP = np.zeros((M, M))
for i in range(M):
    flipP[2*site[i] + 1 - spin[i], i] = 1   # (r,s)->(r,sbar)


def spin_diag(Asp_up, Asp_dn):
    A = np.zeros((M, M), complex)
    A[0::2, 0::2] = Asp_up; A[1::2, 1::2] = Asp_dn
    return A


def blockmask(conserve_pairs=((0, 1), (2, 3))):
    s = spin
    m = np.zeros((M,) * 4, bool)
    a, b = conserve_pairs
    S = (s[:, None, None, None], s[None, :, None, None], s[None, None, :, None], s[None, None, None, :])
    return (S[a[0]] + S[a[1]]) == (S[b[0]] + S[b[1]])


SZ = blockmask()   # sigma1+sigma2 = sigma3+sigma4


def flip4(T):
    return es('ai,bj,ck,dl,ijkl->abcd', flipP, flipP, flipP, flipP, T)


def flip2(A):
    return flipP @ A @ flipP.T


def blk(T, s1, s2, s3, s4):
    return T[s1::2, s2::2, s3::2, s4::2]


def su2_check(T):
    """T must equal A d14 d23 + P d13 d24 in spin with A=T^{ud du}, P=T^{ud ud} (spatial indices in place)."""
    A = blk(T, 0, 1, 1, 0); P = blk(T, 0, 1, 0, 1)
    worst = 0.0
    sc = np.abs(T).max()
    for s1 in range(2):
        for s2 in range(2):
            for s3 in range(2):
                for s4 in range(2):
                    pred = A * (s1 == s4) * (s2 == s3) + P * (s1 == s3) * (s2 == s4)
                    worst = max(worst, np.abs(blk(T, s1, s2, s3, s4) - pred).max() / sc)
    return worst


def su2_tensor(A, P):
    T = np.zeros((M,) * 4, complex)
    for s1 in range(2):
        for s2 in range(2):
            for s3 in range(2):
                for s4 in range(2):
                    T[s1::2, s2::2, s3::2, s4::2] = A * (s1 == s4) * (s2 == s3) + P * (s1 == s3) * (s2 == s4)
    return T


for model, sites in (("lattice", range(N)), ("impurity (site 0)", [0])):
    out(f"\n------------------------------ {model}")
    V = vee_model(sites); Wz = np.zeros_like(V)
    out("vee real %.1e ; vee spin values (ud du, ud ud, du du, du ud) at r=0: %s" % (
        np.abs(V.imag).max(), [V[0, 1, 1, 0].real, V[0, 1, 0, 1].real, V[1, 0, 1, 0].real, V[1, 0, 0, 1].real]))
    out("vee nonzero only on S_z-conserving strings: %s ; flip-invariant: %.1e" % (
        bool(np.all(V[~SZ] == 0)), rel(flip4(V), V)))
    A1 = crand(N, N); A2 = crand(N, N)
    rl = spin_diag((A1 + A1.conj().T) / 2, (A2 + A2.conj().T) / 2)
    h = spin_diag(crand(N, N), crand(N, N))
    GR = spin_diag(crand(N, N), crand(N, N))
    o = objects(rl, xi, V, Wz)
    # MF rate
    lMF = 2 * es('iklj,lk->ij', V, rl)
    exp = np.zeros((M, M), complex)
    for r in sites:
        exp[2*r, 2*r] = rl[2*r+1, 2*r+1]; exp[2*r+1, 2*r+1] = rl[2*r, 2*r]
    out("ell^{>,MF}_ij = gamma delta_ij rho^{sbar}_rr (gamma=1): %.1e" % rel(lMF, exp))

    def rnd(symf):
        T = symf(crand(M, M, M, M)); T[~SZ] = 0; return T
    Gpp = rnd(lambda X: sym_pp(X, xi)); GW = rnd(sym_W); G2 = rnd(lambda X: sym_pp(X, xi))
    Fpp = rnd(lambda X: sym_F2(X, xi)); FW = rnd(lambda X: X); F2 = rnd(lambda X: sym_F2(X, xi))
    Fd = rnd(lambda X: sym_F2(X, xi)); Gd = rnd(lambda X: sym_pp(X, xi))
    cases = {
        'TPPX G':  (lambda o_, h_, GR_: rhs_pp(o_, h_, Gpp), Gpp, 'pp'),
        'TPPX F':  (lambda o_, h_, GR_: rhsF_pp(o_, h_, GR_, Fpp, Gpp), Fpp, 'F2'),
        'mGW G^W': (lambda o_, h_, GR_: rhs_W(o_, h_, GW), GW, 'W'),
        'mGW G^W2': (lambda o_, h_, GR_: rhs_W(o_, h_, G2, kern=False), G2, 'pp'),
        'mGW F^W': (lambda o_, h_, GR_: rhsF_W(o_, h_, GR_, FW, GW), FW, None),
        'mGW F^W2': (lambda o_, h_, GR_: rhsF_W(o_, h_, GR_, F2, GW, full=False), F2, 'F2'),
        'DSL G':   (lambda o_, h_, GR_: rhs_dsl(o_, h_, Gd), Gd, 'pp'),
        'DSL F':   (lambda o_, h_, GR_: rhsF_dsl(o_, h_, GR_, Fd, Gd), Fd, 'F2'),
    }
    out("S_z closure (RHS outside the 6 blocks sigma1+sigma2=sigma3+sigma4, inputs inside):")
    for k, (f, T, sy) in cases.items():
        R = f(o, h, GR)
        out(f"  {k:9s}: max|RHS| outside = {np.abs(R[~SZ]).max():.1e}  (|RHS| ~ {np.abs(R).max():.1e})")
    # spin-flip covariance: RHS[flipped inputs] = flip(RHS[inputs])
    o_f = objects(flip2(rl), xi, V, Wz)
    out("Global spin-flip covariance  RHS[flip(all inputs)] = flip(RHS[inputs]):")
    flipped = dict(Gpp=flip4(Gpp), GW=flip4(GW), G2=flip4(G2), Fpp=flip4(Fpp), FW=flip4(FW), F2=flip4(F2), Fd=flip4(Fd), Gd=flip4(Gd))
    hf, GRf = flip2(h), flip2(GR)
    pairs = {
        'TPPX G': (rhs_pp(o, h, Gpp), rhs_pp(o_f, hf, flipped['Gpp'])),
        'TPPX F': (rhsF_pp(o, h, GR, Fpp, Gpp), rhsF_pp(o_f, hf, GRf, flipped['Fpp'], flipped['Gpp'])),
        'mGW G^W': (rhs_W(o, h, GW), rhs_W(o_f, hf, flipped['GW'])),
        'mGW G^W2': (rhs_W(o, h, G2, kern=False), rhs_W(o_f, hf, flipped['G2'], kern=False)),
        'mGW F^W': (rhsF_W(o, h, GR, FW, GW), rhsF_W(o_f, hf, GRf, flipped['FW'], flipped['GW'])),
        'mGW F^W2': (rhsF_W(o, h, GR, F2, GW, full=False), rhsF_W(o_f, hf, GRf, flipped['F2'], flipped['GW'], full=False)),
        'DSL G': (rhs_dsl(o, h, Gd), rhs_dsl(o_f, hf, flipped['Gd'])),
        'DSL F': (rhsF_dsl(o, h, GR, Fd, Gd), rhsF_dsl(o_f, hf, GRf, flipped['Fd'], flipped['Gd'])),
    }
    for k, (R, Rf) in pairs.items():
        out(f"  {k:9s}: {rel(flip4(R), Rf):.1e}")
    out("  history integrals: I[flip]=flip(I): mGW %.1e" % rel(
        flip2(2 * xi * 1j * es('pjko,sopj->sk', V, GW)), 2 * xi * 1j * es('pjko,sopj->sk', V, flipped['GW'])))

    # site permutation covariance (lattice: cyclic permutation of all sites; impurity: swap of bath sites 1,2)
    perm = [1, 2, 0] if len(list(sites)) == N else [0, 2, 1]
    Pi = np.zeros((M, M))
    for i in range(M):
        Pi[2 * perm[site[i]] + spin[i], i] = 1
    p4 = lambda T: es('ai,bj,ck,dl,ijkl->abcd', Pi, Pi, Pi, Pi, T)
    p2 = lambda A: Pi @ A @ Pi.T
    out("vee invariant under the site permutation %s: %.1e" % (perm, rel(p4(V), V)))
    o_p = objects(p2(rl), xi, V, Wz)
    hp_, GRp_ = p2(h), p2(GR)
    chk = {
        'mGW G^W': (rhs_W(o, h, GW), rhs_W(o_p, hp_, p4(GW))),
        'mGW F^W': (rhsF_W(o, h, GR, FW, GW), rhsF_W(o_p, hp_, GRp_, p4(FW), p4(GW))),
        'DSL G': (rhs_dsl(o, h, Gd), rhs_dsl(o_p, hp_, p4(Gd))),
        'DSL F': (rhsF_dsl(o, h, GR, Fd, Gd), rhsF_dsl(o_p, hp_, GRp_, p4(Fd), p4(Gd))),
    }
    out("Site-permutation covariance RHS[perm inputs] = perm(RHS): " + ", ".join(f"{k} {rel(p4(a), b):.1e}" for k, (a, b) in chk.items()))
    # SU(2), paramagnetic
    Bs = crand(N, N); rlp = spin_diag((Bs + Bs.conj().T) / 2, (Bs + Bs.conj().T) / 2)
    hs = crand(N, N); hp = spin_diag(hs, hs)
    gs = crand(N, N); GRp = spin_diag(gs, gs)
    op = objects(rlp, xi, V, Wz)

    def su2rnd(symf):
        for _ in range(3):
            T = su2_tensor(crand(N, N, N, N), crand(N, N, N, N))
            T = symf(T)
        return T
    Gpp = su2rnd(lambda X: sym_pp(X, xi)); GW = su2rnd(sym_W); G2 = su2rnd(lambda X: sym_pp(X, xi))
    Fpp = su2rnd(lambda X: sym_F2(X, xi)); FW = su2rnd(lambda X: X); F2 = su2rnd(lambda X: sym_F2(X, xi))
    out("SU(2) (paramagnetic inputs; inputs of the form A d14 d23 + P d13 d24, input residual %.1e):" % max(
        su2_check(Gpp), su2_check(GW), su2_check(FW), su2_check(F2)))
    Rs = {
        'TPPX G': rhs_pp(op, hp, Gpp), 'TPPX F': rhsF_pp(op, hp, GRp, Fpp, Gpp),
        'mGW G^W': rhs_W(op, hp, GW), 'mGW G^W2': rhs_W(op, hp, G2, kern=False),
        'mGW F^W': rhsF_W(op, hp, GRp, FW, GW), 'mGW F^W2': rhsF_W(op, hp, GRp, F2, GW, full=False),
        'DSL G': rhs_dsl(op, hp, Gpp), 'DSL F': rhsF_dsl(op, hp, GRp, Fpp, Gpp),
    }
    for k, R in Rs.items():
        out(f"  {k:9s}: RHS deviation from SU(2) form = {su2_check(R):.1e}")
    # SU(2) + xi exchange for pp-type: same-spin block = A+P, and P12 relates A and P
    T = Rs['DSL G']
    A = blk(T, 0, 1, 1, 0); P = blk(T, 0, 1, 0, 1)
    out("  pp-type: A_slab = xi P_slba (P34 exchange within SU(2) form): %.1e" % rel(A, xi * P.transpose(0, 1, 3, 2)))
    T = Rs['mGW G^W']
    A = blk(T, 0, 1, 1, 0); P = blk(T, 0, 1, 0, 1)
    out("  mGW G^W: A_sopj = A_osjp %.1e, P_sopj = P_osjp %.1e (double swap within SU(2) form)" % (
        rel(A, A.transpose(1, 0, 3, 2)), rel(P, P.transpose(1, 0, 3, 2))))
out("\ndone")
