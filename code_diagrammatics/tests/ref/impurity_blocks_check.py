"""Brute-force check of the new-order TPPX RTDE spin reduction for the dissipative impurity
(Implementation_verification.tex, sec:tppxrtde).  xi=-1, wedge=0, vee of Eq. (veeimp).
New tensor F^{pp}_{abks} (ladder pair a,b; closing leg k; probe s last).
Blocks: U_{abks} = F^{up dn up dn}_{abks}, D_{abks} = F^{up dn dn up}_{abks}  (old U_{ksab}, D_{ksab}).
"""
import numpy as np
xi = -1
UP, DN = 0, 1
eps = np.array([[0, 1], [-1, 0]])          # eps_{up dn} = 1
OUT = []
def say(s): print(s); OUT.append(s)

def model(N, gam, rng, para=False):
    M = 2*N
    idx = lambda r, s: 2*r + s
    vee = np.zeros((M,)*4)
    for si in (0, 1):
        for sk in (0, 1):
            for sl in (0, 1):
                for sj in (0, 1):
                    vee[idx(0, si), idx(0, sk), idx(0, sl), idx(0, sj)] = gam/2*eps[sk, si]*eps[sl, sj]
    def rh(): A = rng.normal(size=(N, N))+1j*rng.normal(size=(N, N)); return (A+A.conj().T)/2
    def cm(): return rng.normal(size=(N, N))+1j*rng.normal(size=(N, N))
    rl = {UP: rh()}; rl[DN] = rl[UP].copy() if para else rh()
    h = {UP: cm()}; h[DN] = h[UP].copy() if para else cm()
    GR = {UP: cm()}; GR[DN] = GR[UP].copy() if para else cm()
    Gred = rng.normal(size=(N,)*4)+1j*rng.normal(size=(N,)*4)     # G^{pp, up dn up dn}_{slab}
    if para: Gred = (Gred + Gred.transpose(1, 0, 3, 2))/2
    return dict(N=N, M=M, idx=idx, vee=vee, gam=gam, rl=rl, h=h, GR=GR, Gred=Gred)

def spinfull(m):
    """full spin-orbital rho^<, rho^>, h_o, G^R and G^pp (via the G1-G2 block relations, eq. blocks)"""
    N, M, idx = m['N'], m['M'], m['idx']
    def emb(d):
        A = np.zeros((M, M), complex)
        for s in (0, 1): A[s::2, s::2] = d[s]
        return A
    rl = emb(m['rl']); rg = rl + xi*np.eye(M); h = emb(m['h']); GR = emb(m['GR'])
    G = np.zeros((M,)*4, complex); g = m['Gred']
    G[0::2, 1::2, 0::2, 1::2] = g
    G[0::2, 1::2, 1::2, 0::2] = xi*g.transpose(0, 1, 3, 2)
    G[1::2, 0::2, 0::2, 1::2] = xi*g.transpose(1, 0, 2, 3)
    G[1::2, 0::2, 1::2, 0::2] = g.transpose(1, 0, 3, 2)
    return rl, rg, h, GR, G

def Ffull(U, D, M):
    F = np.zeros((M,)*4, complex)
    F[0::2, 1::2, 0::2, 1::2] = U                        # up dn up dn
    F[0::2, 1::2, 1::2, 0::2] = D                        # up dn dn up
    F[1::2, 0::2, 0::2, 1::2] = xi*U.transpose(1, 0, 2, 3)   # P_ab
    F[1::2, 0::2, 1::2, 0::2] = xi*D.transpose(1, 0, 2, 3)
    return F

def full_rhs(m, F, extra=None):
    """new-order general EOM, wedge = 0 (SM TPP_RTDE_FINALEOM / main TPPRTDEEOM in doc letters)"""
    rl, rg, h, GR, G = extra if extra is not None else spinfull(m)
    v = m['vee']; hd = h.conj().T
    Lg = np.einsum('ac,bd->abdc', rg, rg); Ll = np.einsum('ac,bd->abdc', rl, rl); LL = Lg - Ll
    hfr = -np.einsum('abmn,mncd->abcd', LL, v)
    r = (-1j*np.einsum('ax,xbks->abks', h, F) - 1j*np.einsum('bx,axks->abks', h, F)
         + 1j*np.einsum('abxs,xk->abks', F, hd) + np.einsum('abcd,cdks->abks', hfr, F))
    S = (np.einsum('abdc,dckj->abkj', Ll, v) + xi*np.einsum('abdc,dclj,lk->abkj', LL, v, rl)
         - 1j*np.einsum('abef,efkj->abkj', G, -v))
    r = r + np.einsum('abkj,js->abks', S, GR)
    IR = 2j*np.einsum('kiab,abks->is', v, F)
    return r, IR

def old_full_rhs(m, Fold, extra):
    """black (pre-rewrite) SM TPP_RTDE_FINALEOM for F_{ksab}, wedge = 0"""
    rl, rg, h, GR, G = extra
    v = m['vee']; hd = h.conj().T
    Lg = np.einsum('ac,bd->abdc', rg, rg); Ll = np.einsum('ac,bd->abdc', rl, rl); LL = Lg - Ll
    hfr = -np.einsum('abop,opmn->abmn', LL, v)
    r = (1j*np.einsum('xsab,xk->ksab', Fold, hd) - 1j*np.einsum('ax,ksxb->ksab', h, Fold)
         - 1j*np.einsum('bx,ksax->ksab', h, Fold) + np.einsum('ksmn,abmn->ksab', Fold, hfr))
    S = (xi*np.einsum('lk,cdjl,abdc->kjab', rl, v, LL) + np.einsum('cdjk,abdc->kjab', v, Ll)
         - 1j*np.einsum('efkj,abef->kjab', -v, G))
    return r + np.einsum('kjab,js->ksab', S, GR)

def reduced_rhs(m, U, D, GRcol=None):
    """NEW reduced block equations (boxed eq. UDfinal, new form). U, D: (N,N,N,ns) = [a,b,k,s];
    GRcol[sigma]: (N, ns) columns of G^{R sigma}. Returns dU, dD, dGRup, dGRdn (G^R: only the I^R part + drift)."""
    gam = m['gam']; rl, h = m['rl'], m['h']; N = m['N']
    rg = {s: rl[s] + xi*np.eye(N) for s in (0, 1)}
    if GRcol is None: GRcol = {s: m['GR'][s] for s in (0, 1)}
    K = 2j*m['Gred']                                     # K^{pp}_{slab}
    Ql = np.einsum('a,b->ab', rl[UP][:, 0], rl[DN][:, 0])  # Q^<_{ab0}
    Qg = np.einsum('a,b->ab', rg[UP][:, 0], rg[DN][:, 0])
    V = Qg - Ql
    K00 = K[:, :, 0, 0]
    d0 = np.zeros(N); d0[0] = 1
    def blk(X, sk, sign, GRs):
        dr = (-1j*np.einsum('ax,xbks->abks', h[UP], X) - 1j*np.einsum('bx,axks->abks', h[DN], X)
              + 1j*np.einsum('abxs,xk->abks', X, h[sk].conj().T))
        mem = -gam*np.einsum('ab,ks->abks', V, X[0, 0])
        src = sign*gam/2*(np.einsum('ab,k->abk', Ql - K00, d0) - np.einsum('ab,k->abk', V, rl[sk][0, :]))
        return dr + mem + np.einsum('abk,s->abks', src, GRs[0, :])
    dU = blk(U, UP, +1, GRcol[DN])
    dD = blk(D, DN, -1, GRcol[UP])
    dGup = -1j*h[UP] @ GRcol[UP]; dGup[0] += 2*gam*D[0, 0, 0]
    dGdn = -1j*h[DN] @ GRcol[DN]; dGdn[0] += -2*gam*U[0, 0, 0]
    IRup = 2j*gam*D[0, 0, 0]; IRdn = -2j*gam*U[0, 0, 0]
    return dU, dD, dGup, dGdn, IRup, IRdn

def old_reduced_rhs(m, Uo, Do):
    """the PRE-rewrite boxed eq. UDfinal (old order U_{ksab}), three-rho source"""
    gam = m['gam']; rl, h = m['rl'], m['h']; N = m['N']
    rg = {s: rl[s] + xi*np.eye(N) for s in (0, 1)}
    K = 2j*m['Gred']
    def blk(X, sk, sign, GRs):
        r = (1j*np.einsum('xsab,xk->ksab', X, h[sk].conj().T) - 1j*np.einsum('ax,ksxb->ksab', h[UP], X)
             - 1j*np.einsum('bx,ksax->ksab', h[DN], X)
             + gam*np.einsum('ks,ab->ksab', X[:, :, 0, 0], np.outer(rl[UP][:, 0], rl[DN][:, 0]) - np.outer(rg[UP][:, 0], rg[DN][:, 0])))
        d0 = np.zeros(N); d0[0] = 1
        Qg = np.outer(rg[UP][:, 0], rg[DN][:, 0]); Ql = np.outer(rl[UP][:, 0], rl[DN][:, 0])
        br = (np.einsum('ab,k->kab', K[:, :, 0, 0], d0) + np.einsum('ab,k->kab', Qg, rl[sk][0, :])
              + np.einsum('ab,k->kab', Ql, rg[sk][0, :]) - 2*np.einsum('ab,k->kab', Ql, rl[sk][0, :]))
        return r - sign*gam/2*np.einsum('kab,s->ksab', br, GRs[0, :])
    return blk(Uo, UP, +1, m['GR'][DN]), blk(Do, DN, -1, m['GR'][UP])

def rel(a, b): return np.abs(a-b).max()/max(np.abs(b).max(), 1e-300)

def static_checks(N, rng):
    m = model(N, 0.8, rng); M = m['M']
    ext = spinfull(m)
    U = rng.normal(size=(N,)*4)+1j*rng.normal(size=(N,)*4); D = rng.normal(size=(N,)*4)+1j*rng.normal(size=(N,)*4)
    F = Ffull(U, D, M)
    R, IR = full_rhs(m, F, ext)
    # closure: outside sector
    mask = np.zeros((M,)*4, bool)
    for a, b, k, s in [(0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0)]:
        mask[a::2, b::2, k::2, s::2] = True
    outside = np.abs(R[~mask]).max()
    IRoff = max(np.abs(IR[0::2, 1::2]).max(), np.abs(IR[1::2, 0::2]).max())
    Pab = rel(R[1::2, 0::2, 0::2, 1::2], xi*R[0::2, 1::2, 0::2, 1::2].transpose(1, 0, 2, 3))
    dU, dD, _, _, IRup, IRdn = reduced_rhs(m, U, D)
    eU = rel(dU, R[0::2, 1::2, 0::2, 1::2]); eD = rel(dD, R[0::2, 1::2, 1::2, 0::2])
    eIu = rel(np.outer(np.eye(N)[0], IRup), IR[0::2, 0::2]); eId = rel(np.outer(np.eye(N)[0], IRdn), IR[1::2, 1::2])
    # column reduction: s = 0 only
    dU0, dD0, _, _, _, _ = reduced_rhs(m, U[..., :1], D[..., :1], {s: m['GR'][s][:, :1] for s in (0, 1)})
    eC = max(rel(dU0[..., 0], R[0::2, 1::2, 0::2, 1::2][..., 0]), rel(dD0[..., 0], R[0::2, 1::2, 1::2, 0::2][..., 0]))
    # old block equations through the map U_new_abks = U_old_ksab
    oU, oD = old_reduced_rhs(m, U.transpose(2, 3, 0, 1), D.transpose(2, 3, 0, 1))
    eO = max(rel(dU, oU.transpose(2, 3, 0, 1)), rel(dD, oD.transpose(2, 3, 0, 1)))
    # old full (black SM) EOM through F_new_abks = F_old_ksab
    Ro = old_full_rhs(m, F.transpose(2, 3, 0, 1), ext)
    eOF = rel(R, Ro.transpose(2, 3, 0, 1))
    say(f"N={N}: outside-sector |RHS|={outside:.1e}, I^R off-diag={IRoff:.1e}, P_ab of RHS {Pab:.1e}; "
        f"U {eU:.1e}, D {eD:.1e}, I^R up/dn {eIu:.1e}/{eId:.1e}; column s=0 {eC:.1e}; "
        f"vs old blocks {eO:.1e}; full new vs old SM EOM {eOF:.1e}; |RHS|max {np.abs(R).max():.1e}")

def para_check(N, rng):
    m = model(N, 0.8, rng, para=True); M = m['M']; ext = spinfull(m)
    U = rng.normal(size=(N,)*4)+1j*rng.normal(size=(N,)*4); D = xi*U.transpose(1, 0, 2, 3)
    R, IR = full_rhs(m, Ffull(U, D, M), ext)
    dU = R[0::2, 1::2, 0::2, 1::2]; dD = R[1::2, 0::2, 0::2, 1::2][...]  # not used
    dD = R[0::2, 1::2, 1::2, 0::2]
    say(f"paramagnetic N={N}: |dD_abks - xi dU_baks|/|dD| = {rel(dD, xi*dU.transpose(1, 0, 2, 3)):.1e}, "
        f"|I^R up - I^R dn| = {np.abs(IR[0::2, 0::2]-IR[1::2, 1::2]).max():.1e}")

def propagation(N=3, T=1.0, nst=400, seed=5):
    """coupled RK4: reduced (s=0 column) U, D, G^R columns vs full spin-orbital F^{pp}_{abks}, G^R;
    smooth time-dependent prescribed rho(t), h_o(t), K(t)."""
    rng = np.random.default_rng(seed)
    m0 = model(N, 0.8, rng); M = m0['M']
    A = {s: 0.3*(rng.normal(size=(N, N))+1j*rng.normal(size=(N, N))) for s in (0, 1)}
    A = {s: (A[s]+A[s].conj().T)/2 for s in (0, 1)}
    G1 = rng.normal(size=(N,)*4)+1j*rng.normal(size=(N,)*4)
    def m_at(t):
        m = dict(m0)
        m['rl'] = {s: m0['rl'][s]*0.2 + A[s]*np.sin(1.1*t+s) for s in (0, 1)}
        m['h'] = {s: m0['h'][s]*(1+0.2*np.cos(0.9*t)) for s in (0, 1)}
        m['Gred'] = m0['Gred']*np.cos(0.5*t) + G1*0.3*np.sin(t)
        return m
    def f_full(t, y):
        m = m_at(t); F, GR = y
        rl, rg, h, _, G = spinfull(m)
        R, IR = full_rhs(m, F, (rl, rg, h, GR, G))
        return (R, -1j*h @ GR - 1j*IR)
    def f_red(t, y):
        m = m_at(t); U, D, gu, gd = y
        dU, dD, dgu, dgd, _, _ = reduced_rhs(m, U, D, {UP: gu, DN: gd})
        return (dU, dD, dgu, dgd)
    def rk4(f, y, t, dt):
        add = lambda y, k, c: tuple(a + c*b for a, b in zip(y, k))
        k1 = f(t, y); k2 = f(t+dt/2, add(y, k1, dt/2)); k3 = f(t+dt/2, add(y, k2, dt/2)); k4 = f(t+dt, add(y, k3, dt))
        return tuple(a + dt/6*(b+2*c+2*d+e) for a, b, c, d, e in zip(y, k1, k2, k3, k4))
    yF = (np.zeros((M,)*4, complex), -1j*np.eye(M, dtype=complex))
    e0 = np.zeros((N, 1), complex); e0[0, 0] = -1j
    yR = (np.zeros((N, N, N, 1), complex), np.zeros((N, N, N, 1), complex), e0.copy(), e0.copy())
    dt = T/nst; t = 0.0
    for _ in range(nst):
        yF = rk4(f_full, yF, t, dt); yR = rk4(f_red, yR, t, dt); t += dt
    F, GR = yF
    U_f = F[0::2, 1::2, 0::2, 1::2][..., 0:1]; D_f = F[0::2, 1::2, 1::2, 0::2][..., 0:1]
    # spin-orbital column s=(0,up): index 0 ; s=(0,dn): index 1
    eU = rel(yR[0], U_f); eD = rel(yR[1], D_f)
    eGu = rel(yR[2], GR[0::2, 0:1]); eGd = rel(yR[3], GR[1::2, 1:2])
    outside = np.abs(F[0::2, 0::2]).max()
    say(f"RK4 propagation N={N}, t'=0 -> t={T}, {nst} steps: U {eU:.1e}, D {eD:.1e}, G^R up col {eGu:.1e}, G^R dn col {eGd:.1e}; "
        f"|U|max {np.abs(yR[0]).max():.2e}, |D|max {np.abs(yR[1]).max():.2e}; same-spin-pair blocks of full F max {outside:.1e}")

if __name__ == '__main__':
    rng = np.random.default_rng(2026)
    for N in (1, 2, 3, 4):
        static_checks(N, rng)
    for N in (2, 3):
        para_check(N, rng)
    propagation(N=3)
    propagation(N=2, T=2.0, nst=800, seed=9)
    open(__file__.replace('.py', '_output.txt'), 'w').write("\n".join(OUT) + "\n")
