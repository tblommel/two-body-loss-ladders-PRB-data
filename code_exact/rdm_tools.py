"""2RDM tools for the DSL investigation (fermions, xi = -1).

Conventions (driver README "Conventions"): rho_sa = <c^dag_a c_s>, Gamma_slab = <c^dag_a c^dag_b c_l c_s>
= rho_sa rho_lb - rho_la rho_sb + C_slab, C = 2 i xi G = -2 i G (G = the stored pp correlator, full spin-orbital).
Spin-orbital index i = sigma * N + orbital (sigma 0 = up, 1 = down).

Positivity (Gram matrices, valid for any mixed state, also without fixed particle number):
  P_{(ab),(sl)}  = rho_sa rho_lb - rho_la rho_sb + C_slab          (pair density   <O^dag O>, O = c_l c_s)
  Q_{(sl),(ab)}  = rhob_sa rhob_lb - rhob_la rhob_sb + C_slab      (two holes, rhob = 1 - rho)
  G'_{(sa),(lb)} = rho_ba rhob_sl + C_sbal                          (particle-hole covariance)
  Var N = sum_sl C_slsl + tr rho rhob >= 0
"""
import numpy as np


def full_G(P, U=None, D=None, N=None):
    """spin-orbital G (2N)^4 from the driver blocks: P = G^{ud ud}_{slab} [N^4], U/D = same-spin [N^4] (unpacked);
    U = D = None: paramagnetic, G^{uuuu} = G^{dddd} = P - P_slba"""
    if N is None:
        N = int(round(P.size ** 0.25))
    P = P.reshape(N, N, N, N)
    if U is None:
        U = P - P.transpose(0, 1, 3, 2)
        D = U
    U = U.reshape(N, N, N, N); D = D.reshape(N, N, N, N)
    M = 2 * N
    G = np.zeros((M, M, M, M), complex)
    u, d = slice(0, N), slice(N, 2 * N)
    G[u, d, u, d] = P                                  # s up, l dn, a up, b dn
    G[d, u, d, u] = P.transpose(1, 0, 3, 2)            # double swap: G^{dudu}_{slab} = P_lsba
    G[u, d, d, u] = -P.transpose(0, 1, 3, 2)           # G_slab = xi G_slba
    G[d, u, u, d] = -P.transpose(1, 0, 2, 3)           # G_slab = xi G_lsab
    G[u, u, u, u] = U
    G[d, d, d, d] = D
    return G


def rho_full(ru, rd):
    N = ru.shape[0]
    r = np.zeros((2 * N, 2 * N), complex)
    r[:N, :N] = ru; r[N:, N:] = rd
    return r


def matrices(rho, G):
    """P, Q, G' (Hermitian matrices) and Var N for spin-orbital rho [M,M] and G [M,M,M,M]"""
    M = rho.shape[0]
    C = -2j * G
    rb = np.eye(M) - rho
    # P_{(ab),(sl)} = rho_sa rho_lb - rho_la rho_sb + C_slab   -> array [a,b,s,l]
    Pm = (np.einsum("sa,lb->absl", rho, rho) - np.einsum("la,sb->absl", rho, rho) + C.transpose(2, 3, 0, 1)).reshape(M * M, M * M)
    Qm = (np.einsum("sa,lb->slab", rb, rb) - np.einsum("la,sb->slab", rb, rb) + C).reshape(M * M, M * M)
    # G'_{(sa),(lb)} = rho_ba rhob_sl + C_sbal  -> array [s,a,l,b]
    Gm = (np.einsum("ba,sl->salb", rho, rb) + C.transpose(0, 2, 3, 1)).reshape(M * M, M * M)
    varN = float(np.real(np.einsum("slsl->", C) + np.trace(rho @ rb)))
    return Pm, Qm, Gm, varN


def herm_err(A):
    return float(np.abs(A - A.conj().T).max() / max(1e-300, np.abs(A).max()))


def min_eigs(rho, G):
    Pm, Qm, Gm, varN = matrices(rho, G)
    out = {}
    for k, A in (("P", Pm), ("Q", Qm), ("G", Gm)):
        out[k] = float(np.linalg.eigvalsh(0.5 * (A + A.conj().T))[0])
        out[k + "_herm"] = herm_err(A)
    out["varN"] = varN
    out["rho"] = float(np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))[0])
    out["rho_max"] = float(np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))[-1])
    return out


def exact_rdms(state, c):
    """exact 1RDM rho_sa = <c^dag_a c_s> and cumulant C from a many-body density matrix (dense) and annihilators c[i]"""
    M = len(c)
    cd = [x.T.conj().tocsr() for x in c]
    rho = np.zeros((M, M), complex)
    for s in range(M):
        for a in range(M):
            rho[s, a] = (cd[a] @ c[s] @ state).trace()
    Gam = np.zeros((M, M, M, M), complex)
    for s in range(M):
        for l in range(M):
            cl_cs = c[l] @ c[s]
            for a in range(M):
                for b in range(M):
                    Gam[s, l, a, b] = (cd[a] @ cd[b] @ cl_cs @ state).trace()
    C = Gam - np.einsum("sa,lb->slab", rho, rho) + np.einsum("la,sb->slab", rho, rho)
    return rho, C


def unpack_anti(Xp, N):
    A = N * (N - 1) // 2
    Xp = Xp.reshape(A, A)
    idx = {}
    k = 0
    for s in range(N):
        for l in range(s + 1, N):
            idx[(s, l)] = k; k += 1
    X = np.zeros((N, N, N, N), complex)
    pairs = list(idx.items())
    for (s, l), p in pairs:
        for (a, b), q in pairs:
            v = Xp[p, q]
            X[s, l, a, b] = v; X[l, s, a, b] = -v; X[s, l, b, a] = -v; X[l, s, b, a] = v
    return X


def state_from_checkpoint(cp, N):
    """(rho_up, rho_dn, full G) from a driver checkpoint group (G1-G2 blocks rho, G_opp[, G_UU, G_DD]) of a DSL run"""
    names = cp.attrs["block_names"].split(","); sz = cp.attrs["block_sizes"].astype(int)
    off = np.concatenate([[0], np.cumsum(sz)])
    y = cp["y"][()]
    b = {n: y[off[k]:off[k + 1]] for k, n in enumerate(names)}
    r = b["rho"]
    ru = r[:N * N].reshape(N, N)
    rd = r[N * N:].reshape(N, N) if r.size == 2 * N * N else ru
    if "G_UU" in b:
        G = full_G(b["G_opp"], unpack_anti(b["G_UU"], N), unpack_anti(b["G_DD"], N), N)
    else:
        G = full_G(b["G_opp"], None, None, N)
    return ru, rd, G
