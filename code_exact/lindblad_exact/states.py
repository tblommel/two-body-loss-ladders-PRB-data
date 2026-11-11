"""Gaussian (uncorrelated) product initial states.

rho^MB = rho_up^MB (x) rho_dn^MB, rho_sigma^MB = prod_k [w_k b^dag_k b_k + (1 - w_k)(1 - b^dag_k b_k)] with the natural
orbitals rho1_sigma = V diag(w) V^dag, b_k = sum_x conj(V[x, k]) c_x. Then <c^dag_a c_s> = rho1_sigma[s, a] and all
higher cumulants vanish. Each spin factor is block diagonal in N_sigma, so the state is block diagonal in
(N_up, N_dn); its blocks are kron(rho_up[N_up block], rho_dn[N_dn block]) in the basis order of fock.py.
"""
import numpy as np
from .fock import fermion_ops


class CardError(Exception):
    pass


def check_rho1(r, name, N):
    r = np.asarray(r, dtype=float)
    if r.shape != (N, N):
        raise CardError(f"{name}: shape {r.shape}, expected ({N}, {N})")
    if np.abs(r - r.T).max() > 1e-12:
        raise CardError(f"{name}: not symmetric")
    w = np.linalg.eigvalsh(r)
    if w.min() < -1e-12 or w.max() > 1 + 1e-12:
        raise CardError(f"{name}: spectrum outside [0, 1] ({w.min()}, {w.max()})")
    return r


def spin_state(r1):
    """dense 2^N x 2^N Gaussian state of one spin species (local Jordan-Wigner, mode 0 most significant)"""
    N = r1.shape[0]
    c = [x.toarray() for x in fermion_ops(N)]
    w, V = np.linalg.eigh(0.5 * (r1 + r1.T))
    w = np.clip(w, 0.0, 1.0)
    d = 2 ** N
    st = np.identity(d, dtype=complex)
    for k in range(N):
        b = sum(np.conj(V[x, k]) * c[x] for x in range(N))
        B = b.conj().T @ b
        st = st @ (w[k] * B + (1 - w[k]) * (np.identity(d) - B))
    return st


def gaussian_blocks(fm, r_up, r_dn, tol=1e-12):
    """{(a, a): block} of the Gaussian state for the sectors a = (N_up, N_dn) with nonzero weight; asserts block
    diagonality in N_sigma of each spin factor and unit trace."""
    N = fm.N
    pc = np.array([bin(i).count("1") for i in range(2 ** N)])
    loc = {n: np.flatnonzero(pc == n) for n in range(N + 1)}
    S = [spin_state(r_up), spin_state(r_dn)]
    for st in S:
        offn = 0.0
        for n in range(N + 1):
            for m in range(N + 1):
                if n != m:
                    offn = max(offn, np.abs(st[np.ix_(loc[n], loc[m])]).max() if len(loc[n]) and len(loc[m]) else 0.0)
        assert offn < tol, f"Gaussian state not block diagonal in N_sigma ({offn})"
    out = {}
    tr = 0.0
    for a in fm.sectors:
        Bu = S[0][np.ix_(loc[a[0]], loc[a[0]])]; Bd = S[1][np.ix_(loc[a[1]], loc[a[1]])]
        blk = np.kron(Bu, Bd)
        if np.abs(blk).max() > tol:
            out[a] = blk; tr += np.trace(blk).real
    assert abs(tr - 1.0) < 1e-10, f"Gaussian state trace {tr}"
    return out
