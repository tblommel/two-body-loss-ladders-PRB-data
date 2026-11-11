"""Two-body positivity (min eigenvalues of P, Q, G' and Var N) from driver periodic checkpoints.

The 2RDM of a method is Gamma = rho rho - rho rho + C with C = -2i G (G = the stored correlator, full spin-orbital,
assembled by code_exact/rdm_tools.py). HF: G = 0; 2B, TPPX, TPPX_0: block Kpp (same-spin blocks zero); DSL: G_opp
(+ packed same-spin G_UU, G_DD); mGW: not defined (its two-body density matrix is ambiguous) -> None."""
import numpy as np

from . import paths  # noqa: F401
from .readers import checkpoint_blocks
from rdm_tools import full_G, matrices, rho_full, unpack_anti     # noqa: E402


def state(method, b, N):
    r = b["rho"]
    ru = r[:N * N].reshape(N, N); rd = r[N * N:2 * N * N].reshape(N, N)
    rho = rho_full(ru, rd)
    if method == "HF":
        G = np.zeros((2 * N,) * 4, complex)
    elif method in ("2B", "TPPX", "TPPX_0"):
        Z = np.zeros(N ** 4, complex)
        G = full_G(b["Kpp"], Z, Z, N)
    elif method == "DSL":
        G = (full_G(b["G_opp"], unpack_anti(b["G_UU"], N), unpack_anti(b["G_DD"], N), N) if "G_UU" in b
             else full_G(b["G_opp"], None, None, N))
    else:
        return rho, None
    return rho, G


def pqg_min(rho, G):
    Pm, Qm, Gm, varN = matrices(rho, G)
    e = lambda A: float(np.linalg.eigvalsh(0.5 * (A + A.conj().T))[0])
    return dict(P_min=e(Pm), Q_min=e(Qm), Gp_min=e(Gm), varN=varN)


def checkpoint_pqg(path, N):
    """dict of arrays t, P_min, Q_min, Gp_min, varN over the periodic checkpoints (empty for mGW)"""
    method, cps = checkpoint_blocks(path)
    out = {k: [] for k in ("t", "P_min", "Q_min", "Gp_min", "varN")}
    for t, b in cps:
        rho, G = state(method, b, N)
        if G is None:
            continue
        v = pqg_min(rho, G)
        out["t"].append(t)
        for k in v:
            out[k].append(v[k])
    return {k: np.asarray(v, float) for k, v in out.items()}
