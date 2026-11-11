"""Small card builders and a dense full-Fock-space reference for the tests (independent of the sector machinery)."""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as sla

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
RUN = CODE / "run_exact.py"


def model(name, h1, loss=None, gain=None, sites=(0,), gamma=1.0, weights=None):
    h1 = np.asarray(h1, float); N = len(h1)
    m = {"name": name, "n_orb": N, "h1": h1.tolist(),
         "onebody_loss": list(map(float, loss if loss is not None else np.zeros(N))),
         "onebody_gain": list(map(float, gain if gain is not None else np.zeros(N))),
         "pair_loss": {"sites": list(sites), "gamma": float(gamma)}}
    if weights is not None:
        m["pair_loss"]["weights"] = list(map(float, weights))
    return m


def card(run_id, mdl, rup, rdn, tasks):
    return {"schema": "exact_card/1", "run_id": run_id, "model": mdl,
            "initial_state": {"type": "gaussian", "rho1_up": np.asarray(rup, float).tolist(),
                              "rho1_dn": np.asarray(rdn, float).tolist()},
            "tasks": tasks, "output": {"file": "result.h5"}}


def write(path, c):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(c, indent=1))
    return path


# ---------------------------------------------------------------- paper models (values as in the paper's model setups)
def site(gamma):
    return model("site", [[0.0]], sites=[0], gamma=gamma)


RING4 = -np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], float)
PHI0 = np.ones(4) / 2
PHI1 = np.array([-1, 0, 1, 0]) / np.sqrt(2)
PHI2 = np.array([0, -1, 0, 1]) / np.sqrt(2)


def plaq(gamma):
    return model("plaq", RING4, sites=[0, 1, 2, 3], gamma=gamma)


def plaq_ic(name):
    P = lambda *v: sum(np.outer(x, x) for x in v)
    return {"slater": (P(PHI0, PHI1), P(PHI0, PHI1)), "cdw": (np.diag([1., 0, 1, 0]),) * 2,
            "mixed": (0.5 * np.eye(4),) * 2, "one_pair": (np.diag([1., 0, 0, 0]),) * 2,
            "polarized": (0.75 * np.eye(4), 0.25 * np.eye(4)), "3u1d": (P(PHI0, PHI1, PHI2), P(PHI0)),
            "neel": (np.diag([1., 0, 1, 0]), np.diag([0., 1, 0, 1]))}[name]


C3_H1 = [[-0.5, 0.6, 0.5, 0.7, 0.4], [0.6, 0.8, 0, 0, 0], [0.5, 0, 1.6, 0, 0], [0.7, 0, 0, -0.9, 0],
         [0.4, 0, 0, 0, -1.8]]
C3_LOSS = [0, 0.7, 0.9, 0, 0]
C3_GAIN = [0, 0, 0, 0.8, 0.5]


def c3(gamma, weak=None):
    if weak is None:
        return model("c3", C3_H1, C3_LOSS, C3_GAIN, [0], gamma)
    lo = [0, weak, weak, 0, 0]; ga = [0, 0, 0, weak, weak]
    return model("c3w", C3_H1, lo, ga, [0], gamma)


def toyB(gamma):
    h1 = [[-1.5, 0.8, 0.8], [0.8, 1.0, 0], [0.8, 0, -1.0]]
    return model("toyB", h1, [0, 0.3, 0], [0, 0, 0.3], [0], gamma)


def imp_ic(N, gain, ic):
    nu, nd = {"00": (0, 0), "11": (1, 1), "10": (1, 0)}[ic]
    ru = np.zeros((N, N)); rd = np.zeros((N, N))
    ru[0, 0] = nu; rd[0, 0] = nd
    for k in range(N):
        if gain[k] > 0:
            ru[k, k] = rd[k, k] = 1.0
    return ru, rd


# ---------------------------------------------------------------- dense full-space reference
class Dense:
    """full Fock space, dense Liouvillian (tiny models only)"""

    def __init__(self, mdl):
        from lindblad_exact.fock import FockModel
        self.fm = fm = FockModel(mdl)
        self.d = fm.dim
        I = np.eye(self.d)
        Heff = fm.Heff.toarray()
        self.L = np.kron(I, -1j * Heff) + np.kron((1j * Heff.conj().T).T, I)
        self.Lodd = self.L.copy()
        for Lj, _, par in fm.jumps:
            Lj = Lj.toarray()
            term = 2 * np.kron(Lj.conj(), Lj)
            self.L += term
            self.Lodd += (-1 if par else 1) * term

    def gaussian(self, rup, rdn):
        """Gaussian state from the full-space natural orbitals (independent construction)"""
        fm = self.fm; N = fm.N; st = np.eye(self.d, dtype=complex)
        c = [x.toarray() for x in fm.c]
        for s, r1 in enumerate((rup, rdn)):
            w, V = np.linalg.eigh(r1)
            for k in range(N):
                b = sum(np.conj(V[x, k]) * c[s * N + x] for x in range(N))
                B = b.conj().T @ b
                st = st @ (w[k] * B + (1 - w[k]) * (np.eye(self.d) - B))
        return st

    def evolve(self, X, t):
        return (sla.expm(self.L * t) @ X.reshape(-1, order="F")).reshape(self.d, self.d, order="F")
