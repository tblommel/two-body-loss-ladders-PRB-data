"""All model data of the paper in one place (values identical to the data the paper was made with).

A model: H = sum_sigma c^dag_sigma h1 c_sigma (same h1 for both spins), one-body loss rates (jump sqrt(G_r) c_r sigma),
one-body gain rates (jump sqrt(G_r) c^dag_r sigma), pair loss L_r = sqrt(gamma w_r) c_r dn c_r up on the lossy sites.
1RDM rho_sa = <c^dag_a c_s>, spin blocks [2, N, N] (0 = up).

Systems (run-id tokens):
  site    single site, no hopping, pair loss on it                                 (fig_site)
  plaq    2x2 plaquette = 4-site ring, J = 1, pair loss on every site              (fig_plaq, tab_plaqerr, fig_zeno)
  c3      impurity d + 2 'greater' (lossy, empty) + 2 'lesser' (gain, filled) pseudomodes, strongly damped
  c3G0.2  the same impurity with all four mode rates Gamma = 0.2 (weakly damped)  (fig_impss, fig_sigma)
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Model:
    name: str
    h1: np.ndarray
    loss: np.ndarray
    gain: np.ndarray
    lossy_sites: list
    weights: list
    driver_attrs: dict = field(default_factory=dict)
    vertex_impl: str = "general"

    @property
    def N(self):
        return len(self.h1)

    def h_o(self):
        """one-spin h_o = h1 - i diag(loss + gain) (fermions): the gamma = 0 propagator generator"""
        return self.h1 - 1j * np.diag(self.loss + self.gain)


# ----------------------------------------------------------------------------------------------- single site
def site():
    return Model("site", np.zeros((1, 1)), np.zeros(1), np.zeros(1), [0], [1.0])


def site_rho0(n0):
    return np.full((2, 1, 1), n0, complex)


def site_exact_n(n0, gt):
    """analytic single site (validation and tests only): n_sigma(t), gt = gamma t"""
    return n0 - n0 ** 2 * (1 - np.exp(-2 * np.asarray(gt)))


def site_exact_D(n0, gt):
    return n0 ** 2 * np.exp(-2 * np.asarray(gt))


# ----------------------------------------------------------------------------------------------- plaquette
RING4 = -np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], float)
PHI0 = np.ones(4) / 2                                   # k = 0 orbital
PHI1 = np.array([-1, 0, 1, 0]) / np.sqrt(2)             # standing waves of the degenerate E = 0 shell
PHI2 = np.array([0, -1, 0, 1]) / np.sqrt(2)
PLAQ_ICS = ["slater", "cdw", "mixed", "one_pair", "polarized", "3u1d", "neel"]


def plaq():
    return Model("plaq", RING4.copy(), np.zeros(4), np.zeros(4), [0, 1, 2, 3], [1.0] * 4)


def plaq_rho0(ic):
    """Gaussian initial states (per spin 4x4 1RDM):
    slater    half-filled Slater determinant, per spin PHI0 and PHI1 (site densities 0.75/0.25 alternating)
    cdw       sites 0 and 2 doubly occupied;  mixed: rho = 1/2 per spin;  one_pair: up and dn on site 0
    polarized rho_up = 3/4, rho_dn = 1/4;  3u1d: up in PHI0, PHI1, PHI2, dn in PHI0;  neel: up on 0, 2, dn on 1, 3"""
    P = lambda *v: sum(np.outer(x, x) for x in v).astype(complex)
    if ic == "slater":
        r = P(PHI0, PHI1); u, d = r, r.copy()
    elif ic == "cdw":
        u = np.diag([1., 0, 1, 0]).astype(complex); d = u.copy()
    elif ic == "mixed":
        u = 0.5 * np.eye(4, dtype=complex); d = u.copy()
    elif ic == "one_pair":
        u = np.diag([1., 0, 0, 0]).astype(complex); d = u.copy()
    elif ic == "polarized":
        u, d = 0.75 * np.eye(4, dtype=complex), 0.25 * np.eye(4, dtype=complex)
    elif ic == "3u1d":
        u, d = P(PHI0, PHI1, PHI2), P(PHI0)
    elif ic == "neel":
        u, d = np.diag([1., 0, 1, 0]).astype(complex), np.diag([0., 1, 0, 1]).astype(complex)
    else:
        raise ValueError(ic)
    return np.stack([u, d])


# ----------------------------------------------------------------------------------------------- impurity models
# impurity level ed, greater modes [(eps, g, Gamma)] (one-body loss, initially empty), lesser modes (one-body gain,
# initially filled); orbital order per spin: 0 = d, greater modes, lesser modes; no mode-mode hopping.
_C3 = dict(ed=-0.5, gt=[(0.8, 0.6, 0.7), (1.6, 0.5, 0.9)], lt=[(-0.9, 0.7, 0.8), (-1.8, 0.4, 0.5)])
IMP_SYSTEMS = {"c3": _C3,
               "c3G0.2": dict(ed=_C3["ed"], gt=[(e, g, 0.2) for e, g, _ in _C3["gt"]],
                              lt=[(e, g, 0.2) for e, g, _ in _C3["lt"]])}
IMP_ICS = {"00": (0.0, 0.0), "11": (1.0, 1.0), "10": (1.0, 0.0)}      # (n_d up, n_d dn) at t = 0


def impurity(system):
    m = IMP_SYSTEMS[system]; ms = m["gt"] + m["lt"]; N = 1 + len(ms); ng = len(m["gt"])
    h = np.zeros((N, N)); h[0, 0] = m["ed"]
    lo = np.zeros(N); ga = np.zeros(N)
    for k, (e, g, G) in enumerate(ms):
        h[k + 1, k + 1] = e; h[0, k + 1] = h[k + 1, 0] = g
        (lo if k < ng else ga)[k + 1] = G
    return Model(system, h, lo, ga, [0], [1.0],
                 driver_attrs=dict(N_greater=ng, N_lesser=len(m["lt"]), ordering="d,p>...,p<..."),
                 vertex_impl="impurity")


def imp_rho0(system, ic):
    """impurity in IC (00 empty, 11 doubly occupied, 10 one up particle); greater modes empty, lesser modes filled"""
    mdl = impurity(system); nu, nd = IMP_ICS[ic]
    r = np.zeros((2, mdl.N, mdl.N), complex)
    r[0, 0, 0] = nu; r[1, 0, 0] = nd
    for k in range(mdl.N):
        if mdl.gain[k] > 0:
            r[:, k, k] = 1.0
    return r


def get(system):
    if system == "site":
        return site()
    if system == "plaq":
        return plaq()
    return impurity(system)


def model_default_rho0(system):
    """the model file's /initial/rho (kept as in the old model files: zeros for site/plaq, IC 00 for impurities)"""
    mdl = get(system)
    if system in IMP_SYSTEMS:
        return imp_rho0(system, "00")
    return np.zeros((2, mdl.N, mdl.N), complex)
