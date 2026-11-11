"""Fourier transform of G^R(tau) on a uniform grid and the interaction self-energy Sigma^R(w) = 1/G0(w) - 1/G(w).

ft(tau, G, w, eta): int_0^{tau_end} dtau e^{(i w - eta) tau} G(tau) by piecewise-cubic Filon quadrature:
  on each interval [tau_j, tau_j + h] G is replaced by the cubic through 4 neighbouring samples (stencil j-1..j+2,
  one-sided at the ends) and e^{z tau} * cubic is integrated exactly (I_p(kappa) = int_0^1 u^p e^{kappa u} du by its
  power series). Error O(h^4 G''''), independent of w (no (w h)^2 error as in the plain trapezoid rule).
g0_resolvent(ho, w, eta): [(w + i eta - h_o)^{-1}]_00 (the gamma = 0 impurity propagator, exact).
g0_time(ho, tau): G0(tau) = -i [e^{-i h_o tau}]_00 on the grid (to transform G0 with the same quadrature/window)."""
import numpy as np
from math import factorial
import scipy.linalg as sla

_VS = {}


def _coef_mat(nodes):
    """C with a = C @ g: coefficients of the cubic sum_p a_p u^p through (nodes, g)"""
    key = tuple(nodes)
    if key not in _VS:
        V = np.vander(np.array(nodes, float), 4, increasing=True)
        _VS[key] = np.linalg.inv(V)
    return _VS[key]


def _I(kappa, nterm=40):
    """I_p(kappa) = int_0^1 u^p e^{kappa u} du, p = 0..3, |kappa| <~ 3 (series)"""
    kappa = np.asarray(kappa, complex)
    out = np.zeros(kappa.shape + (4,), complex)
    term = np.ones_like(kappa)
    for k in range(nterm):
        for p in range(4):
            out[..., p] += term / (p + k + 1)
        term = term * kappa / (k + 1)
    return out


def ft(tau, G, w, eta=0.0, chunk=4000):
    tau = np.asarray(tau, float); G = np.asarray(G, complex); w = np.asarray(w, float)
    h = tau[1] - tau[0]; n = len(tau) - 1
    assert np.allclose(np.diff(tau), h, rtol=0, atol=1e-9 * max(1, tau[-1])), "uniform grid required"
    assert n >= 3
    # per-interval cubic coefficients
    a = np.zeros((n, 4), complex)
    Cm = _coef_mat([-1, 0, 1, 2])
    j = np.arange(1, n - 1)
    st = np.stack([G[j - 1], G[j], G[j + 1], G[j + 2]], axis=1)
    a[1:n - 1] = st @ Cm.T
    a[0] = _coef_mat([0, 1, 2, 3]) @ G[0:4]
    a[n - 1] = _coef_mat([-2, -1, 0, 1]) @ G[n - 3:n + 1]
    z = 1j * w - eta
    S = np.zeros((len(w), 4), complex)
    t0 = tau[:-1]
    for s in range(0, n, chunk):
        ph = np.exp(np.outer(z, t0[s:s + chunk]))
        S += ph @ a[s:s + chunk]
    return h * np.sum(S * _I(z * h), axis=1)


def ft_trap(tau, G, w, eta=0.0):
    """plain trapezoid (reference, as in the old figure but with both endpoint halves)"""
    h = tau[1] - tau[0]; wt = np.full(len(tau), h); wt[0] *= 0.5; wt[-1] *= 0.5
    return np.exp(np.outer(1j * w - eta, tau)) @ (G * wt)


def g0_resolvent(ho, w, eta=0.0):
    lam, V = np.linalg.eig(ho); Vi = np.linalg.inv(V)
    c = V[0, :] * Vi[:, 0]
    return np.array([np.sum(c / (x + 1j * eta - lam)) for x in w])


def g0_time(ho, tau):
    lam, V = np.linalg.eig(ho); Vi = np.linalg.inv(V)
    c = V[0, :] * Vi[:, 0]
    return -1j * (np.exp(-1j * np.outer(tau, lam)) @ c)


def sigma(G_w, G0_w):
    return 1.0 / G0_w - 1.0 / G_w
