"""Fock space, operators and the Lindblad model.

Conventions (CONVENTIONS.md §0): fermions; spin-orbital index i = sigma*N + r inside this package only (sigma 0 = up);
Jordan-Wigner order with mode 0 the most significant bit, so a basis index is i_up * 2^N + i_dn.
Lindblad equation  d rho/dt = -i[H, rho] + sum_L (2 L rho L^dag - {L^dag L, rho})  with
  H = sum_sigma c^dag_sigma h1 c_sigma,
  one-body loss  L = sqrt(G_r) c_{r sigma},  one-body gain  L = sqrt(G_r) c^dag_{r sigma}  (both spins),
  pair loss      L = sqrt(gamma w_r) c_{r dn} c_{r up}  on the listed sites.
Both N_up and N_dn are weak U(1) symmetries: every jump shifts the sector label (N_up, N_dn) by a fixed vector.
"""
import numpy as np
import scipy.sparse as sp


def fermion_ops(M):
    """annihilators c_j (csr, real) on 2^M states, Jordan-Wigner, mode 0 = most significant bit"""
    a = sp.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]]))
    Z = sp.csr_matrix(np.diag([1.0, -1.0]))
    I = sp.identity(2, format="csr")
    out = []
    for j in range(M):
        m = sp.identity(1, format="csr")
        for k in range(M):
            m = sp.kron(m, Z if k < j else (a if k == j else I), format="csr")
        out.append(m.tocsr())
    return out


class FockModel:
    """Model operators on the full Fock space (2N spin-orbitals) and the sector bookkeeping."""

    def __init__(self, model):
        h1 = np.asarray(model["h1"], dtype=float)
        N = int(model["n_orb"])
        self.N, self.M, self.dim = N, 2 * N, 4 ** N
        self.h1 = h1
        self.loss = np.asarray(model["onebody_loss"], float)
        self.gain = np.asarray(model["onebody_gain"], float)
        pl = model["pair_loss"]
        self.pair_sites = [int(r) for r in pl["sites"]]
        self.pair_gamma = float(pl["gamma"])
        self.pair_weights = [float(w) for w in pl.get("weights", [1.0] * len(self.pair_sites))]
        M, dim = self.M, self.dim
        c = fermion_ops(M)
        self.c = [x.astype(complex).tocsr() for x in c]
        self.cd = [x.T.tocsr() for x in self.c]
        bits = (np.arange(dim)[None, :] >> (M - 1 - np.arange(M))[:, None]) & 1      # [M, dim]
        self.nup_of = bits[:N].sum(0)
        self.ndn_of = bits[N:].sum(0)
        self.sectors = [(a, b) for a in range(N + 1) for b in range(N + 1)]
        self.idx = {s: np.flatnonzero((self.nup_of == s[0]) & (self.ndn_of == s[1])) for s in self.sectors}
        # Hamiltonian
        H = sp.csr_matrix((dim, dim), dtype=complex)
        for s in range(2):
            for i in range(N):
                for j in range(N):
                    if h1[i, j] != 0.0:
                        H = H + h1[i, j] * (self.cd[s * N + i] @ self.c[s * N + j])
        self.H = H.tocsr()
        # jumps: (operator, sector shift, fermion parity)
        jumps = []
        if self.pair_gamma > 0:
            for r, w in zip(self.pair_sites, self.pair_weights):
                if w > 0:
                    jumps.append((np.sqrt(self.pair_gamma * w) * (self.c[N + r] @ self.c[r]), (-1, -1), 0))
        for s in range(2):
            e = (1, 0) if s == 0 else (0, 1)
            for r in range(N):
                if self.loss[r] > 0:
                    jumps.append((np.sqrt(self.loss[r]) * self.c[s * N + r], (-e[0], -e[1]), 1))
                if self.gain[r] > 0:
                    jumps.append((np.sqrt(self.gain[r]) * self.cd[s * N + r], e, 1))
        self.jumps = [(L.tocsr(), d, p) for L, d, p in jumps]
        Heff = self.H.copy()
        for L, _, _ in self.jumps:
            Heff = Heff - 1j * (L.conj().T @ L)
        self.Heff = Heff.tocsr()
        self.shifts = sorted({d for _, d, _ in self.jumps})
        self._blk = {}

    # ------------------------------------------------------------------ helpers
    def valid(self, s):
        return 0 <= s[0] <= self.N and 0 <= s[1] <= self.N

    def block(self, key, O, a, b):
        """dense-free sub-block O[idx_a, idx_b] (csr), cached by key"""
        k = (key, a, b)
        if k not in self._blk:
            self._blk[k] = O[self.idx[a]][:, self.idx[b]].tocsr()
        return self._blk[k]

    def mode(self, orbital, spin):
        return (0 if spin in ("up", 0) else 1) * self.N + int(orbital)

    # ------------------------------------------------------------------ observables (operators)
    def number_ops(self):
        return [(self.cd[i] @ self.c[i]).tocsr() for i in range(self.M)]

    def double_occ_ops(self):
        n = self.number_ops()
        return [(n[r] @ n[self.N + r]).tocsr() for r in range(self.N)]

    def S2_op(self):
        N = self.N
        n = self.number_ops()
        Sp = sum(self.cd[r] @ self.c[N + r] for r in range(N))
        Sz = 0.5 * sum(n[r] - n[N + r] for r in range(N))
        return (Sp @ Sp.conj().T + Sz @ Sz - Sz).tocsr()
