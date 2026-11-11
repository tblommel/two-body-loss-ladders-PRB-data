"""Observables of even-space vectors (states): one linear-functional matrix for the whole vocabulary, plus the
2RDM functionals for the positivity matrices P, Q, G' (rdm_tools.matrices)."""
import numpy as np
import scipy.sparse as sp
import rdm_tools

VOCAB = ("N", "N_sigma", "n", "D", "S2", "Mz", "rho1", "rho1_eig")


class EvenObservables:
    def __init__(self, fm, space):
        self.fm, self.space = fm, space
        N, M = fm.N, fm.M
        n = fm.number_ops()
        ops, self.rows = [], {}

        def add(name, lst):
            self.rows[name] = (len(ops), len(ops) + len(lst)); ops.extend(lst)
        add("trace", [sp.identity(fm.dim, format="csr", dtype=complex)])
        add("n", n)                                                     # mode order sigma*N + r
        add("D", fm.double_occ_ops())
        add("S2", [fm.S2_op()])
        add("rho1", [(fm.cd[s * N + a] @ fm.c[s * N + q]).tocsr()          # rho^sigma_{q a} = <c^dag_a c_q>
                     for s in range(2) for q in range(N) for a in range(N)])
        self.F = space.functionals(ops)
        self._F2 = None

    def raw(self, X):
        """X: [dim] or [dim, m] -> functional values [k] or [k, m]"""
        return self.F @ X

    def from_raw(self, R, names):
        """R: [k, m] functional values -> dict name -> array with leading T axis (vocabulary shapes)"""
        N = self.fm.N
        g = lambda k: R[self.rows[k][0]:self.rows[k][1]]
        n = g("n").real                                                  # [2N, m]
        m = R.shape[1]
        out = {}
        for k in names:
            if k == "N":
                out[k] = n.sum(0)
            elif k == "N_sigma":
                out[k] = np.stack([n[:N].sum(0), n[N:].sum(0)], 1)
            elif k == "n":
                out[k] = n.reshape(2, N, m).transpose(2, 0, 1)
            elif k == "D":
                out[k] = g("D").real.T
            elif k == "S2":
                out[k] = g("S2").real[0]
            elif k == "Mz":
                out[k] = 0.5 * (n[:N].sum(0) - n[N:].sum(0))
            elif k == "rho1":
                out[k] = g("rho1").reshape(2, N, N, m).transpose(3, 0, 1, 2)
            elif k == "rho1_eig":
                r = g("rho1").reshape(2, N, N, m).transpose(3, 0, 1, 2)
                r = 0.5 * (r + r.conj().transpose(0, 1, 3, 2))
                out[k] = np.sort(np.linalg.eigvalsh(r).reshape(m, 2 * N), axis=1)
            elif k == "trace":
                out[k] = g("trace")[0]
            else:
                raise KeyError(k)
        return out

    def evaluate(self, X, names):
        """X: [dim] (one state) -> dict without T axis; [dim, m] -> with T axis"""
        one = X.ndim == 1
        R = self.raw(X[:, None] if one else X)
        out = self.from_raw(R, names)
        return {k: v[0] for k, v in out.items()} if one else out

    # ------------------------------------------------------------------ 2RDM / positivity
    def _build_F2(self):
        fm = self.fm; N, M = fm.N, fm.M
        spin = lambda i: i // N
        pairs = [(s, l) for s in range(M) for l in range(s + 1, M)]
        self._pairs = pairs
        ops, keys = [], []
        for (s, l) in pairs:
            cc = fm.c[l] @ fm.c[s]
            for (a, b) in pairs:
                if spin(a) + spin(b) != spin(s) + spin(l):
                    continue                                             # changes (N_up, N_dn): zero on even space
                ops.append((fm.cd[a] @ fm.cd[b] @ cc).tocsr()); keys.append((s, l, a, b))
        self._F2 = self.space.functionals(ops)
        self._F2keys = np.array(keys, int)

    def rdm12(self, x):
        """spin-orbital 1RDM rho_sa [M, M] and 2RDM Gamma_slab = <c^dag_a c^dag_b c_l c_s> [M]^4 of one state"""
        if self._F2 is None:
            self._build_F2()
        N, M = self.fm.N, self.fm.M
        r = (self.F[self.rows["rho1"][0]:self.rows["rho1"][1]] @ x).reshape(2, N, N)
        rho = np.zeros((M, M), complex); rho[:N, :N] = r[0]; rho[N:, N:] = r[1]
        v = self._F2 @ x
        G = np.zeros((M,) * 4, complex)
        s, l, a, b = self._F2keys.T
        G[s, l, a, b] = v; G[l, s, a, b] = -v; G[s, l, b, a] = -v; G[l, s, b, a] = v
        return rho, G

    def pqg(self, x):
        """min eigenvalues of P, Q, G' and Var N (rdm_tools.matrices; C = Gamma - rho rho + rho rho, G = C/(-2i))"""
        rho, Gam = self.rdm12(x)
        C = Gam - np.einsum("sa,lb->slab", rho, rho) + np.einsum("la,sb->slab", rho, rho)
        Pm, Qm, Gm, varN = rdm_tools.matrices(rho, C / (-2j))
        e = lambda A: float(np.linalg.eigvalsh(0.5 * (A + A.conj().T))[0])
        return dict(P_min=e(Pm), Q_min=e(Qm), Gp_min=e(Gm), varN=varN)
