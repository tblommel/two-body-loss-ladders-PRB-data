"""Sector-restricted superoperators.

An operator X on Fock space is stored by its blocks X[idx_a, idx_b] for a set of sector pairs (a, b) with a constant
shift a - b (EVEN space: shift (0, 0), states; ODD space: shift e_sigma, the regression vector {c^dag_sigma, rho}).
Each block is vectorized column-major (vec(A X B) = (B^T kron A) vec X) and the blocks are concatenated.
The block set is closed under the jump shifts (closure()), so the Lindblad superoperator maps the space into itself.

Superoperator on the space:  S X = -i Heff X + i X Heff^dag + sum_L sign_L 2 L X L^dag,
sign_L = (-1)^(parity(L) * parity(shift)): one-body jumps acting on a fermion-odd vector get a minus sign
(quantum regression for fermionic operators, as in imp_configs/exact_cfg.py).
"""
import numpy as np
import scipy.sparse as sp


def closure(fm, seed):
    """smallest set of sector pairs containing `seed` and closed under X -> L X L^dag (shift both by the jump shift)"""
    out = set(seed)
    todo = list(seed)
    while todo:
        a, b = todo.pop()
        for d in fm.shifts:
            na = (a[0] + d[0], a[1] + d[1]); nb = (b[0] + d[0], b[1] + d[1])
            if fm.valid(na) and fm.valid(nb) and (na, nb) not in out:
                out.add((na, nb)); todo.append((na, nb))
    return sorted(out)


def all_blocks(fm, shift):
    out = []
    for b in fm.sectors:
        a = (b[0] + shift[0], b[1] + shift[1])
        if fm.valid(a):
            out.append((a, b))
    return sorted(out)


class BlockSpace:
    def __init__(self, fm, blocks):
        self.fm = fm
        self.blocks = list(blocks)
        sh = {(a[0] - b[0], a[1] - b[1]) for a, b in self.blocks}
        assert len(sh) == 1, "block set must have one shift"
        self.shift = sh.pop()
        self.parity = (self.shift[0] + self.shift[1]) % 2
        self.off, self.shape = {}, {}
        o = 0
        for a, b in self.blocks:
            da, db = len(fm.idx[a]), len(fm.idx[b])
            self.off[(a, b)] = o; self.shape[(a, b)] = (da, db); o += da * db
        self.dim = o
        self.key = (self.shift, tuple(self.blocks))

    # --------------------------------------------------------- conversions
    def vec_from_full(self, X):
        x = np.zeros(self.dim, complex)
        for a, b in self.blocks:
            o = self.off[(a, b)]; da, db = self.shape[(a, b)]
            x[o:o + da * db] = np.asarray(X[np.ix_(self.fm.idx[a], self.fm.idx[b])]).ravel(order="F")
        return x

    def full_from_vec(self, x):
        X = np.zeros((self.fm.dim, self.fm.dim), complex)
        for a, b in self.blocks:
            o = self.off[(a, b)]; da, db = self.shape[(a, b)]
            X[np.ix_(self.fm.idx[a], self.fm.idx[b])] = x[o:o + da * db].reshape(da, db, order="F")
        return X

    def block_of(self, x, a, b):
        o = self.off[(a, b)]; da, db = self.shape[(a, b)]
        return x[o:o + da * db].reshape(da, db, order="F")

    def diag_blocks(self):
        return [(a, b) for a, b in self.blocks if a == b]

    # --------------------------------------------------------- linear functionals Tr[O X]
    def functionals(self, ops):
        """csr [len(ops), dim]: row k . x = Tr[ops[k] X]. Tr[O X] = sum_ij O[j, i] X[i, j] with X[i, j] in block (a, b)
        and O[j, i] in O[idx_b, idx_a]."""
        rows, cols, vals = [], [], []
        for k, O in enumerate(ops):
            O = sp.csr_matrix(O)
            for a, b in self.blocks:
                Ob = sp.coo_matrix(O[self.fm.idx[b]][:, self.fm.idx[a]])
                if Ob.nnz == 0:
                    continue
                da = self.shape[(a, b)][0]
                # entry Ob[j, i] multiplies X[i, j] -> vec index i + j*da
                rows.append(np.full(Ob.nnz, k)); cols.append(self.off[(a, b)] + Ob.col + Ob.row * da)
                vals.append(Ob.data)
        if not rows:
            return sp.csr_matrix((len(ops), self.dim), dtype=complex)
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(len(ops), self.dim))

    def trace_functional(self):
        rows = np.zeros(self.dim, complex)
        for a, b in self.diag_blocks():
            o = self.off[(a, b)]; d = self.shape[(a, b)][0]
            rows[o + np.arange(d) * (d + 1)] = 1.0
        return rows

    def hermitize(self, x):
        """even space: replace each diagonal block by its Hermitian part"""
        y = x.copy()
        for a, b in self.diag_blocks():
            B = self.block_of(x, a, b)
            o = self.off[(a, b)]; d = B.shape[0]
            y[o:o + d * d] = (0.5 * (B + B.conj().T)).ravel(order="F")
        return y


def superoperator(fm, space):
    """sparse csr superoperator on `space` (see module doc)"""
    rows, cols, vals = [], [], []
    for a, b in space.blocks:
        o = space.off[(a, b)]
        Ha = fm.block("Heff", fm.Heff, a, a)
        Hb = fm.block("Heff", fm.Heff, b, b)
        T = (sp.kron(sp.identity(len(fm.idx[b])), -1j * Ha) + sp.kron(1j * Hb.conj(), sp.identity(len(fm.idx[a])))).tocoo()
        rows.append(o + T.row); cols.append(o + T.col); vals.append(T.data)
    bset = set(space.blocks)
    for q, (L, d, par) in enumerate(fm.jumps):
        sign = -1.0 if (par * space.parity) % 2 else 1.0
        for a, b in space.blocks:                     # source block (a, b) -> target (a + d, b + d)
            na = (a[0] + d[0], a[1] + d[1]); nb = (b[0] + d[0], b[1] + d[1])
            if not (fm.valid(na) and fm.valid(nb)):
                continue
            assert (na, nb) in bset, "block set not closed under the jumps"
            La = fm.block(("L", q), L, na, a)
            Lb = fm.block(("L", q), L, nb, b)
            if La.nnz == 0 or Lb.nnz == 0:
                continue
            T = (2.0 * sign * sp.kron(Lb.conj(), La)).tocoo()
            rows.append(space.off[(na, nb)] + T.row); cols.append(space.off[(a, b)] + T.col); vals.append(T.data)
    S = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(space.dim, space.dim))
    S.sum_duplicates()
    return S
