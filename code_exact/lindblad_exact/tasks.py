"""The four tasks of exact_card/1: transient, ness, n_inf, greens_R. Each returns (datasets, attrs) for its group."""
import time
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .fock import FockModel
from .superop import BlockSpace, closure, all_blocks, superoperator
from .states import gaussian_blocks, CardError
from .observables import EvenObservables
from .card import EIG_MAX_DIM

ZERO_TOL = 1e-9          # |lambda| below this = zero mode (as plaquette_zeno/exact_sector_v1.py); gap = smallest
                         # -Re lambda among the damped modes (|Re lambda| >= ZERO_TOL)


class Progress:
    def __init__(self, path, run_id, n_tasks):
        self.path, self.run_id, self.n = path, run_id, n_tasks
        self.t0 = time.time(); self.last = 0.0; self.k = 0; self.name = ""; self.ts = time.time()

    def task(self, k, name):
        self.k, self.name, self.ts = k, name, time.time()
        self.update(0.0, "start", force=True)

    def update(self, frac, msg="", force=False):
        now = time.time()
        if not force and now - self.last < 2.0:
            return
        self.last = now
        el = now - self.ts
        eta = f"ETA {el / frac - el:.0f}s" if frac > 1e-6 else "ETA ?"
        line = (f"{self.run_id} | task {self.k + 1}/{self.n} {self.name} | {100 * frac:.1f}% | {msg} | "
                f"task elapsed {el:.0f}s {eta} | total elapsed {now - self.t0:.0f}s | {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        try:
            with open(self.path, "w") as f:
                f.write(line)
        except OSError:
            pass

    def done(self, status):
        try:
            with open(self.path, "w") as f:
                f.write(f"{self.run_id} | {status} | wall {time.time() - self.t0:.1f}s | {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        except OSError:
            pass


class Context:
    def __init__(self, card, progress):
        self.card, self.prog = card, progress
        self.fm = FockModel(card["model"])
        ist = card["initial_state"]
        self.x0_blocks = gaussian_blocks(self.fm, ist["rho1_up"], ist["rho1_dn"])
        self.even = BlockSpace(self.fm, closure(self.fm, [(a, a) for a in self.x0_blocks]))
        x0 = np.zeros(self.even.dim, complex)
        for a, B in self.x0_blocks.items():
            o = self.even.off[(a, a)]; x0[o:o + B.size] = B.ravel(order="F")
        self.x0 = x0
        self._S, self._eig = {}, {}
        self.obs = EvenObservables(self.fm, self.even)
        self.ness = None

    def S(self, space):
        if space.key not in self._S:
            self._S[space.key] = superoperator(self.fm, space)
        return self._S[space.key]

    def eig_even(self):
        """dense eigendecomposition of the even superoperator (cached)"""
        if "even" not in self._eig:
            d = self.even.dim
            if d > EIG_MAX_DIM:
                raise CardError(f"dense eig refused: even dimension {d} > {EIG_MAX_DIM}")
            self.prog.update(0.0, f"dense eig of the even superoperator (dim {d})", force=True)
            w, V = sla.eig(self.S(self.even).toarray(), overwrite_a=True, check_finite=False)
            c = np.linalg.solve(V, self.x0)
            recon = float(np.abs(V @ c - self.x0).max())
            self._eig["even"] = (w, V, c, recon)
        return self._eig["even"]


def model_attrs(fm):
    even_full = BlockSpace(fm, all_blocks(fm, (0, 0)))
    odd_full = BlockSpace(fm, all_blocks(fm, (1, 0)))
    return dict(n_orb=fm.N, dim_fock=fm.dim, dim_even=even_full.dim, dim_odd=odd_full.dim)


# ---------------------------------------------------------------------------------------------- transient
def frame_times(segs):
    ts = [np.array([0.0])]
    for (a, b, dt, n) in segs:
        t = a + dt * np.arange(1, n + 1); t[-1] = b
        ts.append(t)
    return np.concatenate(ts)


def _pqg_mask(t, every):
    if every <= 0:
        return np.zeros(len(t), bool)
    q = t / every
    return np.abs(q - np.round(q)) <= 1e-9 * np.maximum(1.0, np.abs(q))


def transient(ctx, task):
    names = task["observables"]
    segs = task["segments"]
    t = frame_times(segs)
    T = len(t)
    pmask = _pqg_mask(t, task["pqg_every"])
    res = {k: [] for k in names}
    trace = []
    pq = {k: [] for k in ("t", "P_min", "Q_min", "Gp_min", "varN")}
    ob = ctx.obs

    def record(X, tt, mask):
        """X: [dim, m] states at times tt"""
        R = ob.raw(X)
        o = ob.from_raw(R, names + ["trace"])
        for k in names:
            res[k].append(o[k])
        trace.append(o["trace"])
        for j in np.flatnonzero(mask):
            q = ob.pqg(X[:, j])
            pq["t"].append(tt[j])
            for k in ("P_min", "Q_min", "Gp_min", "varN"):
                pq[k].append(q[k])

    attrs = dict(method=task["method"], dim=ctx.even.dim, n_frames=T, pqg_every=task["pqg_every"],
                 grid_t_end=np.array([s[1] for s in segs]), grid_dt=np.array([s[2] for s in segs]))
    if task["method"] == "eig":
        w, V, c, recon = ctx.eig_even()
        attrs["eig_recon"] = recon
        FV = ob.F @ V                                    # [k, dim]
        A = FV * c[None, :]
        B = 1000
        for i in range(0, T, B):
            tt = t[i:i + B]
            E = np.exp(np.outer(w, tt))                  # [dim, m]
            R = A @ E
            o = ob.from_raw(R, names + ["trace"])
            for k in names:
                res[k].append(o[k])
            trace.append(o["trace"])
            for j in np.flatnonzero(pmask[i:i + B]):
                x = V @ (c * E[:, j])
                q = ob.pqg(x)
                pq["t"].append(tt[j])
                for k in ("P_min", "Q_min", "Gp_min", "varN"):
                    pq[k].append(q[k])
            ctx.prog.update(min(1.0, (i + B) / T), f"eig frames {min(T, i + B)}/{T}")
    else:
        S = ctx.S(ctx.even)
        x = ctx.x0.copy()
        record(x[:, None], t[:1], pmask[:1])
        done, chunk = 1, task["chunk"]
        for (a, b, dt, n) in segs:
            j = 0
            while j < n:
                m = min(chunk, n - j)
                Y = spla.expm_multiply(S, x, start=0.0, stop=m * dt, num=m + 1, endpoint=True)
                X = np.ascontiguousarray(Y[1:].T)
                record(X, t[done:done + m], pmask[done:done + m])
                x = Y[-1]; j += m; done += m
                ctx.prog.update(done / T, f"t = {t[done - 1]:.6g}/{t[-1]:.6g}")
    data = {"t": t}
    for k in names:
        data[k] = np.concatenate(res[k], axis=0)
    tr = np.concatenate(trace)
    attrs["max_trace_err"] = float(np.abs(tr - 1.0).max())
    if task["pqg_every"] > 0:
        data["pqg"] = {k: np.array(v) for k, v in pq.items()}
    return data, attrs


# ---------------------------------------------------------------------------------------------- NESS
def _spectrum(S, k):
    d = S.shape[0]
    if d <= 5000:
        ev = sla.eigvals(S.toarray(), overwrite_a=True, check_finite=False); kind = "dense"
    else:
        ev = spla.eigs(S, k=k, which="LR", return_eigenvectors=False, tol=1e-9, maxiter=20000,
                       ncv=max(60, 2 * k + 1)); kind = "arnoldi_LR"
    ev = ev[np.argsort(-ev.real)]
    nz = int(np.sum(np.abs(ev) < ZERO_TOL))
    damped = -ev.real[np.abs(ev.real) >= ZERO_TOL]
    gap = float(damped.min()) if damped.size else np.nan
    return ev[:max(k, nz + 1)], nz, gap, kind


def ness(ctx, task):
    sp_ = ctx.even
    S = ctx.S(sp_)
    method = task["method"]
    if method == "auto":
        method = "lu" if sp_.dim <= 10000 else "propagate"
    trf = sp_.trace_functional()
    attrs = dict(method=method, dim=sp_.dim, t_prop=0.0)
    if method == "lu":
        ctx.prog.update(0.0, f"sparse LU (dim {sp_.dim})", force=True)
        r0 = int(np.flatnonzero(trf)[0])               # a diagonal entry: its equation is redundant (trace)
        A = S.tolil(); A[r0, :] = trf[None, :]; A = A.tocsc()
        b = np.zeros(sp_.dim, complex); b[r0] = 1.0
        try:
            x = spla.splu(A).solve(b)
        except RuntimeError as e:
            raise RuntimeError(f"NESS sparse LU failed ({e}): the stationary state is probably not unique "
                               "(several zero modes); use task n_inf for the t -> infinity limit") from None
    else:
        p = task["propagate"]
        x = ctx.x0.copy(); T = 0.0
        while True:
            xn = spla.expm_multiply(S * p["block"], x); T += p["block"]
            ch = float(np.abs(xn - x).max()); x = xn
            ctx.prog.update(min(1.0, T / p["t_max"]), f"propagation T = {T:g}, change {ch:.1e}")
            if ch < p["tol"] or T >= p["t_max"]:
                break
        attrs["t_prop"] = T; attrs["last_change"] = ch; attrs["converged"] = int(ch < p["tol"])
    x = x / (trf @ x)
    x = sp_.hermitize(x)
    attrs["residual"] = float(np.abs(S @ x).max())
    attrs["mb_eig_min"] = float(min(np.linalg.eigvalsh(sp_.block_of(x, a, b))[0] for a, b in sp_.diag_blocks()))
    data = dict(ctx.obs.evaluate(x, task["observables"]))
    if task["pqg"]:
        data["pqg"] = ctx.obs.pqg(x)
    if "spectrum" in task:
        ctx.prog.update(0.5, "spectrum", force=True)
        ev, nz, gap, kind = _spectrum(S, task["spectrum"]["k"])
        data["spectrum"] = {"eigs": ev}
        attrs.update(spectrum_kind=kind, n_zero_modes=nz, gap=gap)
    ctx.ness = x
    return data, attrs


# ---------------------------------------------------------------------------------------------- N_inf
def n_inf(ctx, task):
    """t -> infinity limit = spectral projection of x0 on ker S along range S. The zero eigenvalue of a Lindblad
    generator is semisimple, so P = R (L^H R)^{-1} L^H with R, L orthonormal bases of the right / left null spaces,
    both from one SVD of S (accurate even when the eigenvector matrix of S is ill-conditioned). The dense
    eigenvalues give the number of zero modes (|lambda| < ZERO_TOL) and the gap; the two counts must agree."""
    d = ctx.even.dim
    if d > EIG_MAX_DIM:
        raise CardError(f"n_inf refused: even dimension {d} > {EIG_MAX_DIM}")
    ob = ctx.obs
    Sd = ctx.S(ctx.even).toarray()
    ctx.prog.update(0.0, f"eigenvalues (dim {d})", force=True)
    w = sla.eigvals(Sd, check_finite=False)
    zero = np.abs(w) < ZERO_TOL
    nz = int(zero.sum())
    ctx.prog.update(0.5, f"SVD null spaces (dim {d})", force=True)
    U, sv, Vh = sla.svd(Sd, check_finite=False, lapack_driver="gesdd")
    R = Vh[d - nz:].conj().T                              # right null space (smallest singular values last)
    L = U[:, d - nz:]                                     # left null space: L^H S = 0
    x = R @ np.linalg.solve(L.conj().T @ R, L.conj().T @ ctx.x0)
    x = ctx.even.hermitize(x)
    o = ob.from_raw((ob.F @ x)[:, None], task["observables"] + ["trace"])
    data = {k: v[0] for k, v in o.items() if k != "trace"}
    osc = (~zero) & (np.abs(w.real) < ZERO_TOL)              # undamped oscillating modes (Re = 0, Im != 0)
    nzw = -w.real[np.abs(w.real) >= ZERO_TOL]
    attrs = dict(n_zero_modes=nz, n_oscillating_modes=int(osc.sum()), gap=float(nzw.min()) if nzw.size else np.nan, dim=d,
                 trace=float(o["trace"][0].real), residual=float(np.abs(ctx.S(ctx.even) @ x).max()),
                 sv_null_max=float(sv[d - nz:].max()) if nz else 0.0,
                 sv_next=float(sv[d - nz - 1]) if nz < d else np.nan,
                 max_abs_imag_zero=float(np.abs(w[zero].imag).max()) if nz else 0.0)
    sv_ok = attrs["sv_null_max"] < 1e-8 * max(1.0, sv[0])  # null singular values are numerically zero ...
    if nz < d:                                              # ... and, unless S = 0 (e.g. the vacuum), the next one is not
        sv_ok = sv_ok and 1e-8 * max(1.0, sv[0]) < attrs["sv_next"]
    if nz and not sv_ok:
        raise RuntimeError(f"n_inf: zero-mode count {nz} from the eigenvalues does not match the singular values "
                           f"(largest null sv {attrs['sv_null_max']:.2e}, next {attrs['sv_next']:.2e})")
    return data, attrs


# ---------------------------------------------------------------------------------------------- G^R
def greens_R(ctx, task):
    fm = ctx.fm
    X = ctx.even.full_from_vec(ctx.ness)
    n = task["n_tau"]; dtau = task["dtau"]
    tau = dtau * np.arange(n + 1); tau[-1] = task["tau_max"]
    G = np.zeros((n + 1, len(task["spins"])), complex)
    res = task.get("resolvent")
    if res:
        omega = np.linspace(res["omega_min"], res["omega_max"], res["n"])
        Gw = np.zeros((res["n"], len(task["spins"])), complex); info = np.zeros((res["n"], len(task["spins"])), int)
    attrs = dict(orbital=task["orbital"], spins=",".join(task["spins"]), state=task["state"])
    for k, spn in enumerate(task["spins"]):
        s = fm.mode(task["orbital"], spn)
        Y = fm.cd[s] @ X + (fm.cd[s].T @ X.T).T          # {c^dag_s, rho}
        e = (1, 0) if spn == "up" else (0, 1)
        seed = [(a, b) for a, b in all_blocks(fm, e) if np.abs(Y[np.ix_(fm.idx[a], fm.idx[b])]).max() > 0]
        space = BlockSpace(fm, closure(fm, seed))
        y = space.vec_from_full(Y)
        S = ctx.S(space)
        w = space.functionals([fm.c[s]])                  # Tr[c_s Y]
        attrs[f"dim_odd_{spn}"] = space.dim
        G[0, k] = -1j * (w @ y)[0]
        j = 0; chunk = 200
        while j < n:
            m = min(chunk, n - j)
            Yt = spla.expm_multiply(S, y, start=0.0, stop=m * dtau, num=m + 1, endpoint=True)
            G[j + 1:j + m + 1, k] = -1j * (w @ Yt[1:].T)[0]
            y = Yt[-1]; j += m
            ctx.prog.update((k + j / n) / len(task["spins"]) * (0.7 if res else 1.0), f"G^R {spn}: tau {j * dtau:.6g}")
        if res:
            y0 = space.vec_from_full(Y)
            wv = w.toarray()[0]
            if res["solver"] == "eig":
                if space.dim > EIG_MAX_DIM:
                    raise CardError(f"resolvent solver eig refused: odd dimension {space.dim} > {EIG_MAX_DIM}")
                lam, Vo = sla.eig(S.toarray(), check_finite=False)
                aa = np.linalg.solve(Vo, y0); bb = wv @ Vo
                Gw[:, k] = [1j * np.sum(bb * aa / (lam + 1j * om)) for om in omega]
            else:
                Id = sp.identity(space.dim, format="csr")
                for q, om in enumerate(omega):
                    v, inf = spla.gmres((S + 1j * om * Id).tocsr(), y0, rtol=res["rtol"], atol=0.0, restart=300,
                                        maxiter=200)
                    Gw[q, k] = 1j * (wv @ v); info[q, k] = inf
                    ctx.prog.update(0.7 + 0.3 * (k + (q + 1) / len(omega)) / len(task["spins"]),
                                    f"resolvent {spn}: omega {om:.3f} info {inf}")
    data = {"tau": tau, "GR": G}
    if res:
        data["resolvent"] = {"omega": omega, "GR": Gw, "info": info}
        attrs["resolvent_solver"] = res["solver"]
    return data, attrs


RUNNERS = {"transient": transient, "ness": ness, "n_inf": n_inf, "greens_R": greens_R}
