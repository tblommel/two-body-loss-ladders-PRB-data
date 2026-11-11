"""Tests of code_exact. Run:  python -I code_exact/tests/test_exact.py   (or pytest code_exact/tests)

1. analytic single site (every observable, several n0 / gamma, both time-stepping methods, n_inf);
2. Gaussian initial state: 1RDM reproduced, C2 = C3 = 0, equal to an independent full-space construction;
3. trace preservation and Hermiticity along a transient;
4. sector-restricted vs dense full-space Liouvillian (transient, NESS, G^R) on a tiny model with all jump types;
5. G^R(tau) by regression vs its spectral representation, and resolvent (eig vs gmres vs Fourier of G^R(tau));
6. card validation and the command line (exit codes, result.h5 layout).
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.linalg as sla

sys.path.insert(0, str(Path(__file__).resolve().parent))
import helpers as H                                                     # noqa: E402
import h5py                                                             # noqa: E402
from lindblad_exact.card import parse                                   # noqa: E402
from lindblad_exact.tasks import Context, Progress, RUNNERS            # noqa: E402
import rdm_tools                                                         # noqa: E402

ALL = ["N", "N_sigma", "n", "D", "S2", "Mz", "rho1", "rho1_eig"]


def _ctx(c, tmp):
    card = parse(json.dumps(c))
    return Context(card, Progress(Path(tmp) / "progress.txt", c["run_id"], len(card["tasks"]))), card


def _run(c, tmp):
    ctx, card = _ctx(c, tmp)
    return {t["name"]: RUNNERS[t["type"]](ctx, t) for t in card["tasks"]}, ctx


def _rand_rho(N, rng):
    A = rng.normal(size=(N, N)); Q, _ = np.linalg.qr(A)
    return Q @ np.diag(rng.uniform(0, 1, N)) @ Q.T


# ---------------------------------------------------------------- 1. analytic single site
def site_analytic(a, b, g, t):
    """single site, pair loss only: D = a b e^{-2 g t}, n_up = a - a b (1 - e^{-2 g t}) (likewise dn)"""
    e = np.exp(-2 * g * t)
    D = a * b * e; nu = a - a * b * (1 - e); nd = b - a * b * (1 - e)
    p11 = D; p10 = nu - D; p01 = nd - D; p00 = 1 - p11 - p10 - p01
    N = nu + nd
    return dict(N=N, n_up=nu, n_dn=nd, D=D, S2=0.75 * (p10 + p01), Mz=0.5 * (nu - nd),
                varN=4 * p11 + p10 + p01 - N ** 2, probs=(p00, p10, p01, p11))


def test_site_analytic():
    with tempfile.TemporaryDirectory() as tmp:
        for (a, b) in [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9), (1.0, 1.0), (0.7, 0.3),
                       (0.0, 0.0), (1.0, 0.0), (0.0, 0.6), (0.7, 0.0), (1.0, 0.4)]:   # edge: vacuum, one spin only
            for g in (0.5, 1.0, 4.0, 100.0):
                grid = [{"t_end": 2.0 / g, "dt": 0.01 / g}, {"t_end": 50.0 / g, "dt": 1.0 / g}]
                c = H.card("site", H.site(g), [[a]], [[b]], [
                    {"type": "transient", "name": "x", "grid": grid, "observables": ALL, "pqg_every": 1.0 / g},
                    {"type": "transient", "name": "e", "grid": grid, "observables": ALL, "method": "eig"},
                    {"type": "n_inf", "name": "inf", "observables": ALL}])
                R, _ = _run(c, tmp)
                for key in ("x", "e"):
                    d, at = R[key]
                    t = d["t"]; A = site_analytic(a, b, g, t)
                    err = max(np.abs(d["N"] - A["N"]).max(), np.abs(d["n"][:, 0, 0] - A["n_up"]).max(),
                              np.abs(d["n"][:, 1, 0] - A["n_dn"]).max(), np.abs(d["D"][:, 0] - A["D"]).max(),
                              np.abs(d["S2"] - A["S2"]).max(), np.abs(d["Mz"] - A["Mz"]).max(),
                              np.abs(d["N_sigma"][:, 0] - A["n_up"]).max(),
                              np.abs(d["rho1"][:, 0, 0, 0] - A["n_up"]).max(),
                              np.abs(d["rho1_eig"] - np.sort(np.stack([A["n_up"], A["n_dn"]], 1), 1)).max())
                    assert err < 1e-12, (a, b, g, key, err)
                    assert at["max_trace_err"] < 1e-12
                d, _ = R["x"]
                tp = d["pqg"]["t"]; A = site_analytic(a, b, g, tp)
                assert np.abs(d["pqg"]["varN"] - A["varN"]).max() < 1e-12
                # P, Q: Gram matrices of c_l c_s / c^dag c^dag; antisymmetry gives exact zero modes -> min = 0
                assert np.abs(d["pqg"]["P_min"]).max() < 1e-12 and np.abs(d["pqg"]["Q_min"]).max() < 1e-12
                # G': from the analytic diagonal state through rdm_tools (independent of the sector code)
                for j, tt in enumerate(tp):
                    p = A["probs"]
                    st = np.diag([p[0][j], p[2][j], p[1][j], p[3][j]])        # basis |n_up n_dn>, up = MSB
                    from lindblad_exact.fock import fermion_ops
                    rho, C = rdm_tools.exact_rdms(st, [x.astype(complex) for x in fermion_ops(2)])
                    gm = rdm_tools.min_eigs(rho, C / (-2j))["G"]
                    assert abs(gm - d["pqg"]["Gp_min"][j]) < 1e-12
                di, at = R["inf"]
                Ninf = (a + b) - 2 * a * b
                assert abs(di["N"] - Ninf) < 1e-12 and abs(di["D"][0]) < 1e-12, (di["N"], Ninf)
                assert abs(di["S2"] - 0.75 * Ninf) < 1e-12
                # one zero mode per reachable sector that is not drained: (0,0) [fed by (1,1)], (1,0), (0,1)
                p00, p10, p01, p11 = ((1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b)
                nz_ref = int(p00 > 0 or p11 > 0) + int(p10 > 0) + int(p01 > 0)
                assert at["n_zero_modes"] == nz_ref, (a, b, at["n_zero_modes"], nz_ref)   # reachable sectors only
    print("test_site_analytic ok")


def test_n_inf_zero_generator():
    """S = 0 on the whole reachable space (every mode is a zero mode, no 'next' singular value): the vacuum, a single
    spin, and the single site at gamma = 0 (dimension 4); n_inf must return the initial state."""
    with tempfile.TemporaryDirectory() as tmp:
        for (a, b, g) in [(0.0, 0.0, 1.0), (1.0, 0.0, 3.0), (0.0, 1.0, 0.5), (0.5, 0.5, 0.0), (0.3, 0.8, 0.0), (1.0, 1.0, 0.0)]:
            c = H.card("z", H.site(g), [[a]], [[b]], [{"type": "n_inf", "name": "inf", "observables": ALL},
                       {"type": "transient", "name": "x", "grid": [{"t_end": 5.0, "dt": 0.5}], "observables": ALL}])
            R, ctx = _run(c, tmp)
            di, at = R["inf"]
            A = site_analytic(a, b, g, np.array([np.inf if g > 0 else 0.0]))
            assert at["n_zero_modes"] == at["dim"], (a, b, g, at)
            assert abs(at["trace"] - 1) < 1e-13 and at["residual"] < 1e-13
            assert abs(di["N"] - A["N"][0]) < 1e-13 and abs(di["D"][0] - A["D"][0]) < 1e-13, (a, b, g, di["N"], A["N"])
            assert abs(di["Mz"] - A["Mz"][0]) < 1e-13 and abs(di["S2"] - A["S2"][0]) < 1e-13
            d, _ = R["x"]
            assert np.abs(d["N"] - di["N"]).max() < 1e-13      # nothing evolves
    print("test_n_inf_zero_generator ok")


# ---------------------------------------------------------------- 2. Gaussian sanity
def test_gaussian():
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as tmp:
        for N in (2, 3):
            ru, rd = _rand_rho(N, rng), _rand_rho(N, rng)
            mdl = H.model("rand", np.zeros((N, N)), sites=[0], gamma=1.0)
            c = H.card("g", mdl, ru, rd, [{"type": "n_inf", "name": "i", "observables": ["N"]}])
            ctx, _ = _ctx(c, tmp)
            x0 = ctx.x0
            rho, Gam = ctx.obs.rdm12(x0)
            M = 2 * N
            ref = np.zeros((M, M)); ref[:N, :N] = ru; ref[N:, N:] = rd
            assert np.abs(rho - ref).max() < 1e-13
            C2 = Gam - np.einsum("sa,lb->slab", rho, rho) + np.einsum("la,sb->slab", rho, rho)
            assert np.abs(C2).max() < 1e-13
            # 3RDM <c^dag_a c^dag_b c^dag_c c_n c_l c_s> vs the Wick determinant (C3 = 0)
            X = ctx.even.full_from_vec(x0)
            fm = ctx.fm
            D = H.Dense(mdl)
            assert np.abs(X - D.gaussian(ru, rd)).max() < 1e-13         # independent full-space construction
            c_ = [x.toarray() for x in fm.c]
            idx = [(s, l, n) for s in range(M) for l in range(s + 1, M) for n in range(l + 1, M)]
            for (s, l, n) in idx:
                A3 = c_[n] @ c_[l] @ c_[s]
                for (a, b, cc) in idx:
                    v = np.trace(c_[a].T @ c_[b].T @ c_[cc].T @ A3 @ X)
                    w = np.linalg.det(np.array([[rho[i, j] for j in (a, b, cc)] for i in (s, l, n)]))
                    assert abs(v - w) < 1e-13, (v, w)
    print("test_gaussian ok")


# ---------------------------------------------------------------- 3./4. trace, hermiticity, sector vs full space
def tiny_model(g=1.3):
    h1 = [[0.2, 0.7], [0.7, -0.4]]
    return H.model("tiny", h1, loss=[0.0, 0.6], gain=[0.0, 0.0], sites=[0], gamma=g), \
        H.model("tiny2", [[0.2, 0.7], [0.7, -0.4]], loss=[0.3, 0.6], gain=[0.0, 0.4], sites=[0, 1], gamma=g,
                weights=[1.0, 0.5])


def test_sector_vs_full():
    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory() as tmp:
        for mdl in tiny_model():
            ru, rd = _rand_rho(2, rng), _rand_rho(2, rng)
            times = [0.37, 1.1, 2.5]
            grid = [{"t_end": 0.37, "dt": 0.37}, {"t_end": 1.1, "dt": 0.73}, {"t_end": 2.5, "dt": 1.4}]
            c = H.card("t", mdl, ru, rd, [
                {"type": "transient", "name": "x", "grid": grid, "observables": ["rho1", "D", "S2", "N"]},
                {"type": "transient", "name": "e", "grid": grid, "observables": ["rho1", "D", "S2", "N"], "method": "eig"}])
            R, ctx = _run(c, tmp)
            D = H.Dense(mdl); X0 = D.gaussian(ru, rd)
            fm = ctx.fm; N = fm.N
            for j, tt in enumerate(times):
                X = D.evolve(X0, tt)
                assert abs(np.trace(X) - 1) < 1e-12 and np.abs(X - X.conj().T).max() < 1e-12
                for key in ("x", "e"):
                    d, at = R[key]
                    for s in range(2):
                        for q in range(N):
                            for a in range(N):
                                ref = np.trace(fm.cd[s * N + a].toarray() @ fm.c[s * N + q].toarray() @ X)
                                assert abs(d["rho1"][j + 1, s, q, a] - ref) < 1e-12
                    S2 = np.trace(fm.S2_op().toarray() @ X).real
                    assert abs(d["S2"][j + 1] - S2) < 1e-12
                    assert at["max_trace_err"] < 1e-12
            # hermiticity of the propagated sector state
            import scipy.sparse.linalg as spla
            x = spla.expm_multiply(ctx.S(ctx.even) * 1.7, ctx.x0)
            Xs = ctx.even.full_from_vec(x)
            assert np.abs(Xs - Xs.conj().T).max() < 1e-13
            assert np.abs(Xs - D.evolve(X0, 1.7)).max() < 1e-12
    print("test_sector_vs_full ok")


def test_ness_and_GR_vs_full():
    """model with one-body loss and gain (unique NESS): NESS (lu, propagate) and G^R vs the dense full space"""
    with tempfile.TemporaryDirectory() as tmp:
        mdl = tiny_model()[1]
        ru = np.diag([0.0, 1.0]); rd = np.diag([0.0, 1.0])
        tasks = [{"type": "ness", "name": "ss", "method": "lu", "observables": ["rho1", "D"], "pqg": True,
                  "spectrum": {"k": 4}},
                 {"type": "greens_R", "name": "GR", "state": "ness", "orbital": 0, "spins": ["up", "dn"],
                  "tau_max": 6.0, "dtau": 0.5,
                  "resolvent": {"omega_min": -3.0, "omega_max": 3.0, "n": 7, "solver": "eig"}},
                 {"type": "ness", "name": "ss2", "method": "propagate",
                  "propagate": {"block": 20.0, "tol": 1e-14, "t_max": 2000.0}, "observables": ["rho1", "D"]}]
        R, ctx = _run(H.card("n", mdl, ru, rd, tasks), tmp)
        D = H.Dense(mdl); d = D.d; fm = ctx.fm
        w, V = np.linalg.eig(D.L)
        k0 = np.argmin(np.abs(w)); Xss = V[:, k0].reshape(d, d, order="F"); Xss = Xss / np.trace(Xss)
        assert np.sort(np.abs(w))[1] > 1e-6                                 # unique NESS
        ds, at = R["ss"]
        for s in range(2):
            ref = np.trace(fm.cd[s * 2].toarray() @ fm.c[s * 2].toarray() @ Xss)
            assert abs(ds["rho1"][s, 0, 0] - ref) < 1e-12
            assert abs(R["ss2"][0]["rho1"][s, 0, 0] - ref) < 1e-12
        assert at["residual"] < 1e-13 and at["n_zero_modes"] == 1
        # G^R by the dense odd Liouvillian
        dg, ag = R["GR"]
        for k, s in enumerate((0, 2)):
            cs, cds = fm.c[s].toarray(), fm.cd[s].toarray()
            Y = cds @ Xss + Xss @ cds
            for j, tau in enumerate(dg["tau"]):
                Yt = (sla.expm(D.Lodd * tau) @ Y.reshape(-1, order="F")).reshape(d, d, order="F")
                assert abs(dg["GR"][j, k] - (-1j) * np.trace(cs @ Yt)) < 1e-12
            assert abs(dg["GR"][0, k] + 1j) < 1e-13                            # G^R(0+) = -i
            # resolvent vs dense solve
            for q, om in enumerate(dg["resolvent"]["omega"]):
                v = np.linalg.solve(D.Lodd + 1j * om * np.eye(d * d), Y.reshape(-1, order="F"))
                ref = 1j * np.trace(cs @ v.reshape(d, d, order="F"))
                assert abs(dg["resolvent"]["GR"][q, k] - ref) < 1e-11
    print("test_ness_and_GR_vs_full ok")


def test_resolvent_vs_time():
    """toy-B-like model: resolvent (eig and gmres) vs a Fourier transform of G^R(tau)"""
    with tempfile.TemporaryDirectory() as tmp:
        mdl = H.toyB(2.0)
        ru, rd = H.imp_ic(3, [0, 0, 0.3], "00")
        res = {"omega_min": -2.0, "omega_max": 2.0, "n": 5}
        tasks = [{"type": "ness", "name": "ss", "method": "lu", "observables": ["N"]},
                 {"type": "greens_R", "name": "A", "state": "ness", "orbital": 0, "spins": ["up"], "tau_max": 300.0,
                  "dtau": 0.01, "resolvent": dict(res, solver="eig")},
                 {"type": "greens_R", "name": "B", "state": "ness", "orbital": 0, "spins": ["up"], "tau_max": 0.5,
                  "dtau": 0.5, "resolvent": dict(res, solver="gmres", rtol=1e-13)}]
        R, _ = _run(H.card("r", mdl, ru, rd, tasks), tmp)
        dA = R["A"][0]; dB = R["B"][0]
        assert np.abs(dA["resolvent"]["GR"] - dB["resolvent"]["GR"]).max() < 1e-10
        tau, G = dA["tau"], dA["GR"][:, 0]
        assert abs(G[-1]) < 1e-8                                                # decayed
        for q, om in enumerate(dA["resolvent"]["omega"]):
            f = np.exp(1j * om * tau) * G
            h = tau[1] - tau[0]
            fp0 = (-3 * f[0] + 4 * f[1] - f[2]) / (2 * h)                      # f'(0), f'(tau_max) ~ 0
            ft = h * (f.sum() - 0.5 * (f[0] + f[-1])) + h ** 2 / 12 * fp0       # trapezoid + Euler-Maclaurin
            assert abs(ft - dA["resolvent"]["GR"][q, 0]) < 1e-6, (om, ft, dA["resolvent"]["GR"][q, 0])
    print("test_resolvent_vs_time ok")


# ---------------------------------------------------------------- 6. card validation and CLI
def test_cli():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        c = H.card("cli_ok", H.site(1.0), [[0.5]], [[0.5]],
                   [{"type": "transient", "name": "tr", "grid": [{"t_end": 1.0, "dt": 0.1}], "observables": ["N"]}])
        H.write(tmp / "ok.json", c)
        p = subprocess.run([sys.executable, "-I", str(H.RUN), str(tmp / "ok.json"), str(tmp / "o1")],
                           capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        with h5py.File(tmp / "o1" / "result.h5") as f:
            assert f.attrs["schema"] == "exact_result/1" and f.attrs["status"] == "Ok"
            assert f.attrs["run_id"] == "cli_ok" and "+" in f.attrs["release_version"]
            assert json.loads(f.attrs["card_json"]) == c
            assert f["model"].attrs["dim_fock"] == 4 and f["tr/t"].shape == (11,)
        assert (tmp / "o1" / "progress.txt").read_text().startswith("cli_ok | Ok")
        bad = json.loads(json.dumps(c)); bad["tasks"][0]["typo"] = 1
        H.write(tmp / "bad.json", bad)
        p = subprocess.run([sys.executable, "-I", str(H.RUN), str(tmp / "bad.json"), str(tmp / "o2")],
                           capture_output=True, text=True)
        assert p.returncode == 2 and "unknown key 'typo'" in p.stderr
        with h5py.File(tmp / "o2" / "result.h5") as f:
            assert f.attrs["status"] == "Error"
        for mut in (lambda d: d["tasks"][0]["grid"].__setitem__(0, {"t_end": 1.0, "dt": 0.3}),
                    lambda d: d["initial_state"].__setitem__("rho1_up", [[1.5]]),
                    lambda d: d["model"].pop("h1"),
                    lambda d: d.__setitem__("schema", "exact_card/2"),
                    lambda d: d["tasks"].append({"type": "greens_R", "name": "g", "state": "ness", "orbital": 0,
                                                 "spins": ["up"], "tau_max": 1.0, "dtau": 0.1})):
            b = json.loads(json.dumps(c)); mut(b)
            try:
                parse(json.dumps(b))
            except Exception as e:
                assert type(e).__name__ == "CardError"
            else:
                raise AssertionError("invalid card accepted")
    print("test_cli ok")


if __name__ == "__main__":
    test_site_analytic()
    test_n_inf_zero_generator()
    test_gaussian()
    test_sector_vs_full()
    test_ness_and_GR_vs_full()
    test_resolvent_vs_time()
    test_cli()
    print("ALL PASSED")
