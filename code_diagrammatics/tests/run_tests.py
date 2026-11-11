"""Step-1 tests of dp5drive (HF end to end: model pipeline, HF G1-G2 and RTDE against solve_ivp, output schema,
bad input files). Run from anywhere:

    python3 -I tests/run_tests.py            (after cmake --build build)

Uses at most 4 OpenMP threads. Work files go to tests/work/ (recreated). The development suite's restart, rtde_frozen,
paramagnetic-layout and legacy-binary checks test removed features and are not included.
"""
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))
import dp5model  # noqa: E402
import read_output  # noqa: E402
import runcard  # noqa: E402

EXE = os.environ.get("DP5_EXE", os.path.join(ROOT, "build", "dp5drive"))
WORK = os.path.join(HERE, "work")
ENV = dict(os.environ, OMP_NUM_THREADS="4")
FAIL = []


def check(cond, msg):
    print(("   ok   " if cond else "   FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def run(rc, out, *extra, expect=0):
    p = subprocess.run([EXE, "--runcard", rc, "--output", out, *extra], cwd=WORK, env=ENV, capture_output=True, text=True)
    if p.returncode != expect:
        print(p.stdout[-2000:], p.stderr[-2000:])
    return p


def rho_series(path):
    with h5py.File(path, "r") as f:
        return f["time/t"][()], f["rho/rho"][()]


# ---------------------------------------------------------------- models
def synthetic_impurity(path, N=7, seed=1):
    """impurity + (N-1)/2 greater (lossy) + (N-1)/2 lesser (filled) bath modes, spin independent, impurity empty"""
    rng = np.random.default_rng(seed)
    n = (N - 1) // 2
    H = np.zeros((N, N), complex)
    H[0, 0] = -1.0
    H[0, 1:] = 0.4 * rng.normal(size=N - 1)
    H[1:, 0] = H[0, 1:]
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    B = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H[1:1 + n, 1:1 + n] = 0.3 * (A + A.conj().T)
    H[1 + n:, 1 + n:] = 0.3 * (B + B.conj().T)
    G = np.zeros((N, N), complex)
    G[1:1 + n, 1:1 + n] = np.diag(0.3 + 0.4 * rng.random(n))
    L = np.zeros((N, N), complex)
    L[1 + n:, 1 + n:] = np.diag(0.3 + 0.4 * rng.random(n))
    rho = np.zeros((2, N, N), complex)
    rho[:, 1 + n:, 1 + n:] = np.eye(N - 1 - n)
    hsh = dp5model.write_model(path, np.stack([H, H]), np.stack([G, G]), np.stack([L, L]), rho, [0], [1.0],
                               dict(N_greater=n, N_lesser=N - 1 - n, ordering="d,p>,p<"), dict(generator="run_tests.py"))
    return H, G, L, hsh


def rc_imp(rc, m, gamma, t_end, dt_out, occ, **kw):
    """run card on imp.h5 with the impurity occupation (n_up, n_dn) set in rho0 (ic_mode full)"""
    rho0 = m["rho"].copy()
    rho0[0, 0, 0], rho0[1, 0, 0] = occ
    return runcard.write_runcard(rc, os.path.join(WORK, "imp.h5"), gamma, t_end, dt_out, rho0=rho0, ic_mode="full", **kw)


def ring_model(path, Ns=4, seed=3):
    rng = np.random.default_rng(seed)
    h = np.zeros((2, Ns, Ns), complex)
    for s, zeeman in enumerate((+0.2, -0.2)):
        for r in range(Ns):
            h[s, r, (r + 1) % Ns] = h[s, (r + 1) % Ns, r] = -1.0
            h[s, r, r] = 0.1 * rng.normal() + zeeman
    rho = np.zeros((2, Ns, Ns), complex)
    for s in range(2):  # random spin-broken pure-ish state: U diag(n) U^dag
        X = rng.normal(size=(Ns, Ns)) + 1j * rng.normal(size=(Ns, Ns))
        U, _ = np.linalg.qr(X)
        nocc = [0.9, 0.7, 0.2, 0.05] if s == 0 else [0.8, 0.1, 0.1, 0.0]
        rho[s] = U @ np.diag(nocc) @ U.conj().T
    z = np.zeros_like(h)
    return dp5model.write_model(path, h, z, z, rho, list(range(Ns)), None,
                                dict(ordering="ring", units="J = 1"), dict(generator="run_tests.py ring"))


# ---------------------------------------------------------------- references
def hf_rhs_factory(m, gamma, ns_full=True, n_gr=0, cols=()):
    h, gt, lt = m["h"], m["ell_greater"], m["ell_lesser"]
    N = h.shape[1]
    hb = h - 1j * (gt + lt)
    sites, w = m["lossy_sites"], m["site_weight"]

    def ho_of(rho):
        hos = []
        for s in range(2):
            ho = hb[s].copy()
            opp = rho[1 - s]
            for r, ww in zip(sites, w):
                ho[r, r] -= 1j * gamma * ww * opp[r, r].real
            hos.append(ho)
        return hos

    def f(t, y):
        rho = y[:2 * N * N].reshape(2, N, N)
        hos = ho_of(rho)
        d = np.empty_like(rho)
        for s in range(2):
            d[s] = -1j * hos[s] @ rho[s] + 1j * rho[s] @ hos[s].conj().T + 2 * lt[s]
        out = [d.ravel()]
        if n_gr:
            g = y[2 * N * N:].reshape(n_gr, 2, N)
            out.append(np.stack([[-1j * hos[s] @ g[c, s] for s in range(2)] for c in range(n_gr)]).ravel())
        return np.concatenate(out)
    return f


def hf_reference(m, gamma, rho0, t_eval):
    f = hf_rhs_factory(m, gamma)
    sol = solve_ivp(f, (t_eval[0], t_eval[-1]), rho0.ravel().astype(complex), method="DOP853", rtol=1e-12,
                    atol=1e-14, t_eval=t_eval)
    N = rho0.shape[1]
    return sol.y.T.reshape(len(t_eval), 2, N, N)


# ---------------------------------------------------------------- tests
def test_pipeline():
    print("[1] pipeline: model writer, validator, hashes")
    H, G, L, hsh = synthetic_impurity(os.path.join(WORK, "imp.h5"))
    m = dp5model.load_model(os.path.join(WORK, "imp.h5"))
    N = H.shape[0]
    check(np.array_equal(m["h"][0], H) and np.array_equal(m["h"][1], H) and m["hash"] == hsh and
          m["hash"] == dp5model.content_hash(m["h"], m["ell_greater"], m["ell_lesser"], m["rho"], m["lossy_sites"], m["site_weight"]),
          "model written and read back, content hash reproducible")
    good = dict(h=m["h"], ell_greater=m["ell_greater"], ell_lesser=m["ell_lesser"], rho=m["rho"], lossy_sites=[0],
                site_weight=[1.0])
    bad_cases = {
        "non-Hermitian h": dict(good, h=m["h"] + np.triu(np.ones((N, N)), 1)[None] * 0.1),
        "ell_greater not PSD": dict(good, ell_greater=m["ell_greater"] - 2 * np.eye(N)[None]),
        "rho spectrum > 1": dict(good, rho=m["rho"] * 1.5),
        "shape mismatch": dict(good, ell_lesser=m["ell_lesser"][:, :-1, :-1]),
        "lossy site out of range": dict(good, lossy_sites=[N]),
    }
    for name, kw in bad_cases.items():
        try:
            dp5model.validate(kw["h"], kw["ell_greater"], kw["ell_lesser"], kw["rho"], kw["lossy_sites"], kw["site_weight"])
            check(False, f"validator rejects {name}")
        except dp5model.ValidationError:
            check(True, f"validator rejects {name}")
    try:
        dp5model.validate(**good, N_greater=2, N_lesser=2, ordering="d,p>,p<")
        check(False, "validator rejects ordering inconsistent with N_greater/N_lesser")
    except dp5model.ValidationError:
        check(True, "validator rejects ordering inconsistent with N_greater/N_lesser")
    # C++ side: tampered model and wrong run-card hash
    shutil.copy(os.path.join(WORK, "imp.h5"), os.path.join(WORK, "tampered.h5"))
    with h5py.File(os.path.join(WORK, "tampered.h5"), "a") as f:
        x = f["model/h"][()]
        x[0, 0, 0] += 1e-9
        f["model/h"][...] = x
    runcard.write_runcard(os.path.join(WORK, "rc_t.h5"), os.path.join(WORK, "tampered.h5"), 1.0, 1.0, 0.1)
    p = run("rc_t.h5", "t.h5", expect=2)
    check(p.returncode == 2 and "hash mismatch" in p.stderr, "dp5drive refuses a model whose content hash does not match")
    return m


def test_hf_g1g2(m_imp):
    print("[2] HF G1-G2 vs solve_ivp (DOP853, rtol 1e-12); impurity and 4-site ring; both vertex implementations")
    t = np.linspace(0, 6, 61)
    res = {}
    for label, occ in (("spin-broken", (1.0, 0.0)), ("symmetric-start", (0.7, 0.7))):
        rho0 = m_imp["rho"].copy()
        rho0[0, 0, 0], rho0[1, 0, 0] = occ
        ref = hf_reference(m_imp, 2.0, rho0, t)
        for vi in ("impurity", "general"):
            rc = os.path.join(WORK, f"rc_imp_{vi}_{label}.h5")
            rc_imp(rc, m_imp, 2.0, 6.0, 0.1, occ, vertex_impl=vi)
            out = f"imp_{vi}_{label}.h5"
            run(rc, out)
            tt, rho = rho_series(os.path.join(WORK, out))
            err = np.max(np.abs(rho - ref))
            res[(vi, label)] = rho
            check(np.allclose(tt, t, atol=1e-14) and err < 2e-6, f"impurity {label}, vertex {vi}: max|rho - ref| = {err:.2e}")
        d = np.max(np.abs(res[("impurity", label)] - res[("general", label)]))
        check(d == 0.0, f"[3] general vertex with sites {{0}} == impurity fast path: max diff {d:.1e} (bitwise expected)")
    # ring
    ring_model(os.path.join(WORK, "ring.h5"))
    mr = dp5model.load_model(os.path.join(WORK, "ring.h5"))
    ref = hf_reference(mr, 1.0, mr["rho"], t)
    rc = os.path.join(WORK, "rc_ring.h5")
    runcard.write_runcard(rc, os.path.join(WORK, "ring.h5"), 1.0, 6.0, 0.1)
    run(rc, "ring_out.h5")
    _, rho = rho_series(os.path.join(WORK, "ring_out.h5"))
    err = np.max(np.abs(rho - ref))
    check(err < 2e-6, f"4-site ring, on-site loss on all sites, spin-broken start: max|rho - ref| = {err:.2e}")
    rc = os.path.join(WORK, "rc_ring_imp.h5")
    runcard.write_runcard(rc, os.path.join(WORK, "ring.h5"), 1.0, 6.0, 0.1, vertex_impl="impurity")
    p = run(rc, "ring_imp.h5", expect=2)
    check(p.returncode == 2 and "lossy_sites" in p.stderr, "impurity fast path refuses a model with lossy sites != {0}")


def test_rtde(m_imp):
    print("[4] HF RTDE")
    N = m_imp["h"].shape[1]
    # (a) gamma = 0 against exp(-i h_o (t - t')) e_s, several columns (one run per probe column)
    cols = [0, 2, 5]
    ho = m_imp["h"] - 1j * (m_imp["ell_greater"] + m_imp["ell_lesser"])
    err, t0ok = 0.0, True
    for s in cols:
        rc = os.path.join(WORK, f"rc_g0_{s}.h5")
        rc_imp(rc, m_imp, 0.0, 8.0, 0.05, (1.0, 0.0), mode="rtde", t_prime=[2.0], probe_columns=[s])
        run(rc, f"g0_{s}.h5")
        with h5py.File(os.path.join(WORK, f"g0_{s}.h5"), "r") as f:
            tt, GR = f["rtde/tp=0/t"][()], f["rtde/tp=0/GR"][()]
        t0ok = t0ok and abs(tt[0] - 2.0) < 1e-14
        for k, t in enumerate(tt):
            for sp in range(2):
                ex = -1j * expm(-1j * ho[sp] * (t - 2.0))[:, s]
                err = max(err, np.max(np.abs(GR[k, 0, sp] - ex)))
    check(t0ok and err < 1e-7, f"gamma=0: G^R vs -i exp(-i h_o (t-t')) e_s for columns {cols}: max err {err:.2e} ({len(tt)} frames)")
    # (b) gamma > 0, co-propagated, columns 0 and 3, against an augmented solve_ivp reference
    cols = (0, 3)
    GR = []
    for s in cols:
        rc = os.path.join(WORK, f"rc_g1_{s}.h5")
        rc_imp(rc, m_imp, 1.5, 6.0, 0.05, (1.0, 0.0), mode="rtde", t_prime=[1.5], probe_columns=[s])
        run(rc, f"g1_{s}.h5")
        with h5py.File(os.path.join(WORK, f"g1_{s}.h5"), "r") as f:
            tt = f["rtde/tp=0/t"][()]
            GR.append(f["rtde/tp=0/GR"][()][:, 0])
            rho_tp = f["checkpoints/tp0/y"][()][:2 * N * N].reshape(2, N, N)
    GR = np.stack(GR, axis=1)  # [T, column, spin, x]
    y0 = np.concatenate([rho_tp.ravel(), np.zeros(2 * 2 * N, complex)])
    g0 = y0[2 * N * N:].reshape(2, 2, N)
    for c, s in enumerate(cols):
        g0[c, :, s] = -1j
    y0[2 * N * N:] = g0.ravel()
    sol = solve_ivp(hf_rhs_factory(m_imp, 1.5, n_gr=2), (1.5, 6.0), y0, method="DOP853", rtol=1e-12, atol=1e-14,
                    t_eval=tt)
    ref = sol.y.T[:, 2 * N * N:].reshape(len(tt), 2, 2, N)
    err = np.max(np.abs(GR - ref))
    check(err < 2e-6, f"gamma=1.5, co-propagated rho and G^R, columns [0, 3]: max err {err:.2e}")


def test_output():
    print("[6] output schema and reader")
    m_imp = dp5model.load_model(os.path.join(WORK, "imp.h5"))
    rc_imp(os.path.join(WORK, "rc_rs.h5"), m_imp, 2.0, 10.0, 0.1, (1.0, 0.0), mode="rtde", t_prime=[5.0], probe_columns=[2],
           checkpoint_interval=2.0)
    run(os.path.join(WORK, "rc_rs.h5"), "rs_A.h5")
    path = os.path.join(WORK, "rs_A.h5")
    with h5py.File(path, "r") as f:
        need = ["input/run", "time/t", "rho/rho", "observables/n_d", "observables/N", "observables/rho_dot_max",
                "rtde/tp=0/t", "rtde/tp=0/GR", "checkpoints/tp0/y", "checkpoints/periodic_1/y", "final/y",
                "profile/time/integrate_seconds", "profile/steps/accepted", "profile/memory/register_bytes"]
        missing = [k for k in need if k not in f]
        check(not missing, f"output has all groups/datasets (missing: {missing})")
        check(f.attrs["dp5_driver_output_schema"] == 1 and f["profile"].attrs["schema_version"] == 1,
              "schema versions present (output 1, profile 1)")
        check(f["status"].attrs["status"] == "Ok" and "git_describe" in f["provenance"].attrs and
              "release_version" in f.attrs and
              "model_hash" in f["input"].attrs and "cli_overrides" in f["input"].attrs,
              "status, provenance (git hash, flags, host), model hash and CLI overrides recorded")
        st = f["status"].attrs
        check(all(k in st for k in ("diag_max_dN_dt", "diag_max_two_body_dN_dt", "diag_min_D", "diag_min_rho_eig",
                                    "diag_max_rho_eig", "diag_warnings")) and "rho_eig_min" in f["observables"],
              f"diagnostics in /status (min D {st['diag_min_D']:.3e}, eig rho in [{st['diag_min_rho_eig']:.2e}, "
              f"{st['diag_max_rho_eig']:.3f}], warnings: {st['diag_warnings']}) and /observables/rho_eig_min|max")
        check(f["rho/rho"].dtype == np.complex128 and f["rho/rho"].chunks is not None,
              "complex data in h5py {r,i} convention, chunked extendible datasets")
    rc = os.path.join(WORK, "rc_thin.h5")
    rc_imp(rc, m_imp, 2.0, 6.0, 0.1, (1.0, 0.0), rho_stride=7)
    run(rc, "thin.h5")
    with h5py.File(os.path.join(WORK, "thin.h5"), "r") as f, \
            h5py.File(os.path.join(WORK, "imp_general_spin-broken.h5"), "r") as g:
        nf, nr = f["time/t"].shape[0], f["rho/rho"].shape[0]
        same = np.array_equal(f["rho/rho"][()], g["rho/rho"][()][::7]) and np.array_equal(f["rho/t"][()], g["time/t"][()][::7])
        obs_same = np.array_equal(f["observables/n_d"][()], g["observables/n_d"][()])
    check(nf == 61 and nr == 9 and same and obs_same,
          f"rho thinning (rho_stride = 7): {nr} rho frames of {nf}, equal to every 7th frame of the full run; observables on all frames")
    ok = read_output.main(path)
    check(ok, "python reader prints status + profile summary; profile export consistent")
    p = run(os.path.join(WORK, "rc_rs.h5"), "rs_A.h5", expect=2)
    check(p.returncode == 2 and "cannot create output" in p.stderr, "existing output file is never overwritten (fail fast)")
    rc = os.path.join(WORK, "rc_tppx.h5")
    runcard.write_runcard(rc, os.path.join(WORK, "imp.h5"), 1.0, 1.0, 0.1, approximation="DSL", mode="rtde",
                          t_prime=[0.5])
    p = run(rc, "dsl_rtde.h5", expect=0)
    with h5py.File(os.path.join(WORK, "dsl_rtde.h5"), "r") as f:
        st = f["status"].attrs["status"]
    check(p.returncode == 0 and st == "Ok", "every RTDE is implemented: a DSL RTDE run ends with status Ok (was NotImplemented)")
    for a in ("TPPX", "mGW", "DSL"):
        for sp in ("full",):
            rc = os.path.join(WORK, f"rc_lay_{a}_{sp}.h5")
            runcard.write_runcard(rc, os.path.join(WORK, "imp.h5"), 1.0, 1.0, 0.1, approximation=a, spin_mode=sp)
            q = subprocess.run([EXE, "--runcard", rc, "--print-layout"], capture_output=True, text=True, env=ENV)
            print("      " + q.stdout.strip().replace("\n", "\n      "))


def test_bad_files():
    print("[8] missing / unreadable / non-HDF5 input files: clean error, exit code 2, no crash")
    imp = os.path.join(WORK, "imp.h5")
    bad = os.path.join(WORK, "bad"); os.makedirs(bad, exist_ok=True)
    txt = os.path.join(bad, "not_hdf5.h5")
    open(txt, "w").write("this is not an HDF5 file\n")
    noread = os.path.join(bad, "noread.h5")
    shutil.copy(imp, noread); os.chmod(noread, 0)
    root = hasattr(os, "geteuid") and os.geteuid() == 0
    def rc_with_model(name, path):
        rc = os.path.join(bad, name)
        runcard.write_runcard(rc, imp, 1.0, 1.0, 0.1)
        with h5py.File(rc, "a") as f:
            f["run"].attrs["model_path"] = path
        return rc
    cases = [("model path does not exist", rc_with_model("rc_missing.h5", os.path.join(bad, "nope.h5")), [], "does not exist"),
             ("model file not HDF5", rc_with_model("rc_txt.h5", txt), [], "is not an HDF5 file"),
             ("model file is a directory", rc_with_model("rc_dir.h5", bad), [], "is not a regular file"),
             ("model path is an HDF5 file but not a model", rc_with_model("rc_wrong.h5", os.path.join(WORK, "rc_rs.h5")), [],
              "missing attribute 'dp5_model_schema'"),
             ("run card does not exist", os.path.join(bad, "no_rc.h5"), [], "does not exist"),
             ("run card not HDF5", txt, [], "is not an HDF5 file")]
    if not root:
        cases.insert(2, ("model file not readable", rc_with_model("rc_noread.h5", noread), [], "not readable"))
    for k, (label, rc, extra, msg) in enumerate(cases):
        out = os.path.join(bad, f"out_{k}.h5")
        p = subprocess.run([EXE, "--runcard", rc, "--output", out, *extra], cwd=WORK, env=ENV, capture_output=True, text=True)
        err = p.stderr.strip().splitlines()
        clean = p.returncode == 2 and any(msg in l for l in err) and "terminate" not in p.stderr and len(err) <= 3
        check(clean, f"{label}: exit {p.returncode}, message: {err[-1] if err else ''}")
    if root:
        print("   (running as root: the unreadable-file case is skipped)")
    os.chmod(noread, 0o644)


if __name__ == "__main__":
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK)
    m = test_pipeline()
    test_hf_g1g2(m)
    test_rtde(m)
    test_output()
    test_bad_files()
    print(f"\n{'ALL DRIVER TESTS PASSED' if not FAIL else 'FAILURES: ' + str(len(FAIL))}")
    sys.exit(1 if FAIL else 0)
