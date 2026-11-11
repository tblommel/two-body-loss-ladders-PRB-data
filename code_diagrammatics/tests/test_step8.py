"""Step-8 tests (trimmed for the release): automatic steady-state detection (run card t_prime = "auto").

    python3 -I tests/test_step8.py [test names]

[1] lambda_ref against a direct diagonalization and the timeout paths, [2] an auto-detected t' gives the same state
and G^R bitwise as a fixed-t' run at that time, [7] default vs tight tolerances against the long-time state.
The development suite's checks of removed options (second t', freeze, Arnoldi, --tprime, restart) and its
Kondo-bath and PRB1-data cases are not part of the release. Uses at most 4 threads. Writes tests/work8/.
"""
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_step2 as T2  # noqa: E402

EXE = T2.EXE
WORK = os.path.join(HERE, "work8")
T2.WORK = WORK
ENV = dict(os.environ, OMP_NUM_THREADS="4")
FAIL = []
IMP = os.path.join(WORK, "imp7.h5")  # generated impurity model (N = 7: impurity + 3 + 3 bath orbitals)
TIGHT = dict(tol_rho=[1e-13, 1e-11], tol_tensor=[1e-13, 1e-11], tol_rtde=[1e-13, 1e-11])


def check(cond, msg):
    print(("   ok   " if cond else "   FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def run(rc, out, *extra):
    p = os.path.join(WORK, out)
    if os.path.exists(p):
        os.remove(p)
    return subprocess.run([EXE, "--runcard", rc, "--output", out, *extra], cwd=WORK, env=ENV, capture_output=True, text=True)


def ss(path):
    with h5py.File(os.path.join(WORK, path), "r") as f:
        return dict(f["steady_state"].attrs) if "steady_state" in f else {}


def lam_ref_direct(m, rho, gamma, mf=True):
    """min |Im eig h_o^MF| over spins; h_o^MF = h - i(l^> + l^<) - i gamma w_r rho^{s-bar}_rr on lossy sites"""
    lam, re = np.inf, []
    for s in range(2):
        ho = m["h"][s] - 1j * (m["ell_greater"][s] + m["ell_lesser"][s])
        if mf:
            for r, w in zip(m["lossy_sites"], m["site_weight"]):
                ho[r, r] -= 1j * gamma * w * rho[1 - s][r, r].real
        e = np.linalg.eigvals(ho)
        lam = min(lam, np.abs(e.imag).min())
        re += list(e.real)
    return lam, max(re) - min(re)


# ---------------------------------------------------------------- tests
def test_lambda_ref_and_timeout():
    print("[1] lambda_ref vs direct diagonalization; timeout path (fail / start)")
    m = T2.impurity_model(IMP, 7)
    cases = (("impurity N=7, gamma=0", 0.0, "model", None), ("impurity N=7, gamma=2, n_d=(1,0)", 2.0, "full", [1.0, 0.0]))
    for k, (label, g, ic, occ) in enumerate(cases):
        kw = dict(mode="rtde", t_prime="auto", rtde_tau=1.0, vertex_impl="impurity", c_min=0.01, c_max=0.02, ic_mode=ic)
        rho0 = None
        if occ is not None:
            rho0 = m["rho"].copy()
            rho0[0][0, 0], rho0[1][0, 0] = occ
        rc = T2.write_rc(f"rc_lam{k}.h5", IMP, g, "HF", rho0=rho0, **kw)
        p = run(rc, f"lam{k}.h5")
        a = ss(f"lam{k}.h5")
        rho = m["rho"].copy()
        if occ is not None:
            rho[0][0, 0], rho[1][0, 0] = occ
        lmf, spread = lam_ref_direct(m, rho, g, True)
        lb, _ = lam_ref_direct(m, rho, g, False)
        e = max(abs(a["lambda_ref_t0"] - lmf), abs(a["lambda_ref_bare_t0"] - lb), abs(a["re_spread_t0"] - spread))
        check(e < 1e-10, f"{label}: lambda_ref(t0) MF {a['lambda_ref_t0']:.6f} (direct {lmf:.6f}), bare {a['lambda_ref_bare_t0']:.6f} "
                         f"(direct {lb:.6f}), Re spread {a['re_spread_t0']:.3f}: {e:.1e}")
        if k == 0:
            with h5py.File(os.path.join(WORK, "lam0.h5"), "r") as f:
                st = f["status"].attrs["status"]
            check(p.returncode == 6 and st == "NoSteadyState" and "no stationary state found" in p.stderr and a["timeout"] == 1.0,
                  f"timeout (ss_on_timeout = fail): exit {p.returncode}, status {st}, message: {p.stderr.strip()[:110]}...")
    rc = T2.write_rc("rc_start.h5", IMP, 0.0, "HF", mode="rtde", t_prime="auto", rtde_tau=1.0, vertex_impl="impurity",
                     c_min=0.01, c_max=0.02, ss_on_timeout="start", dt_out=0.1)
    p = run(rc, "start.h5")
    a = ss("start.h5")
    with h5py.File(os.path.join(WORK, "start.h5"), "r") as f:
        st, nfr = f["status"].attrs["status"], f["rtde/tp=0/t"].shape[0]
    check(p.returncode == 0 and st == "Ok" and "warning" in a and nfr == 11,
          f"timeout (ss_on_timeout = start): exit {p.returncode}, status {st}, RTDE opened at t = {a['t_timeout']:.2f}, {nfr} frames, "
          f"warning attribute set")


def small_model():
    return T2.impurity_model(os.path.join(WORK, "imp5.h5"), 5)


def auto_rc(name, ap="TPPX", **kw):
    base = dict(mode="rtde", t_prime="auto", rtde_tau=4.0, dt_out=0.1, probe_columns=[2], **TIGHT)
    base.update(kw)
    return T2.write_rc(name, os.path.join(WORK, "imp5.h5"), 1.0, ap, **base)


def test_auto_vs_fixed():
    print("[2] auto trigger (impurity N=5, TPPX, gamma=1) == fixed-t' run at the detected time, bitwise")
    small_model()
    p = run(auto_rc("rc_auto.h5"), "auto.h5")
    a = ss("auto.h5")
    print("      " + [l for l in p.stdout.splitlines() if "steady state detected" in l][0])
    tp = a["t_prime"]
    check(p.returncode == 0 and a["timeout"] == 0.0 and a["max_ratio_window"] < 1e-6,
          f"detected t' = {tp:.4f} (lambda_ref(t0) {a['lambda_ref_t0']:.4f}, lambda_ref(t') {a['lambda_ref_t']:.4f}, lambda_fit "
          f"{a['lambda_fit']:.4f}, lambda_used {a['lambda_used']:.4f}; window max (drift + noise)/lambda_used {a['max_ratio_window']:.1e}; "
          f"drift window Delta {a['delta']:.2f}; per-block drift {np.array2string(a['drift_at_tprime'], precision=2)}, noise "
          f"{np.array2string(a['noise_at_tprime'], precision=2)})")
    tend = tp + 4.0
    fx = T2.write_rc("rc_fixed.h5", os.path.join(WORK, "imp5.h5"), 1.0, "TPPX", mode="rtde", t_prime=[tp], probe_columns=[2],
                     t_end=tend, dt_out=tend / 400, **TIGHT)
    run(fx, "fixed.h5")
    with h5py.File(os.path.join(WORK, "auto.h5"), "r") as A, h5py.File(os.path.join(WORK, "fixed.h5"), "r") as B:
        s0 = np.array_equal(A["checkpoints/tp0/y"][()], B["checkpoints/tp0/y"][()])
        s1 = np.array_equal(A["final/y"][()], B["final/y"][()])
        dmax = np.abs(A["final/y"][()] - B["final/y"][()]).max()
        tA, tB = A["final"].attrs["t"], B["final"].attrs["t"]
    check(s0 and s1 and tA == tB, f"start state at t' bitwise equal: {s0}; final state (incl. G^R) at t' + 4 bitwise equal: {s1} "
                                  f"(max diff {dmax:.1e})")


def test_default_tolerances():
    print("[7] impurity N = 7, HF, gamma = 0: DEFAULT tolerances vs tight tolerances")
    if not os.path.exists(IMP):
        T2.impurity_model(IMP, 7)
    res = {}
    for tag, extra in (("default", {}), ("tight", dict(tol_rho=[1e-13, 1e-11]))):
        rc = T2.write_rc(f"rc_n25_{tag}.h5", IMP, 0.0, "HF", mode="rtde", t_prime="auto", rtde_tau=10.0, dt_out=0.1,
                         vertex_impl="impurity", **extra)
        p = run(rc, f"n25_{tag}.h5")
        a = ss(f"n25_{tag}.h5")
        print("      " + tag + ": " + [l for l in p.stdout.splitlines() if "steady state" in l][0])
        with h5py.File(os.path.join(WORK, f"n25_{tag}.h5"), "r") as f:
            res[tag] = (a, f["checkpoints/tp0/y"][()])
        check(p.returncode == 0 and a["timeout"] == 0.0, f"{tag} tolerances: detected t' = {a['t_prime']:.1f}, Delta = {a['delta']:.2f}, "
                                                         f"lambda_fit {a['lambda_fit']:.4f}, lambda_used {a['lambda_used']:.4f}")
    rcl = T2.write_rc("rc_n25_long.h5", IMP, 0.0, "HF", mode="g1g2", t_end=800.0, dt_out=800.0, vertex_impl="impurity",
                      tol_rho=[1e-13, 1e-11])
    run(rcl, "n25_long.h5")
    with h5py.File(os.path.join(WORK, "n25_long.h5"), "r") as f:
        yinf = f["final/y"][()]
    d_def = np.abs(res["default"][1] - yinf).max()
    d_tig = np.abs(res["tight"][1] - yinf).max()
    tol = res["default"][0]["tol_ss"]
    check(d_def < 10 * tol and d_tig < 10 * tol, f"state at t' vs the stationary state (tight run to t = 800): default {d_def:.1e}, "
                                                 f"tight {d_tig:.1e} (tol_ss = {tol:.0e}); t' default {res['default'][0]['t_prime']:.1f} vs tight "
                                                 f"{res['tight'][0]['t_prime']:.1f}")


if __name__ == "__main__":
    tests = [test_lambda_ref_and_timeout, test_auto_vs_fixed, test_default_tolerances]
    sel = sys.argv[1:]
    if not sel:
        shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK, exist_ok=True)
    for t in tests:
        if not sel or t.__name__ in sel:
            t()
    print(f"\n{'ALL STEP-8 TESTS PASSED' if not FAIL else 'FAILURES: ' + str(len(FAIL))}")
    sys.exit(1 if FAIL else 0)
