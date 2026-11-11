"""Step-9 tests (trimmed for the release): automatic RTDE end (rtde_tau = "auto"), checkpoint_keep, divergence guard.

    python3 -I tests/test_step9.py [test names]

Uses at most 4 threads. Writes tests/work9/. Test [3] is opt-in: with DP5_STEP9_REF_EXE set to another build (e.g.
the development driver built with the same flags) it checks bitwise identity of fixed-t', auto-t' and g1g2 runs.
The development suite's extension/restart and wall-time tests ([4], [6]) test removed features and are not included.
"""
import os
import subprocess
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_step2 as T2  # noqa: E402

EXE = os.environ.get("DP5_EXE", T2.EXE)
WORK = os.path.join(HERE, "work9")
T2.WORK = WORK
REF_EXE = os.environ.get("DP5_STEP9_REF_EXE")
ENV = dict(os.environ, OMP_NUM_THREADS="4")
FAIL = []
TIGHT = dict(tol_rho=[1e-12, 1e-10], tol_tensor=[1e-12, 1e-10], tol_rtde=[1e-12, 1e-10])
MODEL = os.path.join(WORK, "imp5.h5")
TP = 20.0
DT = 0.1


def check(cond, msg):
    print(("   ok   " if cond else "   FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def run(rc, out, *extra, exe=None):
    p = os.path.join(WORK, out)
    if os.path.exists(p):
        os.remove(p)
    r = subprocess.run([exe or EXE, "--runcard", rc, "--output", out, *extra], cwd=WORK, env=ENV, capture_output=True, text=True)
    if r.returncode not in (0, 4):
        print(r.stdout[-2000:], r.stderr[-2000:])
    return r


def rc(name, gamma=1.0, ap="TPPX", **kw):
    base = dict(mode="rtde", dt_out=DT, probe_columns=[0], **TIGHT)
    base.update(kw)
    return T2.write_rc(name, MODEL, gamma, ap, **base)


def attrs(path, g):
    with h5py.File(os.path.join(WORK, path), "r") as f:
        return dict(f[g].attrs) if g in f else {}


def gr(path, k=0):
    with h5py.File(os.path.join(WORK, path), "r") as f:
        return f[f"rtde/tp={k}/t"][()], f[f"rtde/tp={k}/GR"][()]


def final(path):
    with h5py.File(os.path.join(WORK, path), "r") as f:
        return f["final/y"][()], float(f["final"].attrs["t"])


def spectrum(t, G, tp, tau_end, w):
    """A(w) = -Im int_0^tau_end e^{i w tau} G^R_00(tau) dtau / pi per spin (trapezoid on the frame grid)"""
    tau = t - tp
    m = tau <= tau_end + 1e-9
    tau, g = tau[m], G[m][:, 0, :, 0]  # [T, spin]
    ph = np.exp(1j * np.outer(w, tau))  # [w, T]
    wts = np.full(tau.size, tau[1] - tau[0])
    wts[0] *= 0.5
    wts[-1] *= 0.5
    return -np.imag(ph @ (g * wts[:, None])) / np.pi  # [w, spin]


# ---------------------------------------------------------------- tests
def test_auto_stop():
    print("[1] auto stop (impurity N=5, TPPX, gamma=1, t' = 20) vs a long fixed run: A(omega) truncation error")
    T2.impurity_model(MODEL, 5)
    cap = 300.0
    p = run(rc("rc_auto.h5", t_prime=[TP], rtde_tau="auto", rtde_tau_max=cap, tol_A=1e-4), "auto.h5")
    print("      " + "\n      ".join(l for l in p.stdout.splitlines() if "RTDE end" in l))
    a = attrs("auto.h5", "rtde_end")
    p2 = run(rc("rc_long.h5", t_prime=[TP], t_end=TP + cap), "long.h5")
    tA, GA = gr("auto.h5")
    tL, GL = gr("long.h5")
    tau_end = a["tau_end"]
    y, tf = final("auto.h5")
    n = tA.size
    same = np.array_equal(tA, tL[:n]) and np.array_equal(GA, GL[:n])
    check(p.returncode == 0 and a["cap_reached"] == 0 and abs(tf - (TP + tau_end)) < 1e-12 and abs(tA[-1] - tf) < 1e-12
          and abs(round(tau_end / DT) * DT - tau_end) < 1e-9,
          f"stopped at tau_end = {tau_end:.1f} (t = {tf:.1f}, last frame {tA[-1]:.1f}, on the dt_out grid; tau_min "
          f"{a['rtde_tau_min_used']:.1f}, window {a['window']:.1f}, lambda_GR {a['lambda_GR']:.4f}, lambda_ref(t') "
          f"{a['lambda_ref_tprime']:.4f}, estimated truncation error {a['truncation_error']:.2e}, monitor: {a['monitor']})")
    check(p2.returncode == 0 and same, f"G^R frames of the auto run = the long fixed run's first {n} frames, bitwise: {same}")
    w = np.linspace(-6, 6, 241)
    Aa = spectrum(tA, GA, TP, tau_end, w)
    Al = spectrum(tL, GL, TP, cap, w)
    d = np.abs(Aa - Al).max()
    tail = np.abs(GL[:, 0, :, 0][tL - TP > tau_end]).max()
    check(d <= 1e-4 and d <= 3 * a["truncation_error"] + 1e-12,
          f"max_w |A_auto - A_long| = {d:.2e} <= tol_A = 1e-4 and consistent with the estimate {a['truncation_error']:.2e} "
          f"(max |G^R_00| beyond tau_end {tail:.1e})")
    # t_prime = auto with rtde_tau = auto
    p = run(rc("rc_auto2.h5", t_prime="auto", rtde_tau="auto", rtde_tau_max=cap, tol_A=1e-4), "auto2.h5")
    a2, s2 = attrs("auto2.h5", "rtde_end"), attrs("auto2.h5", "steady_state")
    t2, _ = gr("auto2.h5")
    _, tf2 = final("auto2.h5")
    check(p.returncode == 0 and a2["cap_reached"] == 0 and abs(tf2 - s2["t_prime"] - a2["tau_end"]) < 1e-9 and abs(t2[-1] - tf2) < 1e-9,
          f"t_prime = auto (t' = {s2['t_prime']:.3f}) + rtde_tau = auto: tau_end = {a2['tau_end']:.1f}, lambda_GR {a2['lambda_GR']:.4f}, "
          f"error {a2['truncation_error']:.1e}")


def test_cap():
    print("[2] cap: rtde_tau_max reached before the criterion (tol_A = 1e-12)")
    p = run(rc("rc_cap.h5", t_prime=[TP], rtde_tau="auto", rtde_tau_max=60.0, tol_A=1e-12), "cap.h5")
    a = attrs("cap.h5", "rtde_end")
    with h5py.File(os.path.join(WORK, "cap.h5"), "r") as f:
        st = f["status"].attrs["status"]
    print("      " + "\n      ".join(l for l in p.stdout.splitlines() if "RTDE end" in l))
    check(p.returncode == 0 and st == "Ok" and a["cap_reached"] == 1 and "warning" in a and abs(a["tau_end"] - 60) < 1e-9
          and np.isfinite(a["truncation_error"]),
          f"status {st}, cap_reached, tau_end = {a['tau_end']:.1f}, warning: '{a['warning']}', estimated truncation error "
          f"{a['truncation_error']:.2e} (lambda_GR {a['lambda_GR']:.4f})")


def test_fixed_identical():
    print("[3] numeric rtde_tau / t_end, t_prime = auto, g1g2: bitwise identical to a reference build")
    if not REF_EXE:
        print("      SKIPPED: opt-in, set DP5_STEP9_REF_EXE to a reference build")
        return
    if not os.path.exists(REF_EXE):
        check(False, f"reference binary {REF_EXE} missing")
        return
    probe = subprocess.run([REF_EXE, "--help"], capture_output=True, text=True)
    if probe.returncode == 127 or "error while loading shared libraries" in probe.stderr:
        print(f"      SKIPPED: reference binary {REF_EXE} does not load on this host ({probe.stderr.strip()[:160]}); "
              "set DP5_STEP9_REF_EXE to a loadable build")
        return
    cases = [("fixed t'", dict(t_prime=[TP], t_end=TP + 30)),
             ("t_prime = auto, rtde_tau = 30", dict(t_prime="auto", rtde_tau=30.0)),
             ("g1g2 mode", dict(mode="g1g2", t_end=30.0))]
    for k, (label, kw) in enumerate(cases):
        r = rc(f"rc_id{k}.h5", **kw)
        p1 = run(r, f"id{k}_new.h5")
        p0 = run(r, f"id{k}_ref.h5", exe=REF_EXE)
        y1, t1 = final(f"id{k}_new.h5")
        y0, t0 = final(f"id{k}_ref.h5")
        same = np.array_equal(y1, y0) and t1 == t0
        if kw.get("mode") != "g1g2":
            (u1, a1), (u0, a0) = gr(f"id{k}_new.h5"), gr(f"id{k}_ref.h5")
            same = same and np.array_equal(u1, u0) and np.array_equal(a1, a0)
        check(p1.returncode == 0 and p0.returncode == 0 and same,
              f"{label}: final state{' and G^R frames' if kw.get('mode') != 'g1g2' else ''} bitwise identical")


def test_F_decay():
    print("[5] sanity: the RTDE tensors decay with G^R")
    out = []
    for tau in (20.0, None):
        kw = dict(t_prime=[TP], t_end=TP + tau) if tau else dict(t_prime=[TP], rtde_tau="auto", rtde_tau_max=300.0, tol_A=1e-4)
        name = f"dec{0 if tau else 1}.h5"
        run(rc("rc_" + name, **kw), name)
        with h5py.File(os.path.join(WORK, name), "r") as f:
            y = f["final/y"][()]
            names = f["final"].attrs["block_names"].split(",")
            sizes = f["final"].attrs["block_sizes"].astype(int)
            t = float(f["final"].attrs["t"])
            G = f["rtde/tp=0/GR"][()]
        off = np.concatenate([[0], np.cumsum(sizes)])
        fmax = max(np.abs(y[off[i]:off[i + 1]]).max() for i, n in enumerate(names) if "_tp0_" in n and not n.startswith("GR"))
        out.append((t - TP, fmax, np.abs(G[-1, 0, :, 0]).max()))
    (t0, f0, g0), (t1, f1, g1) = out
    check(f1 / f0 < 10 * g1 / g0 and f1 < f0,
          f"max|F| at tau = {t0:.0f}: {f0:.2e}, at tau_end = {t1:.0f}: {f1:.2e} (ratio {f1 / f0:.1e}); |G^R_00| ratio {g1 / g0:.1e}")


def test_checkpoint_keep():
    print("[7] checkpoint_keep = 1: only the newest periodic checkpoint stays; results unchanged")
    out = []
    for k, kw in ((0, {}), (1, dict(checkpoint_keep=1))):
        run(rc(f"rc_keep{k}.h5", t_prime=[5.0], t_end=40.0, checkpoint_interval=3.0, **kw), f"keep{k}.h5")
        with h5py.File(os.path.join(WORK, f"keep{k}.h5"), "r") as f:
            out.append((sorted(f["checkpoints"].keys()), f["final/y"][()], os.path.getsize(os.path.join(WORK, f"keep{k}.h5"))))
    (c0, y0, s0), (c1, y1, s1) = out
    check(np.array_equal(y0, y1) and sum(c.startswith("periodic_") for c in c1) == 1 and sum(c.startswith("periodic_") for c in c0) == 13,
          f"checkpoints kept: {c1} (all: {len(c0)} groups); final state bitwise equal; file {s1} vs {s0} bytes")


def test_diverge():
    print("[8] divergence guard: mGW_0 on the 4-site ring (loss on every site, gamma = 1) leaves [-1, 2] and stops")
    rho = np.stack([np.diag([0.9, 0.8, 0.7, 0.6]).astype(complex), np.diag([0.6, 0.5, 0.6, 0.5]).astype(complex)])
    path = os.path.join(WORK, "plaq.h5")
    T2.ring_model(path, 4, [0, 1, 2, 3], [1, 1, 1, 1], zeeman=0.0, rho=rho)
    r = T2.write_rc("rc_div.h5", path, 1.0, "mGW_0", mode="g1g2", t_end=6.0, dt_out=0.1)
    p = run(r, "div.h5")
    with h5py.File(os.path.join(WORK, "div.h5"), "r") as f:
        st = dict(f["status"].attrs)
        e = f["observables/rho_eig_min"][()]
        fin = "final/y" in f
    check(p.returncode == 7 and st["status"] == "Diverged" and fin and "eigenvalue of rho outside" in st["message"],
          f"exit {p.returncode}, status {st['status']} at t = {st['t_reached']:.1f}: '{st['message']}'; state kept in /final; "
          f"min eigenvalue 10 frames before: {e[-10]:.3f}")


TESTS = {"auto_stop": test_auto_stop, "cap": test_cap, "fixed_identical": test_fixed_identical, "F_decay": test_F_decay,
         "checkpoint_keep": test_checkpoint_keep, "diverge": test_diverge}

if __name__ == "__main__":
    os.makedirs(WORK, exist_ok=True)
    names = sys.argv[1:] or list(TESTS)
    for n in names:
        TESTS[n]()
    print(f"\n{'ALL STEP-9 TESTS PASSED' if not FAIL else 'FAILURES: ' + str(len(FAIL))}")
    sys.exit(1 if FAIL else 0)
