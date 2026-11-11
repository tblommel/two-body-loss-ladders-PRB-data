"""Read a dp5drive output file: status, run summary, integrator profile summary.

    python3 read_output.py RUN.h5

The profile summary is computed from the /profile export alone, with the same logic as
integrator/tests/profile_summary.py (summarize() below), and checked for consistency.
"""
import sys

import h5py
import numpy as np

USER = {"stop_callback", "observer", "step_observer", "interrupt"}
PASS = {"pass_stage2", "pass_stage3", "pass_stage4", "pass_stage5", "pass_stage6", "pass_solution",
        "pass_error", "pass_init_step", "dense"}


def load_profile(f):
    num, st = {}, {}

    def walk(g, prefix):
        for k, v in g.items():
            key = f"{prefix}{k}"
            if isinstance(v, h5py.Group):
                walk(v, key + "/")
            else:
                num[key] = np.atleast_1d(v[()]).astype(float).tolist()
    walk(f["profile"], "")
    for k, v in f["profile"].attrs.items():
        st[k.replace(".", "/")] = v if isinstance(v, str) else v
    return num, st


def summarize(num, st, out=print):
    v = lambda k: num[k][0]
    sections = st["time/section_names"].split(",")
    T = v("time/integrate_seconds")
    Ts = T if T > 0 else 1.0
    sec = {s: v(f"time/{s}/seconds") for s in sections}
    calls = {s: v(f"time/{s}/calls") for s in sections}
    rhs = sum(t for s, t in sec.items() if s.startswith("rhs_"))
    user = sum(t for s, t in sec.items() if s in USER)
    over = T - rhs - user
    closure = sum(sec.values()) + v("time/untimed_seconds") - T
    out(f"  integrate {T:.4f} s: RHS {rhs:.4f} s ({100*rhs/Ts:.1f}%), user {user:.4f} s, overhead {over:.4f} s "
        f"({100*over/Ts:.1f}%); closure {closure:.1e}")
    for s in sections:
        if calls[s] > 0:
            out(f"    {s:<17} {sec[s]:9.4f} s {100*sec[s]/Ts:6.2f}% {int(calls[s]):7d} calls")
    acc, att = v("steps/accepted"), v("steps/attempts")
    rej = v("steps/rejected_error") + v("steps/rejected_nonfinite")
    out(f"  steps: attempts {int(att)} = accepted {int(acc)} + rejected {int(rej)}; RHS calls {int(v('steps/rhs_calls'))}; "
        f"h min {v('steps/h_min'):.3e} geomean {v('steps/h_geomean'):.3e} max {v('steps/h_max'):.3e}")
    names = st.get("blocks/names", "").split(",")
    lim = num.get("steps/limiting_block", [])
    if acc > 0:
        out("  limiting block: " + ", ".join(f"{names[i] if i < len(names) else i} {100*c/acc:.1f}%"
                                         for i, c in enumerate(lim) if c))
    for k in range(len(num["segments/t_begin"])):
        S = num["segments/integrate_seconds"][k]
        S1 = S if S > 0 else 1.0
        out(f"  segment {k}: t {num['segments/t_begin'][k]:.4g}..{num['segments/t_end'][k]:.4g} {S:.4f} s, "
            f"{int(num['segments/accepted'][k])} steps, RHS {100*num['segments/rhs_seconds'][k]/S1:.1f}% "
            f"overhead {100*num['segments/overhead_seconds'][k]/S1:.1f}%")
    out(f"  memory: registers {v('memory/register_bytes')/1e6:.2f} MB, peak RSS {v('memory/maxrss_bytes')/1e6:.1f} MB")
    ok = abs(closure) < 1e-9 and abs(over - v("time/overhead_seconds")) < 1e-9 + 1e-9 * T and sum(lim) == acc
    out("  profile export " + ("CONSISTENT" if ok else "INCONSISTENT"))
    return ok


# ---------------------------------------------------------------- two-time post-processing (step 3b)
def _gr_series(f, k, column, spin):
    g = f[f"rtde/tp={k}"]
    tp = float(g.attrs["t_prime"])
    t = g["t"][()]
    GR = g["GR"][()]  # [T, column, spin, x]
    cols = list(g.attrs["probe_columns"])
    sp = min(spin, GR.shape[2] - 1)  # paramagnetic layout: one spin
    return tp, t - tp, GR[:, :, sp, :], cols


def _window(tau, window):
    if window is None:
        return np.ones_like(tau)
    if window == "hann":
        return 0.5 * (1 + np.cos(np.pi * tau / tau[-1]))
    raise ValueError("window: None or 'hann'")


def _half_fourier(tau, f, w, eta, window):
    """int_0^T dtau e^{i w tau} e^{-eta tau} win(tau) f(tau), trapezoid (half weight at both ends); uniform grid."""
    d = np.diff(tau)
    if np.abs(d - d[0]).max() > 1e-9 * max(1.0, tau[-1]):
        raise ValueError("non-uniform tau grid")
    wt = np.full(len(tau), d[0]); wt[0] *= 0.5; wt[-1] *= 0.5
    g = f * np.exp(-eta * tau) * _window(tau, window) * wt
    return np.exp(1j * np.outer(w, tau)) @ g


def spectral_function(path, k=0, column=0, spin=0, eta=0.05, window=None, w=None):
    """A_ss(w) = -(1/pi) Im G^R_ss(w), G^R(w) = int_0^T dtau e^{i w tau} e^{-eta tau} G^R_ss(t'+tau, t'),
    s = probe_columns[column]. Returns (w, A). Default w grid: [-pi/dtau, pi/dtau], 4 x the number of samples.
    With G^R(t'+, t') = -i and the half trapezoid weight at tau = 0, int A dw = 1 up to the damping/truncation."""
    with h5py.File(path, "r") as f:
        tp, tau, GR, cols = _gr_series(f, k, column, spin)
    s = cols[column]
    if w is None:
        dt = tau[1] - tau[0]
        w = np.linspace(-np.pi / dt, np.pi / dt, 4 * len(tau))
    return w, -_half_fourier(tau, GR[:, column, s], w, eta, window).imag / np.pi


def method_label(f):
    """reporting label of an output file: approximation (TPPX + flag_2B = 2B) + "_0" with bare_propagators; old
    outputs with approximation BARE (removed 2026-10-28) are TPPX_0."""
    if "method" in f["input"].attrs:
        return str(f["input"].attrs["method"])
    r = f["input/run"].attrs
    a = str(r["approximation"])
    if a == "BARE":
        return "TPPX_0"
    a = "2B" if (a == "TPPX" and int(r.get("flag_2B", 0))) else a
    return a + ("_0" if int(r.get("bare_propagators", 0)) else "")


def main(path):
    with h5py.File(path, "r") as f:
        s = f["status"].attrs
        print(f"{path}: schema {f.attrs['dp5_driver_output_schema']}, run {f.attrs['run_id']}")
        print(f"  status {s.get('status')} at t = {s.get('t_reached')}: {s.get('message', '')}")
        r = f["input/run"].attrs
        print(f"  {method_label(f)} {r['mode']} spin={r.get('spin_mode', 'full')} vertex={r.get('vertex_impl', 'general')} "
              f"gamma={r['gamma']} model={f['input'].attrs['model_hash']}")
        if "t" in f["time"]:
            t = f["time/t"][()]
            nd = f["observables/n_d"][()]
            print(f"  {len(t)} frames t = {t[0]:.4g}..{t[-1]:.4g}; final n_d = {nd[-1]}; "
                  f"max|drho/dt| = {f['observables/rho_dot_max'][-1]:.2e}")
        for k in f["rtde"]:
            g = f["rtde"][k]
            print(f"  rtde {k}: t' = {g.attrs['t_prime']}, columns {list(g.attrs['probe_columns'])}, "
                  f"{g['t'].shape[0]} frames, GR {g['GR'].shape}")
        print(f"  checkpoints: {list(f['checkpoints'].keys())}; final: {'y' in f['final']}")
        p = f["provenance"].attrs
        print(f"  provenance: git {p['git_hash']}, {p['build_flags']}, host {p['host']}, threads {p['threads']}")
        if "time" in f["profile"]:
            num, st = load_profile(f)
            return summarize(num, st)
        print("  (no profile)")
        return True


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
