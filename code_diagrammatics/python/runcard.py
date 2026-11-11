"""Run-card writer for dp5drive (schema 1).

Run card layout (HDF5): root attr dp5_runcard_schema=1; group /run with attributes (the key set is closed: dp5drive
refuses any other /run attribute with exit code 2; README "Run-card keys"):
  model_path, model_hash            model file (relative to the run card, or absolute) and its content hash
                                    (read from the model file)
  gamma                             gamma of L = sqrt(gamma) d_dn d_up (vee = gamma/2 eps eps)
  approximation                     HF | TPPX | mGW | DSL;  flag_2B (0/1, TPPX only: 2B);  bare_propagators (0/1, not
                                    HF): correlator drifts with the bare h_o (TPPX + 1 = TPPX_0)
  mode                              g1g2 | rtde
  spin_mode                         full;   vertex_impl: general | impurity (requires lossy_sites == {0})
  ic_mode                           model (ρ(0) = the model's /initial/rho) | full (dataset /run/rho0 [2,N,N])
  t0, t_end, dt_out                 output grid t0 + k dt_out (integer number of intervals)
  probe_columns [1]                 the probe column j of G^R_{xj} (exactly one)
  tol_rho, tol_tensor, tol_rtde     [atol, rtol] per block kind
  checkpoint_interval, checkpoint_keep, rho_stride, diverge_rho_lo, diverge_rho_hi, threads
  mode rtde:
  t_prime                           one number, or "auto" (steady-state detection: tol_ss, c_hold, c_min, c_max,
                                    ss_fit_rtol, ss_window_c, ss_hold_records, ss_on_timeout)
  rtde_tau                          a number, or "auto" (automatic RTDE end: tol_A, rtde_tau_min, c_tau_min,
                                    rtde_tau_max (required), rtde_window_beats, rtde_monitor = auto | impurity | all)
"""
import h5py
import numpy as np

SCHEMA = 1
# the closed key set of /run (besides model_path, model_hash, gamma, t_end, dt_out, which are arguments)
KEYS = {"approximation", "flag_2B", "bare_propagators", "mode", "spin_mode", "vertex_impl", "ic_mode", "t0",
        "probe_columns", "tol_rho", "tol_tensor", "tol_rtde", "checkpoint_interval", "checkpoint_keep", "rho_stride",
        "diverge_rho_lo", "diverge_rho_hi", "threads",
        "t_prime", "ss_on_timeout", "tol_ss", "c_hold", "c_min", "c_max", "ss_fit_rtol", "ss_window_c",
        "ss_hold_records", "rtde_tau", "tol_A", "rtde_tau_min", "c_tau_min", "rtde_tau_max", "rtde_window_beats",
        "rtde_monitor"}
DEFAULTS = dict(approximation="HF", flag_2B=0, mode="g1g2", spin_mode="full", vertex_impl="general", t0=0.0,
                probe_columns=[0], tol_rho=[1e-9, 1e-7], tol_tensor=[1e-8, 1e-6], tol_rtde=[1e-10, 1e-6],
                checkpoint_interval=0.0, ic_mode="model", threads=0)


def write_runcard(path, model_path, gamma, t_end, dt_out, rho0=None, **kw):
    """Write a run card; unknown keys raise ValueError (the driver would refuse them)."""
    bad = sorted(set(kw) - KEYS)
    if bad:
        raise ValueError(f"run card: unknown keys {bad}")
    with h5py.File(model_path, "r") as m:
        mhash = m.attrs["content_hash"]
    p = dict(DEFAULTS)
    p.update(kw)
    if rho0 is not None and "ic_mode" not in kw:
        p["ic_mode"] = "full"
    with h5py.File(path, "w") as f:
        f.attrs["dp5_runcard_schema"] = SCHEMA
        g = f.create_group("run")
        g.attrs["model_path"] = str(model_path)
        g.attrs["model_hash"] = mhash
        g.attrs["gamma"] = float(gamma)
        g.attrs["t_end"] = float(t_end)
        g.attrs["dt_out"] = float(dt_out)
        for k, v in p.items():
            if v is None:
                continue
            if isinstance(v, (list, tuple, np.ndarray)):
                if len(v) == 0:
                    continue
                g.attrs[k] = np.asarray(v, dtype=float)
            else:
                g.attrs[k] = v
        if rho0 is not None:
            g.create_dataset("rho0", data=np.asarray(rho0, complex))
    return path
