"""Run-card writers (driver: CONVENTIONS section 3; exact: section 2), run ids and runcards/index.csv.

Every driver card key is written explicitly (no card relies on a driver default). Model files live in
runcards/models/<system>.h5 and are referenced by the relative path ../models/<system>.h5.
"""
import csv
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np

from . import paths  # noqa: F401  (puts code_diagrammatics/python and code_exact on sys.path)
from .methods import DRIVER_FLAGS, token
from . import models as M
import dp5model                      # noqa: E402
import runcard as _rc                # noqa: E402

INDEX_COLUMNS = ["run_id", "kind", "system", "ic", "gamma", "method", "segment", "cost_s"]

# tolerance classes ([atol, rtol]; CONVENTIONS section 3)
TOL_DEFAULT = dict(tol_rho=[1e-9, 1e-7], tol_tensor=[1e-8, 1e-6], tol_rtde=[1e-10, 1e-6])
TOL_TIGHT = dict(tol_rho=[1e-12, 1e-10], tol_tensor=[1e-11, 1e-9], tol_rtde=[1e-12, 1e-10])
# RTDE detector / automatic end of the impurity cards (c3 column of the table; ss_hold_records, rtde_window_beats = 0)
RTDE_C3 = dict(t_prime="auto", ss_on_timeout="start", tol_ss=1e-8, c_hold=1.0, c_min=1.0, c_max=500.0,
               ss_fit_rtol=0.2, ss_window_c=1.0, ss_hold_records=0, rtde_tau="auto", tol_A=1e-7, rtde_tau_min=0.0,
               c_tau_min=3.0, rtde_tau_max=1200.0, rtde_window_beats=0, rtde_monitor="impurity")


def run_id(system, ic, gamma, method, segment=""):
    s = f"{system}_{ic}_g{float(gamma):g}_{token(method)}"
    return s + (f"_{segment}" if segment else "")


def fresh_tree(runcards_dir):
    """delete and recreate runcards/ (make_runcards.py writes it from scratch)"""
    d = Path(runcards_dir)
    if d.exists():
        shutil.rmtree(d)
    (d / "models").mkdir(parents=True)
    return d


def write_model(runcards_dir, system):
    """runcards/models/<system>.h5 from common.models; provenance 'created' = the release VERSION (no timestamp, so
    the cards do not churn). Returns the path."""
    mdl = M.get(system)
    p = Path(runcards_dir) / "models" / f"{system}.h5"
    p.parent.mkdir(parents=True, exist_ok=True)
    N = mdl.N
    h = np.stack([mdl.h1, mdl.h1]).astype(complex)
    g = np.stack([np.diag(mdl.loss)] * 2).astype(complex)
    l = np.stack([np.diag(mdl.gain)] * 2).astype(complex)
    dp5model.write_model(str(p), h, g, l, M.model_default_rho0(system), list(mdl.lossy_sites), list(mdl.weights),
                         dict(mdl.driver_attrs), dict(generator="common/runcards.py", system=system))
    with h5py.File(p, "r+") as f:
        f["provenance"].attrs["created"] = paths.version()
    assert N == mdl.N
    return p


def write_driver_card(runcards_dir, rid, system, gamma, method, rho0, t_end, dt_out, mode, tol, checkpoint_interval,
                      rtde=None, vertex_impl=None):
    """runcards/<rid>/rc.h5 with every key of CONVENTIONS section 3 explicit"""
    d = Path(runcards_dir) / rid
    d.mkdir(parents=True, exist_ok=True)
    mdl = M.get(system)
    kw = dict(DRIVER_FLAGS[method])
    kw.update(mode=mode, spin_mode="full", vertex_impl=vertex_impl or mdl.vertex_impl, ic_mode="full", t0=0.0,
              probe_columns=[0], checkpoint_interval=float(checkpoint_interval), checkpoint_keep=0, rho_stride=1,
              diverge_rho_lo=-1.0, diverge_rho_hi=2.0, threads=1)
    kw.update(tol)
    if mode == "rtde":
        kw.update(rtde)
    else:
        assert rtde is None
    cwd = os.getcwd()
    os.chdir(d)
    try:
        _rc.write_runcard("rc.h5", f"../models/{system}.h5", gamma, t_end, dt_out, rho0=np.asarray(rho0, complex), **kw)
    finally:
        os.chdir(cwd)
    return d / "rc.h5"


def exact_model(system, gamma):
    mdl = M.get(system)
    return {"name": system, "n_orb": mdl.N, "h1": np.asarray(mdl.h1, float).tolist(),
            "onebody_loss": [float(x) for x in mdl.loss], "onebody_gain": [float(x) for x in mdl.gain],
            "pair_loss": {"sites": [int(s) for s in mdl.lossy_sites], "gamma": float(gamma),
                          "weights": [float(w) for w in mdl.weights]}}


def write_exact_card(runcards_dir, rid, system, gamma, rho0, tasks):
    """runcards/<rid>/card.json (schema exact_card/1); validated with code_exact's own card parser"""
    from lindblad_exact.card import parse
    r = np.asarray(rho0)
    assert np.abs(r.imag).max() == 0
    card = {"schema": "exact_card/1", "run_id": rid, "model": exact_model(system, gamma),
            "initial_state": {"type": "gaussian", "rho1_up": r[0].real.tolist(), "rho1_dn": r[1].real.tolist()},
            "tasks": tasks, "output": {"file": "result.h5"}}
    text = json.dumps(card, indent=1)
    parse(text)
    d = Path(runcards_dir) / rid
    d.mkdir(parents=True, exist_ok=True)
    (d / "card.json").write_text(text + "\n")
    return d / "card.json"


def write_index(runcards_dir, rows):
    """rows: dicts with the INDEX_COLUMNS; written sorted by run_id"""
    rows = sorted(rows, key=lambda r: r["run_id"])
    ids = [r["run_id"] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate run ids"
    with open(Path(runcards_dir) / "index.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(INDEX_COLUMNS)
        for r in rows:
            g = r["gamma"]
            w.writerow([r["run_id"], r["kind"], r["system"], r["ic"], f"{float(g):g}", r["method"],
                        r.get("segment", ""), int(round(r["cost_s"]))])


def _h5_content(path):
    out = {}

    def visit(name, obj):
        if name.startswith("provenance"):
            return
        out[name + "@attrs"] = {k: np.asarray(v).tolist() for k, v in obj.attrs.items()}
        if isinstance(obj, h5py.Dataset):
            out[name] = np.asarray(obj[()]).tolist()
    with h5py.File(path, "r") as f:
        out["@attrs"] = {k: np.asarray(v).tolist() for k, v in f.attrs.items()}
        f.visititems(visit)
    return out


def cards_equal(a, b):
    """content equality of two runcards/ trees (HDF5 by content excluding /provenance, JSON and CSV by text)"""
    a, b = Path(a), Path(b)
    fa = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    fb = sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    if fa != fb:
        return False
    for p in fa:
        if p.suffix == ".h5":
            if _h5_content(a / p) != _h5_content(b / p):
                return False
        elif (a / p).read_bytes() != (b / p).read_bytes():
            return False
    return True
