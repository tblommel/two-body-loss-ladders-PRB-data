"""Readers: driver output files (old and release driver alike), exact result files, runcards/index.csv."""
import csv
from pathlib import Path

import h5py
import numpy as np

from .methods import method_from_output

FINISHED_DRIVER = {0, 5, 7}


def _s(v):
    return v.decode() if isinstance(v, bytes) else (str(v) if v is not None else None)


def read_index(dirpath):
    """rows of <dir>/runcards/index.csv as dicts (gamma float, cost_s int)"""
    with open(Path(dirpath) / "runcards" / "index.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["gamma"] = float(r["gamma"]); r["cost_s"] = int(r["cost_s"])
    return rows


def exit_code(output_dir, rid):
    p = Path(output_dir) / rid / "exit_code"
    try:
        return int(p.read_text().strip())
    except (OSError, ValueError):
        return None


def load(path, with_rho=False, with_rtde=True):
    """driver output -> dict. Quantities follow the data names of CONVENTIONS 1.6:
    t, N, N_sigma [T,2], Mz, n_site [T,2,N], D_sites [T,M] (route a, spin-averaged), D (sum over lossy sites),
    D_b [T,N], S2, eig_min, eig_max, dN_dt; status, t_reached, wall, method, version; rtde: t_prime, tau, GR;
    steady_state / rtde_end attrs."""
    d = {}
    with h5py.File(path, "r") as f:
        st = dict(f["status"].attrs)
        d["status"] = _s(st.get("status")); d["message"] = _s(st.get("message", ""))
        d["t_reached"] = float(st.get("t_reached", np.nan)); d["wall"] = float(st.get("wall_seconds", np.nan))
        d["method"] = method_from_output(f)
        d["version"] = _s(f.attrs.get("release_version", "pre-release"))
        d["t"] = f["time/t"][()]
        o = f["observables"]; e = o["equal_time"]
        d["N"] = e["N_tot"][()]; d["N_sigma"] = e["N_sigma"][()]; d["Mz"] = e["Mz"][()]
        d["n_site"] = e["n_site"][()]
        Da = e["D_a"][()]                                      # [T, M, 2]
        d["D_sites"] = Da.mean(axis=2)
        d["D"] = d["D_sites"].sum(axis=1)
        d["D_b"] = e["D_b"][()]
        d["S2"] = e["S2_total"][()]
        d["dN_dt"] = o["dN_dt"][()]
        d["eig_min"] = o["rho_eig_min"][()]; d["eig_max"] = o["rho_eig_max"][()]
        if with_rho and "rho" in f:
            d["rho_t"] = f["rho/t"][()]; d["rho"] = f["rho/rho"][()]
        if with_rtde and "rtde" in f and len(f["rtde"]):
            k = sorted(f["rtde"].keys())[0]
            g = f["rtde"][k]
            tp = float(g.attrs["t_prime"]); t = g["t"][()]
            d["rtde"] = dict(t_prime=tp, t=t, tau=t - tp, GR=g["GR"][()])
        if "steady_state" in f:
            d["ss"] = {k: f["steady_state"].attrs[k] for k in f["steady_state"].attrs}
        if "rtde_end" in f:
            d["rtde_end"] = {k: f["rtde_end"].attrs[k] for k in f["rtde_end"].attrs}
    return d


def checkpoint_blocks(path):
    """list of (t, {block: array}) of the periodic checkpoints, sorted by t; also the approximation and method"""
    res = []
    with h5py.File(path, "r") as f:
        method = method_from_output(f)
        if "checkpoints" in f:
            for k in f["checkpoints"]:
                if not k.startswith("periodic_"):
                    continue
                cp = f["checkpoints"][k]
                names = _s(cp.attrs["block_names"]).split(",")
                sz = cp.attrs["block_sizes"].astype(int)
                off = np.concatenate([[0], np.cumsum(sz)])
                y = cp["y"][()]
                res.append((float(cp.attrs["t"]), {n: y[off[j]:off[j + 1]] for j, n in enumerate(names)}))
    res.sort(key=lambda x: x[0])
    return method, res


def load_exact(path):
    """exact result.h5 -> nested dict {task: {dataset: array, '@attrs': {...}}}, root '@attrs'"""
    out = {}

    def grp(g):
        r = {"@attrs": {k: g.attrs[k] for k in g.attrs}}
        for k, v in g.items():
            r[k] = grp(v) if isinstance(v, h5py.Group) else v[()]
        return r
    with h5py.File(path, "r") as f:
        out = grp(f)
    st = _s(out["@attrs"].get("status"))
    if st != "Ok":
        raise RuntimeError(f"{path}: exact status {st}: {_s(out['@attrs'].get('message'))}")
    out["version"] = _s(out["@attrs"].get("release_version"))
    return out
