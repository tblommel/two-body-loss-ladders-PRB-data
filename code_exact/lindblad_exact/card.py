"""Strict reader of exact run cards (schema exact_card/1, CONVENTIONS.md §2). Unknown keys are an error."""
import json
import math
import numpy as np
from .states import CardError, check_rho1
from .observables import VOCAB

SCHEMA = "exact_card/1"
R, O = True, False          # required / optional

SPEC = {
    "card": {"schema": R, "run_id": R, "model": R, "initial_state": R, "tasks": R, "output": R},
    "model": {"name": R, "n_orb": R, "h1": R, "onebody_loss": R, "onebody_gain": R, "pair_loss": R},
    "pair_loss": {"sites": R, "gamma": R, "weights": O},
    "initial_state": {"type": R, "rho1_up": R, "rho1_dn": R},
    "output": {"file": R},
    "transient": {"type": R, "name": R, "grid": R, "observables": R, "pqg_every": O, "method": O, "chunk": O},
    "grid": {"t_end": R, "dt": R},
    "ness": {"type": R, "name": R, "method": R, "propagate": O, "observables": R, "pqg": O, "spectrum": O},
    "propagate": {"block": R, "tol": R, "t_max": R},
    "spectrum": {"k": R},
    "n_inf": {"type": R, "name": R, "observables": R},
    "greens_R": {"type": R, "name": R, "state": R, "orbital": R, "spins": R, "tau_max": R, "dtau": R, "resolvent": O},
    "resolvent": {"omega_min": R, "omega_max": R, "n": R, "solver": R, "rtol": O},
}
DEFAULTS = {
    "transient": {"pqg_every": 0.0, "method": "expm_multiply", "chunk": 200},
    "ness": {"pqg": False},
    "pair_loss": {},
    "resolvent": {"rtol": 1e-13},
}
EIG_MAX_DIM = 8000      # dense eig of the even space above this is refused (memory ~ 16 * 3 * dim^2 bytes)


def _keys(d, kind, where):
    if not isinstance(d, dict):
        raise CardError(f"{where}: expected an object")
    spec = SPEC[kind]
    for k in d:
        if k not in spec:
            raise CardError(f"{where}: unknown key '{k}'")
    for k, req in spec.items():
        if req and k not in d:
            raise CardError(f"{where}: missing key '{k}'")
    out = dict(DEFAULTS.get(kind, {})); out.update(d)
    return out


def _num(x, where, pos=False, nonneg=False):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise CardError(f"{where}: expected a finite number, got {x!r}")
    if pos and not x > 0:
        raise CardError(f"{where}: must be > 0")
    if nonneg and x < 0:
        raise CardError(f"{where}: must be >= 0")
    return float(x)


def _multiple(a, b, tol=1e-9):
    q = a / b
    return abs(q - round(q)) <= tol * max(1.0, abs(q))


def _obs(lst, where, allowed=VOCAB):
    if not isinstance(lst, list) or not lst:
        raise CardError(f"{where}: observables must be a non-empty list")
    for o in lst:
        if o not in allowed:
            raise CardError(f"{where}: unknown observable '{o}' (allowed: {', '.join(allowed)})")
    if len(set(lst)) != len(lst):
        raise CardError(f"{where}: duplicate observable")
    return list(lst)


def parse(text):
    try:
        card = json.loads(text)
    except json.JSONDecodeError as e:
        raise CardError(f"card is not valid JSON: {e}")
    card = _keys(card, "card", "card")
    if card["schema"] != SCHEMA:
        raise CardError(f"schema '{card['schema']}' (expected {SCHEMA})")
    if not isinstance(card["run_id"], str) or not card["run_id"]:
        raise CardError("run_id must be a non-empty string")
    # model
    m = _keys(card["model"], "model", "model")
    N = m["n_orb"]
    if isinstance(N, bool) or not isinstance(N, int) or not 1 <= N <= 6:
        raise CardError("model.n_orb must be an integer in 1..6")
    h1 = np.asarray(m["h1"], dtype=float)
    if h1.shape != (N, N) or not np.all(np.isfinite(h1)):
        raise CardError(f"model.h1: shape {h1.shape}, expected ({N}, {N})")
    if np.abs(h1 - h1.T).max() > 0:
        raise CardError("model.h1 must be real symmetric")
    for k in ("onebody_loss", "onebody_gain"):
        v = np.asarray(m[k], dtype=float)
        if v.shape != (N,) or np.any(v < 0) or not np.all(np.isfinite(v)):
            raise CardError(f"model.{k}: expected {N} non-negative rates")
    pl = _keys(m["pair_loss"], "pair_loss", "model.pair_loss")
    sites = pl["sites"]
    if not isinstance(sites, list) or any(isinstance(r, bool) or not isinstance(r, int) or not 0 <= r < N for r in sites) \
            or len(set(sites)) != len(sites):
        raise CardError("model.pair_loss.sites: distinct orbital indices in 0..n_orb-1")
    _num(pl["gamma"], "model.pair_loss.gamma", nonneg=True)
    if "weights" in pl:
        w = pl["weights"]
        if not isinstance(w, list) or len(w) != len(sites):
            raise CardError("model.pair_loss.weights: one weight per site")
        for x in w:
            _num(x, "model.pair_loss.weights", nonneg=True)
    m["pair_loss"] = pl
    card["model"] = m
    # initial state
    ist = _keys(card["initial_state"], "initial_state", "initial_state")
    if ist["type"] != "gaussian":
        raise CardError("initial_state.type: only 'gaussian'")
    try:
        ist["rho1_up"] = check_rho1(ist["rho1_up"], "initial_state.rho1_up", N)
        ist["rho1_dn"] = check_rho1(ist["rho1_dn"], "initial_state.rho1_dn", N)
    except ValueError as e:
        raise CardError(f"initial_state: {e}")
    card["initial_state"] = ist
    out = _keys(card["output"], "output", "output")
    if not isinstance(out["file"], str) or "/" in out["file"] or not out["file"].endswith(".h5"):
        raise CardError("output.file: a plain file name ending in .h5")
    # tasks
    if not isinstance(card["tasks"], list) or not card["tasks"]:
        raise CardError("tasks: non-empty list")
    names, tasks, seen_ness = set(), [], False
    for i, t in enumerate(card["tasks"]):
        where = f"tasks[{i}]"
        if not isinstance(t, dict) or t.get("type") not in ("transient", "ness", "n_inf", "greens_R"):
            raise CardError(f"{where}: type must be transient | ness | n_inf | greens_R")
        kind = t["type"]
        t = _keys(t, kind, where)
        if not isinstance(t["name"], str) or not t["name"] or "/" in t["name"] or t["name"] in names \
                or t["name"] in ("model",):
            raise CardError(f"{where}: name must be a unique non-empty string without '/' (not 'model')")
        names.add(t["name"])
        if kind == "transient":
            t["observables"] = _obs(t["observables"], where)
            if not isinstance(t["grid"], list) or not t["grid"]:
                raise CardError(f"{where}.grid: non-empty list of segments")
            t0, segs = 0.0, []
            for j, s in enumerate(t["grid"]):
                s = _keys(s, "grid", f"{where}.grid[{j}]")
                te = _num(s["t_end"], f"{where}.grid[{j}].t_end"); dt = _num(s["dt"], f"{where}.grid[{j}].dt", pos=True)
                if te <= t0 or not _multiple(te - t0, dt):
                    raise CardError(f"{where}.grid[{j}]: (t_end - t_start)/dt must be a positive integer")
                segs.append((t0, te, dt, int(round((te - t0) / dt)))); t0 = te
            t["segments"] = segs
            p = _num(t["pqg_every"], f"{where}.pqg_every", nonneg=True)
            if p > 0:
                for (a, b, dt, n) in segs:
                    if not _multiple(p, dt):
                        raise CardError(f"{where}.pqg_every must be a multiple of every segment's dt")
            if t["method"] not in ("expm_multiply", "eig"):
                raise CardError(f"{where}.method: expm_multiply | eig")
            c = t["chunk"]
            if isinstance(c, bool) or not isinstance(c, int) or c < 1:
                raise CardError(f"{where}.chunk: positive integer")
        elif kind == "ness":
            t["observables"] = _obs(t["observables"], where)
            if t["method"] not in ("lu", "propagate", "auto"):
                raise CardError(f"{where}.method: lu | propagate | auto")
            if t["method"] in ("propagate", "auto"):
                if "propagate" not in t:
                    raise CardError(f"{where}: method {t['method']} needs 'propagate'")
            if "propagate" in t:
                p = _keys(t["propagate"], "propagate", f"{where}.propagate")
                for k in p:
                    _num(p[k], f"{where}.propagate.{k}", pos=True)
                t["propagate"] = p
            if not isinstance(t["pqg"], bool):
                raise CardError(f"{where}.pqg: bool")
            if "spectrum" in t:
                sp_ = _keys(t["spectrum"], "spectrum", f"{where}.spectrum")
                if isinstance(sp_["k"], bool) or not isinstance(sp_["k"], int) or sp_["k"] < 1:
                    raise CardError(f"{where}.spectrum.k: positive integer")
                t["spectrum"] = sp_
            seen_ness = True
        elif kind == "n_inf":
            t["observables"] = _obs(t["observables"], where)
        else:
            if t["state"] != "ness":
                raise CardError(f"{where}.state: only 'ness'")
            if not seen_ness:
                raise CardError(f"{where}: state 'ness' needs a ness task earlier in the list")
            o = t["orbital"]
            if isinstance(o, bool) or not isinstance(o, int) or not 0 <= o < N:
                raise CardError(f"{where}.orbital: integer in 0..n_orb-1")
            if not isinstance(t["spins"], list) or not t["spins"] or any(s not in ("up", "dn") for s in t["spins"]) \
                    or len(set(t["spins"])) != len(t["spins"]):
                raise CardError(f"{where}.spins: non-empty list of 'up'/'dn'")
            tm = _num(t["tau_max"], f"{where}.tau_max", pos=True); dtau = _num(t["dtau"], f"{where}.dtau", pos=True)
            if not _multiple(tm, dtau):
                raise CardError(f"{where}: tau_max/dtau must be an integer")
            t["n_tau"] = int(round(tm / dtau))
            if "resolvent" in t:
                r = _keys(t["resolvent"], "resolvent", f"{where}.resolvent")
                _num(r["omega_min"], f"{where}.resolvent.omega_min"); _num(r["omega_max"], f"{where}.resolvent.omega_max")
                if isinstance(r["n"], bool) or not isinstance(r["n"], int) or r["n"] < 1:
                    raise CardError(f"{where}.resolvent.n: positive integer")
                if r["solver"] not in ("eig", "gmres"):
                    raise CardError(f"{where}.resolvent.solver: eig | gmres")
                _num(r["rtol"], f"{where}.resolvent.rtol", pos=True)
                t["resolvent"] = r
        tasks.append(t)
    card["tasks"] = tasks
    return card
