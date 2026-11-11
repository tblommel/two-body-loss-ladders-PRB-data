"""data/ I/O (CONVENTIONS 1.6): CSV curves/tables, HDF5 (G^R, Sigma), deterministic meta.json.

build() of a postprocess.py returns {"x.csv": {column: 1-D array}, "y.h5": {group: {...}, "@attrs": {...}}}.
"""
import csv
import json
import math
from pathlib import Path

import h5py
import numpy as np

from .paths import version_string


def _fmt(x):
    if isinstance(x, (str, np.str_)):
        return str(x)
    if isinstance(x, (bool, np.bool_)):
        return str(int(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    x = float(x)
    return "nan" if math.isnan(x) else repr(x)


def write_csv(path, cols):
    names = list(cols)
    n = {len(np.atleast_1d(cols[c])) for c in names}
    assert len(n) == 1, f"{path}: columns of unequal length {n}"
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(names)
        arrs = [np.atleast_1d(cols[c]) for c in names]
        for row in zip(*arrs):
            w.writerow([_fmt(v) for v in row])


def _conv(v):
    if v == "nan":
        return float("nan")
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def read_csv(path):
    with open(path) as f:
        r = csv.reader(f)
        names = next(r)
        rows = list(r)
    out = {}
    for j, c in enumerate(names):
        vals = [_conv(row[j]) for row in rows]
        if all(isinstance(v, (int, float)) for v in vals) and any(isinstance(v, float) for v in vals):
            vals = [float(v) for v in vals]
        out[c] = np.array(vals) if vals else np.array([])
    return out


def _h5_write(g, d):
    for k, v in d.items():
        if k == "@attrs":
            for a, x in v.items():
                g.attrs[a] = x
        elif isinstance(v, dict):
            _h5_write(g.create_group(k), v)
        else:
            g.create_dataset(k, data=np.asarray(v), track_times=False)


def _h5_read(g):
    out = {"@attrs": {k: (v.decode() if isinstance(v, bytes) else v) for k, v in g.attrs.items()}}
    for k, v in g.items():
        out[k] = _h5_read(v) if isinstance(v, h5py.Group) else v[()]
    return out


def write_data(data_dir, files, generated_by, descriptions, producers):
    """files: build() result; descriptions: {file: {"description": str, "columns": {col: str}}};
    producers: {"dp5drive": [...], "code_exact": [...]} version strings found in the outputs"""
    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    for old in list(d.glob("*.csv")) + list(d.glob("*.h5")):
        if old.name not in files:
            old.unlink()
    vs = version_string()
    for name, content in files.items():
        if name.endswith(".csv"):
            write_csv(d / name, content)
        elif name.endswith(".h5"):
            with h5py.File(d / name, "w", track_order=True) as f:
                _h5_write(f, content)
                f.attrs["release_version"] = vs
        else:
            raise ValueError(name)
    meta = {"release_version": vs, "generated_by": generated_by,
            "producers": {k: sorted(set(v)) for k, v in producers.items()},
            "files": {n: descriptions.get(n, {}) for n in sorted(files)}}
    (d / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=False) + "\n")


def read_data(data_dir):
    d = Path(data_dir)
    out = {}
    for p in sorted(d.glob("*.csv")):
        out[p.name] = read_csv(p)
    for p in sorted(d.glob("*.h5")):
        with h5py.File(p, "r") as f:
            out[p.name] = _h5_read(f)
    return out


def select(cols, **where):
    """rows of a CSV dict matching key=value conditions -> dict of arrays"""
    n = len(next(iter(cols.values())))
    m = np.ones(n, bool)
    for k, v in where.items():
        c = cols[k]
        m &= np.isclose(c.astype(float), float(v), rtol=0, atol=1e-12) if isinstance(v, float) else (c == v)
    return {k: v[m] for k, v in cols.items()}
