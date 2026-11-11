#!/usr/bin/env python3
"""Exact Lindblad solver: run one exact card.

usage:  python -I code_exact/run_exact.py CARD.json OUTDIR/

Reads CARD.json (schema exact_card/1, see code_exact/README.md), writes OUTDIR/<output.file> (schema exact_result/1)
and OUTDIR/progress.txt (one line, overwritten, with ETA).
Exit codes: 0 = Ok; 1 = runtime failure (result file written with status Error and the message);
            2 = invalid card or command line (result file with status Error if OUTDIR is usable).
"""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # code_exact/
ROOT = HERE.parent                                     # release root (holds VERSION)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np                                     # noqa: E402
import h5py                                            # noqa: E402
from lindblad_exact import RESULT_SCHEMA               # noqa: E402
from lindblad_exact.card import parse                  # noqa: E402
from lindblad_exact.states import CardError            # noqa: E402
from lindblad_exact.tasks import Context, Progress, RUNNERS, model_attrs   # noqa: E402


def version_string():
    try:
        v = (ROOT / "VERSION").read_text().strip()
    except OSError:
        v = "unknown"
    try:
        g = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT, capture_output=True, text=True,
                           timeout=20)
        d = g.stdout.strip() if g.returncode == 0 and g.stdout.strip() else "nogit"
    except Exception:
        d = "nogit"
    return f"{v}+{d}"


def _write(grp, d):
    for k, v in d.items():
        if isinstance(v, dict):
            _write(grp.require_group(k), v)
        else:
            grp.create_dataset(k, data=np.asarray(v))


def _attr(v):
    if isinstance(v, (list, tuple)):
        if all(isinstance(x, str) for x in v):
            return ",".join(v)
        return np.asarray(v)
    return v


def task_param_attrs(raw):
    """the card's task entry as attributes (nested objects flattened with '_')"""
    out = {}
    for k, v in raw.items():
        if k == "grid":
            out["grid_json"] = json.dumps(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                out[f"{k}_{kk}"] = _attr(vv)
        else:
            out[k] = _attr(v)
    return out


def main(argv):
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    card_path, outdir = Path(argv[1]), Path(argv[2])
    t0 = time.time()
    try:
        outdir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"cannot create {outdir}: {e}", file=sys.stderr)
        return 2
    version = version_string()
    try:
        text = card_path.read_text()
    except OSError as e:
        print(f"cannot read card: {e}", file=sys.stderr)
        return 2
    try:
        raw = json.loads(text)
        run_id = str(raw.get("run_id", "?")); out_name = str(raw.get("output", {}).get("file", "result.h5"))
    except Exception:
        run_id, out_name = "?", "result.h5"
    if "/" in out_name or not out_name.endswith(".h5"):
        out_name = "result.h5"
    out_file = outdir / out_name
    part = outdir / (out_name + ".part")
    prog = Progress(outdir / "progress.txt", run_id, 1)
    status, message, code = "Ok", "", 0
    results = []
    model = None
    try:
        card = parse(text)
        prog.n = len(card["tasks"])
        print(f"[run_exact] {run_id}  version {version}", flush=True)
        ctx = Context(card, prog)
        model = model_attrs(ctx.fm)
        model["even_dim_used"] = ctx.even.dim
        print(f"[run_exact] model n_orb {ctx.fm.N}, Fock dim {ctx.fm.dim}, even dim {model['dim_even']} "
              f"(reachable from the initial state {ctx.even.dim})", flush=True)
        for k, (t, traw) in enumerate(zip(card["tasks"], raw["tasks"])):
            prog.task(k, t["name"])
            ts = time.time()
            data, attrs = RUNNERS[t["type"]](ctx, t)
            attrs = {**task_param_attrs(traw), **attrs, "wall_seconds": time.time() - ts}
            results.append((t["name"], data, attrs))
            print(f"[run_exact] task {t['name']} ({t['type']}) done in {time.time() - ts:.1f}s", flush=True)
    except CardError as e:
        status, message, code = "Error", f"card error: {e}", 2
    except Exception as e:
        status, message, code = "Error", f"{type(e).__name__}: {e}\n{traceback.format_exc()}", 1
    if code:
        print(f"[run_exact] {message}", file=sys.stderr, flush=True)
    try:
        with h5py.File(part, "w") as f:
            f.attrs["schema"] = RESULT_SCHEMA
            f.attrs["run_id"] = run_id
            f.attrs["release_version"] = version
            f.attrs["status"] = status
            f.attrs["message"] = message
            f.attrs["wall_seconds"] = time.time() - t0
            f.attrs["card_json"] = text
            f.attrs["python_versions"] = (f"python {sys.version.split()[0]}, numpy {np.__version__}, "
                                          f"scipy {__import__('scipy').__version__}, h5py {h5py.__version__}")
            if model is not None:
                g = f.create_group("model")
                for k, v in model.items():
                    g.attrs[k] = v
            for name, data, attrs in results:
                g = f.create_group(name)
                _write(g, data)
                for k, v in attrs.items():
                    g.attrs[k] = v
        os.replace(part, out_file)
    except Exception as e:
        print(f"[run_exact] cannot write {out_file}: {e}", file=sys.stderr)
        code = code or 1
    prog.done(f"{status} (exit {code})")
    print(f"[run_exact] {status}, wall {time.time() - t0:.1f}s -> {out_file}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
