"""Method labels (data), run-id tokens, driver flags and display labels (CONVENTIONS section 0)."""
import h5py

METHODS = ["HF", "2B", "TPPX", "TPPX_0", "mGW", "DSL"]
DRIVER_FLAGS = {
    "HF": dict(approximation="HF", flag_2B=0, bare_propagators=0),
    "2B": dict(approximation="TPPX", flag_2B=1, bare_propagators=0),
    "TPPX": dict(approximation="TPPX", flag_2B=0, bare_propagators=0),
    "TPPX_0": dict(approximation="TPPX", flag_2B=0, bare_propagators=1),
    "mGW": dict(approximation="mGW", flag_2B=0, bare_propagators=0),
    "DSL": dict(approximation="DSL", flag_2B=0, bare_propagators=0),
}


def token(method):
    """run-id token: data label without underscore (TPPX_0 -> TPPX0)"""
    return method.replace("_", "")


def method_from_run_attrs(r):
    """data label from the /input/run attributes (old files: approximation BARE = TPPX_0)"""
    a = r["approximation"]
    a = a.decode() if isinstance(a, bytes) else str(a)
    if a == "BARE":
        return "TPPX_0"
    if a == "TPPX" and int(r.get("flag_2B", 0)):
        return "2B"
    return a + ("_0" if int(r.get("bare_propagators", 0)) else "")


def method_from_output(path_or_file):
    """data label of a driver output file (understands the old BARE files)"""
    f = h5py.File(path_or_file, "r") if not isinstance(path_or_file, h5py.File) else path_or_file
    try:
        if "method" in f["input"].attrs:
            m = f["input"].attrs["method"]
            m = m.decode() if isinstance(m, bytes) else str(m)
            return "TPPX_0" if m == "BARE" else m
        return method_from_run_attrs(dict(f["input/run"].attrs))
    finally:
        if not isinstance(path_or_file, h5py.File):
            f.close()
