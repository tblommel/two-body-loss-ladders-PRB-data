"""Model files for dp5drive (schema 1): writer, validator, content hash, loader.

Model file layout (HDF5):
  /                 attrs dp5_model_schema=1, content_hash (FNV-1a 64, hex), hash_algorithm
  /model            h, ell_greater, ell_lesser: complex [2,N,N] (spin-resolved, spin-diagonal one-body
                    matrices); attrs N, N_greater, N_lesser, mu, Gamma, beta, ordering, units, fit_info, xi=-1
  /model/vertex     lossy_sites int64 [M], site_weight float64 [M]; attrs type, gamma_convention
  /initial/rho      complex [2,N,N] default rho(0) (bath block from the fit; impurity set by the run card)
  /provenance       attrs: generator, created, assumptions, source, ...
The content hash covers h, ell_greater, ell_lesser, rho (complex128 C order), lossy_sites (int64),
site_weight (float64), in that order -- identical to drv::model_hash in the C++ driver.
"""
import datetime
import h5py
import numpy as np

SCHEMA = 1
GAMMA_CONVENTION = ("on-site pair loss L_r = sqrt(gamma w_r) c_{r,dn} c_{r,up}; vee = (gamma w_r/2) eps eps; "
                    "l^{>,MF,sigma}_{rr} = gamma w_r rho^{sigma-bar}_{rr} (spin-diagonal rho), xi = -1")


def fnv1a64(data: bytes, h: int = 1469598103934665603) -> int:
    # vectorized FNV-1a is not possible; process in pure Python but on bytes (fine for model sizes)
    mask = (1 << 64) - 1
    for b in data:
        h ^= b
        h = (h * 1099511628211) & mask
    return h


def content_hash(h, ell_greater, ell_lesser, rho, lossy_sites, site_weight) -> str:
    hv = 1469598103934665603
    for a in (h, ell_greater, ell_lesser, rho):
        hv = fnv1a64(np.ascontiguousarray(a, dtype=np.complex128).tobytes(), hv)
    hv = fnv1a64(np.ascontiguousarray(lossy_sites, dtype=np.int64).tobytes(), hv)
    hv = fnv1a64(np.ascontiguousarray(site_weight, dtype=np.float64).tobytes(), hv)
    return f"{hv:016x}"


class ValidationError(ValueError):
    pass


def validate(h, ell_greater, ell_lesser, rho, lossy_sites, site_weight, N_greater=None, N_lesser=None,
             ordering=None, tol=1e-10):
    """Raise ValidationError on any inconsistency (shapes, Hermiticity, positivity, rho spectrum, ordering)."""
    h, gt, lt, rho = (np.asarray(x, dtype=complex) for x in (h, ell_greater, ell_lesser, rho))
    if h.ndim != 3 or h.shape[0] != 2 or h.shape[1] != h.shape[2]:
        raise ValidationError(f"h must be [2,N,N], got {h.shape}")
    N = h.shape[1]
    for name, a in (("ell_greater", gt), ("ell_lesser", lt), ("rho", rho)):
        if a.shape != h.shape:
            raise ValidationError(f"{name} must be [2,{N},{N}], got {a.shape}")
    for name, a in (("h", h), ("ell_greater", gt), ("ell_lesser", lt), ("rho", rho)):
        for s in range(2):
            err = np.max(np.abs(a[s] - a[s].conj().T))
            if err > tol * max(1.0, np.max(np.abs(a[s]))):
                raise ValidationError(f"{name}[{s}] not Hermitian (max |A - A^dag| = {err:.2e})")
    for name, a in (("ell_greater", gt), ("ell_lesser", lt)):
        for s in range(2):
            ev = np.linalg.eigvalsh(0.5 * (a[s] + a[s].conj().T))
            if ev.min() < -tol:
                raise ValidationError(f"{name}[{s}] not positive semidefinite (min eigenvalue {ev.min():.2e})")
    for s in range(2):
        ev = np.linalg.eigvalsh(0.5 * (rho[s] + rho[s].conj().T))
        if ev.min() < -tol or ev.max() > 1 + tol:
            raise ValidationError(f"rho[{s}] spectrum outside [0,1]: [{ev.min():.3e}, {ev.max():.3e}]")
    ls = np.asarray(lossy_sites, dtype=np.int64)
    sw = np.asarray(site_weight, dtype=float)
    if ls.ndim != 1 or len(ls) == 0 or len(set(ls.tolist())) != len(ls) or ls.min() < 0 or ls.max() >= N:
        raise ValidationError(f"lossy_sites must be distinct orbitals in [0,{N}), got {ls}")
    if sw.shape != ls.shape or np.any(sw < 0) or not np.all(np.isfinite(sw)):
        raise ValidationError("site_weight must be finite, >= 0, one per lossy site")
    if ordering is not None and ordering == "d,p>,p<":
        if N_greater is None or N_lesser is None or N != 1 + N_greater + N_lesser:
            raise ValidationError(f"ordering d,p>,p< needs N = 1 + N_greater + N_lesser (N={N}, "
                                  f"N_greater={N_greater}, N_lesser={N_lesser})")
    return N


def write_model(path, h, ell_greater, ell_lesser, rho, lossy_sites=(0,), site_weight=None, attrs=None,
                provenance=None):
    """Validate and write a model file; returns the content hash."""
    attrs = dict(attrs or {})
    site_weight = np.ones(len(lossy_sites)) if site_weight is None else np.asarray(site_weight, dtype=float)
    N = validate(h, ell_greater, ell_lesser, rho, lossy_sites, site_weight, attrs.get("N_greater"),
                 attrs.get("N_lesser"), attrs.get("ordering"))
    arrs = [np.ascontiguousarray(a, dtype=np.complex128) for a in (h, ell_greater, ell_lesser, rho)]
    ls = np.ascontiguousarray(lossy_sites, dtype=np.int64)
    sw = np.ascontiguousarray(site_weight, dtype=np.float64)
    hsh = content_hash(*arrs, ls, sw)
    with h5py.File(path, "w") as f:
        f.attrs["dp5_model_schema"] = SCHEMA
        f.attrs["content_hash"] = hsh
        f.attrs["hash_algorithm"] = "FNV-1a 64 over h, ell_greater, ell_lesser, rho (complex128), lossy_sites (int64), site_weight (float64)"
        g = f.create_group("model")
        for n, a in zip(("h", "ell_greater", "ell_lesser"), arrs[:3]):
            g.create_dataset(n, data=a)
        g.attrs["N"] = N
        g.attrs["xi"] = -1
        for k, v in attrs.items():
            if v is not None:
                g.attrs[k] = v
        v = g.create_group("vertex")
        v.create_dataset("lossy_sites", data=ls)
        v.create_dataset("site_weight", data=sw)
        v.attrs["type"] = "onsite_pair_loss"
        v.attrs["gamma_convention"] = GAMMA_CONVENTION
        f.create_group("initial").create_dataset("rho", data=arrs[3])
        p = f.create_group("provenance")
        p.attrs["created"] = datetime.datetime.now().isoformat(timespec="seconds")
        for k, val in (provenance or {}).items():
            p.attrs[k] = val
    return hsh


def load_model(path):
    with h5py.File(path, "r") as f:
        g = f["model"]
        m = dict(h=g["h"][()], ell_greater=g["ell_greater"][()], ell_lesser=g["ell_lesser"][()],
                 rho=f["initial/rho"][()], lossy_sites=g["vertex/lossy_sites"][()],
                 site_weight=g["vertex/site_weight"][()], hash=f.attrs["content_hash"],
                 attrs={k: g.attrs[k] for k in g.attrs})
    hc = content_hash(m["h"], m["ell_greater"], m["ell_lesser"], m["rho"], m["lossy_sites"], m["site_weight"])
    if hc != m["hash"]:
        raise ValidationError(f"content hash mismatch: stored {m['hash']}, computed {hc}")
    return m


def from_spin_orbital(A, lossy_sites=(), tol=1e-12, what="rho"):
    """Convert a spin-orbital matrix [2N,2N] (index 2r + sigma, sigma = 0 up) into the [2,N,N] spin blocks
    used by model files and run cards. The driver assumes (A1): spin-diagonal one-body matrices. The MF loss
    rate implemented, l^{>,MF,sigma}_{rr} = gamma w_r rho^{sigma-bar}_{rr}, is main's Eq. (MFrates)
    gamma [tr rho_r 1 - rho_r] restricted to spin-diagonal rho_r. Spin-off-diagonal elements on a lossy site
    are therefore refused (ValidationError); elsewhere they are refused as well, since no spin-off-diagonal
    block can be represented (a warning names the largest one)."""
    A = np.asarray(A, dtype=complex)
    M = A.shape[0]
    if A.shape != (M, M) or M % 2:
        raise ValidationError(f"{what}: expected a [2N,2N] spin-orbital matrix, got {A.shape}")
    off = np.abs(A[0::2, 1::2]) + np.abs(A[1::2, 0::2].T)
    for r in lossy_sites:
        if off[r, r] > tol:
            raise ValidationError(f"{what}: spin-off-diagonal element on lossy site {r} ({off[r, r]:.2e}); "
                                  "the MF rate gamma[tr rho_r 1 - rho_r] would need the full 2x2 block (A1 violated)")
    if off.max() > tol:
        i, j = np.unravel_index(np.argmax(off), off.shape)
        raise ValidationError(f"{what}: spin-off-diagonal block not representable (largest |{what}_(({i},up),({j},dn))| = "
                              f"{off.max():.2e}); the driver assumes spin-diagonal one-body matrices (A1)")
    return np.stack([A[0::2, 0::2], A[1::2, 1::2]])
