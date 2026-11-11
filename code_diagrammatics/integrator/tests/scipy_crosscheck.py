"""Cross-check dp5_integrator against scipy.integrate.solve_ivp.

Usage: python3 -I scipy_crosscheck.py DIR   (DIR written by `./test_dp5 scipy DIR`)

Same nonlinear complex test ODE as scipy_export() in test_dp5.cpp:
    y_i' = lam_i y_i + 0.3 i |y_i|^2 y_i,  t in [0, 10], rtol=1e-6, atol=1e-9.
Checks
  1. step control: the 'standard' variant (scipy controller) must reproduce
     scipy RK45's accepted step sequence;
  2. dense output: our grid frames vs scipy RK45 dense output and vs a DOP853
     reference at rtol=1e-13.
"""
import sys
import numpy as np
from scipy.integrate import solve_ivp

d = sys.argv[1]
lam = np.array([-0.3 + 1j, -0.1 + 3j, -1 + 0.5j, 0.2j])
y0 = np.array([1, 0.5 + 0.5j, -0.3j, 0.8], dtype=complex)


def f(t, y):
    return lam * y + 0.3j * np.abs(y) ** 2 * y


ok = True
ref = solve_ivp(f, (0, 10), y0, method="DOP853", rtol=1e-13, atol=1e-15, dense_output=True)
sp = solve_ivp(f, (0, 10), y0, method="RK45", rtol=1e-6, atol=1e-9, dense_output=True)
t_sp = sp.t  # accepted step end points (incl. t0)

for tag in ["standard", "pi"]:
    st = np.loadtxt(f"{d}/steps_{tag}.txt")
    acc = st[st[:, 3] == 1]
    rej = st[st[:, 3] == 0]
    t_ours = np.concatenate([[0.0], acc[:, 0][1:], [acc[-1, 0] + acc[-1, 1]]])
    fr = np.loadtxt(f"{d}/frames_{tag}.txt")
    tg = fr[:, 0]
    yo = fr[:, 1::2] + 1j * fr[:, 2::2]
    e_ref = np.max(np.abs(yo - ref.sol(tg).T))
    e_sp = np.max(np.abs(yo - sp.sol(tg).T))
    e_sp_ref = np.max(np.abs(sp.sol(tg).T - ref.sol(tg).T))
    print(f"[{tag}] accepted={len(acc)} rejected={len(rej)}   scipy RK45: accepted={len(t_sp)-1} nfev={sp.nfev}")
    if tag == "standard":
        same = len(t_ours) == len(t_sp)
        dt = np.max(np.abs(t_ours - t_sp)) if same else np.inf
        print(f"   step sequence identical length: {same};  max |t_ours - t_scipy| = {dt:.2e}")
        ok &= same and dt < 1e-9
    print(f"   dense frames (201 pts): max|ours - DOP853 ref| = {e_ref:.2e};  max|ours - scipy RK45| = {e_sp:.2e};"
          f"  (scipy RK45 vs ref: {e_sp_ref:.2e})")
    ok &= e_ref < 10 * max(e_sp_ref, 1e-9)
    if tag == "standard":
        ok &= e_sp < 1e-9
print("SCIPY CROSS-CHECK", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
