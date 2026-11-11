"""quoted_numbers: run cards for the numbers of the text that no figure or table needs. Writes runcards/, index.csv.
(1) Stability of the MF and TPPX steady states of the strongly damped impurity (Sec. VII C, "an initial polarization
    of 5x10^-4 grows ..."): G1-G2 runs to t = 600 from a slightly polarized impurity and from the polarized start.
(2) Exact plaquette runs at gamma/J = 50, 100 (Sec. VII B, "additional exact runs give gamma kappa / J^2 = ...")."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import runcards as RC, models as M           # noqa: E402

# ----------------------------------------------------------------------------------------- parameters (only here)
STAB_SYSTEM = "c3"
STAB_GAMMAS = {"HF": [4.0, 5.0, 6.0, 7.0, 8.0, 12.0, 24.0], "TPPX": [12.0, 16.0, 20.0, 22.0, 24.0]}
STAB_STARTS = {"pert": (0.101, 0.100), "ic10": (1.0, 0.0)}      # impurity (n_up, n_dn) at t = 0; modes as IC 00
STAB_T_END, STAB_DT = 600.0, 0.5
STAB_TOL = RC.TOL_DEFAULT
BIG_SYSTEM = "plaq"
BIG_ICS = ["mixed", "neel"]
BIG_GAMMAS = [50.0, 100.0]
BIG_GRID = [(20.0, 0.001), (3000.0, 1.0)]                      # exact transient segments (t_end, dt) [1/J]
COST_S = {"stab": 10, "big": 400}
# ---------------------------------------------------------------------------------------------------------------


def main():
    D = RC.fresh_tree(HERE / "runcards")
    rows = []
    RC.write_model(D, STAB_SYSTEM)
    for m, gs in STAB_GAMMAS.items():
        for g in gs:
            for tag, (nu, nd) in STAB_STARTS.items():
                r0 = M.imp_rho0(STAB_SYSTEM, "00"); r0[0, 0, 0] = nu; r0[1, 0, 0] = nd
                rid = RC.run_id(STAB_SYSTEM, tag, g, m, "stab")
                RC.write_driver_card(D, rid, STAB_SYSTEM, g, m, r0, STAB_T_END, STAB_DT, "g1g2", STAB_TOL, 0.0)
                rows.append(dict(run_id=rid, kind="driver", system=STAB_SYSTEM, ic=tag, gamma=g, method=m,
                                 segment="stab", cost_s=COST_S["stab"]))
    RC.write_model(D, BIG_SYSTEM)
    for ic in BIG_ICS:
        for g in BIG_GAMMAS:
            rid = RC.run_id(BIG_SYSTEM, ic, g, "exact")
            RC.write_exact_card(D, rid, BIG_SYSTEM, g, M.plaq_rho0(ic).real,
                                [{"type": "transient", "name": "transient",
                                  "grid": [{"t_end": T, "dt": dt} for T, dt in BIG_GRID], "observables": ["N", "n", "D"]},
                                 {"type": "n_inf", "name": "n_inf", "observables": ["N", "D"]}])
            rows.append(dict(run_id=rid, kind="exact", system=BIG_SYSTEM, ic=ic, gamma=g, method="exact",
                             cost_s=COST_S["big"]))
    RC.write_index(D, rows)
    print(len(rows), "cards ->", D)


if __name__ == "__main__":
    main()
