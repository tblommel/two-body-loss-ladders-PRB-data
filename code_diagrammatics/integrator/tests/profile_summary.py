"""Summarize a dp5 profile from its flat export alone.

Usage: python3 -I profile_summary.py profile_dump.json

The JSON has {"num": {key: [values]}, "str": {key: string}} with the keys of
dp5::Profile::visit() / strings() (schema in DESIGN.md sec. 10.5). Everything
below is recomputed from raw keys and checked against the derived keys that the
integrator exports, which shows that the export is sufficient on its own.
"""
import json
import sys

d = json.load(open(sys.argv[1]))
num, st = d["num"], d["str"]


def v(k):
    return num[k][0]


assert int(v("profile/schema_version")) == 1, "unknown schema version"
sections = st["time/section_names"].split(",")
USER = {"stop_callback", "observer", "step_observer", "interrupt"}
PASS = {"pass_stage2", "pass_stage3", "pass_stage4", "pass_stage5", "pass_stage6", "pass_solution",
        "pass_error", "pass_init_step", "dense"}
names = st["blocks/names"].split(",")

T = v("time/integrate_seconds")
sec = {s: v(f"time/{s}/seconds") for s in sections}
calls = {s: v(f"time/{s}/calls") for s in sections}
rhs = sum(t for s, t in sec.items() if s.startswith("rhs_"))
user = sum(t for s, t in sec.items() if s in USER)
over = T - rhs - user
passes = sum(t for s, t in sec.items() if s in PASS)
closure = sum(sec.values()) + v("time/untimed_seconds") - T

print(f"profile schema {int(v('profile/schema_version'))}, {int(v('context/threads'))} threads, "
      f"{st['context/compiler']} | {st['context/build_flags']}")
print(f"integrate {T:.4f} s: RHS {rhs:.4f} s ({100*rhs/T:.1f}%), user {user:.4f} s ({100*user/T:.1f}%), "
      f"overhead {over:.4f} s ({100*over/T:.1f}%; passes {passes:.4f} s); closure {closure:.1e} s")
print("  section            seconds   %total   calls")
for s in sections:
    if calls[s] > 0:
        print(f"  {s:<17} {sec[s]:9.4f} {100*sec[s]/T:7.2f}% {int(calls[s]):7d}")

acc, att = v("steps/accepted"), v("steps/attempts")
rej = v("steps/rejected_error") + v("steps/rejected_nonfinite")
hist = num["steps/h_hist"]
lo, per = v("steps/h_hist_lo_log10"), v("steps/h_bins_per_decade")
cum, med = 0, 0.0
for b, c in enumerate(hist):
    cum += c
    if 2 * cum >= acc and acc > 0:
        med = 10 ** (lo + (b + 0.5) / per)
        break
rhs_expected = 6 * att + v("steps/slope_recomputes") + v("steps/init_step_estimates")
print(f"steps: attempts {int(att)} = accepted {int(acc)} + rejected {int(rej)}; clipped {int(v('steps/clipped'))}; "
      f"RHS calls {int(v('steps/rhs_calls'))} (6*attempts + slope + h0 = {int(rhs_expected)}); "
      f"rejection rate {100*rej/att:.2f}%")
print(f"h: min {v('steps/h_min'):.3e} median~ {med:.3e} geomean {v('steps/h_geomean'):.3e} max {v('steps/h_max'):.3e}")
lim = num["steps/limiting_block"]
print("limiting block: " + ", ".join(f"{names[i]} {100*c/acc:.1f}%" for i, c in enumerate(lim) if c))
print("segments:   t_begin     t_end   seconds  accepted   RHS%  user%   ovh%")
for k in range(len(num["segments/t_begin"])):
    S = num["segments/integrate_seconds"][k]
    r = sum(num[f"segments/time/{s}/seconds"][k] for s in sections if s.startswith("rhs_"))
    u = sum(num[f"segments/time/{s}/seconds"][k] for s in sections if s in USER)
    o = S - r - u
    assert abs(o - num["segments/overhead_seconds"][k]) <= 1e-12 + 1e-9 * S
    S1 = S if S > 0 else 1.0
    print(f"  {k:<3} {num['segments/t_begin'][k]:9.4g} {num['segments/t_end'][k]:9.4g} {S:9.4f} "
          f"{int(num['segments/accepted'][k]):9d} {100*r/S1:6.1f} {100*u/S1:6.1f} {100*o/S1:6.1f}")
print(f"memory: registers {v('memory/register_bytes')/1e6:.1f} MB (peak {v('memory/register_bytes_peak')/1e6:.1f}, "
      f"transient {v('memory/layout_transient_peak_bytes')/1e6:.1f}), dense {v('memory/dense_bytes')/1e6:.2f} MB, "
      f"peak RSS {v('memory/maxrss_bytes')/1e6:.1f} MB")
print(f"end: t={v('end/t'):.6g} status={st['end/status_name']} worst block {int(v('end/last_worst_block'))}")

# consistency of recomputed vs exported derived numbers
ok = (abs(rhs - v("time/rhs_seconds")) < 1e-12 + 1e-9 * T and abs(over - v("time/overhead_seconds")) < 1e-12 + 1e-9 * T
      and abs(closure) < 1e-9 and int(rhs_expected) == int(v("steps/rhs_calls"))
      and sum(lim) == acc and sum(hist) == acc)
print("EXPORT CONSISTENT" if ok else "EXPORT INCONSISTENT")
sys.exit(0 if ok else 1)
