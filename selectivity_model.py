"""
selectivity_model.py
====================
Phase 4 of the AS-ALD in-silico co-scientist (Challenge 4).

Closes the loop to the actual problem-statement metric: given the inhibitor
binding energetics from Phase 2/3, predict film thickness on the growth and
non-growth surfaces cycle-by-cycle and report the ASD selectivity

    S = (Thk_GS - Thk_NGS) / (Thk_GS + Thk_NGS)          (slide 8 definition)

together with whether the target "90% selectivity at 10 nm of oxide" is met.

Physical model (phenomenological, transparent, and calibratable)
----------------------------------------------------------------
On each surface the inhibitor blocks a fraction of sites. We map the inhibitor
adsorption energy dE (eV, negative = stronger binding) to a blocked-site
fraction theta via a Boltzmann-style sigmoid:

    theta(dE) = 1 / (1 + exp((dE - dE_half) / width))

    - strongly binding (very negative dE)  -> theta -> 1 (well blocked)
    - weakly/non binding (dE >~ 0)          -> theta -> 0 (not blocked)
    - theta = 0.5 exactly at dE = dE_half (the half-blocking energy)

Growth per cycle is suppressed in proportion to open sites, and blocked
surfaces additionally show a nucleation delay (cycles before defect-driven
break-through nucleates):

    GPC_eff(surface) = GPC_intrinsic * (1 - theta)
    n_delay(surface) = delay_max * theta
    Thk(n, surface)  = GPC_eff * max(0, n - n_delay)

The inhibitor is applied to BOTH surfaces (as in a real ASD cycle); selectivity
emerges because theta_NGS >> theta_GS when the inhibitor is well chosen. All
parameters are exposed and documented so they can be fit to experimental
break-through curves — the model is a reasoning scaffold, not a claim of
first-principles accuracy.

Pure numpy; no GPU.
"""

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Default kinetic parameters. Sources / rationale in comments; override freely.
# ---------------------------------------------------------------------------
@dataclass
class GrowthParams:
    gpc_intrinsic_A: float = 1.2   # SiO2 ALD growth-per-cycle, ~1.0-1.4 A/cyc
    dE_half_eV: float = -0.50      # dE giving 50% site blocking
    width_eV: float = 0.25         # sigmoid width (thermal/heterogeneity spread)
    delay_max_cycles: float = 25.0 # max nucleation delay on a fully blocked surface
    target_thickness_nm: float = 10.0
    target_selectivity: float = 0.90


def blocked_fraction(dE_eV, p: GrowthParams):
    """Fraction of sites blocked by the inhibitor given its adsorption energy."""
    if dE_eV is None or np.isnan(dE_eV):
        return 0.0
    return float(1.0 / (1.0 + np.exp((dE_eV - p.dE_half_eV) / p.width_eV)))


def thickness_curve(dE_eV, p: GrowthParams, n_cycles):
    """Return (theta, gpc_eff, n_delay, thickness_A[array]) for one surface."""
    theta = blocked_fraction(dE_eV, p)
    gpc_eff = p.gpc_intrinsic_A * (1.0 - theta)
    n_delay = p.delay_max_cycles * theta
    n = np.arange(n_cycles + 1)
    thk = gpc_eff * np.maximum(0.0, n - n_delay)
    return theta, gpc_eff, n_delay, thk


def selectivity_curve(dE_GS, dE_NGS, p: GrowthParams, n_cycles=200):
    """
    Cycle-resolved ASD selectivity for an inhibitor with the given adsorption
    energies on the growth (GS) and non-growth (NGS) surfaces.

    Returns a dict with the per-cycle arrays and the headline numbers the
    challenge asks for.
    """
    theta_gs, gpc_gs, delay_gs, thk_gs = thickness_curve(dE_GS, p, n_cycles)
    theta_ngs, gpc_ngs, delay_ngs, thk_ngs = thickness_curve(dE_NGS, p, n_cycles)

    denom = thk_gs + thk_ngs
    S = np.divide(thk_gs - thk_ngs, denom, out=np.ones_like(denom), where=denom > 0)

    # cycle at which the GS reaches the target oxide thickness
    target_A = p.target_thickness_nm * 10.0
    if gpc_gs > 0:
        n_target = target_A / gpc_gs + delay_gs
    else:
        n_target = np.inf
    n_target_int = int(np.clip(np.ceil(n_target), 0, n_cycles))
    S_at_target = float(S[n_target_int]) if np.isfinite(n_target) else float("nan")
    thk_ngs_at_target = float(thk_ngs[n_target_int]) if np.isfinite(n_target) else float("nan")

    # max GS thickness while selectivity still >= target
    ok = S >= p.target_selectivity
    if ok.any():
        last_ok = np.where(ok)[0].max()
        max_thk_at_target_S_nm = float(thk_gs[last_ok] / 10.0)
    else:
        max_thk_at_target_S_nm = 0.0

    return {
        "n": np.arange(n_cycles + 1),
        "thk_GS_A": thk_gs,
        "thk_NGS_A": thk_ngs,
        "selectivity": S,
        "theta_GS": theta_gs,
        "theta_NGS": theta_ngs,
        "gpc_GS_A": gpc_gs,
        "gpc_NGS_A": gpc_ngs,
        "n_delay_GS": delay_gs,
        "n_delay_NGS": delay_ngs,
        "n_target": float(n_target),
        "selectivity_at_target": S_at_target,
        "thk_NGS_at_target_nm": thk_ngs_at_target / 10.0 if np.isfinite(thk_ngs_at_target) else float("nan"),
        "max_thickness_at_target_selectivity_nm": max_thk_at_target_S_nm,
        "meets_target": bool(np.isfinite(S_at_target)
                             and S_at_target >= p.target_selectivity),
    }


def summarize(name, res, p: GrowthParams):
    """Human-readable summary of a selectivity result."""
    lines = []
    lines.append("-" * 66)
    lines.append(f"Selectivity model — inhibitor: {name}")
    lines.append(f"  blocked fraction: GS(theta)={res['theta_GS']:.2f}  "
                 f"NGS(theta)={res['theta_NGS']:.2f}")
    lines.append(f"  GPC_eff: GS={res['gpc_GS_A']:.2f} A/cyc  "
                 f"NGS={res['gpc_NGS_A']:.2f} A/cyc  "
                 f"(nucleation delay NGS={res['n_delay_NGS']:.0f} cyc)")
    lines.append(f"  cycles to {p.target_thickness_nm:.0f} nm oxide (GS): "
                 f"{res['n_target']:.0f}")
    lines.append(f"  selectivity at {p.target_thickness_nm:.0f} nm: "
                 f"{res['selectivity_at_target']*100:.1f}%  "
                 f"(NGS parasitic film: {res['thk_NGS_at_target_nm']:.2f} nm)")
    lines.append(f"  max oxide thickness while S >= "
                 f"{p.target_selectivity*100:.0f}%: "
                 f"{res['max_thickness_at_target_selectivity_nm']:.2f} nm")
    verdict = "MEETS" if res["meets_target"] else "DOES NOT MEET"
    lines.append(f"  => {verdict} the target "
                 f"({p.target_selectivity*100:.0f}% @ "
                 f"{p.target_thickness_nm:.0f} nm)")
    lines.append("-" * 66)
    return "\n".join(lines)


def write_curve_csv(path, res):
    """Dump the cycle-resolved curve for external plotting."""
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cycle", "thk_GS_A", "thk_NGS_A", "selectivity"])
        for i in range(len(res["n"])):
            w.writerow([int(res["n"][i]),
                        round(float(res["thk_GS_A"][i]), 3),
                        round(float(res["thk_NGS_A"][i]), 3),
                        round(float(res["selectivity"][i]), 4)])


# ===========================================================================
# Demo
# ===========================================================================

if __name__ == "__main__":
    p = GrowthParams()
    # A selective inhibitor (weak on GS, strong on NGS) vs a poor one.
    for name, dE_GS, dE_NGS in [
        ("DMATMS (selective)", -0.15, -0.95),
        ("TMCS (wrong-way)",   -1.10, -0.40),
        ("NH3 (non-selective)", -0.20, -0.25),
    ]:
        res = selectivity_curve(dE_GS, dE_NGS, p)
        print(summarize(name, res, p))
    # write one curve
    res = selectivity_curve(-0.15, -0.95, p)
    write_curve_csv("_selectivity_demo.csv", res)
    print("Wrote _selectivity_demo.csv")
