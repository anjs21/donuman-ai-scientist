"""
selection_agent.py
==================
Phase 3 of the AS-ALD in-silico co-scientist (Challenge 4).

Agentic selection logic for inhibitor / precursor candidates. Consumes the
per-reagent, per-surface energetics from energetics.py and applies the
transparent, criteria-driven decision procedure documented in
`selection_criteria.md`:

  Growth surface (GS)     = SiO2  -> we WANT growth  -> inhibitor should spare it
  Non-growth surface (NGS)= SiNx  -> we want BLOCKED -> inhibitor should bind it

  contrast = dE_ads(GS) - dE_ads(NGS)   (large positive = selective blocker)

The "agent" is a glass-box reasoner: every candidate gets a score decomposed
into named terms (binding, sparing, contrast, volatility, strain), each
candidate is accepted/rejected against editable thresholds, and the agent emits
a ranked shortlist plus a natural-language rationale for its top pick. This is
deliberately auditable rather than a black box, which is what an in-silico
co-scientist needs to be trustworthy.

Depends only on the numbers produced upstream; no GPU needed.
"""

import re
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import inhibitor_library as lib


def _clean(x):
    """Normalise missing/NaN energies to None so gates handle them explicitly."""
    if x is None:
        return None
    try:
        if math.isnan(float(x)):
            return None
    except (TypeError, ValueError):
        return None
    return float(x)


# Which model material plays which role in the target process.
GROWTH_SURFACE = "SiO2"       # GS: keep reactive, grow SiOx film here
NONGROWTH_SURFACE = "SiNx"    # NGS: passivate/block here

DEFAULT_CONFIG = {
    "bind_threshold_eV": -0.30,
    "spare_threshold_eV": -0.20,
    "precursor_threshold_eV": -0.30,
    "contrast_weight": 1.0,
    "volatility_bonus": 0.15,
    "volatility_penalty": 0.15,
    "strain_penalty": 0.10,
    "min_contrast_eV": 0.20,
}


def load_config(md_path="selection_criteria.md"):
    """Parse the ```selection-config block from the criteria markdown."""
    cfg = dict(DEFAULT_CONFIG)
    try:
        with open(md_path, encoding="utf-8") as fh:
            text = fh.read()
    except FileNotFoundError:
        print(f"  [agent] {md_path} not found; using built-in defaults.")
        return cfg
    m = re.search(r"```selection-config(.*?)```", text, re.S)
    if not m:
        return cfg
    for line in m.group(1).splitlines():
        line = line.split("#")[0].strip()
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        try:
            cfg[k] = float(v.strip())
        except ValueError:
            pass
    return cfg


# ===========================================================================
# Candidate evaluation
# ===========================================================================

@dataclass
class Candidate:
    name: str
    role: str                     # 'inhibitor' | 'precursor'
    dE_GS: Optional[float]        # adsorption energy on growth surface
    dE_NGS: Optional[float]       # adsorption energy on non-growth surface
    contrast: Optional[float]
    score: float
    accepted: bool
    reasons: list = field(default_factory=list)
    terms: dict = field(default_factory=dict)
    volatility: str = "medium"


def _score_inhibitor(reagent, dE_GS, dE_NGS, strain_dominated, cfg):
    """Return (score, accepted, reasons, terms) for an inhibitor candidate."""
    reasons, terms = [], {}

    if dE_NGS is None:
        return -1e9, False, ["no NGS data"], terms
    contrast = (dE_GS if dE_GS is not None else 0.0) - dE_NGS

    # hard gates
    accepted = True
    if dE_NGS > cfg["bind_threshold_eV"]:
        accepted = False
        reasons.append(f"does not bind NGS (dE={dE_NGS:+.2f} > "
                       f"{cfg['bind_threshold_eV']:+.2f} eV)")
    if dE_GS is not None and dE_GS < cfg["spare_threshold_eV"]:
        # binds the growth surface too strongly -> would suppress wanted growth
        reasons.append(f"also caps GS (dE={dE_GS:+.2f} < "
                       f"{cfg['spare_threshold_eV']:+.2f} eV)")
    if contrast < cfg["min_contrast_eV"]:
        accepted = False
        reasons.append(f"insufficient contrast ({contrast:+.2f} < "
                       f"{cfg['min_contrast_eV']:+.2f} eV)")

    # score terms
    terms["contrast"] = cfg["contrast_weight"] * contrast
    # reward strong NGS binding beyond threshold
    terms["ngs_binding"] = 0.5 * max(0.0, -(dE_NGS - cfg["bind_threshold_eV"]))
    # penalise GS capping (below spare threshold)
    terms["gs_penalty"] = -0.5 * max(0.0, cfg["spare_threshold_eV"] -
                                     (dE_GS if dE_GS is not None else 0.0))
    terms["volatility"] = (cfg["volatility_bonus"] if reagent.volatility == "high"
                           else -cfg["volatility_penalty"] if reagent.volatility == "low"
                           else 0.0)
    terms["strain"] = -cfg["strain_penalty"] if strain_dominated else 0.0

    score = sum(terms.values())
    return score, accepted, reasons, terms


def _score_precursor(reagent, dE_GS, cfg):
    reasons, terms = [], {}
    if dE_GS is None:
        return -1e9, False, ["no GS data"], terms
    accepted = dE_GS <= cfg["precursor_threshold_eV"]
    if not accepted:
        reasons.append(f"too weak on GS (dE={dE_GS:+.2f} > "
                       f"{cfg['precursor_threshold_eV']:+.2f} eV)")
    terms["gs_binding"] = -dE_GS                       # more negative dE -> higher score
    terms["volatility"] = (cfg["volatility_bonus"] if reagent.volatility == "high"
                           else -cfg["volatility_penalty"] if reagent.volatility == "low"
                           else 0.0)
    return sum(terms.values()), accepted, reasons, terms


# ===========================================================================
# Top-level agent
# ===========================================================================

def select(energetics_results, cfg=None, strain_dominated_names=None):
    """
    Rank inhibitors and precursors from an energetics screen.

    energetics_results : output of energetics.screen_reagents, i.e.
        {reagent_name: {material: {"dE_mean","dE_min", ...}}}
    cfg                : thresholds dict (from load_config()).
    strain_dominated_names : optional set of reagent names whose favourable
        binding was judged strain-dominated (down-weighted).

    Returns dict with ranked "inhibitors", ranked "precursors", and a chosen
    "recommendation" for the target process.
    """
    cfg = cfg or load_config()
    strain_dominated_names = strain_dominated_names or set()

    inhibitors, precursors = [], []
    for name, per_mat in energetics_results.items():
        reagent = lib.LIBRARY.get(name)
        if reagent is None:
            continue
        dE_GS = _clean(per_mat.get(GROWTH_SURFACE, {}).get("dE_mean"))
        dE_NGS = _clean(per_mat.get(NONGROWTH_SURFACE, {}).get("dE_mean"))

        if reagent.category == "inhibitor":
            score, ok, reasons, terms = _score_inhibitor(
                reagent, dE_GS, dE_NGS, name in strain_dominated_names, cfg)
            inhibitors.append(Candidate(
                name, "inhibitor", dE_GS, dE_NGS,
                None if dE_NGS is None else (dE_GS or 0.0) - dE_NGS,
                round(score, 3), ok, reasons, {k: round(v, 3) for k, v in terms.items()},
                reagent.volatility))
        elif reagent.category == "precursor":
            score, ok, reasons, terms = _score_precursor(reagent, dE_GS, cfg)
            precursors.append(Candidate(
                name, "precursor", dE_GS, dE_NGS, None,
                round(score, 3), ok, reasons, {k: round(v, 3) for k, v in terms.items()},
                reagent.volatility))

    inhibitors.sort(key=lambda c: c.score, reverse=True)
    precursors.sort(key=lambda c: c.score, reverse=True)

    best_inhibitor = next((c for c in inhibitors if c.accepted), None)
    best_precursor = next((c for c in precursors if c.accepted), None)

    recommendation = {
        "inhibitor": best_inhibitor.name if best_inhibitor else None,
        "precursor": best_precursor.name if best_precursor else None,
        "inhibitor_dE_NGS": best_inhibitor.dE_NGS if best_inhibitor else None,
        "inhibitor_dE_GS": best_inhibitor.dE_GS if best_inhibitor else None,
        "inhibitor_contrast": best_inhibitor.contrast if best_inhibitor else None,
    }
    return {
        "config": cfg,
        "inhibitors": [asdict(c) for c in inhibitors],
        "precursors": [asdict(c) for c in precursors],
        "recommendation": recommendation,
    }


def explain(selection):
    """Human-readable rationale for the agent's decision."""
    lines = []
    rec = selection["recommendation"]
    lines.append("=" * 66)
    lines.append("AS-ALD SELECTION AGENT — RECOMMENDATION")
    lines.append("=" * 66)
    lines.append(f"Target: passivate {NONGROWTH_SURFACE} (NGS), grow SiOx on "
                 f"{GROWTH_SURFACE} (GS).")
    lines.append("")

    lines.append("Inhibitor ranking (want: strong on NGS, weak on GS):")
    lines.append(f"  {'reagent':16s} {'dE_GS':>7s} {'dE_NGS':>7s} "
                 f"{'contrast':>9s} {'score':>7s}  status")
    for c in selection["inhibitors"]:
        gs = "  n/a" if c["dE_GS"] is None else f"{c['dE_GS']:+.2f}"
        ng = "  n/a" if c["dE_NGS"] is None else f"{c['dE_NGS']:+.2f}"
        ct = "   n/a" if c["contrast"] is None else f"{c['contrast']:+.2f}"
        status = "ACCEPT" if c["accepted"] else "reject: " + "; ".join(c["reasons"])
        lines.append(f"  {c['name']:16s} {gs:>7s} {ng:>7s} {ct:>9s} "
                     f"{c['score']:+7.2f}  {status}")

    lines.append("")
    lines.append("Precursor ranking (want: strong on GS -OH):")
    for c in selection["precursors"]:
        gs = "  n/a" if c["dE_GS"] is None else f"{c['dE_GS']:+.2f}"
        status = "ACCEPT" if c["accepted"] else "reject: " + "; ".join(c["reasons"])
        lines.append(f"  {c['name']:16s} dE_GS={gs:>7s}  score={c['score']:+.2f}  {status}")

    lines.append("")
    if rec["inhibitor"]:
        lines.append(f"--> Recommended inhibitor: {rec['inhibitor']}  "
                     f"(dE_NGS={rec['inhibitor_dE_NGS']:+.2f} eV, "
                     f"dE_GS={rec['inhibitor_dE_GS']:+.2f} eV, "
                     f"contrast={rec['inhibitor_contrast']:+.2f} eV)")
        r = lib.LIBRARY[rec["inhibitor"]]
        lines.append(f"    rationale: binds the {NONGROWTH_SURFACE} sites "
                     f"{r.targets} and is comparatively inert on {GROWTH_SURFACE}, "
                     f"so it passivates the nitride while leaving the oxide -OH "
                     f"free for the precursor. Volatility: {r.volatility}.")
    else:
        lines.append("--> No inhibitor met the acceptance criteria. Widen the "
                     "library or relax thresholds in selection_criteria.md.")
    if rec["precursor"]:
        lines.append(f"--> Recommended precursor: {rec['precursor']} for the "
                     f"SiOx film on {GROWTH_SURFACE}.")
    lines.append("=" * 66)
    return "\n".join(lines)


# ===========================================================================
# Demo with synthetic energetics (no GPU needed) to exercise the logic
# ===========================================================================

if __name__ == "__main__":
    # Synthetic dE_mean values (eV) illustrating a selective vs non-selective case.
    demo = {
        "DMATMS":        {"SiO2": {"dE_mean": -0.15}, "SiNx": {"dE_mean": -0.95}},
        "TMCS":          {"SiO2": {"dE_mean": -1.10}, "SiNx": {"dE_mean": -0.40}},
        "pyridine":      {"SiO2": {"dE_mean": -0.10}, "SiNx": {"dE_mean": -0.55}},
        "NH3":           {"SiO2": {"dE_mean": -0.20}, "SiNx": {"dE_mean": -0.25}},
        "BDMAS":         {"SiO2": {"dE_mean": -0.85}, "SiNx": {"dE_mean": -0.30}},
        "SiCl4":         {"SiO2": {"dE_mean": -1.20}, "SiNx": {"dE_mean": -0.60}},
    }
    cfg = load_config()
    sel = select(demo, cfg)
    print(explain(sel))
    with open("_selection_demo.json", "w") as fh:
        json.dump(sel, fh, indent=2)
    print("\nWrote _selection_demo.json")
