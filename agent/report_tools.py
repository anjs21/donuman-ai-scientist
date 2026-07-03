"""Report loading, prompt-shaping, and the instant re-selection loop.

Bridges the pipeline's artefacts (report.json + selection_criteria.md) and the
LLM surface:

  * load_report / find_reports    -- locate and read a pipeline report.
  * report_digest                 -- compact, token-cheap text for the LLM.
  * current_config / config_bounds -- the editable selection thresholds.
  * reselect                      -- re-run Phase 3 (selection) and Phase 4
                                     (selectivity) from a report's already-
                                     computed energetics, with overridden
                                     thresholds. No GPU, no MACE, sub-second.
  * apply_config_to_md            -- persist chosen thresholds back into the
                                     fenced block of selection_criteria.md.

reselect is what makes Job 3 usable on an 8 GB laptop: threshold tuning only
touches phases that consume energetics, so it never re-runs the MD/MLIP work.
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import selection_agent as agent
import selectivity_model as sm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CRITERIA_MD = os.path.join(REPO_ROOT, "selection_criteria.md")


# --------------------------------------------------------------------------
# Report I/O
# --------------------------------------------------------------------------

def find_reports(root: str = REPO_ROOT) -> List[str]:
    """All report*.json under `root` (recursive), newest first."""
    hits = glob.glob(os.path.join(root, "**", "report*.json"), recursive=True)
    return sorted(set(hits), key=os.path.getmtime, reverse=True)


def load_report(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def find_surface_sets(root: str = REPO_ROOT) -> Dict[str, List[str]]:
    """Directories holding reusable surface *slabs* keyed by relative dir.

    A slab file is a material-named ``*.xyz`` or ``*.vasp`` (contains "SiO2",
    "SiNx", "oxide", or "nitride" in the filename) that is NOT a bulk cache
    entry -- bulk structures have no surface and cannot be reused as slabs.
    Passing one of these dirs' globs to run_pipeline's ``--use-existing`` skips
    Phase 1 (the melt-quench MD), which is the expensive MLIP step.
    Newest set first.
    """
    sets: Dict[str, List[str]] = {}
    for ext in ("*.xyz", "*.vasp"):
        for p in glob.glob(os.path.join(root, "**", ext), recursive=True):
            base = os.path.basename(p)
            if "bulk_cache" in p or ".venv" in p or "_bulk_" in base:
                continue
            if _material_of(base) is None:
                continue
            sets.setdefault(os.path.dirname(p), []).append(p)

    def _n_materials(files: List[str]) -> int:
        return len({_material_of(os.path.basename(f))
                    for f in files
                    if _material_of(os.path.basename(f))})

    # Complete sets (both materials) first, then newest.
    return dict(sorted(
        ((d, sorted(f)) for d, f in sets.items()),
        key=lambda kv: (_n_materials(kv[1]),
                        max(os.path.getmtime(f) for f in kv[1])),
        reverse=True))




def reagent_names() -> List[str]:
    """Inhibitor + precursor names available in the library (for re-screening)."""
    import inhibitor_library as lib
    return [r.name for r in lib.get_reagents()]


def reagents_by_category() -> Dict[str, List[str]]:
    """Library reagent names grouped by category (inhibitor/precursor/coreactant)."""
    import inhibitor_library as lib
    out: Dict[str, List[str]] = {}
    for r in lib.get_reagents():
        out.setdefault(r.category, []).append(r.name)
    return out


_MATERIAL_KEYWORDS: Dict[str, str] = {
    "SiO2": "SiO2", "oxide": "SiO2",
    "SiNx": "SiNx", "nitride": "SiNx",
}


def _material_of(basename: str) -> Optional[str]:
    """Return canonical material name ('SiO2' or 'SiNx') for a filename, or None."""
    for kw, mat in _MATERIAL_KEYWORDS.items():
        if kw in basename:
            return mat
    return None


def set_materials(files: List[str]) -> List[str]:
    """Which substrate materials a surface set covers (from slab file names)."""
    return sorted({m for f in files
                   if (m := _material_of(os.path.basename(f))) is not None})


# --------------------------------------------------------------------------
# Prompt shaping
# --------------------------------------------------------------------------

def _fmt(x, spec="+.2f"):
    return "n/a" if x is None else format(x, spec)


def report_digest(report: Dict[str, Any]) -> str:
    """A compact plain-text summary of a report for the LLM context window.

    Deliberately terse: the whole point of the on-device model is a small
    context, so we hand it the decision-relevant numbers, not the raw JSON.
    """
    L: List[str] = []
    L.append(f"Run mode: {report.get('mode')}   "
             f"materials: {report.get('materials')}")
    if report.get("placeholder_calc"):
        L.append("WARNING: run used a Lennard-Jones placeholder calculator -- "
                 "energies are NOT physical.")
    L.append(f"Runtime: {report.get('runtime_s')} s")

    # Surfaces / validation
    val = report.get("validation", {})
    for mat, entries in val.items():
        areas = ", ".join(f"{e.get('area_nm2')} nm^2"
                          f"{'' if e.get('structure_ok') else ' (off-target)'}"
                          for e in entries)
        L.append(f"  {mat}: {len(entries)} surface(s); area {areas}")

    # Selection ranking
    sel = report.get("selection", {})
    rec = sel.get("recommendation", {})
    L.append("")
    L.append("Inhibitor ranking (want strong binding on SiNx/NGS, weak on "
             "SiO2/GS; contrast = dE_GS - dE_NGS, large positive = selective):")
    L.append("  reagent           dE_GS   dE_NGS  contrast   score  status")
    for c in sel.get("inhibitors", []):
        status = "ACCEPT" if c.get("accepted") else \
            "reject (" + "; ".join(c.get("reasons", [])) + ")"
        L.append(f"  {c.get('name',''):16s} {_fmt(c.get('dE_GS')):>6s} "
                 f"{_fmt(c.get('dE_NGS')):>7s} {_fmt(c.get('contrast')):>8s} "
                 f"{_fmt(c.get('score')):>7s}  {status}")
    L.append("Precursor ranking (want strong binding on SiO2/GS -OH):")
    for c in sel.get("precursors", []):
        status = "ACCEPT" if c.get("accepted") else \
            "reject (" + "; ".join(c.get("reasons", [])) + ")"
        L.append(f"  {c.get('name',''):16s} dE_GS={_fmt(c.get('dE_GS')):>6s}  "
                 f"score={_fmt(c.get('score')):>6s}  {status}")

    # Selectivity vs target
    gp = report.get("growth_params", {})
    tgt_s = gp.get("target_selectivity")
    tgt_t = gp.get("target_thickness_nm")
    L.append("")
    L.append(f"Selectivity vs target "
             f"({'?' if tgt_s is None else f'{tgt_s*100:.0f}%'} @ "
             f"{tgt_t} nm oxide):")
    for name, s in report.get("selectivity", {}).items():
        sat = s.get("selectivity_at_target")
        mx = s.get("max_thickness_at_target_selectivity_nm")
        L.append(f"  {name:16s} S@target="
                 f"{'n/a' if sat is None else f'{sat*100:.1f}%'}  "
                 f"max_thk@targetS="
                 f"{'n/a' if mx is None else f'{mx:.2f} nm'}  "
                 f"meets={'yes' if s.get('meets_target') else 'no'}")

    L.append("")
    L.append(f"Agent recommendation: inhibitor={rec.get('inhibitor')}, "
             f"precursor={rec.get('precursor')}")
    L.append(f"Thresholds used: {json.dumps(sel.get('config', {}))}")
    return "\n".join(L)


# --------------------------------------------------------------------------
# Config (selection thresholds)
# --------------------------------------------------------------------------

# Sensible edit ranges so a small model can't propose nonsense values, and the
# UI can render sliders. (eV for energies, dimensionless for weights.)
CONFIG_BOUNDS: Dict[str, Tuple[float, float]] = {
    "bind_threshold_eV": (-2.0, 0.0),
    "spare_threshold_eV": (-1.0, 0.5),
    "precursor_threshold_eV": (-2.0, 0.0),
    "contrast_weight": (0.0, 3.0),
    "volatility_bonus": (0.0, 1.0),
    "volatility_penalty": (0.0, 1.0),
    "strain_penalty": (0.0, 1.0),
    "min_contrast_eV": (0.0, 1.0),
}


def current_config() -> Dict[str, float]:
    return agent.load_config(CRITERIA_MD)


def clamp_config(overrides: Dict[str, float]) -> Dict[str, float]:
    """Keep only known keys and clamp each to its allowed range."""
    out = {}
    for k, v in overrides.items():
        if k not in CONFIG_BOUNDS:
            continue
        lo, hi = CONFIG_BOUNDS[k]
        try:
            out[k] = max(lo, min(hi, float(v)))
        except (TypeError, ValueError):
            continue
    return out


# --------------------------------------------------------------------------
# Instant re-selection (Job 3, cheap path)
# --------------------------------------------------------------------------

def reselect(report: Dict[str, Any],
             overrides: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Re-run selection + selectivity from a report's energetics.

    Returns a dict shaped like the report's own "selection"/"selectivity"
    sections so the UI can diff old vs new. Pure CPU; typically <1 s.
    """
    energetics = report.get("energetics", {})
    if not energetics:
        raise ValueError("report has no 'energetics' block to re-select from")

    cfg = dict(current_config())
    cfg.update(clamp_config(overrides or {}))

    selection = agent.select(energetics, cfg)

    # Phase 4: selectivity curves for inhibitors that have NGS data (mirrors
    # run_pipeline.main so numbers are consistent with a real run).
    p = sm.GrowthParams()
    selectivity: Dict[str, Any] = {}
    for cand in selection["inhibitors"]:
        dE_GS, dE_NGS = cand["dE_GS"], cand["dE_NGS"]
        if dE_NGS is None:
            continue
        res = sm.selectivity_curve(dE_GS if dE_GS is not None else 0.0, dE_NGS, p)
        selectivity[cand["name"]] = {
            k: (float(v) if hasattr(v, "__float__") and not hasattr(v, "__len__")
                else None)
            for k, v in res.items() if not hasattr(v, "__len__")
        }
    return {"config": cfg, "selection": selection, "selectivity": selectivity}


# --------------------------------------------------------------------------
# Persist chosen thresholds back to the criteria file
# --------------------------------------------------------------------------

def apply_config_to_md(overrides: Dict[str, float],
                       md_path: str = CRITERIA_MD) -> List[Tuple[str, float, float]]:
    """Rewrite matched keys inside the ```selection-config fenced block.

    Preserves inline comments and every other line. Returns a list of
    (key, old_value, new_value) for the keys actually changed.
    """
    overrides = clamp_config(overrides)
    if not overrides:
        return []
    with open(md_path, encoding="utf-8") as fh:
        text = fh.read()

    m = re.search(r"(```selection-config\n)(.*?)(```)", text, re.S)
    if not m:
        raise ValueError("no ```selection-config block found in " + md_path)

    changed: List[Tuple[str, float, float]] = []
    body_lines = m.group(2).splitlines(keepends=True)
    out_lines = []
    line_re = re.compile(r"^(\s*)([A-Za-z_]+)(\s*:\s*)([-\d.]+)(.*)$")
    for line in body_lines:
        lm = line_re.match(line)
        if lm and lm.group(2) in overrides:
            key = lm.group(2)
            old = float(lm.group(4))
            new = float(overrides[key])
            if old != new:
                changed.append((key, old, new))
            newtext = format(new, "g")
            out_lines.append(f"{lm.group(1)}{key}{lm.group(3)}{newtext}{lm.group(5)}\n")
        else:
            out_lines.append(line)

    new_text = text[:m.start(2)] + "".join(out_lines) + text[m.end(2):]
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(new_text)
    return changed


# --------------------------------------------------------------------------
# Structured-suggestion schema for Job 3
# --------------------------------------------------------------------------

SUGGESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string"},
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "enum": list(CONFIG_BOUNDS.keys())},
                    "to": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["key", "to", "reason"],
            },
        },
        "expected_effect": {"type": "string"},
    },
    "required": ["diagnosis", "changes", "expected_effect"],
}
