"""Control-gated ranking, method consensus, and baseline lift.

A ranking is only labelled control-calibrated when every literature control
reproduces its expected surface preference with the same method and settings.
Baselines answer "would a simpler rule have picked the same molecule?".
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from .common import clean
from .process import ProcessParams, design


def direction(kinetics):
    """prefers_SiNx when capping is faster on the nitride (lower barrier)."""
    d = (kinetics.get("summary") or {}).get("dEa_GS_minus_NGS_eV")
    if d is None:
        return None, None
    return ("prefers_SiNx" if d > 0 else "prefers_SiO2"), d


def consensus(records):
    """Merge per-method records of one candidate; disagreement becomes sigma."""
    by_method = {r["method"]: r for r in records}
    deltas = {m: (r.get("summary") or {}).get("dEa_GS_minus_NGS_eV") for m, r in by_method.items()}
    known = [d for d in deltas.values() if d is not None]
    primary = by_method.get("dft") or by_method.get("xtb") or next(iter(by_method.values()))
    merged = dict(primary)
    merged["methods"] = deltas
    merged["method_disagreement_eV"] = float(np.ptp(known)) if len(known) > 1 else None
    merged["direction_agreement"] = (len({d > 0 for d in known}) == 1) if len(known) > 1 else None
    return merged


def control_gate(kinetics_by_name, controls):
    rows = []
    for c in controls:
        k = kinetics_by_name.get(c["name"])
        got, d = direction(k) if k else (None, None)
        rows.append({"name": c["name"], "expected": c["expected"], "computed": got, "dEa_eV": d,
                     "match": None if got is None else got == c["expected"],
                     "source": c["source"], "source_verified": bool(c.get("source_verified"))})
    if any(r["match"] is False for r in rows):
        status = "fail"
    elif any(r["match"] is None for r in rows) or not rows:
        status = "incomplete"
    else:
        status = "pass"
    return {"status": status, "controls": rows,
            "note": "Controls with unverified sources calibrate direction only after the citation is checked."}


def electrophilicity(smiles):
    """Gasteiger charge on the mapped anchor atom: the naive reactivity baseline."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.ComputeGasteigerCharges(mol)
    anchor = next(a for a in mol.GetAtoms() if a.GetAtomMapNum() == 1)
    return float(anchor.GetDoubleProp("_GasteigerCharge"))


def thermo_contrast(kinetics):
    """Reaction-energy contrast from the same endpoints: the thermodynamics-only baseline."""
    out = {}
    for material in ("SiO2", "SiNx"):
        rows = [s for s in kinetics["sites"] if s["material"] == material and s["status"] == "valid"]
        if not rows:
            return None
        w = np.array([s["population"] for s in rows], float)
        out[material] = float(np.average([s["dE_rxn_eV"] for s in rows], weights=w))
    return out["SiO2"] - out["SiNx"]


def score(row):
    rec = row.get("design", {}).get("recommended") or {}
    return (rec.get("P_success") or 0., rec.get("S_median") or -1.)


def assess(records, controls=(), params=None, design_options=None, control_names=()):
    params = params or ProcessParams()
    grouped = {}
    for r in records:
        grouped.setdefault(r["candidate_id"], []).append(r)
    merged = [consensus(rs) for rs in grouped.values()]
    by_name = {m["name"]: m for m in merged}
    gate = control_gate(by_name, controls)
    controls_set = set(control_names) | {c["name"] for c in controls}
    rows = []
    for m in merged:
        row = {"candidate_id": m["candidate_id"], "name": m["name"], "smiles": m["smiles"],
               "template": m["template"], "summary": m.get("summary"),
               "methods": m.get("methods"), "method_disagreement_eV": m.get("method_disagreement_eV"),
               "direction_agreement": m.get("direction_agreement"),
               "is_control": m["name"] in controls_set,
               "surface_admission": m.get("surface_admission", True)}
        try:
            row["design"] = design(m, params, **(design_options or {}))
            row["status"] = "assessed"
        except ValueError as exc:
            row["status"] = "insufficient_evidence"; row["reason"] = str(exc)
        try:
            row["electrophilicity"] = electrophilicity(m["smiles"])
        except Exception:
            row["electrophilicity"] = None
        row["thermo_contrast_eV"] = thermo_contrast(m)
        rows.append(row)
    pool = [r for r in rows if r["status"] == "assessed" and not r["is_control"] and r["surface_admission"]]
    ranking = sorted(pool, key=score, reverse=True)
    baselines = {}
    if pool:
        objective = lambda r: score(r)[0]
        picks = {
            "electrophilicity_only": max((r for r in pool if r["electrophilicity"] is not None),
                                         key=lambda r: r["electrophilicity"], default=None),
            "thermodynamics_only": max((r for r in pool if r["thermo_contrast_eV"] is not None),
                                       key=lambda r: r["thermo_contrast_eV"], default=None),
            "barrier_contrast_only": max((r for r in pool if (r["summary"] or {}).get("dEa_GS_minus_NGS_eV") is not None),
                                         key=lambda r: r["summary"]["dEa_GS_minus_NGS_eV"], default=None)}
        top = objective(ranking[0])
        for name, pick in picks.items():
            baselines[name] = None if pick is None else {
                "pick": pick["name"], "P_success": objective(pick), "lift_P_success": top - objective(pick)}
        baselines["random_expected"] = {"P_success": float(np.mean([objective(r) for r in pool])),
                                        "lift_P_success": top - float(np.mean([objective(r) for r in pool]))}
    label = "control_calibrated" if gate["status"] == "pass" else "uncalibrated_" + gate["status"]
    return clean({"label": label, "scientific_success": False, "control_gate": gate,
                  "ranking": [{"rank": i + 1, "name": r["name"], "P_success": score(r)[0],
                               "S_median": score(r)[1],
                               "recommended": r["design"]["recommended"],
                               "measure_first": r["design"]["measure_first"][:3]} for i, r in enumerate(ranking)],
                  "baselines": baselines, "candidates": rows,
                  "limitation": "Probabilities are conditional on stated priors and single-site models."})


def markdown(result):
    lines = ["# WARDEN v3 kinetic assessment", "",
             f"**Label: {result['label']}** — screening only; no experimental AS-ALD success is claimed.", "",
             "## Control gate", "", "| Control | Expected | Computed | dEa (GS-NGS) eV | Match | Source verified |",
             "|---|---|---|---:|---|---|"]
    for c in result["control_gate"]["controls"]:
        d = f"{c['dEa_eV']:.2f}" if c["dEa_eV"] is not None else "—"
        lines.append(f"| {c['name']} | {c['expected']} | {c['computed']} | {d} | {c['match']} | {c['source_verified']} |")
    designed = [c for c in result["candidates"] if c["is_control"] and c.get("design")]
    if designed:
        lines += ["", "Process design for controls (not ranked):", ""]
        for c in designed:
            rec = c["design"]["recommended"]
            nominal_s = rec["nominal"]["S_at_target"]
            lines.append(f"- {c['name']}: P(success) {rec['P_success']:.2f}, nominal S "
                         f"{'—' if nominal_s is None else f'{nominal_s:.3f}'} at {rec['T_K']:.0f} K, "
                         f"{rec['dose_s']} s dose, redose every {rec['redose_every']}; measure first: "
                         + ", ".join(m["input"] for m in c["design"]["measure_first"][:3]))
    lines += ["", "## Ranking (robust probability of S >= target at target thickness)", "",
              "| # | Candidate | P(success) | median S | T (K) | dose (s) | redose every | measure first |",
              "|---:|---|---:|---:|---:|---:|---:|---|"]
    for r in result["ranking"]:
        rec = r["recommended"]
        mf = ", ".join(m["input"] for m in r["measure_first"])
        s = f"{r['S_median']:.3f}" if r["S_median"] is not None and r["S_median"] > -1 else "—"
        lines.append(f"| {r['rank']} | {r['name']} | {r['P_success']:.2f} | {s} | {rec.get('T_K')} | "
                     f"{rec.get('dose_s')} | {rec.get('redose_every')} | {mf} |")
    lines += ["", "## Baselines", "", "| Baseline | Pick | P(success) | Lift of WARDEN pick |", "|---|---|---:|---:|"]
    for name, b in result["baselines"].items():
        if b:
            lines.append(f"| {name} | {b.get('pick', '—')} | {b['P_success']:.2f} | {b['lift_P_success']:+.2f} |")
    lines += ["", "## Per-candidate kinetics", "",
              "| Candidate | Ea SiO2 (eV) | Ea SiNx (eV) | dEa | methods | practical dose window |", "|---|---:|---:|---:|---|---|"]
    for c in result["candidates"]:
        s = c.get("summary") or {}
        f = lambda x: f"{x['Ea_mean_eV']:.2f}" if x else "—"
        windows = (c.get("design") or {}).get("dose_windows", [])
        practical = [f"{w['T_K']:.0f} K" for w in windows if w["practical"]]
        d = s.get("dEa_GS_minus_NGS_eV")
        lines.append(f"| {c['name']} | {f(s.get('SiO2'))} | {f(s.get('SiNx'))} | "
                     f"{'—' if d is None else f'{d:.2f}'} | {c.get('methods')} | {', '.join(practical) or 'none'} |")
    lines += ["", result["limitation"]]
    return "\n".join(lines) + "\n"
