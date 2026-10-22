"""Literature loop: computed outcomes -> literature questions -> evidence-backed library.

lit_brief() reads stored kinetics records (re-scored with the current rules) and
separates what each calculation says about the CHEMISTRY from what it says about
the METHOD. Only chemical outcomes are turned into literature questions; numerical
failures become method actions and are never treated as evidence against a molecule.
No LLM is involved, so the brief is reproducible.

import_candidates() validates literature-backed proposals against stored evidence
(exact quotes, checked on the readable text of the retrieved record) and writes a
library for `warden decide --library`, literature-tested molecules first.
"""
import json
import re
import unicodedata
from .common import clean

# How an attempt outcome should be read. Statuses come from kinetics.classify;
# reasons from the guards in kinetics.endpoints/check_reactant/check_product.
CHEMICAL = {"valid", "barrierless", "reactant_reacted_spontaneously"}
INCONCLUSIVE = {"no_barrier_resolved", "product_not_formed", "proton_transfer_not_completed",
                "byproduct_proton_captured", "reactant_connectivity_changed"}
SETUP = {"open_shell_cluster", "isolated_site", "could_not_embed_site_model", "could_not_embed_candidate",
         "no_surface_proton", "no_lone_pair_direction"}
# Everything else (SCF not converged, endpoint_not_converged, neb_not_converged,
# band_unresolved, unexpected exceptions) is numerical.

FAST_EV = 0.10          # an oxide path this fast caps silanols during any practical dose
GROWTH, NON_GROWTH = "SiO2", "SiNx"


def outcome_class(attempt):
    status, reason = attempt.get("status"), attempt.get("reason") or ""
    if status in CHEMICAL or reason in CHEMICAL:
        return "chemistry"
    if status in INCONCLUSIVE or reason in INCONCLUSIVE:
        return "inconclusive"
    if reason in SETUP or reason.startswith("pathway "):
        return "setup"
    return "numerical"


def _site_view(site):
    counts = {}
    for a in site.get("attempts", []):
        key = outcome_class(a)
        counts[key] = counts.get(key, 0) + 1
    if not site.get("attempts") and site.get("reason"):
        counts[outcome_class({"status": "failed", "reason": site["reason"]})] = 1
    spontaneous = any(a.get("reason") == "reactant_reacted_spontaneously" for a in site.get("attempts", []))
    return {"label": site["label"], "material": site["material"], "site_type": site.get("site_type"),
            "status": site.get("status"), "Ea_eV": site.get("Ea_eV"),
            "barrierless": bool(site.get("barrierless")), "reacts_on_contact": spontaneous,
            "outcomes": counts}


def window_margin(ea_gs_fastest, ea_ngs_slowest):
    """Dose-window margin (decades, best temperature) under decide's default
    process settings; negative means no practical dose separates the surfaces."""
    import numpy as np
    from .decide import DEFAULTS, window_width
    return float(window_width(np.array([[ea_gs_fastest]]), np.array([[ea_ngs_slowest]]), DEFAULTS, clip=False)[0])


def findings(record):
    """Deterministic statements about one candidate, each tagged chemistry or method."""
    sites = [_site_view(s) for s in record["sites"]]
    out = []
    by = {m: [s for s in sites if s["material"] == m] for m in (GROWTH, NON_GROWTH)}
    fastest = {m: min((s["Ea_eV"] for s in by[m] if s["status"] == "valid"), default=None) for m in by}
    # Every nitride site must be capped, so the slowest one sets the window (as decide.window_width).
    slowest_ngs = max((s["Ea_eV"] for s in by[NON_GROWTH] if s["status"] == "valid"), default=None)
    partial = any(s["status"] != "valid" for s in by[NON_GROWTH])
    gs_fast = [s for s in by[GROWTH] if s["status"] == "valid" and (s["barrierless"] or s["Ea_eV"] <= FAST_EV)]
    gs_fast += [s for s in by[GROWTH] if s["reacts_on_contact"] and s not in gs_fast]
    if gs_fast:
        out.append({"kind": "chemistry", "code": "oxide_too_reactive",
                    "text": f"Caps SiO2 silanol with no meaningful barrier ({', '.join(s['label'] for s in gs_fast)}): "
                            "it would block the growth surface during any practical dose.",
                    "direction": "milder or more chemoselective reagent (N-H over O-H)"})
    if fastest[GROWTH] is not None and slowest_ngs is not None:
        d = fastest[GROWTH] - slowest_ngs
        margin = window_margin(fastest[GROWTH], slowest_ngs)
        scope = " (resolved nitride sites only)" if partial else ""
        if d <= 0:
            out.append({"kind": "chemistry", "code": "no_kinetic_contrast",
                        "text": f"Fastest oxide path is not slower than the slowest nitride site{scope} "
                                f"(dEa {d:+.2f} eV): no kinetic selectivity in this model.",
                        "direction": "different reaction class or leaving group"})
        elif margin <= 0:
            out.append({"kind": "chemistry", "code": "contrast_outside_dose_window",
                        "text": f"Oxide slower than the slowest nitride site by {d:.2f} eV{scope}, but no practical "
                                f"dose caps the nitride before the oxide (window misses by {-margin:.1f} decades).",
                        "direction": "keep the N-H preference, shift both barriers into the practical band"})
        else:
            out.append({"kind": "chemistry", "code": "kinetic_contrast",
                        "text": f"Oxide slower than the slowest nitride site by {d:.2f} eV{scope}; "
                                f"practical dose window {margin:.1f} decades in this model.",
                        "direction": "look for experimental confirmation and process conditions"})
    for m in (GROWTH, NON_GROWTH):
        other = NON_GROWTH if m == GROWTH else GROWTH
        if not by[m] and fastest[other] is not None:
            out.append({"kind": "method", "code": f"missing_{m}",
                        "text": f"Only {other} computed (fastest {fastest[other]:.2f} eV); no {m} sites yet.",
                        "direction": f"compute {m} sites before judging selectivity"})
    for s in sites:
        if s["status"] != "valid":
            dominant = max(s["outcomes"], key=s["outcomes"].get) if s["outcomes"] else "numerical"
            kind = "method" if dominant in ("numerical", "setup") else "chemistry"
            out.append({"kind": kind, "code": f"unresolved_{dominant}",
                        "text": f"{s['label']}: no usable barrier ({', '.join(f'{k} x{v}' for k, v in s['outcomes'].items())}).",
                        "direction": ("rerun with another orientation or a transition-state search; "
                                      "not evidence against the molecule") if kind == "method"
                                     else "inspect the path; product stability may be the issue"})
    return sites, out


QUESTION_TEMPLATES = {
    "oxide_too_reactive": [
        "chemoselective {reaction} of amines over alcohols or silanols",
        "{reaction} reagents that react with Si-NH2 but not Si-OH surface groups",
        "vapor-phase small-molecule inhibitor selective for silicon nitride over silicon oxide",
    ],
    "no_kinetic_contrast": [
        "area-selective deposition inhibitor SiO2 growth on SiO2 not on SiN",
        "surface functionalization selectivity silicon nitride versus silicon oxide",
    ],
    "contrast_outside_dose_window": [
        "milder {reaction} reagent with the same selectivity for amines over alcohols",
        "{name} analogs with slower {reaction} kinetics for vapor-phase surface modification",
    ],
    "kinetic_contrast": [
        "{name} reaction with silicon nitride surface amine groups",
        "{name} vapor phase surface modification selectivity",
    ],
}
REACTION = {"acylation": "acylation", "silyl_substitution": "silylation", "physisorption": "adsorption"}


def questions(record, found):
    """Literature questions implied by chemical findings (method findings get none)."""
    out, seen = [], set()
    fill = {"name": record.get("name", ""), "reaction": REACTION.get(record.get("template"), "reaction")}
    for f in found:
        for t in QUESTION_TEMPLATES.get(f["code"], []):
            q = t.format(**fill)
            if q not in seen:
                seen.add(q)
                out.append({"query": q, "because": f["code"], "candidate": record.get("name")})
    return out


def gather(kinetics_records, observations=(), candidates=None, rules=None):
    """One record per candidate, combining stored kinetics screens with the decision
    agent's own observations; site barriers re-derived with kinetics.select_best."""
    from . import kinetics as kin
    candidates = candidates or {}
    material_of = {f"gas:{t}": m for m, sites in kin.GAS_SITES.items() for t in sites}
    out, skipped = {}, 0
    for k in kinetics_records:
        rec = out.setdefault(k["candidate_id"], {"candidate_id": k["candidate_id"], "name": k.get("name"),
                                                  "smiles": k.get("smiles"), "template": k.get("template"),
                                                  "methods": set(), "sites": {}})
        rec["methods"].add(k.get("method"))
        for site in k["sites"]:
            material_of.setdefault(site["label"], site["material"])
            row = rec["sites"].setdefault(site["label"], {"label": site["label"], "material": site["material"],
                                                          "site_type": site.get("site_type"), "attempts": []})
            row["attempts"] += site.get("attempts") or ([{"status": "failed", "reason": site.get("reason")}]
                                                        if site.get("reason") else [])
    for o in observations:
        material = material_of.get(o["site"])
        spec = candidates.get(o["candidate"])
        if material is None or spec is None:
            skipped += 1
            continue
        rec = out.setdefault(o["candidate"], {"candidate_id": o["candidate"], "name": spec.get("name"),
                                              "smiles": spec.get("smiles"), "template": spec.get("template"),
                                              "methods": set(), "sites": {}})
        rec["methods"].add(o.get("method"))
        row = rec["sites"].setdefault(o["site"], {"label": o["site"], "material": material,
                                                  "site_type": o["site"].split(":")[-1], "attempts": []})
        if o.get("status") == "valid":
            status = "barrierless" if o.get("reason") == "barrierless" else "valid"
            row["attempts"].append({"status": status, "Ea_eV": o.get("Ea_eV"), "dE_rxn_eV": o.get("dE_rxn_eV"),
                                    "stepwise_indication": None, "pathway": o.get("pathway")})
        else:
            row["attempts"].append({"status": o.get("status"), "reason": o.get("reason"), "pathway": o.get("pathway")})
    records = []
    for rec in out.values():
        sites = []
        for row in rec["sites"].values():
            attempts = [{"dE_rxn_eV": None, "stepwise_indication": None, **a} for a in row["attempts"]]
            sites.append({**row, "attempts": attempts, **kin.site_fields(kin.select_best(attempts, row["material"], rules))})
        methods = sorted(m for m in rec["methods"] if m)
        records.append({**{k: v for k, v in rec.items() if k not in ("sites", "methods")},
                        "method": "+".join(methods) or None, "sites": sites})
    return records, skipped


def lit_brief(records, objective="Block SiNx while preserving SiO2 growth (SiO2 on SiO2, not on SiNx)."):
    rows, all_q, seen = [], [], set()
    for r in records:
        sites, found = findings(r)
        rows.append({"candidate_id": r.get("candidate_id"), "name": r.get("name"), "smiles": r.get("smiles"),
                     "template": r.get("template"), "method": r.get("method"), "sites": sites, "findings": found})
        for q in questions(r, found):
            if q["query"] not in seen:
                seen.add(q["query"]); all_q.append(q)
    method_actions = [f"{row['name']}: {f['text']} -> {f['direction']}"
                      for row in rows for f in row["findings"] if f["kind"] == "method"]
    tried = sorted({r.get("smiles") for r in records if r.get("smiles")})
    return clean({"objective": objective, "candidates": rows, "questions": all_q,
                  "method_actions": method_actions, "already_computed_smiles": tried,
                  "rules": {"fast_oxide_barrier_eV": FAST_EV,
                            "numerical_failures": "never used as evidence against a molecule"}})


def brief_markdown(b):
    lines = ["# WARDEN literature brief", "", f"**Objective:** {b['objective']}", "",
             "Generated from stored calculations without an LLM. Chemical outcomes become literature "
             "questions; numerical failures become method actions only.", "", "## What the calculations say", ""]
    for c in b["candidates"]:
        lines.append(f"### {c['name']} ({c['template']}, {c['method']})")
        for f in c["findings"]:
            lines.append(f"- [{f['kind']}] {f['text']} Next: {f['direction']}.")
        if not c["findings"]:
            lines.append("- No usable outcomes yet.")
        lines.append("")
    lines += ["## Literature questions", ""]
    lines += [f"{i}. {q['query']}  _(from {q['candidate']}: {q['because']})_" for i, q in enumerate(b["questions"], 1)]
    if not b["questions"]:
        lines.append("None: no chemical findings yet. Resolve the method actions first.")
    lines += ["", "## Method actions (not literature questions)", ""]
    lines += [f"- {m}" for m in b["method_actions"]] or ["- None."]
    lines += ["", "## Already computed (do not re-propose)", ""] + [f"- `{s}`" for s in b["already_computed_smiles"]]
    return "\n".join(lines) + "\n"


# --- evidence text and quotes -------------------------------------------------

def readable(text):
    """All human-readable strings in an evidence record.

    bond-call stores the MCP result as JSON, whose tool payload is often itself a
    JSON string, so 'SiO2' with a subscript is stored as an escape sequence and
    quotes as \\". Quotes must be matched against the decoded strings, not the
    escaped serialisation."""
    parts = [text]

    def walk(value, depth=0):
        if isinstance(value, str):
            parts.append(value)
            s = value.strip()
            if depth < 4 and s[:1] in "[{":
                try:
                    walk(json.loads(s), depth + 1)
                except ValueError:
                    pass
        elif isinstance(value, dict):
            for v in value.values():
                walk(v, depth)
        elif isinstance(value, list):
            for v in value:
                walk(v, depth)
    try:
        walk(json.loads(text))
    except (ValueError, TypeError):
        pass
    return "\n".join(parts)


def normalize(s):
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u2010", "-").replace("\u2011", "-").replace("\u2013", "-").replace("\u2212", "-")
    s = re.sub(r"[\u2018\u2019]", "'", re.sub(r"[\u201c\u201d]", '"', s))
    return re.sub(r"\s+", " ", s).strip().lower()


def quote_in(quote, text):
    q = normalize(quote or "")
    return len(q) >= 12 and q in normalize(readable(text))


# --- literature-backed candidates ------------------------------------------------

def prior_check(p):
    """Optional own prior for `decide`: barriers (eV) a source reports or implies
    for this molecule, replacing what its reaction class has shown so far."""
    prior = p.get("prior_Ea_eV")
    if prior is None:
        return
    if not isinstance(prior, dict) or not prior or set(prior) - {GROWTH, NON_GROWTH}:
        raise ValueError("prior_Ea_eV must map SiO2 and/or SiNx to a barrier in eV")
    for value in [*prior.values(), p.get("prior_sd_eV", 0.1)]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 5:
            raise ValueError("prior_Ea_eV values and prior_sd_eV must be numbers between 0 and 5 eV")
    if p.get("prior_sd_eV") == 0:
        raise ValueError("prior_sd_eV must be positive")


def import_candidates(proposals, evidence, known_smiles=(), validate=None):
    """Validate proposals and order them for `decide --library`.

    Returns (library, rejected). Each library entry keeps its hypothesis,
    falsification, evidence references and literature_status."""
    from .chemistry import validate_candidate
    validate = validate or validate_candidate
    library, rejected, seen = [], [], set(known_smiles)
    for p in proposals:
        name = p.get("name", "?")
        try:
            status = p.get("literature_status")
            if status not in ("tested", "analog"):
                raise ValueError("literature_status must be 'tested' (molecule itself studied in the source) "
                                 "or 'analog' (extrapolated from a related molecule)")
            prior_check(p)
            spec = validate(p, evidence, require_evidence=True)
            if spec["canonical_smiles"] in seen:
                raise ValueError("already computed or proposed")
            seen.add(spec["canonical_smiles"])
            library.append(spec)
        except ValueError as exc:
            rejected.append({"name": name, "reason": str(exc)})
    library.sort(key=lambda s: (s["literature_status"] != "tested", -len(s.get("evidence", []))))
    return clean(library), rejected
