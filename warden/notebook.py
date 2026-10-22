"""Campaign memory: a lab notebook and deterministic hypothesis tracking.

The notebook holds short entries from two sources: lessons the planner writes
with its actions, and automatic entries for outcomes (results, failures,
rejections). Recent entries go back into every planning context so the agent
does not rediscover the same failure.

Hypothesis status is never decided by the model. Each agent candidate states
a predicted surface preference; the code compares it with computed evidence:

    supported     every valid computed contrast agrees with the prediction and,
                  for a SiNx-selective prediction, at least one barrier screen
                  has a positive dose-window margin (a practical dose caps the
                  slowest nitride site before the fastest oxide site)
    direction_only  the contrasts agree but no practical dose window exists
    refuted       any valid computed contrast contradicts it
    inconclusive  computations exist but none produced a usable contrast
    untested      no computation yet

"Supported" means agreement with a screening surrogate, never experimental
confirmation.
"""
from datetime import datetime, timezone
from .common import digest

NOTEBOOK_ITEMS = 15
DIRECTIONS = ("prefers_SiNx", "prefers_SiO2")


def write(store, step, source, text, **extra):
    text = str(text).strip()[:600]
    if not text:
        return None
    key = "nb_" + digest({"step": step, "source": source, "text": text})[:16]
    store.put("notebook", key, {"id": key, "step": step, "source": source, "text": text,
                                "time": datetime.now(timezone.utc).isoformat(), **extra})
    return key


def recent(store, limit=NOTEBOOK_ITEMS):
    return [{k: e[k] for k in ("step", "source", "text")} for e in store.recent("notebook", limit)]


def _evidence(store, candidate_id):
    """(method, contrast_eV, direction) for each valid computed contrast of one candidate."""
    rows = []
    for r in store.all("result"):
        if r.get("candidate_id") == candidate_id and r.get("mode") == "mace":
            if r.get("status") == "complete" and r.get("contrast_eV") is not None:
                # dE_GS - dE_NGS > 0: binds the nitride more strongly.
                rows.append({"kind": "adsorption", "fidelity": r.get("fidelity", 0),
                             "value_eV": r["contrast_eV"],
                             "direction": "prefers_SiNx" if r["contrast_eV"] > 0 else "prefers_SiO2"})
            else:
                rows.append({"kind": "adsorption", "fidelity": r.get("fidelity", 0), "value_eV": None,
                             "direction": None, "status": r.get("status")})
    from .kinetics import reclassify
    from .lit import window_margin
    for k in store.all("kinetics"):
        if k.get("candidate_id") != candidate_id:
            continue
        # Re-scored under current rules: a barrierless oxide path is a fast reaction,
        # not a failed calculation, so stored summaries may overstate selectivity.
        if k.get("sites"):
            k = reclassify(k)
        d = (k.get("summary") or {}).get("dEa_GS_minus_NGS_eV")
        valid = {m: [s["Ea_eV"] for s in k.get("sites", []) if s["material"] == m and s["status"] == "valid"]
                 for m in ("SiO2", "SiNx")}
        margin = (window_margin(min(valid["SiO2"]), max(valid["SiNx"]))
                  if valid["SiO2"] and valid["SiNx"] else None)
        # Ea_GS - Ea_NGS > 0: caps the nitride faster.
        rows.append({"kind": f"kinetics:{k.get('method')}:{k.get('sites_source', 'gas')}", "value_eV": d,
                     "direction": None if d is None else ("prefers_SiNx" if d > 0 else "prefers_SiO2"),
                     "window_margin_decades": margin,
                     "admitted_surfaces": k.get("surface_admission", True)})
    return rows


def hypothesis_status(store, candidate):
    predicted = candidate.get("predicted_direction") or "prefers_SiNx"
    evidence = _evidence(store, candidate["id"])
    usable = [e for e in evidence if e["direction"] is not None]
    if not evidence:
        status = "untested"
    elif not usable:
        status = "inconclusive"
    elif any(e["direction"] != predicted for e in usable):
        status = "refuted"
    elif predicted == "prefers_SiNx" and not any((e.get("window_margin_decades") or 0) > 0 for e in usable):
        status = "direction_only"     # right preference, but no practical dose separates the surfaces
    else:
        status = "supported"
    return {"candidate_id": candidate["id"], "name": candidate.get("name"),
            "hypothesis": candidate.get("hypothesis"), "falsification": candidate.get("falsification"),
            "predicted_direction": predicted, "status": status, "evidence": evidence,
            "note": "Status compares a prediction with screening surrogates only."}


def update_hypotheses(store, step=None):
    """Re-evaluate every candidate that carries a hypothesis; log status changes."""
    out = []
    for c in store.all("candidate"):
        if not c.get("hypothesis"):
            continue
        row = hypothesis_status(store, c)
        old = store.get("hypothesis", c["id"])
        if old is None or old["status"] != row["status"]:
            write(store, step, "hypothesis_tracker",
                  f"{c.get('name')}: hypothesis now {row['status']} "
                  f"(predicted {row['predicted_direction']}; {len(row['evidence'])} computed records)")
        store.put("hypothesis", c["id"], row)
        out.append(row)
    return out
