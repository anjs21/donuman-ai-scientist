"""Bounded, resumable propose--critique--validate--compute loop."""
from pathlib import Path
import subprocess
import sys
import jsonschema
from .chemistry import validate_candidate
from .common import dumps, digest, write_json
from .metrics import diagnose
from .planner import SCHEMA


def context(store, state, tools):
    evidence = [{**e, "text": e["text"][:18000]} for e in store.all("evidence")[-12:]]
    latest = {}
    for result in store.all("result"):
        key = (result.get("mode"), result["candidate_id"])
        if key not in latest or result.get("fidelity", 0) >= latest[key].get("fidelity", 0):
            latest[key] = result
    rows = list(latest.values()) + store.all("historical")
    return {"objective": "Block SiNx while preserving SiO2 growth; 90% at 10nm is a surrogate target only.",
            "budget": {k: v for k, v in state.items() if k != "pending"},
            "candidates": store.all("candidate"),
            "diagnoses": diagnose(rows),
            "results": [{k: r.get(k) for k in ("candidate_id", "mode", "status", "dE_GS", "dE_NGS", "metrics", "reason")} for r in rows],
            "kinetics": kinetic_context(store),
            "evidence": evidence, "literature_tools": tools,
            "last_error": state.get("last_error")}


def kinetic_context(store):
    """Compact barrier and process-window facts for the planner."""
    assessment = store.get("assessment", "latest") or {}
    designs = {c["candidate_id"]: c.get("design") or {} for c in assessment.get("candidates", [])}
    out = []
    for k in store.all("kinetics"):
        d = designs.get(k["candidate_id"], {})
        rec = d.get("recommended") or {}
        out.append({"candidate_id": k["candidate_id"], "method": k["method"], "summary": k["summary"],
                    "failed_sites": [s["label"] for s in k["sites"] if s["status"] != "valid"],
                    "P_success": rec.get("P_success"), "recommended": {x: rec.get(x) for x in ("T_K", "dose_s", "redose_every")},
                    "dose_windows": [{x: w[x] for x in ("T_K", "practical", "target_Ea_NGS_max_eV", "target_Ea_GS_min_eV")}
                                     for w in d.get("dose_windows", [])],
                    "measure_first": [m["input"] for m in d.get("measure_first", [])[:3]]})
    return {"label": assessment.get("label"), "control_gate": (assessment.get("control_gate") or {}).get("status"),
            "candidates": out}


def run(store, planner, runner, config, literature=None, mode="live"):
    limits = config.get("campaign", {})
    settings = {"max_steps": int(limits.get("max_steps", 10)),
                "max_candidates": int(limits.get("max_candidates", 6)),
                "max_searches": int(limits.get("max_searches", 4)),
                "max_refinements": int(limits.get("max_refinements", 2)),
                "config_hash": digest(config), "mode": mode}
    if any(settings[k] < 1 for k in ("max_steps", "max_candidates")):
        raise ValueError("Positive campaign budgets required")
    state = store.get("state", "campaign")
    if state and any(state.get(k) != v for k, v in settings.items()):
        raise ValueError("Campaign configuration changed: use a new database")
    if state is None:
        state = {**settings, "step": 0, "submitted": [], "searches": 0,
                 "refinements": {}, "pending": None, "status": "running"}
    if state["status"] in {"finished", "budget_exhausted"}:
        return state
    tools = []
    if literature:
        tools = [t for t in literature.tools() if t["name"] in literature.config.get("allowed_tools", [])]
        if not tools:
            raise ValueError("No allowed BOND tools configured. Run bond-tools and set allowed_tools first.")
    store.put("state", "campaign", state)
    while state["pending"] is not None or state["step"] < state["max_steps"]:
        ctx = context(store, state, tools)
        if state["pending"] is None:
            state["step"] += 1
            # Count invalid outputs and failed model calls against the step budget too.
            store.put("state", "campaign", state)
            try:
                action = planner.plan(ctx)
                jsonschema.validate(action, SCHEMA)
                state["pending"] = action
                store.event("plan", {"step": state["step"], "action": action})
                store.put("state", "campaign", state)
            except Exception as exc:
                state["last_error"] = str(exc)
                store.event("planner_error", {"error": str(exc)})
                store.put("state", "campaign", state)
                continue
        action = state["pending"]
        try:
            kind = action["action"]
            if kind == "finish":
                if not state["submitted"]:
                    raise ValueError("Cannot finish before testing any new candidate; unsupported stop refused")
                state["status"] = "finished"
                store.event("verified_stop", {"reason": action["reason"],
                            "scientific_success": False,
                            "verdict": "See deterministic screening report; model prose cannot certify success."})
            elif kind == "search":
                if literature is None:
                    raise ValueError("Live search is not configured; use supplied evidence or configure BOND")
                if state["searches"] >= state["max_searches"]:
                    raise ValueError("Literature call budget exhausted")
                state["searches"] += 1
                store.put("state", "campaign", state)
                result = literature.call(action["tool"], action.get("arguments", {}))
                evidence_id = store.evidence(dumps(result),
                    {"provider": "BOND", "tool": action["tool"], "arguments": action.get("arguments", {})}, "bond_result")
                store.event("retrieval", {"evidence_id": evidence_id, "tool": action["tool"]})
            elif kind == "propose":
                if "_ids" not in action:
                    proposals = action.get("candidates", [])
                    if not proposals or len(proposals) > 3:
                        raise ValueError("Propose 1-3 candidates")
                    if len(state["submitted"]) + len(proposals) > state["max_candidates"]:
                        raise ValueError("New-candidate budget would be exceeded")
                    evidence = {e["id"]: e for e in store.all("evidence")}
                    checked = [validate_candidate(c, evidence, require_evidence=True) for c in proposals]
                    if any(c["role"] != "inhibitor" for c in checked):
                        raise ValueError("This campaign optimizes inhibitors; evaluate precursors separately")
                    known = {(c["canonical_smiles"], c["template"]) for c in store.all("candidate")}
                    for c in checked:
                        identity = (c["canonical_smiles"], c["template"])
                        if identity in known:
                            raise ValueError("Duplicate molecule/mechanism: use refine instead")
                        known.add(identity)
                    review = planner.critique(action, ctx)
                    store.event("critique", review)
                    if not review.get("approve"):
                        raise ValueError("Critic rejected proposal: " + review.get("reason", ""))
                    for c in checked:
                        store.put("candidate", c["id"], c)
                    action["_ids"] = [c["id"] for c in checked]
                    state["submitted"].extend(action["_ids"])
                    store.put("state", "campaign", state)
                runner(action["_ids"], 0)
            elif kind == "refine":
                candidate_id = action.get("candidate_id")
                if not store.get("candidate", candidate_id):
                    raise ValueError("Unknown candidate")
                if "_fidelity" not in action:
                    if sum(state["refinements"].values()) >= state["max_refinements"]:
                        raise ValueError("Refinement budget exhausted")
                    level = state["refinements"].get(candidate_id, 0) + 1
                    state["refinements"][candidate_id] = level
                    action["_fidelity"] = level
                    store.put("state", "campaign", state)
                runner([candidate_id], action["_fidelity"])
            state["last_error"] = None
        except subprocess.TimeoutExpired:
            state["status"] = "paused"
            store.event("worker_timeout", {"pending": action, "note": "Resume command reuses completed jobs."})
            store.put("state", "campaign", state)
            return state
        except Exception as exc:
            state["last_error"] = str(exc)
            store.event("action_rejected_or_failed", {"error": str(exc), "action": action})
        state["pending"] = None
        store.put("state", "campaign", state)
        if state["status"] == "finished":
            return state
    state["status"] = "budget_exhausted"
    store.event("budget_stop", {"scientific_success": False})
    store.put("state", "campaign", state)
    return state


def subprocess_runner(db, manifest, config_path, timeout, kinetics=False, controls=None):
    def runner(ids, fidelity):
        command = [sys.executable, "-m", "warden", "screen", "--db", str(db),
                   "--manifest", str(manifest), "--config", str(config_path),
                   "--ids", *ids, "--fidelity", str(fidelity)]
        # The worker owns MACE; its exit frees GPU allocations before the next LLM call.
        subprocess.run(command, check=True, timeout=timeout)
        if kinetics:
            subprocess.run([sys.executable, "-m", "warden", "kinetics", "--db", str(db),
                            "--config", str(config_path), "--ids", *ids, "--sites", "manifest",
                            "--manifest", str(manifest)], check=True, timeout=timeout)
            subprocess.run([sys.executable, "-m", "warden", "assess", "--db", str(db),
                            "--config", str(config_path), "--out", str(Path(db).parent / "assess"),
                            *(["--controls", str(controls)] if controls else [])], check=True, timeout=timeout)
    return runner
