"""Bounded, resumable plan--critique--validate--compute loop.

Actions the planner can take (all budgeted, all logged):

    search    call an allowlisted literature tool (e.g. Valency ChemRxiv search)
    analogs   run the deterministic analog designer around a candidate
    propose   register 1-3 new molecules, cited by verbatim quotes, then compute
    refine    recompute a candidate with more sites/conformers
    repair    launch the evidence-acquisition agent on failed or uncertain barriers
    finish    stop (refused before any new candidate was tested)

Every action may carry a one-line `lesson`, kept in the campaign notebook and
shown back to the planner. Hypotheses are re-scored by code after each
computation (see notebook.py).
"""
from pathlib import Path
import subprocess
import sys
import jsonschema
from . import notebook
from .chemistry import validate_candidate
from .common import digest
from .metrics import diagnose
from .planner import SCHEMA


EVIDENCE_ITEMS = 12
EVIDENCE_CHARS = 2000   # per item; keeps the prompt well inside the planner's num_ctx
TOOL_DESCRIPTION_CHARS = 300


def context(store, state, tools):
    # Newest evidence first: store.all() orders by content-hash id, which is
    # unrelated to when a record was retrieved.
    evidence = [{**e, "text": e["text"][:EVIDENCE_CHARS],
                 "truncated": len(e["text"]) > EVIDENCE_CHARS}
                for e in store.recent("evidence", EVIDENCE_ITEMS)]
    latest = {}
    for result in store.all("result"):
        key = (result.get("mode"), result["candidate_id"])
        if key not in latest or result.get("fidelity", 0) >= latest[key].get("fidelity", 0):
            latest[key] = result
    rows = list(latest.values()) + store.all("historical")
    return {"objective": "Block SiNx while preserving SiO2 growth; 90% at 10nm is a surrogate target only.",
            "budget": {k: v for k, v in state.items() if k not in ("pending", "search_log")},
            "candidates": [{k: c.get(k) for k in ("id", "name", "smiles", "template", "role",
                                                  "predicted_direction")} for c in store.all("candidate")],
            "diagnoses": diagnose(rows),
            "results": [{k: r.get(k) for k in ("candidate_id", "mode", "status", "dE_GS", "dE_NGS", "metrics", "reason")} for r in rows],
            "kinetics": kinetic_context(store),
            "agent_ranking": decision_context(store),
            "hypotheses": [{k: h[k] for k in ("candidate_id", "name", "predicted_direction", "status")}
                           for h in store.all("hypothesis")],
            "notebook": notebook.recent(store),
            "literature": literature_context(store, state),
            "evidence": evidence,
            "literature_tools": [{"name": t["name"], "description": (t.get("description") or "")[:TOOL_DESCRIPTION_CHARS],
                                  "arguments": compact_schema(t.get("inputSchema"))} for t in tools],
            "last_error": state.get("last_error")}


def compact_schema(schema, description_chars=80):
    """Argument summary for the planner: name -> type/enum/short description.

    Full provider schemas run to kilobytes per tool; the call itself is still
    validated against the full schema in literature.py.
    """
    schema = schema or {}
    required = set(schema.get("required", []))
    out = {}
    for name, prop in (schema.get("properties") or {}).items():
        item = {"type": prop.get("type", "any")}
        if prop.get("enum"):
            item["enum"] = prop["enum"]
        if prop.get("description"):
            item["about"] = prop["description"][:description_chars]
        if name in required:
            item["required"] = True
        out[name] = item
    return out


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


def decision_context(store):
    """Latest evidence-acquisition ranking and next calculations (from `decide`), if any."""
    d = store.get("decision", "latest")
    if not d:
        return None
    return {k: d.get(k) for k in ("pick", "pcs", "stop", "spent", "budget", "scores", "next", "meaning")}


def literature_context(store, state):
    papers = store.recent("paper", 15)
    return {"papers_seen": len(store.all("paper")),
            "recent_papers": [p["doi"] for p in papers],
            "searches_done": [s for s in state.get("search_log", [])][-8:]}


def _record_papers(store, text, evidence_id, step):
    from .literature import dois
    new = 0
    for doi in dois(text):
        if store.get("paper", doi) is None:
            store.put("paper", doi, {"doi": doi, "evidence_id": evidence_id, "first_step": step})
            new += 1
    return new


def _summarize_outcome(store, ids):
    """Deterministic notebook line per computed candidate."""
    names = {c["id"]: c.get("name", c["id"]) for c in store.all("candidate")}
    lines = []
    for cid in ids:
        results = [r for r in store.all("result") if r.get("candidate_id") == cid]
        best = max(results, key=lambda r: r.get("fidelity", 0), default=None)
        part = [names.get(cid, cid) + ":"]
        if best:
            part.append(f"adsorption {best.get('status')}"
                        + (f", contrast {best['contrast_eV']:+.2f} eV" if best.get("contrast_eV") is not None else "")
                        + (f" ({best.get('reason')})" if best.get("reason") else ""))
        for k in store.all("kinetics"):
            if k.get("candidate_id") == cid:
                failed = [s["label"] for s in k["sites"] if s["status"] != "valid"]
                d = (k.get("summary") or {}).get("dEa_GS_minus_NGS_eV")
                part.append(f"kinetics[{k['method']}] dEa {'n/a' if d is None else f'{d:+.2f} eV'}"
                            + (f", failed sites {failed}" if failed else ""))
        lines.append(" ".join(part))
    return lines


def run(store, planner, runner, config, literature=None, mode="live", repairer=None):
    limits = config.get("campaign", {})
    settings = {"max_steps": int(limits.get("max_steps", 10)),
                "max_candidates": int(limits.get("max_candidates", 6)),
                "max_searches": int(limits.get("max_searches", 4)),
                "max_refinements": int(limits.get("max_refinements", 2)),
                "max_repairs": int(limits.get("max_repairs", 2)),
                "max_analog_calls": int(limits.get("max_analog_calls", 3)),
                "repair_budget": float(limits.get("repair_budget", 6.0)),
                "config_hash": digest(config), "mode": mode}
    if any(settings[k] < 1 for k in ("max_steps", "max_candidates")):
        raise ValueError("Positive campaign budgets required")
    state = store.get("state", "campaign")
    if state and any(state.get(k) != v for k, v in settings.items()):
        raise ValueError("Campaign configuration changed: use a new database")
    if state is None:
        state = {**settings, "step": 0, "submitted": [], "searches": 0, "search_log": [],
                 "refinements": {}, "repairs": 0, "analog_calls": 0, "pending": None, "status": "running"}
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
                if action.get("lesson"):
                    notebook.write(store, state["step"], "planner", action["lesson"])
                store.put("state", "campaign", state)
            except Exception as exc:
                state["last_error"] = str(exc)
                store.event("planner_error", {"error": str(exc)})
                store.put("state", "campaign", state)
                continue
        action = state["pending"]
        computed = []
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
                from .literature import result_text
                if literature is None:
                    raise ValueError("Live search is not configured; use supplied evidence or configure BOND")
                if state["searches"] >= state["max_searches"]:
                    raise ValueError("Literature call budget exhausted")
                request = {"tool": action.get("tool"), "arguments": action.get("arguments", {})}
                if request in state["search_log"]:
                    raise ValueError("Identical search already run; its result is in the evidence records")
                state["searches"] += 1
                state["search_log"].append(request)
                store.put("state", "campaign", state)
                result = literature.call(request["tool"], request["arguments"])
                text = result_text(result)
                evidence_id = store.evidence(text, {"provider": "BOND", **request}, "bond_result")
                new = _record_papers(store, text, evidence_id, state["step"])
                store.event("retrieval", {"evidence_id": evidence_id, "tool": request["tool"], "new_papers": new})
                notebook.write(store, state["step"], "literature",
                               f"{request['tool']}({request['arguments']}) -> {evidence_id}, {new} new DOIs")
            elif kind == "analogs":
                from .analogs import enumerate_analogs, as_evidence_text
                if state["analog_calls"] >= state["max_analog_calls"]:
                    raise ValueError("Analog-design budget exhausted")
                parent = store.get("candidate", action.get("candidate_id"))
                if not parent:
                    raise ValueError("Unknown candidate for analog design")
                direction = action.get("direction", "all")
                known = [c["canonical_smiles"] for c in store.all("candidate")]
                designs = enumerate_analogs(parent, direction, limit=int(action.get("limit", 6)), known=known)
                state["analog_calls"] += 1
                if not designs:
                    raise ValueError(f"No new analogs of {parent['name']} in direction {direction}")
                evidence_id = store.evidence(as_evidence_text(parent, designs, direction),
                                             {"provider": "warden.analogs", "parent": parent["id"],
                                              "direction": direction}, "analog_design")
                store.event("analog_design", {"evidence_id": evidence_id, "parent": parent["id"],
                                              "direction": direction, "n": len(designs)})
                notebook.write(store, state["step"], "analogs",
                               f"{len(designs)} {direction} analogs of {parent['name']} in {evidence_id}; "
                               "propose by quoting a design line (analog hypotheses are extrapolations)")
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
                computed = action["_ids"]
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
                computed = [candidate_id]
            elif kind == "repair":
                if repairer is None:
                    raise ValueError("Self-repair needs barrier kinetics (set campaign.kinetics=true)")
                ids = action.get("candidate_ids") or ([action["candidate_id"]] if action.get("candidate_id") else [])
                if not ids or any(store.get("candidate", c) is None for c in ids):
                    raise ValueError("Repair needs known candidate_ids")
                if "_budget" not in action:
                    if state["repairs"] >= state["max_repairs"]:
                        raise ValueError("Repair budget exhausted")
                    state["repairs"] += 1
                    action["_budget"] = min(float(action.get("budget", state["repair_budget"])),
                                            state["repair_budget"])
                    store.put("state", "campaign", state)
                repairer(ids, action["_budget"])
                computed = ids
            state["last_error"] = None
        except subprocess.CalledProcessError as exc:
            # A crashed worker is not a chemical result. Retry the stored action once;
            # after that, record the molecules as unscreened instead of losing them.
            action["_worker_failures"] = action.get("_worker_failures", 0) + 1
            store.event("worker_failed", {"action": action, "returncode": exc.returncode})
            if action["_worker_failures"] < 2:
                state["pending"] = action
                store.put("state", "campaign", state)
                continue
            ids = action.get("_ids") or [action.get("candidate_id")]
            state.setdefault("unscreened", []).extend(i for i in ids if i not in state.get("unscreened", []))
            state["last_error"] = f"worker failed twice for {ids}; marked unscreened (not a chemical rejection)"
        except subprocess.TimeoutExpired:
            state["status"] = "paused"
            store.event("worker_timeout", {"pending": action, "note": "Resume command reuses completed jobs."})
            store.put("state", "campaign", state)
            return state
        except Exception as exc:
            state["last_error"] = str(exc)
            store.event("action_rejected_or_failed", {"error": str(exc), "action": action})
            notebook.write(store, state["step"], "system", f"{action.get('action')} rejected/failed: {exc}")
        if computed:
            for line in _summarize_outcome(store, computed):
                notebook.write(store, state["step"], "results", line)
        notebook.update_hypotheses(store, state["step"])
        state["pending"] = None
        store.put("state", "campaign", state)
        if state["status"] == "finished":
            return state
    state["status"] = "budget_exhausted"
    store.event("budget_stop", {"scientific_success": False})
    store.put("state", "campaign", state)
    return state


def subprocess_runner(db, manifest, config_path, timeout, kinetics=False, controls=None, kinetics_sites="manifest"):
    """Worker launchers. kinetics_sites: "manifest" (clusters carved from the admitted
    slabs) or "gas" (small site models; far cheaper, coarser physics)."""
    if kinetics_sites not in ("manifest", "gas"):
        raise ValueError("campaign.kinetics_sites must be manifest or gas")
    sites = ["--sites", kinetics_sites] + (["--manifest", str(manifest)] if kinetics_sites == "manifest" else [])

    def assess():
        subprocess.run([sys.executable, "-m", "warden", "assess", "--db", str(db),
                        "--config", str(config_path), "--out", str(Path(db).parent / "assess"),
                        *(["--controls", str(controls)] if controls else [])], check=True, timeout=timeout)

    def runner(ids, fidelity):
        command = [sys.executable, "-m", "warden", "screen", "--db", str(db),
                   "--manifest", str(manifest), "--config", str(config_path),
                   "--ids", *ids, "--fidelity", str(fidelity)]
        # The worker owns MACE; its exit frees GPU allocations before the next LLM call.
        subprocess.run(command, check=True, timeout=timeout)
        if kinetics:
            subprocess.run([sys.executable, "-m", "warden", "kinetics", "--db", str(db),
                            "--config", str(config_path), "--ids", *ids, *sites], check=True, timeout=timeout)
            assess()

    def repairer(ids, budget):
        # The decision agent retries failed sites with other orientations/pathways and
        # spends only where the ranking depends on it; assess then merges its barriers.
        from .common import read_json
        method = read_json(config_path).get("kinetics", {}).get("method", "xtb")
        subprocess.run([sys.executable, "-m", "warden", "decide", "--db", str(db), "--config", str(config_path),
                        "--ids", *ids, *sites, "--methods", method, "--budget", str(budget),
                        "--out", str(Path(db).parent / "decide")], check=True, timeout=timeout)
        assess()

    return runner, (repairer if kinetics else None)
