"""Whole-pipeline orchestrator agent (supervised autonomy).

The screening agent in ``agent_loop.py`` drives Phase 2 in-process. This one
sits a level higher: it plans and *runs the whole pipeline* to find a selective
inhibitor + precursor, deciding which reagents to screen and when it is done --
but it keeps the physics deterministic and it gates the one expensive,
irreversible step (fresh Phase 1 surface builds) behind human approval.

GPU discipline (the reason this is subprocess-based)
----------------------------------------------------
Ollama (the agent's brain) and MACE (the pipeline) both want the same ~8 GB of
VRAM, so they must never run at once. This orchestrator therefore holds NO
calculator in-process: every ``run_screen`` shells out to run_pipeline.py with
``--use-existing`` (reusing slabs on disk) and BLOCKS until it finishes, so the
LLM is idle while MACE runs and vice-versa. Phase 1 (melt-quench MD) is never
launched autonomously -- ``request_build`` only records a request for the UI to
approve.

Tools given to the model
-------------------------
    inspect_state()                 surfaces on disk, existing reports, reagents
    run_screen(reagents, ...)       run Phase 2-5 on existing surfaces -> report
    request_build(materials, mode)  GATED: propose a fresh Phase 1 build

``run_agent`` yields trace events (see the docstring in agent_loop.py for the
shared vocabulary) plus one extra: ``{"type":"approval_request", ...}`` when the
model asks for a surface build the human must approve.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Iterator, List, Optional

from . import llm
from . import report_tools as rt

REPO_ROOT = rt.REPO_ROOT
MAX_STEPS_DEFAULT = 12
MAX_RUNS_DEFAULT = 4          # budget: how many pipeline subprocesses the agent may launch
RUN_TIMEOUT_S = 3600          # hard ceiling per run_screen subprocess

SYSTEM_PROMPT = (
    "You are the orchestrator for an area-selective ALD (AS-ALD) co-scientist. "
    "Your job: find the best selective inhibitor (and a precursor) by running "
    "the screening pipeline, deciding which reagents to screen and when the "
    "result is good enough. You do NOT compute energies or edit scoring "
    "thresholds -- the pipeline does the physics; you plan the runs.\n\n"
    "Chemistry: a good inhibitor binds SiNx (NGS) strongly and SiO2 (GS) weakly. "
    "contrast = dE_GS - dE_NGS; large positive with favourable (negative) dE_NGS "
    "is selective. The target is 90% selectivity at 10 nm oxide.\n\n"
    "Tools and discipline:\n"
    "- inspect_state: see what surfaces, reports and reagents exist. Call it "
    "first.\n"
    "- run_screen: runs the pipeline (Phase 2-5) on EXISTING surfaces for a "
    "chosen set of reagents and returns the ranked report. It is the expensive, "
    "GPU step -- screen a deliberate batch, read the contrasts, then run again "
    "only where it could beat the current best. You have a limited run budget.\n"
    "- request_build: propose a FRESH surface build (Phase 1). This is hours of "
    "GPU and is NOT run automatically -- it only asks the human to approve it. "
    "Use it only if existing surfaces are inadequate, and then keep working with "
    "what you have or finalize.\n\n"
    "A sensible plan: inspect, run one broad cheap screen (low sites) over the "
    "inhibitors, identify the top selective candidates, then re-screen just "
    "those at higher confidence, and finalize with the best inhibitor + "
    "precursor, their numbers, and whether the target is met. Never invent "
    "numbers; report only what run_screen returned. Stop when you can name a "
    "winner or explain why none qualifies."
)


# --------------------------------------------------------------------------
# Session: surface selection, run budget, and the blocking pipeline runner
# --------------------------------------------------------------------------

class OrchestratorSession:
    def __init__(self, surface_dir: Optional[str] = None,
                 max_runs: int = MAX_RUNS_DEFAULT):
        sets = rt.find_surface_sets()
        if not sets:
            raise RuntimeError(
                "no reusable surface slabs on disk -- approve a build first.")
        if surface_dir is None or surface_dir not in sets:
            surface_dir = next(
                (d for d, f in sets.items()
                 if {"SiO2", "SiNx"} <= set(rt.set_materials(f))),
                next(iter(sets)))
        self.surface_dir = surface_dir
        self.materials = rt.set_materials(sets[surface_dir])
        self.max_runs = max_runs
        self.runs_used = 0
        self.build_requests: List[Dict[str, Any]] = []
        self.last_report_path: Optional[str] = None

    # -- the one blocking, serialized GPU call -----------------------------
    def run_screen(self, reagents: Optional[List[str]], materials: Optional[List[str]],
                   max_sites: int) -> Dict[str, Any]:
        if self.runs_used >= self.max_runs:
            return {"error": f"run budget exhausted ({self.max_runs} runs). "
                    "Finalize with what you have."}
        self.runs_used += 1

        stamp = time.strftime("%Y%m%d_%H%M%S")
        outdir = os.path.join(REPO_ROOT, "runs", f"agent_{stamp}")
        os.makedirs(outdir, exist_ok=True)
        surface_glob = os.path.join(self.surface_dir, "*.*")
        cmd = [sys.executable, "run_pipeline.py", "--mode", "test",
               "--use-existing", surface_glob,
               "--output-dir", outdir, "--report", os.path.join(outdir, "report"),
               "--max-sites", str(max_sites)]
        mats = [m for m in (materials or self.materials) if m in self.materials]
        if mats:
            cmd += ["--materials", *mats]
        if reagents:
            cmd += ["--reagents", *reagents]

        try:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True,
                                  text=True, timeout=RUN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return {"error": f"run timed out after {RUN_TIMEOUT_S}s"}

        report_path = os.path.join(outdir, "report.json")
        if proc.returncode != 0 or not os.path.exists(report_path):
            tail = (proc.stderr or proc.stdout or "")[-800:]
            return {"error": f"pipeline run failed (rc={proc.returncode})",
                    "log_tail": tail}
        self.last_report_path = report_path
        report = rt.load_report(report_path)
        return {"runs_used": self.runs_used, "runs_left": self.max_runs - self.runs_used,
                "report_path": os.path.relpath(report_path, REPO_ROOT),
                **_summarize_report(report)}


def _summarize_report(report: Dict[str, Any]) -> Dict[str, Any]:
    sel = report.get("selection", {})
    rec = sel.get("recommendation", {})
    inhibitors = [{
        "name": c.get("name"), "dE_GS": c.get("dE_GS"), "dE_NGS": c.get("dE_NGS"),
        "contrast": c.get("contrast"), "accepted": c.get("accepted"),
        "reasons": c.get("reasons", []),
    } for c in sel.get("inhibitors", [])]
    selectivity = {n: {"S_at_target": s.get("selectivity_at_target"),
                       "meets_target": bool(s.get("meets_target"))}
                   for n, s in report.get("selectivity", {}).items()}
    return {
        "placeholder_calc": report.get("placeholder_calc"),
        "recommended_inhibitor": rec.get("inhibitor"),
        "recommended_precursor": rec.get("precursor"),
        "inhibitors": inhibitors,
        "selectivity": selectivity,
    }


# --------------------------------------------------------------------------
# Tool schemas + implementations
# --------------------------------------------------------------------------

def tool_schemas() -> List[Dict[str, Any]]:
    return [
        {"type": "function", "function": {
            "name": "inspect_state",
            "description": ("Report what is available: surface sets on disk, "
                            "existing reports (with their recommendation), and "
                            "the inhibitor/precursor library."),
            "parameters": {"type": "object", "properties": {}},
        }},
        {"type": "function", "function": {
            "name": "run_screen",
            "description": (
                "Run the pipeline (Phase 2 energetics -> selection -> "
                "selectivity -> report) on the EXISTING surfaces for the chosen "
                "reagents. Blocking and GPU-expensive; counts against the run "
                "budget. Returns the ranked inhibitors with dE_GS/dE_NGS/"
                "contrast/accepted and selectivity vs the 90%@10nm target."),
            "parameters": {"type": "object", "properties": {
                "reagents": {"type": "array", "items": {"type": "string"},
                             "description": "reagent names to screen; omit for the "
                             "whole library"},
                "materials": {"type": "array", "items": {"type": "string"},
                              "description": "restrict substrates, e.g. "
                              '["SiO2","SiNx"]'},
                "max_sites": {"type": "integer",
                              "description": "representative sites per site-type "
                              "(1 = fast/coarse, 3 = careful)"}},
                "required": []},
        }},
        {"type": "function", "function": {
            "name": "request_build",
            "description": (
                "Propose a FRESH surface build (Phase 1 melt-quench MD). This is "
                "NOT executed -- it asks the human to approve hours of GPU. Only "
                "use it if existing surfaces are inadequate; then continue with "
                "what you have or finalize."),
            "parameters": {"type": "object", "properties": {
                "materials": {"type": "array", "items": {"type": "string"}},
                "mode": {"type": "string", "enum": ["test", "fast", "full"]},
                "reason": {"type": "string"}},
                "required": ["materials", "mode", "reason"]},
        }},
    ]


def _as_dict(args: Any) -> Dict[str, Any]:
    if isinstance(args, str):
        try:
            return json.loads(args) or {}
        except json.JSONDecodeError:
            return {}
    return args or {}


def build_tools(session: OrchestratorSession,
                emit: Callable[[Dict[str, Any]], None]) -> Dict[str, Callable[..., Any]]:
    """`emit` lets a tool push an out-of-band event (used for approval requests)."""

    def inspect_state(**_) -> Dict[str, Any]:
        sets = rt.find_surface_sets()
        surfaces = [{"dir": os.path.relpath(d, REPO_ROOT),
                     "materials": rt.set_materials(f),
                     "in_use": d == session.surface_dir}
                    for d, f in sets.items()]
        reports = []
        for p in rt.find_reports()[:6]:
            try:
                r = rt.load_report(p)
            except (OSError, json.JSONDecodeError):
                continue
            reports.append({"path": os.path.relpath(p, REPO_ROOT),
                            "recommended_inhibitor":
                                r.get("selection", {}).get("recommendation", {})
                                .get("inhibitor"),
                            "placeholder": r.get("placeholder_calc")})
        return {"surfaces": surfaces, "surface_in_use": os.path.relpath(
                    session.surface_dir, REPO_ROOT),
                "reagents": rt.reagents_by_category(),
                "recent_reports": reports,
                "runs_left": session.max_runs - session.runs_used}

    def run_screen(reagents: Any = None, materials: Any = None,
                   max_sites: Any = 2, **_) -> Dict[str, Any]:
        try:
            max_sites = max(1, min(4, int(max_sites)))
        except (TypeError, ValueError):
            max_sites = 2
        if isinstance(reagents, str):
            reagents = [reagents]
        if isinstance(materials, str):
            materials = [materials]
        emit({"type": "run_start", "reagents": reagents or "whole library",
              "max_sites": max_sites})
        return session.run_screen(reagents, materials, max_sites)

    def request_build(materials: Any = None, mode: str = "fast",
                      reason: str = "", **_) -> Dict[str, Any]:
        if isinstance(materials, str):
            materials = [materials]
        req = {"materials": list(materials or []), "mode": mode, "reason": reason}
        session.build_requests.append(req)
        emit({"type": "approval_request", **req})
        return {"status": "build request logged for human approval; NOT executed. "
                "Continue with existing surfaces or finalize.",
                "request": req}

    return {"inspect_state": inspect_state, "run_screen": run_screen,
            "request_build": request_build}


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

def run_agent(model: str, goal: str, surface_dir: Optional[str] = None,
              max_steps: int = MAX_STEPS_DEFAULT,
              max_runs: int = MAX_RUNS_DEFAULT) -> Iterator[Dict[str, Any]]:
    try:
        session = OrchestratorSession(surface_dir=surface_dir, max_runs=max_runs)
    except Exception as e:
        yield {"type": "error", "step": 0, "message": str(e)}
        return

    yield {"type": "setup", "surfaces": session.materials,
           "dir": session.surface_dir, "runs_budget": session.max_runs}

    pending: List[Dict[str, Any]] = []
    tools = build_tools(session, pending.append)
    schemas = tool_schemas()

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            f"GOAL: {goal}\n\n"
            f"Surfaces in use: {', '.join(session.materials)} "
            f"(from {os.path.relpath(session.surface_dir, REPO_ROOT)}). "
            f"Run budget: {session.max_runs} pipeline runs.\n"
            "Call inspect_state first, then plan your screening."},
    ]

    for step in range(1, max_steps + 1):
        try:
            msg = llm.chat_tools(model, messages, schemas, timeout=RUN_TIMEOUT_S)
        except llm.OllamaError as e:
            yield {"type": "error", "step": step, "message": str(e)}
            return

        content = (msg.get("content") or "").strip()
        tool_calls = msg.get("tool_calls") or []
        messages.append({"role": "assistant", "content": content,
                         "tool_calls": tool_calls})
        if content:
            yield {"type": "assistant", "step": step, "text": content}

        if not tool_calls:
            yield {"type": "final", "step": step,
                   "text": content or "(model ended without a summary)"}
            return

        for tc in tool_calls:
            fn = tc.get("function", {}) or {}
            name = fn.get("name", "")
            args = _as_dict(fn.get("arguments"))
            yield {"type": "tool_call", "step": step, "name": name, "args": args}
            try:
                if name not in tools:
                    raise KeyError(f"unknown tool '{name}'")
                result = tools[name](**args)
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"}
            while pending:                      # surface any approval/run events
                ev = pending.pop(0)
                yield {**ev, "step": step}
            yield {"type": "tool_result", "step": step, "name": name,
                   "result": result}
            messages.append({"role": "tool", "tool_name": name,
                             "content": json.dumps(result, default=str)})

    yield {"type": "final", "step": max_steps,
           "text": "Reached the step budget without a final recommendation."}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

if __name__ == "__main__":
    if not llm.is_up():
        print(f"[orchestrator] Ollama not reachable at {llm.OLLAMA_HOST}."); sys.exit(1)
    goal = ("Find the best selective inhibitor and a precursor, and say whether "
            "the 90%@10nm target is met.")
    for ev in run_agent(llm.DEFAULT_MODEL, goal, max_runs=2, max_steps=8):
        t = ev["type"]
        if t == "setup":
            print(f"[setup] {ev['surfaces']} from {ev['dir'].split('AI-Scientist/')[-1]} "
                  f"budget={ev['runs_budget']}")
        elif t == "assistant":
            print(f"\n[step {ev['step']}] 💭 {ev['text']}")
        elif t == "tool_call":
            print(f"[step {ev['step']}] 🔧 {ev['name']}({json.dumps(ev['args'])})")
        elif t == "run_start":
            print(f"[step {ev['step']}] ⏳ screening {ev['reagents']} "
                  f"(max_sites={ev['max_sites']})…")
        elif t == "approval_request":
            print(f"[step {ev['step']}] 🚧 BUILD APPROVAL NEEDED: "
                  f"{ev['materials']} {ev['mode']} — {ev['reason']}")
        elif t == "tool_result":
            print(f"[step {ev['step']}] ↳ {json.dumps(ev['result'], default=str)[:600]}")
        elif t == "final":
            print(f"\n[final] ✅ {ev['text']}")
        elif t == "error":
            print(f"\n[error] {ev['message']}")
