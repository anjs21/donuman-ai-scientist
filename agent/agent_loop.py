"""Autonomous screening agent: an on-device LLM that drives Phase 2 to find
the best inhibitor.

This agent does NOT touch the selection judgement criteria (thresholds live in
selection_criteria.md and are edited by hand in the Tune tab). Instead it works
one stage upstream, in the energetics screen itself: given the inhibitor
library and a set of surfaces already on disk, it runs an observe -> act ->
observe loop --

    list_inhibitors()          see the candidate space (+ what's screened)
    screen([names])            run Phase 2 energetics on chosen inhibitors;
                               get dE_GS, dE_NGS, contrast back per candidate
    rank_screened()            current best-to-worst by selective contrast
      ... iterate: screen more, narrow down ...
    -> report the single best selective inhibitor with its numbers.

"Best" = strong binding on SiNx (NGS, dE_NGS < 0) and weak on SiO2 (GS), i.e.
the largest positive contrast = dE_GS - dE_NGS. The value of the loop is that
the model can screen incrementally and stop early instead of brute-forcing the
whole library through the expensive MLIP -- the same batching a human would do.

Heavy modules (energetics/surface_builder/ase) are imported lazily so this file
stays cheap to load. ``run_agent`` yields trace events for a UI or the CLI:

    {"type":"setup",       "surfaces", "placeholder", "dir"}
    {"type":"assistant",   "step","text"}          model reasoning
    {"type":"tool_call",   "step","name","args"}   a tool it invoked
    {"type":"tool_result", "step","name","result"} what it got back
    {"type":"final",       "step","text"}          closing answer
    {"type":"error",       "step","message"}       loop-level failure
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Callable, Dict, Iterator, List, Optional

from . import llm
from . import report_tools as rt

MAX_STEPS_DEFAULT = 10

SYSTEM_PROMPT = (
    "You are an autonomous screening agent for area-selective ALD (AS-ALD). "
    "Your job is to find the single BEST inhibitor by running the Phase 2 "
    "energetics screen on candidate molecules and reading the results.\n\n"
    "Chemistry of 'best': a good inhibitor binds SiNx (the non-growth surface, "
    "NGS) strongly and SiO2 (the growth surface, GS) weakly, so the nitride is "
    "passivated while the oxide stays free to grow. Energies dE are in eV, "
    "negative = favourable binding. contrast = dE_GS - dE_NGS; a large POSITIVE "
    "contrast with a favourable (negative) dE_NGS is the selective winner. Some "
    "inhibitors only target one surface's sites, so a surface may come back with "
    "no data (null) -- that is expected.\n\n"
    "Each `screen` call runs a real MLIP relaxation and is the expensive step, "
    "so work like a scientist: call list_inhibitors first, screen a small "
    "promising batch, read the contrasts with rank_screened, then screen more "
    "only where it could beat the current best. Do NOT screen everything blindly "
    "if a clear winner has emerged. When you are confident, stop and name the "
    "best inhibitor with its dE_GS, dE_NGS and contrast, and one line of "
    "chemical justification. Never invent numbers -- only report what screen "
    "returned."
)


# --------------------------------------------------------------------------
# Screening session: holds surfaces + calculator + a per-reagent result cache
# --------------------------------------------------------------------------

class ScreenSession:
    """Loaded surfaces + MLIP calculator + memoised screen results.

    The calculator is built once and reused for every ``screen`` call, and each
    reagent's result is cached so the agent re-reading a candidate is free.
    """

    def __init__(self, surface_dir: Optional[str] = None, max_sites: int = 2,
                 n_workers: int = 1, dtype: str = "float32"):
        import surface_builder as sb  # heavy (torch/ase); import lazily

        sets = rt.find_surface_sets()
        if not sets:
            raise RuntimeError(
                "no reusable surface slabs found on disk -- build surfaces first "
                "(run_pipeline.py without --use-existing, or run_surface_builder.py).")
        if surface_dir is None or surface_dir not in sets:
            surface_dir = next(
                (d for d, f in sets.items()
                 if {"SiO2", "SiNx"} <= set(rt.set_materials(f))),
                next(iter(sets)))
        self.surface_dir = surface_dir
        self.surfaces = self._load(sets[surface_dir])
        self.calc = sb.get_calculator(dtype=dtype)
        self.is_placeholder = type(self.calc).__name__ == "LennardJones"
        self.max_sites = max_sites
        self.n_workers = n_workers
        self.cache: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _load(files: List[str]) -> Dict[str, list]:
        from ase.io import read
        surfaces: Dict[str, list] = {}
        for f in files:
            mat = rt._material_of(os.path.basename(f))
            if mat:
                surfaces.setdefault(mat, []).append(read(f))
        return surfaces

    @property
    def materials(self) -> List[str]:
        return sorted(self.surfaces)

    def _summarize(self, name: str, per_mat: Dict[str, Any]) -> Dict[str, Any]:
        def g(mat: str) -> Optional[float]:
            v = per_mat.get(mat, {}).get("dE_mean")
            if v is None:
                return None
            try:
                v = float(v)
            except (TypeError, ValueError):
                return None
            return None if math.isnan(v) else round(v, 3)

        dE_GS, dE_NGS = g("SiO2"), g("SiNx")
        contrast = (None if dE_GS is None or dE_NGS is None
                    else round(dE_GS - dE_NGS, 3))
        return {
            "name": name,
            "dE_GS": dE_GS, "dE_NGS": dE_NGS, "contrast": contrast,
            "binds_NGS": dE_NGS is not None and dE_NGS < 0,
            "selective": (contrast is not None and contrast > 0
                          and dE_NGS is not None and dE_NGS < 0),
        }

    def screen(self, names: List[str]) -> Dict[str, Dict[str, Any]]:
        import energetics as en
        import inhibitor_library as lib

        todo = [n for n in names if n not in self.cache]
        if todo:
            reagents = lib.get_reagents(names=todo)
            found = {r.name for r in reagents}
            res = en.screen_reagents(self.surfaces, reagents, self.calc,
                                     max_sites=self.max_sites,
                                     n_workers=self.n_workers)
            for name in todo:
                if name in res:
                    self.cache[name] = self._summarize(name, res[name])
                elif name not in found:
                    self.cache[name] = {"name": name, "error": "not an inhibitor "
                                        "in the library"}
        return {n: self.cache[n] for n in names if n in self.cache}

    def ranking(self) -> List[Dict[str, Any]]:
        rows = [r for r in self.cache.values() if "error" not in r]
        rows.sort(key=lambda r: (r["selective"],
                                 r["contrast"] if r["contrast"] is not None
                                 else -1e9),
                  reverse=True)
        return rows


# --------------------------------------------------------------------------
# Tool schemas + implementations (bound to a session)
# --------------------------------------------------------------------------

def tool_schemas() -> List[Dict[str, Any]]:
    return [
        {"type": "function", "function": {
            "name": "list_inhibitors",
            "description": ("List the inhibitor candidates in the library with "
                            "volatility and target site-types, and whether each "
                            "has been screened yet."),
            "parameters": {"type": "object", "properties": {}},
        }},
        {"type": "function", "function": {
            "name": "screen",
            "description": (
                "Run the Phase 2 energetics screen on the named inhibitors "
                "against the loaded surfaces. Expensive (real MLIP relaxations) "
                "so pass a small, deliberate batch. Returns per inhibitor: "
                "dE_GS (on SiO2), dE_NGS (on SiNx), contrast = dE_GS - dE_NGS, "
                "binds_NGS, and selective. Results are cached."),
            "parameters": {"type": "object", "properties": {
                "names": {"type": "array", "items": {"type": "string"},
                          "description": "inhibitor names to screen"}},
                "required": ["names"]},
        }},
        {"type": "function", "function": {
            "name": "rank_screened",
            "description": ("Return everything screened so far, ranked best "
                            "first (selective candidates by descending "
                            "contrast). Use it to decide whether to stop."),
            "parameters": {"type": "object", "properties": {}},
        }},
    ]


def _as_dict(args: Any) -> Dict[str, Any]:
    if isinstance(args, str):
        try:
            return json.loads(args) or {}
        except json.JSONDecodeError:
            return {}
    return args or {}


def build_tools(session: ScreenSession) -> Dict[str, Callable[..., Any]]:
    def list_inhibitors(**_) -> Dict[str, Any]:
        import inhibitor_library as lib
        out = []
        for r in lib.get_reagents(category="inhibitor"):
            out.append({"name": r.name, "volatility": r.volatility,
                        "targets": list(r.targets),
                        "screened": r.name in session.cache})
        return {"inhibitors": out, "surfaces": session.materials}

    def screen(names: Any = None, **_) -> Dict[str, Any]:
        names = names or []
        if isinstance(names, str):
            names = [names]
        if not names:
            return {"error": "no names given"}
        return {"screened": session.screen(list(names))}

    def rank_screened(**_) -> Dict[str, Any]:
        ranked = session.ranking()
        best = next((r for r in ranked if r["selective"]), None)
        return {"best_selective": best, "ranking": ranked}

    return {"list_inhibitors": list_inhibitors, "screen": screen,
            "rank_screened": rank_screened}


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

def run_agent(model: str, goal: str, surface_dir: Optional[str] = None,
              max_steps: int = MAX_STEPS_DEFAULT, max_sites: int = 2,
              n_workers: int = 1) -> Iterator[Dict[str, Any]]:
    """Drive the model through a Phase 2 screening loop; yield trace events."""
    try:
        session = ScreenSession(surface_dir=surface_dir, max_sites=max_sites,
                                n_workers=n_workers)
    except Exception as e:
        yield {"type": "error", "step": 0, "message": str(e)}
        return

    yield {"type": "setup", "surfaces": session.materials,
           "placeholder": session.is_placeholder, "dir": session.surface_dir}

    tools = build_tools(session)
    schemas = tool_schemas()

    inh_names = [r["name"] for r in tools["list_inhibitors"]()["inhibitors"]]
    warn = ("\n\nNOTE: the calculator is a Lennard-Jones PLACEHOLDER -- energies "
            "are not physical; report the mechanics but temper any conclusion."
            if session.is_placeholder else "")
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            f"GOAL: {goal}\n\n"
            f"Surfaces loaded: {', '.join(session.materials)} "
            f"(from {session.surface_dir}).\n"
            f"Inhibitors available: {', '.join(inh_names)}.\n"
            "Start by calling list_inhibitors, then screen a promising batch."
            + warn},
    ]

    for step in range(1, max_steps + 1):
        try:
            msg = llm.chat_tools(model, messages, schemas)
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
            yield {"type": "tool_result", "step": step, "name": name,
                   "result": result}
            messages.append({"role": "tool", "tool_name": name,
                             "content": json.dumps(result, default=str)})

    yield {"type": "final", "step": max_steps,
           "text": "Reached the step budget without naming a best inhibitor."}


# --------------------------------------------------------------------------
# CLI: run the screening agent and print the trace
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if not llm.is_up():
        print(f"[agent] Ollama not reachable at {llm.OLLAMA_HOST}; "
              "start it with `ollama serve`."); sys.exit(1)

    goal = ("Screen the inhibitor library and identify the single best "
            "selective inhibitor for passivating SiNx while sparing SiO2.")
    for ev in run_agent(llm.DEFAULT_MODEL, goal, max_sites=1):
        t = ev["type"]
        if t == "setup":
            print(f"[setup] surfaces={ev['surfaces']} placeholder={ev['placeholder']} "
                  f"dir={ev['dir'].split('AI-Scientist/')[-1]}")
        elif t == "assistant":
            print(f"\n[step {ev['step']}] 💭 {ev['text']}")
        elif t == "tool_call":
            print(f"[step {ev['step']}] 🔧 {ev['name']}({json.dumps(ev['args'])})")
        elif t == "tool_result":
            print(f"[step {ev['step']}] ↳ {json.dumps(ev['result'], default=str)[:700]}")
        elif t == "final":
            print(f"\n[final] ✅ {ev['text']}")
        elif t == "error":
            print(f"\n[error] {ev['message']}")
