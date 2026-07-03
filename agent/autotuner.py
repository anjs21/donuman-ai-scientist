"""Autonomous threshold-tuning agent (closed propose -> verify -> refine loop).

The one-shot "diagnose & suggest" flow (app.py Job 3) asks the model for a
threshold change *once* and trusts it. In practice a small model proposes
changes that don't actually achieve the goal -- e.g. it lowers min_contrast to
0.15 to "accept pyridine", but pyridine's contrast is 0.05, so it stays
rejected, and the model never checks.

This module closes the loop and makes the step autonomous:

    round r:  model proposes threshold changes (structured JSON)
              -> reselect() re-runs selection + selectivity DETERMINISTICALLY
              -> we compute the objective (does the *recommended* inhibitor meet
                 the 90%-at-10nm target?) and the physics ceiling (best
                 selectivity ANY candidate can reach, whatever the thresholds)
              -> that result is fed back to the model for the next round

The verifier is code, not the model, so the agent cannot "win" by loosening a
threshold to admit a non-selective inhibitor: the selectivity simulator still
shows it failing. The loop stops when

  * the recommended inhibitor actually meets the target (SUCCESS), or
  * the model concludes no threshold can help (STOP -- the honest answer when
    the physics ceiling is below target, i.e. the library lacks a selective
    inhibitor), or
  * it runs out of rounds (EXHAUSTED).

Pure CPU (reselect is sub-second); the only GPU-free "cost" is the local LLM
calls. Import-safe: no torch/streamlit.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from agent import llm
from agent import report_tools as rt

TARGET_S = 0.90        # selectivity target (fraction) — matches GrowthParams


# One extra field vs SUGGESTION_SCHEMA: an explicit `stop` the agent sets when
# it judges no threshold change can reach the target (so the loop can end on the
# model's own reasoning, not just a round cap).
LOOP_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "stop": {"type": "boolean"},
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string",
                            "enum": list(rt.CONFIG_BOUNDS.keys())},
                    "to": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["key", "to", "reason"],
            },
        },
    },
    "required": ["reasoning", "stop", "changes"],
}

SYSTEM = (
    "You are an autonomous tuning agent for an area-selective ALD (AS-ALD) "
    "screening pipeline. Goal: choose selection THRESHOLDS so the pipeline "
    "recommends an inhibitor that meets the target of 90% selectivity at 10 nm "
    "oxide. The adsorption energies (dE, eV) are FIXED measurements -- you only "
    "move thresholds, you never invent or change energies.\n"
    "A deterministic simulator re-runs selection + selectivity after every "
    "proposal and reports back: the recommended inhibitor, its predicted "
    "selectivity, and the PHYSICS CEILING = the best selectivity achievable by "
    "ANY candidate under ANY thresholds. Read the feedback and adapt.\n"
    "CRITICAL: loosening a threshold to accept a poorly-selective inhibitor "
    "does NOT help -- the selectivity simulator will still show it below "
    "target. If the physics ceiling is below 90%, NO threshold setting can "
    "succeed; then set \"stop\": true and explain that the molecule library "
    "needs a more selective inhibitor. Propose the SMALLEST change justified by "
    "the numbers. An inhibitor is ACCEPTED (eligible to be recommended) when it "
    "passes the two HARD gates: bind NGS (dE_NGS <= bind_threshold_eV) AND "
    "contrast (dE_GS - dE_NGS >= min_contrast_eV). spare_threshold_eV is only a "
    "soft score penalty for also binding GS, not a hard gate. So to admit a "
    "blocked candidate, relax bind_threshold_eV or min_contrast_eV. Respond "
    "only in the required JSON."
)

# Bounds table handed to the model so it proposes in-range values.
_BOUNDS_TEXT = "; ".join(f"{k} in [{lo},{hi}]"
                        for k, (lo, hi) in rt.CONFIG_BOUNDS.items())


def _objective(report: Dict[str, Any],
               overrides: Dict[str, float]) -> Dict[str, Any]:
    """Deterministically evaluate one threshold set.

    Returns the recommended inhibitor, its predicted selectivity/meets-target,
    the physics ceiling across all candidates, and per-candidate accept/reject
    detail -- everything the loop needs to judge success and to brief the model.
    """
    new = rt.reselect(report, overrides)
    rec = new["selection"]["recommendation"]["inhibitor"]
    sel = new["selectivity"]

    def _s(name):
        s = sel.get(name, {})
        return s.get("selectivity_at_target"), bool(s.get("meets_target"))

    rec_S, rec_meets = _s(rec) if rec else (None, False)

    # ceiling: best selectivity any candidate reaches, thresholds aside
    ceiling_name, ceiling_S = None, None
    for name, s in sel.items():
        v = s.get("selectivity_at_target")
        if v is not None and (ceiling_S is None or v > ceiling_S):
            ceiling_name, ceiling_S = name, v

    cand = [{"name": c["name"], "dE_GS": c["dE_GS"], "dE_NGS": c["dE_NGS"],
             "contrast": c["contrast"], "accepted": c["accepted"],
             "reasons": c.get("reasons", [])}
            for c in new["selection"]["inhibitors"]]

    return {
        "config": new["config"], "recommended": rec,
        "rec_selectivity": rec_S, "meets_target": bool(rec and rec_meets),
        "ceiling_name": ceiling_name, "ceiling_selectivity": ceiling_S,
        "candidates": cand,
    }


def _feedback(obj: Dict[str, Any]) -> str:
    """Render one evaluation as compact feedback text for the model."""
    def pct(x):
        return "n/a" if x is None else f"{x*100:.1f}%"
    L = [f"recommended inhibitor: {obj['recommended'] or 'NONE'}",
         f"  its predicted selectivity @10nm: {pct(obj['rec_selectivity'])} "
         f"(target 90%) -> meets_target={obj['meets_target']}",
         f"physics ceiling: best any candidate can reach = "
         f"{pct(obj['ceiling_selectivity'])} "
         f"(via {obj['ceiling_name'] or 'none'})",
         "candidates:"]
    for c in obj["candidates"]:
        tag = "ACCEPT" if c["accepted"] else "reject: " + "; ".join(c["reasons"])
        L.append(f"  {c['name']:10s} dE_GS={rt._fmt(c['dE_GS'])} "
                 f"dE_NGS={rt._fmt(c['dE_NGS'])} "
                 f"contrast={rt._fmt(c['contrast'])} -> {tag}")
    # Deterministic feasibility verdict — the code does the comparison so the
    # model doesn't have to (a weak model misreads "97.8% vs 90%").
    ceil = obj["ceiling_selectivity"]
    if ceil is not None and ceil >= TARGET_S and not obj["meets_target"]:
        L.append(f"=> FEASIBLE: '{obj['ceiling_name']}' can reach "
                 f"{ceil*100:.1f}% (>= {TARGET_S*100:.0f}% target) but is not yet "
                 f"ACCEPTED. A threshold solution EXISTS — relax the gate(s) in "
                 f"its reject reason to admit it. Do NOT stop.")
    elif ceil is not None and ceil < TARGET_S:
        L.append(f"=> INFEASIBLE: even the best candidate tops out at "
                 f"{ceil*100:.1f}% (< {TARGET_S*100:.0f}% target). No thresholds "
                 f"can reach the target; set stop=true.")
    return "\n".join(L)


def autotune(report: Dict[str, Any], model: str, *, max_rounds: int = 6,
             start_overrides: Optional[Dict[str, float]] = None,
             log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Run the autonomous propose->verify->refine loop.

    Returns {status, rounds, best, final_config, history}. `status` is one of
    "SUCCESS" (recommended inhibitor meets target), "STOP" (agent concluded no
    threshold can help), or "EXHAUSTED" (hit max_rounds).

    start_overrides optionally seeds the threshold set (e.g. to begin from an
    intentionally strict config); defaults to the on-file thresholds.

    The code -- not the model -- decides SUCCESS: whenever the deterministic
    objective reports meets_target, the loop terminates SUCCESS regardless of
    what the model says (a small model sometimes misreads its own feedback).
    """
    digest = rt.report_digest(report)
    overrides: Dict[str, float] = dict(start_overrides or {})  # running set
    history: List[Dict[str, Any]] = []

    # baseline so the agent sees the start state; config judged deterministically
    base = _objective(report, overrides)
    cfg0 = base["config"]
    log("[round 0] baseline thresholds")
    log(_feedback(base))
    best = base

    # If the starting thresholds already recommend a target-meeting inhibitor,
    # there is nothing to tune -- the verifier says SUCCESS, no model call.
    if base["meets_target"]:
        log(f"\n[round 0] SUCCESS at baseline: '{base['recommended']}' already "
            f"meets the 90%@10nm target; no tuning needed.")
        return {"status": "SUCCESS", "rounds": 0, "best": base,
                "final_config": rt.clamp_config(overrides) or cfg0, "history": []}

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"Threshold bounds: {_BOUNDS_TEXT}\n"
            f"Current thresholds: {json.dumps(cfg0)}\n\n"
            f"RUN REPORT\n----------\n{digest}\n\n"
            f"BASELINE RESULT\n---------------\n{_feedback(base)}\n\n"
            "Propose the smallest threshold change that could make the pipeline "
            "recommend a target-meeting inhibitor, or set stop=true if the "
            "physics ceiling shows it is impossible."},
    ]

    status = "EXHAUSTED"
    for r in range(1, max_rounds + 1):
        try:
            step = llm.chat_json(model, messages, LOOP_SCHEMA)
        except llm.OllamaError as e:
            log(f"[round {r}] model error: {e}")
            status = "MODEL_ERROR"
            break

        changes = {c["key"]: c["to"] for c in step.get("changes", [])
                   if c.get("key") in rt.CONFIG_BOUNDS}
        log(f"\n[round {r}] reasoning: {step.get('reasoning','').strip()}")
        if step.get("stop") and not changes:
            # Only honour a stop when it's actually infeasible. If the verifier
            # says a solution exists (ceiling >= target), a stop is a weak-model
            # false surrender — push back instead of quitting.
            ceil = best["ceiling_selectivity"]
            if ceil is not None and ceil >= TARGET_S:
                log(f"[round {r}] agent tried to stop, but '{best['ceiling_name']}' "
                    f"can reach {ceil*100:.1f}% >= target — rejecting the stop, "
                    f"re-prompting.")
                messages.append({"role": "assistant", "content": json.dumps(step)})
                messages.append({"role": "user", "content":
                    f"Do NOT stop. The simulator confirms '{best['ceiling_name']}' "
                    f"reaches {ceil*100:.1f}%, above the {TARGET_S*100:.0f}% "
                    f"target — a threshold solution EXISTS. It is currently "
                    f"rejected only because of its listed gate(s). Propose the "
                    f"specific threshold change (bind_threshold_eV or "
                    f"min_contrast_eV) that admits it."})
                history.append({"round": r, "reasoning": step.get("reasoning"),
                                "changes": {}, "stopped": "rejected"})
                continue
            log(f"[round {r}] agent stopped: no threshold change proposed.")
            status = "STOP"
            history.append({"round": r, "reasoning": step.get("reasoning"),
                            "changes": {}, "stopped": True})
            break

        overrides.update(changes)             # accumulate absolute values
        obj = _objective(report, overrides)
        clamped = rt.clamp_config(overrides)
        log(f"[round {r}] applied {clamped}")
        log(_feedback(obj))
        history.append({"round": r, "reasoning": step.get("reasoning"),
                        "changes": clamped, "result": obj})

        # track the best real selectivity seen
        if (obj["rec_selectivity"] or -1) > (best["rec_selectivity"] or -1):
            best = obj

        if obj["meets_target"]:
            log(f"\n[round {r}] SUCCESS: '{obj['recommended']}' meets the "
                f"90%@10nm target with thresholds {clamped}.")
            status = "SUCCESS"
            break

        messages.append({"role": "assistant", "content": json.dumps(step)})
        messages.append({"role": "user", "content":
            f"RESULT of your change ({clamped}):\n{_feedback(obj)}\n\n"
            "If this meets the target you are done. Otherwise propose the next "
            "smallest change, or stop=true if the physics ceiling proves it "
            "cannot be reached by any thresholds."})
    else:
        log(f"\n[exhausted] {max_rounds} rounds without meeting target.")

    return {"status": status, "rounds": len(history), "best": best,
            "final_config": rt.clamp_config(overrides) or cfg0,
            "history": history}


# ---------------------------------------------------------------------------
# CLI:  python -m agent.autotuner <report.json> [model] [max_rounds]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "results_testpipe/report.json"
    model = sys.argv[2] if len(sys.argv) > 2 else llm.DEFAULT_MODEL
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 6

    if not llm.is_up():
        print(f"Ollama not reachable at {llm.OLLAMA_HOST}; start it with "
              f"`ollama serve` and pull {model}.")
        raise SystemExit(1)

    rep = rt.load_report(path)
    print("=" * 70)
    print(f"AUTONOMOUS THRESHOLD TUNER  (model={model}, max_rounds={rounds})")
    print(f"report: {path}")
    print("=" * 70)
    out = autotune(rep, model, max_rounds=rounds)
    print("\n" + "=" * 70)
    print(f"FINAL: status={out['status']}  rounds={out['rounds']}")
    b = out["best"]
    bs = "n/a" if b["rec_selectivity"] is None else f"{b['rec_selectivity']*100:.1f}%"
    print(f"best recommendation: {b['recommended']}  (selectivity {bs}, "
          f"ceiling {'n/a' if b['ceiling_selectivity'] is None else f'{b['ceiling_selectivity']*100:.1f}%'})")
    print(f"final thresholds: {out['final_config']}")
