"""Offline control-flow fixture. No LLM, BOND, MACE, or scientific predictions."""
from pathlib import Path
from .common import digest
from .metrics import evaluate
from .store import Store
from .campaign import run
from .report import export

DEMO_QUOTE = "Synthetic fixture: compare an analog that poisons GS with an analog that preserves GS."


class FixturePlanner:
    def plan(self, ctx):
        tested = [r for r in ctx["results"] if r["mode"] == "synthetic_demo"]
        if len(tested) >= 2:
            return {"action": "finish", "reason": "Two synthetic feedback rounds completed; no scientific conclusion."}
        poisoned = any(d["issue"] == "growth_surface_poisoning_in_surrogate" for d in ctx["diagnoses"])
        smiles = "CCC[Si:1]([Cl:2])(Cl)Cl" if poisoned else "C[Si:1]([Cl:2])(Cl)Cl"
        return {"action": "propose", "reason": "Respond to GS poisoning" if poisoned else "Exercise registration of a new molecule",
                "candidates": [{"name": "demo_propyltrichlorosilane" if poisoned else "demo_methyltrichlorosilane",
                  "smiles": smiles, "template": "silyl_substitution", "role": "inhibitor",
                  "hypothesis": "Synthetic demonstration only; this is not a claim about this molecule.",
                  "falsification": "The synthetic GS energy indicates growth poisoning.",
                  "evidence": [{"id": ctx["evidence"][0]["id"], "quote": DEMO_QUOTE}]}]}

    def critique(self, action, ctx):
        return {"approve": True, "reason": "Fixture proposal is explicitly synthetic."}


def execute(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    store = Store(out / "campaign.sqlite")
    store.evidence(DEMO_QUOTE, "offline test fixture", "synthetic_demo")
    def runner(ids, fidelity):
        for cid in ids:
            key = digest({"fixture": cid, "fidelity": fidelity})
            if store.get("result", key):
                continue
            c = store.get("candidate", cid)
            gs, ngs = (-2.0, -2.3) if "methyl" in c["name"] else (-.05, -1.4)
            row = {"id": key, "candidate_id": cid, "mode": "synthetic_demo", "status": "complete",
                   "role": "inhibitor", "dE_GS": gs, "dE_NGS": ngs, "fidelity": fidelity,
                   "metrics": evaluate(gs, ngs), "warning": "Fabricated numbers for software tests only."}
            store.put("result", key, row); store.event("synthetic_result", row)
    state = run(store, FixturePlanner(), runner,
                {"campaign": {"max_steps": 4, "max_candidates": 2}}, mode="synthetic_demo")
    export(store, out)
    store.close()
    return state
