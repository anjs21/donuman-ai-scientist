"""One evaluator for every v2 entry point. Predictions remain uncalibrated proxies."""
from dataclasses import asdict
import math
from selectivity_model import GrowthParams, selectivity_curve
from .common import clean


def evaluate(gs, ngs, config=None):
    config = config or {}
    if any(x is None or not math.isfinite(float(x)) for x in (gs, ngs)):
        return {"status": "insufficient_evidence", "model_pass": False,
                "scientific_success": False, "reason": "Both surfaces need valid energies."}
    params = GrowthParams(**config.get("growth", {}))
    result = selectivity_curve(float(gs), float(ngs), params,
                               n_cycles=config.get("max_cycles", 200))
    enough_growth = result["gpc_GS_A"] >= config.get("min_gpc_GS_A", 0.1)
    result = {k: v for k, v in result.items() if not hasattr(v, "__len__")}
    result["model_pass"] = bool(result.pop("meets_target") and enough_growth)
    result["scientific_success"] = False
    result["status"] = "screening_proxy_only"
    result["minimum_growth_rate_met"] = enough_growth
    result["growth_params"] = asdict(params)
    result["limitation"] = "Uncalibrated energy-to-growth surrogate; not proof of AS-ALD performance."
    return clean(result)


def diagnose(rows):
    """Deterministic failure categories for the planner, without invented chemistry."""
    out = []
    for row in rows:
        gs, ngs = row.get("dE_GS"), row.get("dE_NGS")
        if row.get("mode") == "legacy_unverified":
            issue = "historical_result_requires_revalidation"
        elif row.get("status") != "complete":
            issue = "calculation_or_sampling_failure"
        elif gs is None or ngs is None:
            issue = "missing_surface_evidence"
        elif not (row.get("metrics") or {}).get("minimum_growth_rate_met", True):
            issue = "growth_surface_poisoning_in_surrogate"
        elif gs - ngs <= 0:
            issue = "wrong_direction_binding_contrast"
        elif not (row.get("metrics") or {}).get("model_pass"):
            issue = "promising_contrast_but_target_not_met_in_surrogate"
        else:
            issue = "promising_proxy_requires_independent_validation"
        out.append({"candidate_id": row.get("candidate_id"), "issue": issue,
                    "dE_GS": gs, "dE_NGS": ngs})
    return out
