"""Deterministic reports: model prose cannot override calculation records."""
from pathlib import Path
from .common import write_json
from .metrics import diagnose


def export(store, destination):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    candidates = {c["id"]: c for c in store.all("candidate")}
    results = store.all("result")
    eligible = [r for r in results if r.get("mode") == "mace" and r.get("status") == "complete"
                and r.get("role", "inhibitor") == "inhibitor" and r.get("metrics")
                and not any(x.get("candidate_id") == r["candidate_id"] and x.get("mode") == "mace"
                            and x.get("fidelity", 0) > r.get("fidelity", 0) for x in results)]
    # Highest evidence version per candidate, then a feasibility-first ordering.
    best_by_candidate = {}
    for r in eligible:
        previous = best_by_candidate.get(r["candidate_id"])
        if previous is None or r.get("fidelity", 0) > previous.get("fidelity", 0):
            best_by_candidate[r["candidate_id"]] = r
    ranking = sorted(best_by_candidate.values(), key=lambda r: (
        r["metrics"].get("model_pass", False),
        r["metrics"].get("minimum_growth_rate_met", False),
        r["metrics"].get("selectivity_at_target") if r["metrics"].get("selectivity_at_target") is not None else -2,
        r.get("contrast_eV", -1e9)), reverse=True)
    events = [{"seq": seq, "time": time, "kind": kind, "body": __import__('json').loads(body)}
              for seq, time, kind, body in store.db.execute("SELECT * FROM events ORDER BY seq")]
    result = {"scientific_success": False, "status": "screening_report_only",
              "ranked_real_screening_results": ranking, "results": results,
              "candidates": list(candidates.values()), "evidence": store.all("evidence"),
              "historical": store.all("historical"),
              "diagnoses": diagnose(results + store.all("historical")), "campaign": store.get("state", "campaign"),
              "hypotheses": store.all("hypothesis"), "notebook": store.recent("notebook", 10 ** 6)[::-1],
              "papers": store.all("paper"), "repair": store.get("decision", "latest"),
              "events": events}
    write_json(destination / "report.json", result)
    lines = ["# WARDEN v2 campaign report", "",
             "**Screening only. No experimental AS-ALD success is claimed.**",
             "Synthetic demo and unverified historical results are excluded from the real ranking.",
             "Surface admission permits numerical screening only; it does not certify bulk density or chemical accuracy.", "",
             "| Candidate | Mode | Status | GS eV | NGS eV | Proxy pass |",
             "|---|---|---|---:|---:|---|"]
    for r in results:
        name = candidates.get(r["candidate_id"], {}).get("name", r["candidate_id"])
        metrics = r.get("metrics") or {}
        lines.append(f"| {name} | {r.get('mode')} | {r.get('status')} | {r.get('dE_GS')} | {r.get('dE_NGS')} | {metrics.get('model_pass', False)} |")
    if result["hypotheses"]:
        lines += ["", "## Hypotheses (scored by code against screening surrogates)", "",
                  "| Candidate | Predicted | Status | Computed records |", "|---|---|---|---:|"]
        for h in result["hypotheses"]:
            lines.append(f"| {h['name']} | {h['predicted_direction']} | {h['status']} | {len(h['evidence'])} |")
    if result["notebook"]:
        lines += ["", "## Notebook", ""]
        lines += [f"- step {e.get('step')} [{e['source']}]: {e['text']}" for e in result["notebook"]]
    if result["papers"]:
        lines += ["", f"## Papers retrieved ({len(result['papers'])})", ""]
        lines += [f"- https://doi.org/{p['doi']} (evidence {p['evidence_id']})" for p in result["papers"]]
    lines += ["", "## Trace", ""]
    for e in events:
        lines.append(f"- {e['seq']}: **{e['kind']}** — {str(e['body'])[:1500]}")
    lines += ["", "All calculation attempts, geometry paths, configuration hashes, source evidence, "
              "and critic decisions are available in report.json and the SQLite database."]
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
