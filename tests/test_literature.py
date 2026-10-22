"""Literature loop: brief from computed outcomes, robust quotes, read-only tools, library import."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from warden import kinetics as kin
from warden.chemistry import validate_candidate
from warden.common import dumps
from warden.lit import findings, gather, import_candidates, lit_brief, outcome_class, quote_in, window_margin
from warden.literature import read_only
from warden.store import Store

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/kinetics/acetyl_chloride_xtb_gas.json"
PASSAGE = ("Acetyl chloride acylated the N\u2013H groups of the SiN\u2093 surface, while SiO\u2082 "
           "silanols remained \"largely unreacted\" at 150 \u00b0C.")


def bond_call_evidence(passage=PASSAGE):
    """Shape written by `warden bond-call`: the MCP result dumped as JSON, whose
    text content is itself a JSON string from the provider."""
    payload = {"results": [{"title": "Selective acylation", "abstract": passage, "doi": "10.0000/example"}]}
    result = {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": False}
    return {"id": "ev_test", "kind": "bond_result", "text": dumps(result), "source": {"provider": "BOND"}}


# --- quotes -----------------------------------------------------------------------

def test_quote_with_subscripts_and_quotes_matches_bond_call_evidence():
    ev = bond_call_evidence()
    assert "SiO\u2082" not in ev["text"]                          # stored escaped, twice
    assert quote_in('SiO\u2082 silanols remained "largely unreacted"', ev["text"])
    assert quote_in("SiO2 silanols remained", ev["text"])            # NFKC: subscript == digit
    assert not quote_in("SiO2 silanols reacted quickly", ev["text"])  # paraphrase is not a quote
    assert not quote_in("SiO2", ev["text"])                          # too short to identify a passage


def test_candidate_validation_uses_readable_evidence():
    ev = bond_call_evidence()
    spec = {"name": "acetyl chloride", "smiles": "C[C:1](=O)[Cl:2]", "template": "acylation",
            "hypothesis": "h", "falsification": "f",
            "evidence": [{"id": "ev_test", "quote": 'SiO\u2082 silanols remained "largely unreacted"'}]}
    assert validate_candidate(spec, {"ev_test": ev}, require_evidence=True)["canonical_smiles"] == "CC(=O)Cl"
    spec["evidence"][0]["quote"] = "silanols were fully acylated"
    with pytest.raises(ValueError, match="quote"):
        validate_candidate(spec, {"ev_test": ev}, require_evidence=True)


# --- read-only tools ------------------------------------------------------------

@pytest.mark.parametrize("name,ok", [
    ("semantic_search_papers", True), ("search_by_abstract", True), ("get_paper_by_id", True),
    ("find_similar_papers", True), ("export_bibtex", True),
    ("submit_feedback", False), ("mcp__plugin_valency_valency__submit_feedback", False),
    ("send_message", False), ("delete_paper", False)])
def test_write_like_tools_are_refused(name, ok):
    assert read_only(name) is ok


def test_allowlisted_write_tool_is_still_refused(tmp_path):
    server = tmp_path / "server.py"
    server.write_text('''from mcp.server.fastmcp import FastMCP
mcp = FastMCP("t")
@mcp.tool()
def submit_feedback(text: str) -> str:
    return "sent"
if __name__ == "__main__":
    mcp.run(transport="stdio")
''')
    config = tmp_path / "bond.json"
    config.write_text(json.dumps({"transport": "stdio", "command": sys.executable, "args": [str(server)],
                                  "allowed_tools": ["submit_feedback"]}))
    arguments = tmp_path / "a.json"; arguments.write_text(json.dumps({"text": "x"}))
    r = subprocess.run([sys.executable, "-m", "warden", "bond-call", "--config", str(config), "--tool",
                        "submit_feedback", "--arguments", str(arguments), "--out", str(tmp_path / "o.json")],
                       cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "refused" in r.stderr and not (tmp_path / "o.json").exists()


# --- brief ----------------------------------------------------------------------

def test_outcomes_split_chemistry_from_method():
    assert outcome_class({"status": "barrierless"}) == "chemistry"
    assert outcome_class({"status": "failed", "reason": "reactant_reacted_spontaneously"}) == "chemistry"
    assert outcome_class({"status": "failed", "reason": "SCF not converged in 250 cycles"}) == "numerical"
    assert outcome_class({"status": "band_unresolved"}) == "numerical"
    assert outcome_class({"status": "failed", "reason": "open_shell_cluster"}) == "setup"
    assert outcome_class({"status": "failed", "reason": "product_not_formed"}) == "inconclusive"


def test_brief_from_real_record_asks_for_less_reactive_chemistry():
    record = kin.reclassify(json.loads(FIXTURE.read_text()))
    records, _ = gather([record])
    b = lit_brief(records)
    codes = {f["code"] for c in b["candidates"] for f in c["findings"]}
    assert {"oxide_too_reactive", "no_kinetic_contrast"} <= codes
    assert any("amines over alcohols" in q["query"] for q in b["questions"])
    assert b["already_computed_smiles"] == ["C[C:1](=O)[Cl:2]"]


def test_numerical_failures_never_become_literature_questions():
    fail = {"status": "failed", "reason": "SCF not converged in 250 cycles"}
    record = {"candidate_id": "c", "name": "m", "smiles": "C[C:1](=O)[Cl:2]", "template": "acylation",
              "method": "xtb", "sites": [
                  {"label": "gas:OH", "material": "SiO2", "site_type": "OH", "population": 1, "attempts": [fail, fail]},
                  {"label": "gas:NH2", "material": "SiNx", "site_type": "NH2", "population": 1, "attempts": [fail]}]}
    records, _ = gather([record])
    b = lit_brief(records)
    assert b["questions"] == []
    assert all(f["kind"] == "method" for c in b["candidates"] for f in c["findings"])
    assert len(b["method_actions"]) == 2


def test_agent_observations_are_included():
    obs = [{"candidate": "p", "site": "gas:OH", "pathway": "addition", "method": "xtb",
            "status": "valid", "Ea_eV": 0.0, "reason": "barrierless"},
           {"candidate": "p", "site": "gas:NH2", "pathway": "addition", "method": "xtb",
            "status": "valid", "Ea_eV": 0.4, "reason": None},
           {"candidate": "p", "site": "slab:unknown", "pathway": "addition", "method": "xtb", "status": "failed"}]
    cands = {"p": {"name": "pivaloyl chloride", "smiles": "CC(C)(C)[C:1](=O)[Cl:2]", "template": "acylation"}}
    records, skipped = gather([], obs, cands)
    assert skipped == 1
    sites = {s["label"]: s for s in records[0]["sites"]}
    assert sites["gas:OH"]["barrierless"] and sites["gas:OH"]["Ea_eV"] == 0.0
    codes = {f["code"] for f in lit_brief(records)["candidates"][0]["findings"]}
    assert "oxide_too_reactive" in codes and "no_kinetic_contrast" in codes


def site(label, material, ea):
    return {"label": label, "material": material, "site_type": "x", "status": "valid", "Ea_eV": ea,
            "barrierless": False, "attempts": [{"status": "valid", "Ea_eV": ea}]}


def codes(*sites):
    return {f["code"] for f in findings({"name": "m", "template": "acylation", "sites": list(sites)})[1]}


def test_contrast_uses_the_slowest_nitride_site_and_the_dose_window():
    # Fastest nitride site is faster than the oxide, but the slowest one is not: no selectivity.
    assert "no_kinetic_contrast" in codes(site("o", "SiO2", 0.6), site("n1", "SiNx", 0.2), site("n2", "SiNx", 0.8))
    # Contrast, but both barriers too high for any practical dose.
    assert window_margin(2.0, 1.6) < 0
    assert "contrast_outside_dose_window" in codes(site("o", "SiO2", 2.0), site("n", "SiNx", 1.6))
    # Contrast inside the practical dose range.
    assert window_margin(1.2, 0.6) > 0
    assert "kinetic_contrast" in codes(site("o", "SiO2", 1.2), site("n", "SiNx", 0.6))


def test_site_barrier_comes_from_the_most_accurate_method_not_a_cross_method_minimum():
    def rec(method, ea):
        return {"candidate_id": "p", "name": "p", "template": "acylation", "method": method,
                "sites": [{"label": "gas:OH", "material": "SiO2", "site_type": "OH",
                           "attempts": [{"status": "valid", "Ea_eV": ea}]}]}
    records, _ = gather([rec("mace", 0.1), rec("xtb", 0.6)])
    (row,) = records[0]["sites"]
    assert row["method"] == "xtb" and row["Ea_eV"] == 0.6 and len(row["attempts"]) == 2


def test_agent_observation_site_type_is_looked_up_not_parsed():
    k = {"candidate_id": "p", "name": "p", "template": "acylation", "method": "xtb",
         "sites": [{"label": "slab1:12:top", "material": "SiNx", "site_type": "NH2", "attempts": []}]}
    obs = [{"candidate": "p", "site": "slab1:12:top", "pathway": "addition", "method": "xtb",
            "status": "valid", "Ea_eV": 0.4},
           {"candidate": "p", "site": "gas:NH_bridge", "pathway": "addition", "method": "xtb",
            "status": "valid", "Ea_eV": 0.5}]
    cands = {"p": {"name": "p", "smiles": "C[C:1](=O)[Cl:2]", "template": "acylation"}}
    records, _ = gather([k], obs, cands)
    types = {s["label"]: s["site_type"] for s in records[0]["sites"]}
    assert types == {"slab1:12:top": "NH2", "gas:NH_bridge": "NH_bridge"}


def test_decide_total_budget_caps_this_run(tmp_path):
    db = tmp_path / "c.sqlite"
    store = Store(db)
    store.put("kinetics", "k1", json.loads(FIXTURE.read_text()))
    spec = validate_candidate({"name": "acetyl chloride", "smiles": "C[C:1](=O)[Cl:2]", "template": "acylation"})
    store.put("candidate", spec["id"], spec)
    store.put("state", "decide", {"spent": 8.})
    store.close()
    r = subprocess.run([sys.executable, "-m", "warden", "decide", "--db", str(db), "--sites", "gas", "--budget", "5",
                        "--total-budget", "10", "--dry-run", "--out", str(tmp_path / "d")],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    result = json.loads((tmp_path / "d/decisions.json").read_text())
    assert result["budget"] == 2. and result["total_spent"] == 8.
    summary = Store(db).get("decide_summary", "latest")
    assert summary["pick"] == "acetyl chloride" and "scores_decades" in summary


# --- import -----------------------------------------------------------------------

def proposals():
    q = 'SiO\u2082 silanols remained "largely unreacted"'
    base = {"template": "acylation", "role": "inhibitor", "hypothesis": "h", "falsification": "f",
            "evidence": [{"id": "ev_test", "quote": q}]}
    return [{**base, "name": "N-acetylimidazole", "smiles": "C[C:1](=O)[n:2]1ccnc1", "literature_status": "analog"},
            {**base, "name": "acetic anhydride", "smiles": "C[C:1](=O)[O:2]C(C)=O", "literature_status": "tested"},
            {**base, "name": "acetyl chloride", "smiles": "C[C:1](=O)[Cl:2]", "literature_status": "tested"},
            {**base, "name": "no status", "smiles": "CC[C:1](=O)[Cl:2]"},
            {**base, "name": "bad quote", "smiles": "CCC[C:1](=O)[Cl:2]", "literature_status": "tested",
             "evidence": [{"id": "ev_test", "quote": "this sentence is not in the source"}]}]


def test_import_orders_tested_first_and_rejects_unsupported():
    ev = {"ev_test": bond_call_evidence()}
    library, rejected = import_candidates(proposals(), ev, known_smiles={"CC(=O)Cl"})
    assert [s["name"] for s in library] == ["acetic anhydride", "N-acetylimidazole"]
    reasons = {r["name"]: r["reason"] for r in rejected}
    assert "already computed" in reasons["acetyl chloride"]
    assert "literature_status" in reasons["no status"] and "quote" in reasons["bad quote"]
    for spec in library:                         # what `decide --library` will do with it
        assert validate_candidate(spec, ev)["id"] == spec["id"]


def test_import_checks_an_own_prior():
    ev = {"ev_test": bond_call_evidence()}
    ok, bad = proposals()[1], proposals()[0]
    ok = {**ok, "prior_Ea_eV": {"SiO2": 1.2, "SiNx": 0.6}, "prior_sd_eV": 0.25}
    bad = {**bad, "prior_Ea_eV": {"SiO2": "high"}}
    library, rejected = import_candidates([ok, bad], ev)
    assert [s["name"] for s in library] == ["acetic anhydride"]
    assert library[0]["prior_Ea_eV"] == {"SiO2": 1.2, "SiNx": 0.6}
    assert "prior_Ea_eV" in rejected[0]["reason"]


def test_cli_brief_and_import(tmp_path):
    db = tmp_path / "c.sqlite"
    store = Store(db)
    store.put("kinetics", "k1", json.loads(FIXTURE.read_text()))
    store.put("evidence", "ev_test", bond_call_evidence())
    store.close()
    run = lambda *a: subprocess.run([sys.executable, "-m", "warden", *a], cwd=ROOT,
                                    capture_output=True, text=True, timeout=120)
    r = run("lit-brief", "--db", str(db), "--config", str(ROOT / "config/v3.json"), "--out", str(tmp_path / "lit"))
    assert r.returncode == 0, r.stderr
    assert json.loads((tmp_path / "lit/queries/q01.json").read_text())["query"]
    (tmp_path / "props.json").write_text(json.dumps(proposals()))
    r = run("lit-import", "--db", str(db), "--candidates", str(tmp_path / "props.json"),
            "--out", str(tmp_path / "library.json"))
    assert r.returncode == 0, r.stderr
    names = [s["name"] for s in json.loads((tmp_path / "library.json").read_text())]
    assert names == ["acetic anhydride", "N-acetylimidazole"]     # acetyl chloride already computed
    # Re-running on the same proposals (e.g. after fixing one) gives the same library, not an empty one.
    r = run("lit-import", "--db", str(db), "--candidates", str(tmp_path / "props.json"),
            "--out", str(tmp_path / "library2.json"))
    assert r.returncode == 0, r.stderr
    assert [s["name"] for s in json.loads((tmp_path / "library2.json").read_text())] == names


def test_decide_accepts_literature_library(tmp_path):
    db = tmp_path / "c.sqlite"
    store = Store(db)
    store.put("kinetics", "k1", json.loads(FIXTURE.read_text()))
    spec = validate_candidate({"name": "acetyl chloride", "smiles": "C[C:1](=O)[Cl:2]", "template": "acylation"})
    store.put("candidate", spec["id"], spec)
    store.put("evidence", "ev_test", bond_call_evidence())
    store.close()
    library, _ = import_candidates(proposals()[:2], {"ev_test": bond_call_evidence()})
    (tmp_path / "library.json").write_text(json.dumps(library))
    r = subprocess.run([sys.executable, "-m", "warden", "decide", "--db", str(db), "--sites", "gas",
                        "--budget", "5", "--dry-run", "--library", str(tmp_path / "library.json"),
                        "--out", str(tmp_path / "decide")], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    plan = (tmp_path / "decide/decisions.md").read_text()
    assert "add_molecule acetic anhydride" in plan        # literature-tested molecule offered first
