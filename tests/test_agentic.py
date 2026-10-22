"""Regression tests for literature mining, analog design, self-repair and campaign memory."""
import pytest
from rdkit import Chem
from warden import kinetics as kin
from warden import notebook
from warden.analogs import enumerate_analogs, as_evidence_text
from warden.campaign import run, context
from warden.chemistry import validate_candidate
from warden.literature import dois, result_text
from warden.store import Store

ACYL = validate_candidate({"name": "acetyl chloride", "smiles": "C[C:1](=O)[Cl:2]", "template": "acylation"})
DMATMS = validate_candidate({"name": "DMATMS", "smiles": "C[Si:1](C)(C)[N:2](C)C",
                             "template": "silyl_substitution"})
PAPER = ("Area-selective deposition on SiO2 versus SiN: acetyl chloride vapour caps surface amines "
         "selectively over silanols at 150 C (doi:10.26434/chemrxiv-2026-abc12).")


def _identity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


@pytest.mark.parametrize("parent", [ACYL, DMATMS])
def test_analogs_are_valid_new_and_keep_the_template(parent):
    analogs = enumerate_analogs(parent, "all", limit=20)
    assert analogs and len({a["canonical_smiles"] for a in analogs}) == len(analogs)
    assert parent["canonical_smiles"] not in {a["canonical_smiles"] for a in analogs}
    for a in analogs:
        assert validate_candidate({**a, "role": "inhibitor"})["template"] == parent["template"]


def test_analog_directions_follow_leaving_group_and_sterics():
    reactive = enumerate_analogs(DMATMS, "more_reactive")
    assert {_identity(a["smiles"]) for a in reactive} >= {"C[Si](C)(C)Cl"}
    assert all(a.get("pKa_HX_approx", 99) < 36 or a.get("electronic_edit") == "withdrawing" for a in reactive)
    bulky = enumerate_analogs(ACYL, "bulkier")
    assert [a["steric_edit"] for a in bulky] == ["ethyl", "isopropyl", "tert-butyl"]
    known = [a["canonical_smiles"] for a in bulky]
    assert not enumerate_analogs(ACYL, "bulkier", known=known)


def test_analog_evidence_lines_are_quotable():
    designs = enumerate_analogs(DMATMS, "more_reactive")
    text = as_evidence_text(DMATMS, designs, "more_reactive")
    line = text.splitlines()[1]
    evidence = {"ev": {"text": text}}
    spec = {"name": designs[0]["name"], "smiles": designs[0]["smiles"], "template": designs[0]["template"],
            "hypothesis": "h", "falsification": "f", "evidence": [{"id": "ev", "quote": line[:60]}]}
    assert validate_candidate(spec, evidence, require_evidence=True)


def test_result_text_keeps_quotes_verbatim_and_finds_dois():
    result = {"content": [{"type": "text", "text": PAPER}], "isError": False}
    assert result_text(result) == PAPER
    assert dois(PAPER) == ["10.26434/chemrxiv-2026-abc12"]


def test_hypothesis_status_follows_computed_evidence(tmp_path):
    s = Store(tmp_path / "db")
    spec = {**ACYL, "hypothesis": "caps amines faster", "falsification": "no contrast",
            "predicted_direction": "prefers_SiNx"}
    s.put("candidate", spec["id"], spec)
    assert notebook.hypothesis_status(s, spec)["status"] == "untested"
    s.put("result", "r0", {"candidate_id": spec["id"], "mode": "mace", "status": "failed"})
    assert notebook.hypothesis_status(s, spec)["status"] == "inconclusive"
    def kinetics(gs, ngs):
        site = lambda m, ea: {"material": m, "label": m, "population": 1, "status": "valid", "Ea_eV": ea,
                              "attempts": [{"status": "valid", "Ea_eV": ea, "dE_rxn_eV": -.5,
                                            "stepwise_indication": False, "precomplex_shift_eV": 0.}]}
        return {"candidate_id": spec["id"], "method": "xtb", "sites": [site("SiO2", gs), site("SiNx", ngs)]}
    # Right direction, but both barriers too high for any practical dose: not "supported".
    s.put("kinetics", "k0", kinetics(2.0, 1.6))
    assert notebook.hypothesis_status(s, spec)["status"] == "direction_only"
    s.put("kinetics", "k0", kinetics(1.2, 0.6))
    assert notebook.hypothesis_status(s, spec)["status"] == "supported"
    s.put("result", "r1", {"candidate_id": spec["id"], "mode": "mace", "status": "complete",
                           "contrast_eV": -.2, "fidelity": 1})
    assert notebook.hypothesis_status(s, spec)["status"] == "refuted"
    notebook.update_hypotheses(s, 1)
    assert any("refuted" in e["text"] for e in notebook.recent(s))
    s.close()


def test_repair_observations_fill_failed_sites_and_lower_slower_ones():
    sites = [{"material": "SiO2", "label": "gs", "population": 1, "status": "valid", "Ea_eV": 1.0},
             {"material": "SiNx", "label": "ngs", "population": 1, "status": "failed", "Ea_eV": None}]
    record = {"candidate_id": "c", "method": "mace", "sites": sites, "summary": kin.summarize(sites)}
    obs = [{"candidate": "c", "site": "ngs", "pathway": "four_centre", "method": "mace", "status": "valid",
            "Ea_eV": .45}, {"candidate": "c", "site": "ngs", "pathway": "addition", "method": "mace",
                             "status": "valid", "Ea_eV": .30},
           {"candidate": "c", "site": "gs", "pathway": "addition", "method": "mace", "status": "valid", "Ea_eV": .1},
           {"candidate": "c", "site": "ngs", "pathway": "addition", "method": "xtb", "status": "valid", "Ea_eV": .01}]
    out = kin.apply_observations(record, obs)
    # The failed nitride site takes its fastest mace path; the oxide site is lowered by the
    # faster repaired path (0.1 < 1.0): ignoring it would overstate selectivity. xtb is ignored.
    assert out["repaired_sites"] == ["gs", "ngs"]
    assert [s["Ea_eV"] for s in out["sites"]] == [.1, .30]
    assert out["summary"]["dEa_GS_minus_NGS_eV"] == pytest.approx(-.20)
    # A slower observed path never raises a valid site.
    slow = kin.apply_observations(record, [{**obs[2], "Ea_eV": 1.5}])
    assert slow["repaired_sites"] == [] and slow["sites"][0]["Ea_eV"] == 1.0


class Literature:
    config = {"allowed_tools": ["semantic_search_papers"]}

    def __init__(self):
        self.calls = []

    def tools(self):
        return [{"name": "semantic_search_papers", "description": "search", "inputSchema": {"type": "object"}},
                {"name": "post_message", "description": "not allowed", "inputSchema": {"type": "object"}}]

    def call(self, tool, arguments):
        self.calls.append((tool, arguments))
        return {"content": [{"type": "text", "text": PAPER}]}


class Scripted:
    """Planner that replays a fixed list of actions and records what it saw."""

    def __init__(self, actions):
        self.actions, self.seen = list(actions), []

    def plan(self, ctx):
        self.seen.append(ctx)
        if not self.actions:
            return {"action": "finish", "reason": "script done"}
        action = self.actions.pop(0)
        return action(ctx) if callable(action) else action

    def critique(self, action, ctx):
        return {"approve": True, "reason": "scripted"}


def test_search_mining_dedupe_and_memory(tmp_path):
    s = Store(tmp_path / "db")
    query = {"action": "search", "reason": "find inhibitors", "tool": "semantic_search_papers",
             "arguments": {"query": "area-selective deposition SiN inhibitor", "source": "chemrxiv"},
             "lesson": "Search ChemRxiv before proposing."}
    quote = "acetyl chloride vapour caps surface amines selectively over silanols"

    def propose(ctx):
        ev = next(e for e in ctx["evidence"] if e["kind"] == "bond_result")
        return {"action": "propose", "reason": "literature-tested", "candidates": [{
            "name": "acetyl chloride", "smiles": "C[C:1](=O)[Cl:2]", "template": "acylation", "role": "inhibitor",
            "hypothesis": "Amine acylation outpaces silanol esterification.", "falsification": "dEa <= 0",
            "predicted_direction": "prefers_SiNx", "evidence": [{"id": ev["id"], "quote": quote}]}]}

    computed = []
    planner = Scripted([query, query, propose])
    lit = Literature()
    state = run(s, planner, lambda ids, f: computed.append(ids), {"campaign": {"max_steps": 5}}, lit)
    assert state["status"] == "finished" and len(lit.calls) == 1 and computed
    last = planner.seen[-1]
    assert [t["name"] for t in last["literature_tools"]] == ["semantic_search_papers"]
    assert last["literature"]["papers_seen"] == 1
    assert any("Identical search" in e["text"] for e in last["notebook"])
    assert any(e["text"] == "Search ChemRxiv before proposing." for e in last["notebook"])
    assert last["hypotheses"][0]["status"] == "untested"
    s.close()


def test_analog_action_then_cited_proposal(tmp_path):
    s = Store(tmp_path / "db")
    s.put("candidate", DMATMS["id"], DMATMS)

    def propose(ctx):
        ev = next(e for e in ctx["evidence"] if e["kind"] == "analog_design")
        line = ev["text"].splitlines()[1]
        smiles = line.split(": ", 1)[1].split(" via ")[0]
        return {"action": "propose", "reason": "analog", "candidates": [{
            "name": "DMATMS analog", "smiles": smiles, "template": "silyl_substitution", "role": "inhibitor",
            "hypothesis": "Better leaving group lowers the nitride barrier.", "falsification": "dEa <= 0",
            "evidence": [{"id": ev["id"], "quote": line[:50]}]}]}

    computed = []
    state = run(s, Scripted([{"action": "analogs", "reason": "steer", "candidate_id": DMATMS["id"],
                              "direction": "more_reactive"}, propose]),
                lambda ids, f: computed.append(ids), {"campaign": {"max_steps": 4}})
    assert state["status"] == "finished" and len(computed) == 1
    assert len(s.all("candidate")) == 2 and s.all("hypothesis")
    s.close()


def test_tool_schemas_are_compacted_for_the_planner():
    from warden.campaign import compact_schema
    schema = {"type": "object", "required": ["query"],
              "properties": {"query": {"type": "string", "description": "x" * 500},
                             "sort_by": {"type": "string", "enum": ["relevance", "date"]}}}
    out = compact_schema(schema)
    assert out["query"] == {"type": "string", "about": "x" * 80, "required": True}
    assert out["sort_by"]["enum"] == ["relevance", "date"]


def test_gas_site_runner_passes_no_manifest(tmp_path, monkeypatch):
    import subprocess as sp
    from warden.campaign import subprocess_runner
    calls = []
    monkeypatch.setattr(sp, "run", lambda cmd, **kw: calls.append(cmd))
    (tmp_path / "c.json").write_text('{"kinetics": {"method": "mace"}}')
    runner, repairer = subprocess_runner(tmp_path / "db", tmp_path / "m.json", tmp_path / "c.json", 10,
                                         kinetics=True, kinetics_sites="gas")
    runner(["c1"], 0); repairer(["c1"], 2.0)
    kinetics, decide = calls[1], calls[3]
    assert kinetics[kinetics.index("--sites") + 1] == "gas" and "--manifest" not in kinetics
    assert decide[decide.index("--methods") + 1] == "mace" and "--manifest" not in decide
    with pytest.raises(ValueError):
        subprocess_runner(tmp_path / "db", None, None, 1, kinetics_sites="slab")


def test_repair_action_is_budgeted_and_needs_kinetics(tmp_path):
    s = Store(tmp_path / "db")
    s.put("candidate", ACYL["id"], ACYL)
    repair = {"action": "repair", "reason": "failed sites", "candidate_ids": [ACYL["id"]], "budget": 50}
    run(s, Scripted([dict(repair)]), lambda *a: None, {"campaign": {"max_steps": 2}})
    assert any("needs barrier kinetics" in e["text"] for e in s.all("notebook"))
    s2 = Store(tmp_path / "db2")
    s2.put("candidate", ACYL["id"], ACYL)
    calls = []
    run(s2, Scripted([dict(repair) for _ in range(3)]), lambda *a: None,
        {"campaign": {"max_steps": 4, "max_repairs": 2, "repair_budget": 5}},
        repairer=lambda ids, budget: calls.append((ids, budget)))
    assert calls == [([ACYL["id"]], 5.0), ([ACYL["id"]], 5.0)]
    assert any("Repair budget exhausted" in e["text"] for e in s2.all("notebook"))
    s.close(); s2.close()
