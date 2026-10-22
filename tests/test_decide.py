"""Tests for the evidence-acquisition agent and its equal-budget benchmark."""
import numpy as np
import pytest
from warden import decide as dc
from warden.benchmark import benchmark, make_world, true_width
from warden.process import ProcessParams, dose_window, stress_test

SITES = [{"material": "SiO2", "label": "g0", "site_type": "OH", "population": 3},
         {"material": "SiO2", "label": "g1", "site_type": "OH", "population": 1},
         {"material": "SiNx", "label": "n0", "site_type": "NH2", "population": 2},
         {"material": "SiNx", "label": "n1", "site_type": "NH_bridge", "population": 2}]
CANDS = {"a": {"id": "a", "name": "a", "template": "acylation"},
         "b": {"id": "b", "name": "b", "template": "acylation"}}


def obs(c, site, ea, method="xtb", pathway="addition", status="valid"):
    return dc.Observation(c, site, pathway, method, status, ea if status == "valid" else None, ea - .5 if status == "valid" else None)


def test_window_width_matches_process_dose_window():
    rec = {"name": "x", "mass_amu": 80., "sites": [
        {"material": "SiO2", "label": "g", "population": 1, "status": "valid", "Ea_eV": 1.0},
        {"material": "SiNx", "label": "n", "population": 1, "status": "valid", "Ea_eV": 0.7}]}
    from warden.process import site_classes
    w = dose_window(site_classes(rec, "SiO2"), site_classes(rec, "SiNx"), ProcessParams(T_K=423, use_precomplex=False))
    cfg = {**dc.DEFAULTS, "temperatures": [423.]}
    got = dc.window_width(np.array([[1.0]]), np.array([[0.7]]), cfg, clip=False)[0]
    expected = min(np.log10(w["max_dose_s"]), np.log10(30)) - max(np.log10(w["min_dose_s"]), np.log10(.05))
    assert got == pytest.approx(expected, abs=1e-9)


def test_observation_shrinks_uncertainty_and_moves_mean():
    b = dc.Belief(SITES, CANDS, dc.DEFAULTS)
    m0, v0 = b.leaf_marginal("a", "SiNx", 0)
    b.add(obs("a", "n0", 0.3))
    m1, v1 = b.leaf_marginal("a", "SiNx", 0)
    assert v1 < v0 and m1 < m0


def test_reference_pairs_learn_and_correct_cheap_bias():
    b = dc.Belief(SITES, CANDS, dc.DEFAULTS)
    for site, ea in (("n0", 0.5), ("n1", 0.6)):
        b.add(obs("a", site, ea + 0.3, "xtb"))
        b.add(obs("a", site, ea, "dft"))
    mean, var = b.methods.bias("xtb", "SiNx", "acylation")
    assert mean == pytest.approx(0.3, abs=0.08)
    assert var < dc.DEFAULTS["methods"]["xtb"]["bias_sd_eV"] ** 2
    # A later xtb-only result on another candidate is corrected by the learned bias.
    b.add(obs("b", "n0", 0.8, "xtb"))
    assert b.leaf_marginal("b", "SiNx", 0)[0] < 0.8 - 0.1


def test_failures_lower_success_probability_for_that_key():
    b = dc.Belief(SITES, CANDS, dc.DEFAULTS)
    key = ("a", "n0", "addition", "xtb")
    p0 = b.failures.p("xtb", "acylation", "addition", "NH2", key)
    b.add(obs("a", "n0", 0, status="failed"))
    assert b.failures.p("xtb", "acylation", "addition", "NH2", key) < p0
    other = ("a", "n0", "four_centre", "xtb")
    # The class itself is untouched; only the pooled method rate moves.
    assert b.failures.p("xtb", "acylation", "four_centre", "NH2", other) > b.failures.p("xtb", "acylation", "addition", "NH2", key)


def test_action_space_covers_molecule_site_pathway_method():
    b = dc.Belief(SITES, CANDS, dc.DEFAULTS)
    agent = dc.Agent(b, ["xtb", "dft"], [{"id": "c", "name": "c", "template": "acylation"}])
    acts = agent.actions()
    assert {a.kind for a in acts} == {"observe", "add_molecule"}
    assert {a.pathway for a in acts if a.kind == "observe"} == {"addition", "four_centre"}
    assert {a.method for a in acts if a.kind == "observe"} == {"xtb", "dft"}
    assert {a.site for a in acts if a.kind == "observe"} == {s["label"] for s in SITES}


def test_escalation_preferred_only_when_cheap_bias_blocks_the_decision():
    cfg = {**dc.DEFAULTS, "methods": {**dc.DEFAULTS["methods"], "dft": {**dc.DEFAULTS["methods"]["dft"], "cost": 3.0}}}
    close = dc.Belief(SITES, CANDS, cfg)
    for c, shift in (("a", 0.0), ("b", 0.02)):
        for s, ea in (("g0", 1.1), ("g1", 1.15), ("n0", 0.75), ("n1", 0.8)):
            for p in ("addition", "four_centre"):
                for _ in range(3):
                    close.add(obs(c, s, ea + shift, "xtb", p))
    acts = dc.Agent(close, ["xtb", "dft"], (), cfg).actions()
    best_dft = max(a.score() for a in acts if a.method == "dft")
    best_xtb = max(a.score() for a in acts if a.method == "xtb")
    assert best_dft > best_xtb


def test_loop_stops_when_confident():
    cands = {"a": CANDS["a"]}
    b = dc.Belief(SITES, cands, dc.DEFAULTS)
    b.add(obs("a", "n0", 0.5))

    class Never:
        def execute(self, a, belief):
            raise AssertionError("should not run")
    r = dc.run(b, Never(), 10., cfg=dc.DEFAULTS)
    assert r["trace"][-1]["stop"] == "confident"


def test_synthetic_world_truth_is_hidden_and_consistent():
    w = make_world(3)
    assert len(w["candidates"]) == 6 and w["truth"]
    widths = [true_width(w, c, dc.DEFAULTS) for c in w["candidates"]]
    assert all(np.isfinite(widths))


def test_benchmark_runs_equal_budgets_and_beats_random():
    r = benchmark(n_worlds=4, budget=20., seed=11, policies=["warden_voi", "random"])
    s = {x["policy"]: x for x in r["summary"]}
    assert all(x["spent"] <= 20. for x in r["runs"])
    assert s["warden_voi"]["mean_regret_decades"] <= s["random"]["mean_regret_decades"]


def test_stress_test_reports_repassivation_sensitivity():
    rec = {"name": "x", "mass_amu": 80., "sites": [
        {"material": "SiO2", "label": "g", "population": 1, "status": "valid", "Ea_eV": 1.3},
        {"material": "SiNx", "label": "n", "population": 1, "status": "valid", "Ea_eV": 0.6}]}
    r = stress_test(rec, ProcessParams(T_K=423, dose_s=1., use_precomplex=False), 1)
    sweep = r["incomplete_repassivation_theta_max"]["sweep"]
    s = [x["S_at_target"] or 0 for x in sweep]
    assert s == sorted(s)
    assert set(r) == {"cap_loss_per_cycle", "incomplete_repassivation_theta_max",
                      "slower_repassivation_NGS_shift_eV", "oxide_inhibition_GS_shift_eV",
                      "oxide_cap_persistence_k_deg_GS"}
