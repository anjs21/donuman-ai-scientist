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


class Never:
    def execute(self, a, belief):
        raise AssertionError("should not run")


def separated_belief(gs=(1.5, 0.3), ngs=(0.5, 1.2)):
    """Two candidates, every leaf observed: 'a' is clearly better than 'b'.
    gs/ngs give (a, b) barriers on the oxide and nitride sites."""
    b = dc.Belief(SITES, CANDS, dc.DEFAULTS)
    for pathway in ("addition", "four_centre"):
        for _ in range(3):
            for site, good, bad in (("g0", *gs), ("g1", *gs), ("n0", *ngs), ("n1", *ngs)):
                b.add(obs("a", site, good, pathway=pathway))
                b.add(obs("b", site, bad, pathway=pathway))
    return b


def test_loop_stops_when_confident():
    b = separated_belief()
    assert b.pcs() >= dc.DEFAULTS["target_pcs"]
    r = dc.run(b, Never(), 10., cfg=dc.DEFAULTS)
    assert r["trace"][-1]["stop"] == "confident" and r["spent"] == 0


def test_single_candidate_is_never_confident():
    """P(correct selection) is trivially 1 with one candidate; that is not a reason to stop."""
    b = dc.Belief(SITES, {"a": CANDS["a"]}, dc.DEFAULTS)
    b.add(obs("a", "n0", 0.5))
    assert b.pcs() == 1.0
    assert not dc.confident_stop(b, dc.Agent(b, ["xtb"]), b.pcs())


def test_confidence_among_infeasible_candidates_does_not_block_new_molecules():
    library = [{"id": "c", "name": "c", "template": "acylation"}]
    # Both candidates react fast with the oxide: confidently ranked, both infeasible.
    b = separated_belief(gs=(0.8, 0.2), ngs=(0.5, 1.2))
    assert b.pcs() >= dc.DEFAULTS["target_pcs"] and max(b.scores().values()) < 0
    assert not dc.confident_stop(b, dc.Agent(b, ["xtb"], library), b.pcs())
    assert dc.confident_stop(b, dc.Agent(b, ["xtb"]), b.pcs())     # nothing left to try: stop
    # A feasible, confident pick stops even with molecules left in the library.
    good = separated_belief()
    assert max(good.scores().values()) > 0
    assert dc.confident_stop(good, dc.Agent(good, ["xtb"], library), good.pcs())


def test_orientations_take_the_fastest_path_and_repeats_average():
    """Orientations are different paths to one site: the leaf is the fastest of
    them (as kinetics.select_best). Only repeats of one orientation are averaged."""
    def leaf(orientations):
        b = dc.Belief(SITES, CANDS, dc.DEFAULTS)
        for ea, o in zip((0.2, 1.0), orientations):
            b.add(dc.Observation("a", "g0", "addition", "xtb", "valid", ea, -0.5, orientation=o))
        return b.leaf_marginal("a", "SiO2", 0)[0]
    one = dc.Belief(SITES, CANDS, dc.DEFAULTS)
    one.add(dc.Observation("a", "g0", "addition", "xtb", "valid", 0.2, -0.5, orientation=0))
    assert leaf((0, 1)) == pytest.approx(one.leaf_marginal("a", "SiO2", 0)[0], abs=1e-9)
    assert leaf((0, 0)) > leaf((0, 1)) + 0.2              # same orientation: averaged


def test_hypothetical_new_orientation_can_only_lower_a_leaf():
    b = dc.Belief(SITES, CANDS, dc.DEFAULTS)
    b.add(dc.Observation("a", "g0", "addition", "xtb", "valid", 0.5, -0.5, orientation=0))
    ys = np.array([0.2, 0.5, 2.0])
    ybar, _, _, _ = b._data("a", b.leaves("a", "SiO2"), "SiO2", (0, ys, "xtb"))
    bias = b.methods.bias("xtb", "SiO2", "acylation")[0]
    assert ybar[:, 0] + bias == pytest.approx([0.2, 0.5, 0.5])


def test_kinetics_seeds_keep_their_orientation():
    record = {"candidate_id": "a", "template": "acylation", "method": "xtb", "sites": [
        {"label": "g0", "material": "SiO2", "attempts": [
            {"orientation": 0, "status": "valid", "Ea_eV": 0.3, "dE_rxn_eV": -0.4},
            {"orientation": 1, "status": "valid", "Ea_eV": 0.9, "dE_rxn_eV": -0.4}]}]}
    seeds = dc.observations_from_kinetics([record])
    assert [o.orientation for o in seeds] == [0, 1]


def test_latest_kinetics_record_supersedes_reruns():
    def rec(rid, method="xtb", fingerprint="gas_models"):
        return {"id": rid, "candidate_id": "a", "method": method, "surface_fingerprint": fingerprint}
    kept = dc.latest_kinetics([rec("k2"), rec("k1"), rec("k3", "mace"), rec("k4", fingerprint="slab")],
                              finished=["k1", "k2", "k3", "k4"])
    assert sorted(r["id"] for r in kept) == ["k2", "k3", "k4"]
    # Without completion events the last record supplied wins.
    assert [r["id"] for r in dc.latest_kinetics([rec("k2"), rec("k1")])] == ["k1"]


NEW_ACYL = {"id": "n_acyl", "name": "new acyl", "template": "acylation"}
NEW_SILYL = {"id": "n_silyl", "name": "new silyl", "template": "silyl_substitution"}


def add_values(b, library=(NEW_ACYL, NEW_SILYL)):
    acts = dc.Agent(b, ["xtb"], list(library)).actions()
    return {a.extra["id"]: a.value for a in acts if a.kind == "add_molecule"}


def test_every_library_molecule_is_valued_and_equal_before_any_results():
    v = add_values(dc.Belief(SITES, {}, dc.DEFAULTS))
    assert set(v) == {"n_acyl", "n_silyl"} and v["n_acyl"] == pytest.approx(v["n_silyl"])


def test_class_results_inform_the_next_molecule():
    """Every acylating agent so far caps silanols with no barrier: the next
    acylating agent is expected to do the same, so a silylating agent is preferred."""
    bad = separated_belief(gs=(0.05, 0.0), ngs=(0.5, 0.6))
    bad.add_candidate(NEW_ACYL); bad.add_candidate(NEW_SILYL)
    assert bad.prior("n_acyl", "SiO2")[0] < dc.DEFAULTS["prior_mean_eV"]["SiO2"] - 0.3
    assert bad.prior("n_silyl", "SiO2")[0] == pytest.approx(dc.DEFAULTS["prior_mean_eV"]["SiO2"])
    del bad.candidates["n_acyl"], bad.candidates["n_silyl"]
    v = add_values(bad)
    assert v["n_silyl"] > v["n_acyl"]
    # Promising acylation results favour another acylating agent instead.
    good = separated_belief(gs=(1.6, 1.5), ngs=(0.5, 0.55))
    v = add_values(good)
    assert v["n_acyl"] > v["n_silyl"]


def test_a_candidate_does_not_inform_its_own_prior():
    b = dc.Belief(SITES, {"a": CANDS["a"]}, dc.DEFAULTS)
    b.add(obs("a", "g0", 0.0))
    assert b.prior("a", "SiO2")[0] == pytest.approx(dc.DEFAULTS["prior_mean_eV"]["SiO2"])


def test_literature_prior_replaces_the_class_mean():
    b = separated_belief(gs=(0.05, 0.0), ngs=(0.5, 0.6))
    b.add_candidate({**NEW_ACYL, "prior_Ea_eV": {"SiO2": 1.4}, "prior_sd_eV": 0.2})
    assert b.prior("n_acyl", "SiO2") == pytest.approx((1.4, 0.2))
    assert b.prior("n_acyl", "SiNx")[0] < dc.DEFAULTS["prior_mean_eV"]["SiNx"]   # class mean still used


def test_run_adds_the_best_library_molecule_not_the_first():
    b = separated_belief(gs=(0.05, 0.0), ngs=(0.5, 0.6))
    r = dc.run(b, Never(), dc.DEFAULTS["new_molecule_cost"], [NEW_ACYL, NEW_SILYL], methods=(), cfg=dc.DEFAULTS)
    added = [t["added"] for t in r["trace"] if "added" in t]
    assert added == ["new silyl"] and "n_silyl" in b.candidates and "n_acyl" not in b.candidates


def test_benchmark_world_orientations_include_the_true_path():
    w = make_world(5)
    assert all(e.min() == 0. and len(e) == 4 for e in w["excess"].values())


def test_orientation_choice_skips_tried_ones_and_stops_when_exhausted():
    b = dc.Belief(SITES, CANDS, dc.DEFAULTS)
    key = ("a", "g0", "addition", "xtb")
    for o in (0, 1):
        b.add(dc.Observation(*key, "valid", 0.5, -0.5, orientation=o))
    b.add(dc.Observation(*key, "failed", orientation=3))
    assert b.next_orientation(key, 4) == 2
    b.add(dc.Observation(*key, "valid", 0.5, -0.5, orientation=6))   # 6 = 2 (mod 4): same path
    assert b.next_orientation(key, 4) is None
    acts = dc.Agent(b, ["xtb", "dft"], (), dc.DEFAULTS).actions()
    offered = {(a.method, a.orientation) for a in acts if (a.candidate, a.site, a.pathway) == key[:3]}
    assert offered == {("dft", 0)}                     # no xtb repeat of a computed path
    # Results without a recorded orientation occupy the lowest free orientations.
    legacy = dc.Belief(SITES, CANDS, dc.DEFAULTS)
    legacy.add(dc.Observation(*key, "valid", 0.5, -0.5))
    legacy.add(dc.Observation(*key, "valid", 0.5, -0.5, orientation=0))
    assert legacy.next_orientation(key, 4) == 2


def test_method_error_pairs_only_the_same_orientation():
    b = dc.Belief(SITES, CANDS, dc.DEFAULTS)
    b.add(dc.Observation("a", "n0", "addition", "xtb", "valid", 0.9, 0.1, orientation=0))
    b.add(dc.Observation("a", "n0", "addition", "dft", "valid", 0.4, 0.0, orientation=1))
    assert b.methods.report() == []                   # different paths: not a method error
    b.add(dc.Observation("a", "n0", "addition", "dft", "valid", 0.6, 0.0, orientation=0))
    (row,) = [r for r in b.methods.report() if r["quantity"] == "barrier"]
    assert row["n_pairs"] == 1 and row["mean_difference_eV"] == pytest.approx(0.3)


def test_added_molecule_is_identified_when_logged():
    """decide_command saves an added molecule from the log callback, before the run ends."""
    b = separated_belief(gs=(0.05, 0.0), ngs=(0.5, 0.6))
    seen = []
    log = lambda record: seen.append((record.get("added_id"), record.get("added_id") in b.candidates))
    dc.run(b, Never(), dc.DEFAULTS["new_molecule_cost"], [NEW_SILYL], methods=(), cfg=dc.DEFAULTS, log=log)
    assert seen == [("n_silyl", True)]


class Fixed:
    """Executor that returns one barrier for every calculation."""
    def __init__(self, ea=0.5):
        self.ea, self.calls = ea, []

    def execute(self, a, belief):
        self.calls.append(a)
        return dc.Observation(a.candidate, a.site, a.pathway, a.method, "valid", self.ea, -0.5,
                              cost=dc.DEFAULTS["methods"][a.method]["cost"], orientation=a.orientation)


def test_adding_a_molecule_runs_its_first_calculation():
    b = separated_belief(gs=(0.05, 0.0), ngs=(0.5, 0.6))
    ex = Fixed()
    budget = dc.DEFAULTS["new_molecule_cost"] + dc.DEFAULTS["methods"]["xtb"]["cost"]
    r = dc.run(b, ex, budget, [NEW_SILYL], methods=("xtb",), cfg=dc.DEFAULTS)
    step = next(t for t in r["trace"] if "added" in t)
    assert step["action"]["cost"] == pytest.approx(budget)          # costed as molecule + calculation
    assert step["result"]["candidate"] == "n_silyl" and ex.calls[0].candidate == "n_silyl"
    assert r["spent"] == pytest.approx(budget)
    # Every add action is costed as the molecule plus its first calculation.
    small = dc.Agent(b, ["xtb"], [NEW_SILYL]).actions()
    assert all(a.cost == pytest.approx(budget) for a in small if a.kind == "add_molecule")


def test_more_tried_orientations_make_another_one_worth_less():
    def kg(n):
        b = dc.Belief(SITES, CANDS, dc.DEFAULTS)
        for c in CANDS:
            for site, ea in (("g0", 0.9), ("g1", 0.95), ("n1", 0.6)):
                b.add(dc.Observation(c, site, "addition", "xtb", "valid", ea, -0.5, orientation=0))
        for o in range(n):
            b.add(dc.Observation("a", "n0", "addition", "xtb", "valid", 0.7, -0.5, orientation=o))
        acts = dc.Agent(b, ["xtb"], (), dc.DEFAULTS).actions()
        a = next(a for a in acts if (a.candidate, a.site, a.pathway) == ("a", "n0", "addition"))
        return a.extra["knowledge_gradient"] + a.extra["relevance"]
    assert kg(1) > kg(3) > 0


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


def test_benchmark_world_classes_and_library():
    w = make_world(2, world={"library_size": 4})
    assert len(w["initial"]) == 6 and [s["id"] for s in w["library"]] == ["L0", "L1", "L2", "L3"]
    assert set(w["candidates"]) == set(w["initial"]) | {"L0", "L1", "L2", "L3"}
    r = benchmark(n_worlds=2, budget=12., seed=3, policies=["warden_voi", "warden_voi_no_sharing"],
                  world={"library_size": 3})
    assert {x["policy"] for x in r["runs"]} == {"warden_voi", "warden_voi_no_sharing"}
    assert all(x["spent"] <= 12. for x in r["runs"])
