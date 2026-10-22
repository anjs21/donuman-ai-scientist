"""Regression tests for acylation, barriers, process design and control gating."""
from collections import Counter
from dataclasses import replace
from pathlib import Path
import importlib.util
import numpy as np
import pytest
from ase.io import read
from warden.chemistry import validate_candidate
from warden import kinetics as kin
from warden.process import ProcessParams, simulate, at_target, design, dose_window, site_classes, calibrate
from warden.assess import assess, control_gate, consensus

ROOT = Path(__file__).resolve().parents[1]
ACYL = {"name": "acetyl chloride", "smiles": "C[C:1](=O)[Cl:2]", "template": "acylation"}


@pytest.mark.parametrize("smiles", ["C[C:1](=O)[Cl:2]", "C[C:1](=O)[O:2]C(C)=O", "C[C:1](=O)[n:2]1ccnc1"])
def test_acylation_accepts_carbonyl_electrophiles(smiles):
    assert validate_candidate({"name": "x", "smiles": smiles, "template": "acylation"})["id"]


@pytest.mark.parametrize("smiles", ["C[CH:1](C)[Cl:2]", "C[Si:1](C)(C)[Cl:2]", "C[C:1](=O)[CH2:2]C"])
def test_acylation_rejects_non_acyl_chemistry(smiles):
    with pytest.raises(ValueError):
        validate_candidate({"name": "x", "smiles": smiles, "template": "acylation"})


def _record(ea_gs, ea_ngs, status="valid", name="x", method="xtb", binding=-0.3):
    sites = [{"material": "SiO2", "label": "gs", "population": 1, "status": "valid",
              "Ea_eV": ea_gs, "dE_rxn_eV": -0.5, "precomplex_binding_eV": binding},
             {"material": "SiNx", "label": "ngs", "population": 1, "status": status,
              "Ea_eV": ea_ngs if status == "valid" else None, "dE_rxn_eV": -1.0, "precomplex_binding_eV": binding}]
    return {"candidate_id": "c_" + name, "name": name, "smiles": "C[C:1](=O)[Cl:2]", "template": "acylation",
            "method": method, "mass_amu": 78.5, "sites": sites, "summary": kin.summarize(sites)}


def test_gas_sites_and_endpoints_are_atom_matched():
    spec = validate_candidate(ACYL)
    for s in kin.sites_for("gas"):
        r, p, m = kin.endpoints(s["substrate"], s["site"], s["normal"], spec, kin.DEFAULTS)
        assert r.get_chemical_symbols() == p.get_chemical_symbols()
        b_r, b_p = kin.bonded(r), kin.bonded(p)
        assert not b_r[m["site"], m["anchor"]] and b_r[m["anchor"], m["leaving"]]
        assert b_p[m["site"], m["anchor"]] and b_p[m["proton"], m["leaving"]]


def test_cluster_carved_from_amorphous_slab():
    slab = read(ROOT / "zenodo_structures/POSCAR_SiO2_1", format="vasp")
    slab.set_pbc([True, True, False])
    site = next(i for i, n in enumerate(slab.numbers) if n == 8 and
                any(slab.numbers[j] == 1 for j in np.flatnonzero(kin.bonded(slab)[i])))
    cluster, s, frozen, normal = kin.carve_cluster(slab, site, 3)
    assert s == 0 and cluster.numbers[0] == 8 and frozen
    assert np.isclose(np.linalg.norm(normal), 1)
    assert not kin._clash(cluster)
    # Every Si keeps four neighbours after capping.
    b = kin.bonded(cluster, kin.STRUCTURE_FACTOR)
    assert all(b[i].sum() == 4 for i, n in enumerate(cluster.numbers) if n == 14)


def test_sackur_tetrode_matches_nitrogen_reference():
    # N2 translational entropy at 298.15 K, 1 bar is about 150.4 J/(mol K).
    s = kin.sackur_tetrode(28.0134, 298.15, 1e5) * 96485.332
    assert s == pytest.approx(150.4, abs=0.5)


def test_rates_monotonic_in_barrier_and_temperature():
    assert kin.rate_constant(0.5, 473) > kin.rate_constant(0.8, 473)
    assert kin.rate_constant(0.8, 523) > kin.rate_constant(0.8, 423)
    assert 0 < kin.precomplex_fraction(-0.3, 78.5, 473, 133) < 1


def test_selective_inhibitor_reaches_target_and_unselective_fails():
    p = ProcessParams(theta_max=0.999, k_deg=0.0, k_deg_GS=0.0, shielding=3.0, dose_s=1.0, T_K=423)
    good = simulate(site_classes(_record(2.0, 0.6), "SiO2"), site_classes(_record(2.0, 0.6), "SiNx"), p, 1)
    S, cycles, _ = at_target(good["GS"], good["NGS"], p)
    assert S[0] > 0.95 and cycles[0] == pytest.approx(100 / 1.2, rel=0.02)
    bad = simulate(site_classes(_record(0.6, 0.6), "SiO2"), site_classes(_record(0.6, 0.6), "SiNx"), p, 1)
    S, _, _ = at_target(bad["GS"], bad["NGS"], p)
    assert not np.isfinite(S[0]) or S[0] < 0.5


def test_degradation_needs_redosing():
    rec = _record(2.0, 0.6)
    gs, ngs = site_classes(rec, "SiO2"), site_classes(rec, "SiNx")
    p = ProcessParams(theta_max=0.999, k_deg=0.1, shielding=1.0, dose_s=1.0, T_K=423)
    s_once = at_target(*simulate(gs, ngs, p, 1000).values(), p)[0][0]
    s_often = at_target(*simulate(gs, ngs, p, 1).values(), p)[0][0]
    assert s_often > s_once


def test_missing_barrier_is_refused_not_guessed():
    with pytest.raises(ValueError):
        design(_record(1.5, None, status="failed"), n_draws=10)


def test_dose_window_matches_rate_theory():
    rec = _record(1.4, 0.9)
    p = ProcessParams(T_K=473, use_precomplex=False)
    w = dose_window(site_classes(rec, "SiO2"), site_classes(rec, "SiNx"), p)
    assert w["exists"]
    assert w["min_dose_s"] == pytest.approx(np.log(100) / kin.rate_constant(0.9, 473), rel=1e-6)
    assert w["max_dose_s"] == pytest.approx(-np.log(0.99) / kin.rate_constant(1.4, 473), rel=1e-6)


def test_design_is_reproducible_and_ranks_inputs():
    a = design(_record(1.4, 0.8), n_draws=40, temperatures=(473.,), doses=(1.,), intervals=(1, 3))
    b = design(_record(1.4, 0.8), n_draws=40, temperatures=(473.,), doses=(1.,), intervals=(1, 3))
    assert a["recommended"] == b["recommended"]
    assert {m["input"] for m in a["measure_first"]} >= {"p_nuc", "shielding", "k_deg"}


def test_calibration_recovers_synthetic_parameters():
    rec = _record(1.6, 0.8)
    truth = ProcessParams(T_K=423, dose_s=1., p_nuc=0.4, shielding=2.0, k_deg=0.05, theta_max=0.99, max_cycles=60)
    curve = simulate(site_classes(rec, "SiO2"), site_classes(rec, "SiNx"), truth, 2)
    observed = [{"cycle": c, "thk_GS_A": float(curve["GS"][0, c]), "thk_NGS_A": float(curve["NGS"][0, c])}
                for c in range(5, 61, 5)]
    fit = calibrate(rec, observed, replace(truth, p_nuc=1., shielding=1., k_deg=.03), 2,
                    fit=("p_nuc", "k_deg"))
    assert fit["rmse_A"] < 0.05
    assert fit["params"]["p_nuc"] == pytest.approx(0.4, rel=0.1)


def test_control_gate_fails_on_wrong_direction():
    controls = [{"name": "acyl", "expected": "prefers_SiNx", "source": "test"}]
    assert control_gate({"acyl": _record(1.5, 0.8, name="acyl")}, controls)["status"] == "pass"
    assert control_gate({"acyl": _record(0.8, 1.5, name="acyl")}, controls)["status"] == "fail"
    assert control_gate({}, controls)["status"] == "incomplete"


def test_method_disagreement_becomes_uncertainty():
    merged = consensus([_record(1.5, 0.8, method="xtb"), _record(1.2, 0.9, method="mace")])
    assert merged["method_disagreement_eV"] == pytest.approx(0.4)
    assert merged["direction_agreement"] is True


def test_assessment_excludes_controls_and_unadmitted_and_reports_baselines():
    options = {"n_draws": 30, "temperatures": (423.,), "doses": (1.,), "intervals": (1, 2)}
    good, weak = _record(1.6, 0.8, name="good"), _record(0.9, 0.85, name="weak")
    control = _record(1.5, 0.8, name="ctrl")
    hidden = {**_record(2.0, 0.5, name="hidden"), "surface_admission": False}
    r = assess([good, weak, control, hidden], [{"name": "ctrl", "expected": "prefers_SiNx", "source": "t"}],
               design_options=options)
    names = [x["name"] for x in r["ranking"]]
    assert names[0] == "good" and "ctrl" not in names and "hidden" not in names
    assert r["label"] == "control_calibrated" and r["scientific_success"] is False
    assert "barrier_contrast_only" in r["baselines"]


@pytest.mark.skipif(importlib.util.find_spec("tblite") is None, reason="tblite (xTB) not installed")
def test_xtb_acylation_barrier_on_amine_model():
    spec = validate_candidate(ACYL)
    s = next(x for x in kin.sites_for("gas") if x["site_type"] == "NH2")
    r = kin.barrier(s["substrate"], s["site"], s["normal"], s["frozen"], spec,
                    kin.calculator_factory({"method": "xtb"}), {})
    assert r["status"] == "valid"
    assert 0.0 < r["Ea_eV"] < 2.0 and r["dE_rxn_eV"] < 0


def test_barrier_measured_from_lowest_preceding_image():
    r = kin.classify([0., -.15, -.05, .40, -.6, -1.], True, False, {})
    assert r["status"] == "valid" and r["Ea_eV"] == pytest.approx(.55)
    assert r["precomplex_image"] == 1 and r["dE_rxn_eV"] == pytest.approx(-.85)


@pytest.mark.parametrize("profile,status", [
    ([0., -.02, -.10, -.15, -.40, -.5], "no_barrier_resolved"),
    ([0., .1, 4.0, 4.1, .2, -.5], "band_unresolved")])
def test_unresolved_paths_are_not_barriers(profile, status):
    assert kin.classify(profile, True, False, {})["status"] == status


def test_reclassify_updates_stored_record_without_recomputing():
    attempt = {"status": "valid", "Ea_eV": -.02, "profile_eV": [0., -.02, -.10, -.15, -.40, -.5],
               "precomplex_shift_eV": 0.}
    record = {"name": "x", "sites": [{"material": "SiO2", "label": "gs", "population": 1, "status": "valid",
                                      "Ea_eV": -.02, "precomplex_binding_eV": -.1, "attempts": [attempt]}]}
    out = kin.reclassify(record)
    assert out["sites"][0]["status"] == "failed" and out["summary"]["SiO2"] is None
