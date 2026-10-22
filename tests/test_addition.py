"""Carbonyl addition template, rotation input safety, and reference-structure exemptions."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pytest
from warden import kinetics as kin
from warden.analogs import enumerate_analogs
from warden.chemistry import rotation, validate_candidate, geometries
from warden.engine import graph
from warden.surfaces import load_manifest

ROOT = Path(__file__).resolve().parents[1]
PIVALDEHYDE = {"name": "pivaldehyde", "smiles": "CC(C)(C)[CH:1]=[O:2]", "template": "carbonyl_addition"}


@pytest.mark.parametrize("smiles", ["CC(C)(C)[CH:1]=[O:2]", "O=CCCC[CH:1]=[O:2]", "CC(=O)[CH:1]=[O:2]",
                                    "C[C:1](C)=[O:2]"])
def test_addition_accepts_aldehydes_and_ketones(smiles):
    assert validate_candidate({**PIVALDEHYDE, "smiles": smiles})["template"] == "carbonyl_addition"


@pytest.mark.parametrize("smiles", ["C[C:1](=[O:2])Cl", "C[C:1](=[O:2])OC", "C[C:1](=[O:2])N", "C[CH2:1][OH:2]",
                                    "O=[CH:1]c1ccccc1"])
def test_addition_rejects_acyl_and_unmapped_chemistry(smiles):
    with pytest.raises(ValueError):
        validate_candidate({**PIVALDEHYDE, "smiles": smiles})


def test_addition_fragment_is_balanced_tetrahedral_and_protonated():
    spec = validate_candidate(PIVALDEHYDE)
    g = geometries(spec, 1, 42)[0]
    fragment = g["fragment"]
    assert g["byproduct"] is None
    assert Counter(fragment.get_chemical_symbols()) == Counter(g["molecule"].get_chemical_symbols()) + Counter("H")
    assert graph(fragment) == {tuple(sorted(b)) for b in g["bonds"]}       # O-H formed, C-O kept
    neighbours = [i for i in range(1, len(fragment)) if fragment.get_distance(0, i) < 1.7]
    assert len(neighbours) == 3 and all(fragment.positions[i, 2] > 0.2 for i in neighbours)


@pytest.mark.parametrize("pathway", kin.PATHWAYS["carbonyl_addition"])
def test_addition_endpoints_transfer_the_proton_to_oxygen(pathway):
    spec = validate_candidate(PIVALDEHYDE)
    for site in kin.sites_for("gas"):
        reactant, product, m = kin.endpoints(site["substrate"], site["site"], site["normal"], spec,
                                             kin.DEFAULTS, 0, pathway)
        assert reactant.get_chemical_symbols() == product.get_chemical_symbols() and m["lost"] == []
        kin.check_reactant(reactant, m)
        kin.check_product(product, m)
        with pytest.raises(ValueError, match="product_not_formed"):
            kin.check_product(reactant, m)


def test_substitution_product_check_still_requires_bond_cleavage():
    spec = validate_candidate({"name": "acetyl chloride", "smiles": "C[C:1](=O)[Cl:2]", "template": "acylation"})
    site = kin.sites_for("gas")[0]
    reactant, product, m = kin.endpoints(site["substrate"], site["site"], site["normal"], spec, kin.DEFAULTS)
    with pytest.raises(ValueError, match="product_not_formed"):
        kin.check_product(reactant, m)


def test_addition_analogs_only_edit_sterics_and_electronics():
    analogs = enumerate_analogs(validate_candidate(PIVALDEHYDE), "all")
    assert analogs and all("leaving_group" not in a for a in analogs)


def test_rotation_does_not_modify_its_inputs():
    positions = np.array([[0., 0., 0.], [1.77, 0., 0.]])
    rotation(positions[1], [0., 0., -1.])
    assert positions[1, 0] == pytest.approx(1.77)


@pytest.mark.parametrize("pathway", kin.PATHWAYS["acylation"])
def test_reactant_keeps_embedded_leaving_bond(pathway):
    spec = validate_candidate({"name": "acetyl chloride", "smiles": "C[C:1](=O)[Cl:2]", "template": "acylation"})
    mol, anchor, leaving, _ = kin.molecule(spec, kin.DEFAULTS["seed"])
    site = kin.sites_for("gas")[0]
    reactant, _, m = kin.endpoints(site["substrate"], site["site"], site["normal"], spec, kin.DEFAULTS, 0, pathway)
    assert reactant.get_distance(m["anchor"], m["leaving"]) == pytest.approx(mol.get_distance(anchor, leaving))


def _manifest(tmp_path, exemption):
    data = json.loads((ROOT / "examples/surfaces-kim-reference.json").read_text())
    for entry in data["surfaces"]:
        entry["path"] = str((ROOT / "examples" / entry["path"]).resolve())
        if entry["material"] == "SiNx":
            if exemption is None:
                entry.pop("reference_structure")
            else:
                entry["reference_structure"] = exemption
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    return path


def test_reference_exemption_is_recorded_and_limited(tmp_path):
    report, _ = load_manifest(ROOT / "examples/surfaces-kim-reference.json")
    nitride = next(s for s in report["surfaces"] if s["material"] == "SiNx")
    assert nitride["valid"] and nitride["exemptions"]["density_gcc"]["passed_anyway"] is False
    assert nitride["exemptions"]["density_gcc"]["doi"] == "10.5281/zenodo.18081511"
    plain, _ = load_manifest(_manifest(tmp_path, None), require_valid=False)
    assert not plain["valid"] and plain["fingerprint"] != report["fingerprint"]
    base = {"citation": "x", "reason": "y"}
    with pytest.raises(ValueError, match="can be exempted"):
        load_manifest(_manifest(tmp_path, {**base, "exempt_checks": ["density_gcc", "severe atom overlap"]}))
    with pytest.raises(ValueError, match="citation and a reason"):
        load_manifest(_manifest(tmp_path, {"exempt_checks": ["density_gcc"], "citation": "", "reason": "y"}))
