"""Serial, geometry-checked MACE screening with explicit failure records.

No Lennard-Jones fallback, no inaccessible-site fallback, no acceptance of
unconverged optimizations. This is a screening surrogate, not kinetic proof.
"""
from pathlib import Path
import importlib.metadata
import math
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.data import covalent_radii
from ase.geometry import find_mic
from ase.io import write
from ase.optimize import BFGS
from .chemistry import ATTACHING, SUBSTITUTION, geometries, rotation
from .common import digest, file_hash, clean
from .metrics import evaluate
from .surfaces import representatives

DEFAULTS = {"max_sites": 2, "n_conformers": 3, "orientations": 4,
            "seed": 42, "active_radius_A": 5.0, "fmax": 0.05,
            "steps": 250, "min_site_success_fraction": 0.8,
            "energy_review_limit_eV": 15.0, "device": "cpu", "dtype": "float64"}


# Anchor--surface heteroatom bond lengths used for initial placement (Angstrom).
BOND_LENGTH = {("silyl_substitution", 8): 1.65, ("silyl_substitution", 7): 1.75,
               ("acylation", 8): 1.43, ("acylation", 7): 1.47,
               ("carbonyl_addition", 8): 1.43, ("carbonyl_addition", 7): 1.47}


def calculator(config):
    model = Path(config["model_path"]).expanduser().resolve()
    if not model.is_file():
        raise ValueError("model_path must point to a local MACE checkpoint; see RUN_ORDER.md")
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=str(model), device=config.get("device", "cpu"),
                          default_dtype=config.get("dtype", "float64"))


def calculation_key(candidate, surface_report, config):
    model = Path(config["model_path"]).expanduser().resolve()
    if not model.is_file():
        raise ValueError("Missing MACE model checkpoint")
    from . import chemistry, metrics, surfaces, slab_validation
    import selectivity_model
    import surface_builder, structure_validation
    signature = {"surface_admission": surface_report, "candidate": {k: candidate[k] for k in ("smiles", "template", "role")},
                 "surface_manifest": surface_report["fingerprint"],
                 "config": config, "model_sha256": file_hash(model),
                 "code": [file_hash(m.__file__) for m in (chemistry, metrics, surfaces, slab_validation, selectivity_model, surface_builder, structure_validation)],
                 "engine": file_hash(__file__),
                 "versions": {p: importlib.metadata.version(p) for p in ("ase", "rdkit", "numpy", "mace-torch")}}
    return digest(signature), signature


def graph(atoms):
    d = atoms.get_all_distances(mic=bool(atoms.pbc.any()))
    radii = covalent_radii[atoms.numbers]
    return {(i, j) for i in range(len(atoms)) for j in range(i + 1, len(atoms))
            if d[i, j] < 1.3 * (radii[i] + radii[j])}


def overlaps(atoms):
    d = atoms.get_all_distances(mic=bool(atoms.pbc.any()))
    np.fill_diagonal(d, np.inf)
    r = covalent_radii[atoms.numbers]
    return bool((d < .65 * (r[:, None] + r[None, :])).any())


def relax(atoms, calc, config, expected_bonds=None):
    atoms = atoms.copy()
    if overlaps(atoms):
        raise ValueError("initial_geometry_clash")
    atoms.calc = calc
    converged = BFGS(atoms, logfile=None).run(fmax=config["fmax"], steps=config["steps"])
    forces = atoms.get_forces()
    force = float(np.linalg.norm(forces, axis=1).max())
    energy = float(atoms.get_potential_energy())
    if not converged or not math.isfinite(energy) or not np.isfinite(forces).all():
        raise ValueError("optimization_not_converged_or_nonfinite")
    if force > config["fmax"] * 1.001 or overlaps(atoms):
        raise ValueError("final_force_or_geometry_failed")
    if expected_bonds is not None and graph(atoms) != {tuple(sorted(x)) for x in expected_bonds}:
        raise ValueError("molecular_connectivity_changed")
    return atoms, energy, force


def freeze_region(atoms, center, radius, substrate_count):
    _, ds = find_mic(atoms.positions[:substrate_count] - center, atoms.cell, atoms.pbc)
    atoms.set_constraint(FixAtoms(indices=np.flatnonzero(ds > radius)))


def place(slab, site, geo, template, orientation, total):
    atoms = slab.copy()
    site_pos = slab.positions[site].copy()
    site_element = slab.numbers[site]
    zc = (slab.positions[:, 2].max() + slab.positions[:, 2].min()) / 2
    normal = np.array([0., 0., 1. if site_pos[2] >= zc else -1.])
    if template in ATTACHING:
        h = [i for i, n in enumerate(slab.numbers) if n == 1 and
             slab.get_distance(site, i, mic=True) < 1.25 * (covalent_radii[n] + covalent_radii[slab.numbers[site]])]
        if not h:
            raise ValueError("no_surface_proton")
        removed = min(h, key=lambda i: slab.get_distance(site, i, mic=True))
        del atoms[removed]
        site = site - int(removed < site)
        height = BOND_LENGTH[template, site_element]
    else:
        height = 2.8 + .4 * (orientation % 2)
    n_substrate = len(atoms)
    fragment = geo["fragment"].copy()
    theta = 2 * np.pi * orientation / total
    spin = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    # Modest tilts plus azimuthal sampling; the anchor remains fixed initially.
    tilt = (.25 if orientation % 2 else -.25)
    tilted = np.array([np.sin(tilt), 0, np.cos(tilt)])
    pos = fragment.positions @ spin.T @ rotation([0, 0, 1], tilted).T
    fragment.positions = pos @ rotation([0, 0, 1], normal).T + site_pos + height * normal
    atoms += fragment
    # Reject periodic self-interaction of a molecule too large for the surface cell.
    euclid = fragment.get_all_distances(mic=False)
    periodic = fragment.copy(); periodic.set_cell(atoms.cell); periodic.set_pbc(atoms.pbc)
    if np.max(np.abs(euclid - periodic.get_all_distances(mic=True))) > 0.1:
        raise ValueError("molecule_too_large_for_surface_cell")
    return atoms, site, n_substrate


def check_product(atoms, n_substrate, site, geo, template):
    adsorbate = atoms[n_substrate:]
    if graph(adsorbate) != {tuple(sorted(b)) for b in geo["bonds"]}:
        raise ValueError("adsorbate_connectivity_changed")
    full_graph = graph(atoms)
    cross = {(i, j) for i, j in full_graph if i < n_substrate <= j}
    expected = {(site, n_substrate)} if template in ATTACHING else set()
    if cross != expected:
        raise ValueError("attachment_or_reaction_identity_changed")
    if template == "physisorption":
        nearest = atoms.get_all_distances(mic=True)[:n_substrate, n_substrate:].min()
        if nearest > 5.:
            raise ValueError("adsorbate_desorbed")


def screen(candidate, surface_report, atoms_by_id, config, calc, artifact_dir):
    config = {**DEFAULTS, **config}
    if type(calc).__name__ == "LennardJones":
        raise ValueError("Nonphysical calculator is prohibited")
    output = Path(artifact_dir); output.mkdir(parents=True, exist_ok=True)
    geos = geometries(candidate, config["n_conformers"], config["seed"])
    gas_energies, by_energies, gas_failures = [], [], []
    for g in geos:
        try:
            molecule = g["molecule"].copy(); molecule.center(vacuum=8)
            _, energy, _ = relax(molecule, calc, config, g["molecule_bonds"])
            gas_energies.append(energy)
            if g["byproduct"] is not None:
                by = g["byproduct"].copy(); by.center(vacuum=8)
                _, by_energy, _ = relax(by, calc, config, graph(by))
                by_energies.append(by_energy)
        except Exception as exc:
            gas_failures.append(str(exc))
    if not gas_energies or (candidate["template"] in SUBSTITUTION and not by_energies):
        return {"candidate_id": candidate["id"], "mode": "mace", "status": "failed",
                "reason": "gas_reference_failed", "failures": gas_failures,
                "dE_GS": None, "dE_NGS": None}
    e_molecule = min(gas_energies)
    e_byproduct = min(by_energies) if by_energies else 0.
    sites = []
    for surface in surface_report["surfaces"]:
        slab = atoms_by_id[surface["id"]]
        for representative in representatives(slab, surface["material"], config["max_sites"]):
            record = {**representative, "surface_id": surface["id"], "material": surface["material"],
                      "attempts": [], "status": "failed", "energy_eV": None}
            site = representative["site"]
            print(f"  {candidate['name']}: {surface['id']} site={site} "
                  f"type={representative['site_type']} face={representative['face']}", flush=True)
            try:
                bare = slab.copy()
                freeze_region(bare, slab.positions[site], config["active_radius_A"], len(bare))
                bare, e_bare, _ = relax(bare, calc, config)
                # Comparable local relaxation of bare and covered substrates.
                successes = []
                for ci, geo in enumerate(geos):
                    for oi in range(config["orientations"]):
                        attempt = {"conformer": ci, "orientation": oi}
                        try:
                            product, mapped_site, n_sub = place(bare, site, geo, candidate["template"], oi, config["orientations"])
                            freeze_region(product, bare.positions[site], config["active_radius_A"], n_sub)
                            product, e_product, fmax = relax(product, calc, config)
                            check_product(product, n_sub, mapped_site, geo, candidate["template"])
                            de = e_product + e_byproduct - e_bare - e_molecule
                            if abs(de) > config["energy_review_limit_eV"]:
                                raise ValueError("energy_requires_review_not_a_universal_physics_cutoff")
                            filename = output / f"{surface['id']}_{site}_{ci}_{oi}.extxyz"
                            write(filename, product)
                            attempt.update(status="valid", energy_eV=de, max_force=fmax,
                                           geometry=str(filename.resolve()))
                            successes.append(de)
                        except Exception as exc:
                            attempt.update(status="failed", reason=str(exc))
                        record["attempts"].append(attempt)
                if successes:
                    record.update(status="valid", energy_eV=min(successes),
                                  placement_spread_eV=float(np.ptp(successes)))
            except Exception as exc:
                record["reason"] = str(exc)
            sites.append(record)
    # Require sufficient population coverage on EACH surface, not just pooled data.
    per_surface = []
    for surface in surface_report["surfaces"]:
        records = [s for s in sites if s["surface_id"] == surface["id"]]
        valid = [s for s in records if s["status"] == "valid"]
        total = sum(s["population"] for s in records)
        measured = sum(s["population"] for s in valid)
        fraction = measured / total if total else 0.
        mean = float(np.average([s["energy_eV"] for s in valid], weights=[s["population"] for s in valid])) if valid else None
        per_surface.append({"surface_id": surface["id"], "material": surface["material"],
                            "coverage": fraction, "mean_eV": mean,
                            "valid": fraction >= config["min_site_success_fraction"]})
    energies = {}
    for material in ("SiO2", "SiNx"):
        rows = [s for s in per_surface if s["material"] == material]
        energies[material] = float(np.mean([s["mean_eV"] for s in rows])) if rows and all(s["valid"] for s in rows) else None
    gs, ngs = energies["SiO2"], energies["SiNx"]
    return clean({"candidate_id": candidate["id"], "mode": "mace", "role": candidate["role"],
                  "status": "complete" if gs is not None and ngs is not None else "insufficient_evidence",
                  "dE_GS": gs, "dE_NGS": ngs,
                  "contrast_eV": gs - ngs if gs is not None and ngs is not None else None,
                  "gas_reference_failures": gas_failures, "per_surface": per_surface, "sites": sites,
                  "metrics": evaluate(gs, ngs, config.get("objective")) if candidate["role"] == "inhibitor" else None,
                  "uncertainty_note": "Site/placement spread is sampling variability, not calibrated model uncertainty."})
