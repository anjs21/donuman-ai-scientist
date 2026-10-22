"""Explicit manifests, content hashes, and fail-closed surface admission."""
from pathlib import Path
import re
import numpy as np
from ase.io import read
from ase.geometry import find_mic
from ase.data import covalent_radii
import structure_validation as sv
import surface_builder as sb
from .common import read_json, file_hash, digest, clean
from .slab_validation import PROFILE, finite_slab_structure, identify_composition


def load_manifest(path, require_valid=True):
    path = Path(path).resolve()
    manifest = read_json(path)
    profile = manifest.get('validation_profile', 'legacy_bulk_comparison')
    if profile not in {'legacy_bulk_comparison', PROFILE}:
        raise ValueError('Unknown surface validation_profile')
    rows, atoms_by_id, seen = [], {}, set()
    for entry in manifest.get("surfaces", []):
        material = entry.get("material")
        if material not in {"SiO2", "SiNx"}:
            raise ValueError("Manifest material must be SiO2 or SiNx")
        if not entry.get("provenance") or entry.get("stage") != "passivated_slab":
            raise ValueError("Each surface needs provenance and stage=passivated_slab")
        filename = (path.parent / entry["path"]).resolve()
        checksum = file_hash(filename)
        if checksum in seen:
            raise ValueError("Duplicate surface content in manifest")
        if entry.get('expected_sha256') and entry['expected_sha256'] != checksum:
            raise ValueError('Surface checksum differs from the pinned manifest: ' + str(filename))
        if profile == PROFILE and not entry.get('expected_sha256'):
            raise ValueError('Finite-slab screening manifests must pin expected_sha256 for every slab')
        seen.add(checksum)
        atoms = read(filename, format=entry.get("format"))
        atoms.set_pbc([True, True, False])
        reasons = []
        identity, composition_reasons = identify_composition(atoms, material)
        reasons.extend(composition_reasons)
        if not np.isfinite(atoms.positions).all() or abs(np.linalg.det(atoms.cell)) < 1e-6:
            raise ValueError(f"Invalid positions/cell: {filename}")
        if abs(atoms.cell[0, 2]) > 1e-5 or abs(atoms.cell[1, 2]) > 1e-5:
            raise ValueError("This implementation requires slab normal along z")
        vacuum = abs(atoms.cell[2, 2]) - np.ptp(atoms.positions[:, 2])
        if vacuum < 10:
            reasons.append("less than 10 Angstrom vacuum")
        d = atoms.get_all_distances(mic=True)
        np.fill_diagonal(d, np.inf)
        radii = covalent_radii[atoms.numbers]
        if (d < .65 * (radii[:, None] + radii[None, :])).any():
            reasons.append("severe atom overlap")
        # Reject detached molecules/atoms under the same distance-based bond heuristic.
        from scipy.sparse.csgraph import connected_components
        adjacency = d < 1.3 * (radii[:, None] + radii[None, :])
        n_components, _ = connected_components(adjacency, directed=False)
        if n_components != 1:
            reasons.append('disconnected substrate atom network; inspect detached species or bond cutoffs')
        structural = sv.validate_structure(atoms, material, verbose=False)
        density = None
        warnings = []
        if profile == PROFILE:
            structural, density = finite_slab_structure(atoms, material, structural)
            warnings.append('Bulk density is unverified; finite-slab density is diagnostic, not a passed bulk test.')
            warnings.append('One slab per material supports a pipeline smoke test, not ensemble representativeness.')
        exemptions = reference_exemptions(entry)
        if not structural["_ok"]:
            reasons.extend("structural: " + k for k, v in structural.items()
                           if not k.startswith("_") and not v["pass"] and k not in exemptions)
        counts = sb.classify_sites(atoms, material, exposure_filter=True)
        ok, gate_reasons = sb.slab_passes_quality_gate(
            counts, sb.surface_area_nm2(atoms), material, atoms=atoms)
        reasons.extend(gate_reasons)
        row = {"id": entry.get("id", material + "_" + checksum[:12]),
               "material": material, "path": str(filename), "sha256": checksum,
               "provenance": entry["provenance"], "valid": not reasons,
               "screening_admitted": not reasons, "scientific_validity": "unverified",
               "validation_profile": profile, "material_identity": identity,
               "network_components": int(n_components), "density_diagnostics": density,
               "warnings": warnings,
               "reasons": reasons, "structural": clean(structural),
               "sites": {k: v["exposed"] for k, v in counts.items()},
               "exemptions": {k: {"measured": structural.get(k, {}).get("value"),
                                  "passed_anyway": bool(structural.get(k, {}).get("pass")),
                                  **entry["reference_structure"]} for k in exemptions},
               "coordination_defects": coordination_defects(atoms),
               "vacuum_A": float(vacuum)}
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", row["id"]):
            raise ValueError("Surface ID must be a simple filename-safe identifier")
        if row["id"] in atoms_by_id:
            raise ValueError("Surface IDs must be unique")
        rows.append(row); atoms_by_id[row["id"]] = atoms
    if not rows:
        raise ValueError("Empty surface manifest")
    complete = {r["material"] for r in rows if r["valid"]} == {"SiO2", "SiNx"}
    report = {"valid": complete and all(r["valid"] for r in rows), "surfaces": rows,
              "validation_profile": profile, "scientific_success": False,
              "admission_scope": "numerical_screening_only",
              "bulk_density_validated": False}
    report["fingerprint"] = digest({"validation_profile": profile, "surfaces": [
        {k: r[k] for k in ("id", "material", "sha256", "exemptions")} for r in rows]})
    if require_valid and not report["valid"]:
        raise ValueError("Surface admission failed. Run validate-surfaces and inspect its report; no energies computed.")
    return report, atoms_by_id


# Checks a published reference structure may be exempted from. Only the slab
# density estimate qualifies: on thin passivated slabs it is biased by the
# rough surface layers (whole-slab extent under-reads, interior trimming
# over-reads), while geometry, overlap, RDF/coordination and site-density
# checks remain meaningful and always gate.
EXEMPTABLE = {"density_gcc"}


def reference_exemptions(entry):
    """Exempted check names for a manifest entry, validated.

    "reference_structure": {"citation": "...", "doi": "...", "reason": "...",
                            "exempt_checks": ["density_gcc"]}
    """
    ref = entry.get("reference_structure")
    if not ref:
        return set()
    checks = set(ref.get("exempt_checks", []))
    if not checks:
        raise ValueError("reference_structure needs exempt_checks")
    if checks - EXEMPTABLE:
        raise ValueError(f"Only {sorted(EXEMPTABLE)} can be exempted for a reference structure")
    if not str(ref.get("citation", "")).strip() or not str(ref.get("reason", "")).strip():
        raise ValueError("A reference-structure exemption needs a citation and a reason")
    return checks


IDEAL_COORDINATION = {14: 4, 7: 3, 8: 2}
SYMBOL = {14: "Si", 7: "N", 8: "O"}


def coordination_defects(atoms, factor=1.15):
    """Reported, not gated: under/over-coordinated atoms and homopolar Si-Si bonds."""
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    r = covalent_radii[atoms.numbers]
    bonds = d < factor * (r[:, None] + r[None, :])
    out = {}
    for z, ideal in IDEAL_COORDINATION.items():
        idx = np.flatnonzero(atoms.numbers == z)
        if len(idx):
            n = bonds[idx].sum(axis=1)
            out[SYMBOL[z]] = {"under": int((n < ideal).sum()), "over": int((n > ideal).sum()),
                                 "fraction_defective": float((n != ideal).mean())}
    si = np.flatnonzero(atoms.numbers == 14)
    out["Si_Si_bonds"] = int(bonds[np.ix_(si, si)].sum() // 2)
    return out


def representatives(atoms, material, max_sites):
    counts = sb.classify_sites(atoms, material, exposure_filter=True)
    allowed = {"OH"} if material == "SiO2" else {"NH2", "NH_bridge", "OH"}
    groups = []
    middle = (atoms.positions[:, 2].min() + atoms.positions[:, 2].max()) / 2
    for kind in sorted(allowed):
        indices = counts.get(kind, {}).get("indices", [])
        for top in (False, True):
            face = [i for i in indices if (atoms.positions[i, 2] >= middle) == top]
            if not face:
                continue
            # Deterministic farthest-point sampling in local coordination fingerprints.
            features = []
            for i in face:
                delta = atoms.positions - atoms.positions[i]
                _, ds = find_mic(delta, atoms.cell, atoms.pbc)
                feat = []
                for element in (1, 7, 8, 14):
                    vals = sorted(float(v) for j, v in enumerate(ds)
                                  if j != i and atoms.numbers[j] == element and v < 5)
                    feat.extend((vals + [5.] * 4)[:4])
                features.append(feat)
            f = np.array(features)
            chosen = [int(np.argmin(np.linalg.norm(f - f.mean(axis=0), axis=1)))]
            while len(chosen) < min(max_sites, len(face)):
                ds = np.linalg.norm(f[:, None] - f[chosen][None], axis=2).min(axis=1)
                ds[chosen] = -1
                chosen.append(int(ds.argmax()))
            labels = np.linalg.norm(f[:, None] - f[chosen][None], axis=2).argmin(axis=1)
            for cluster, local in enumerate(chosen):
                weight = int((labels == cluster).sum())
                if weight:
                    groups.append({"site": face[local], "site_type": kind,
                                   "face": "top" if top else "bottom", "population": weight})
    return groups
