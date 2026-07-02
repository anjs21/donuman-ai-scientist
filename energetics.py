"""
energetics.py
=============
Phase 2b of the AS-ALD in-silico co-scientist (Challenge 4).

Computes the reaction / adsorption energy of a reagent (inhibitor or precursor
from inhibitor_library.py) at a specific reactive site of an amorphous surface
(from surface_builder.py), using the same MLIP calculator (MACE-MP-0) that
built the surfaces.

Reaction bookkeeping
--------------------
dissociative:   Site-H + R-LG  ->  Site-R + H-LG
    dE = [E(slab_capped) + E(byproduct)] - [E(slab) + E(molecule)]

physisorption:  Site + M      ->  Site...M
    dE = E(slab + M) - E(slab) - E(M)

A negative dE means the reagent binds/reacts favourably at that site.

Cost control
------------
* Gas-phase molecule energies are relaxed once and cached per reagent.
* The clean-slab energy is a single point, cached per surface (id-based).
* Product relaxation is LOCAL: atoms farther than `active_radius` from the
  reaction site are frozen (FixAtoms), so each site costs a small local
  optimisation instead of a full-slab relax. This is the "representative-site"
  saving the challenge brief asks for -- we never relax the whole slab per site.

The engine is MLIP-agnostic: pass any ASE calculator. Without MACE it will run
on the Lennard-Jones placeholder from surface_builder (numbers meaningless, but
the pipeline executes end-to-end for testing).
"""

import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.neighborlist import NeighborList, natural_cutoffs

import inhibitor_library as lib


# ---------------------------------------------------------------------------
# small caches (keyed by object id / reagent name)
# ---------------------------------------------------------------------------
_MOL_ENERGY_CACHE = {}
_SLAB_ENERGY_CACHE = {}


def _rotate_from_z(positions, target_dir):
    """Rotate a set of positions so that +z maps onto `target_dir`."""
    a = np.array([0.0, 0.0, 1.0])
    b = np.asarray(target_dir, float)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.linalg.norm(v) < 1e-8:
        # parallel or antiparallel
        return positions if c > 0 else positions * np.array([1, 1, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
    return positions @ R.T


def _site_normal(atoms, idx):
    """Outward surface normal at a site: +z if in the top half, else -z."""
    z = atoms.get_positions()[:, 2]
    zc = 0.5 * (z.min() + z.max())
    return np.array([0, 0, 1.0]) if atoms[idx].position[2] >= zc else np.array([0, 0, -1.0])


def _bonded_H(atoms, idx):
    """Index of an H bonded to atom `idx` (or None)."""
    cutoffs = natural_cutoffs(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    nbrs, _ = nl.get_neighbors(idx)
    symbols = atoms.get_chemical_symbols()
    hs = [j for j in nbrs if symbols[j] == "H"]
    return hs[0] if hs else None


def _mol_energy(reagent_or_name, build, calc, fmax=0.03, steps=200):
    """Relaxed gas-phase energy of a molecule, cached by name."""
    key = reagent_or_name
    if key in _MOL_ENERGY_CACHE:
        return _MOL_ENERGY_CACHE[key]
    mol = build()
    mol.center(vacuum=6.0)
    mol.calc = calc
    try:
        BFGS(mol, logfile=None).run(fmax=fmax, steps=steps)
        e = float(mol.get_potential_energy())
    except Exception as ex:
        print(f"    [warn] gas relax failed for {key}: {ex}")
        e = float(mol.get_potential_energy())
    _MOL_ENERGY_CACHE[key] = e
    return e


def _slab_energy(atoms, calc):
    """Single-point energy of a clean slab, cached by object id."""
    key = id(atoms)
    if key in _SLAB_ENERGY_CACHE:
        return _SLAB_ENERGY_CACHE[key]
    a = atoms.copy()
    a.calc = calc
    e = float(a.get_potential_energy())
    _SLAB_ENERGY_CACHE[key] = e
    return e


def _freeze_far(atoms, center, active_radius):
    """Freeze all atoms farther than active_radius from `center` (Angstrom)."""
    d = np.linalg.norm(atoms.get_positions() - center, axis=1)
    frozen = [i for i, di in enumerate(d) if di > active_radius]
    atoms.set_constraint(FixAtoms(indices=frozen))
    return len(atoms) - len(frozen)


def _relax_local(atoms, center, calc, active_radius, fmax, steps):
    atoms = atoms.copy()
    n_active = _freeze_far(atoms, center, active_radius)
    atoms.calc = calc
    try:
        BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
    except Exception as ex:
        print(f"    [warn] local relax failed: {ex}")
    return float(atoms.get_potential_energy()), n_active


# ===========================================================================
# Product builders
# ===========================================================================

def build_capped_slab(slab, site_idx, reagent):
    """
    Dissociative product: remove an H from the site atom and attach the
    reagent's surface fragment (anchor atom first) along the outward normal.
    Returns (product_atoms, attach_position) or (None, None) if no removable H.
    """
    h_idx = _bonded_H(slab, site_idx)
    if h_idx is None:
        return None, None
    normal = _site_normal(slab, site_idx)
    site_pos = slab[site_idx].position

    frag = reagent.fragment()
    fpos = frag.get_positions()
    fpos = fpos - fpos[0]                      # anchor (index 0) at origin
    fpos = _rotate_from_z(fpos, normal)        # fragment open bond (-z) -> -normal
    anchor = site_pos + normal * reagent.anchor_bond
    fpos = fpos + anchor

    product = slab.copy()
    del product[h_idx]                         # abstract the H
    for sym, p in zip(frag.get_chemical_symbols(), fpos):
        product += Atoms(sym, positions=[p])
    return product, anchor


def build_physisorbed_slab(slab, site_idx, reagent, height=2.6):
    """Physisorption product: place the whole molecule above the site."""
    normal = _site_normal(slab, site_idx)
    site_pos = slab[site_idx].position
    mol = reagent.build()
    mpos = mol.get_positions()
    mpos = mpos - mpos.mean(axis=0)
    place = site_pos + normal * height
    mpos = mpos + place
    product = slab.copy()
    for sym, p in zip(mol.get_chemical_symbols(), mpos):
        product += Atoms(sym, positions=[p])
    return product, place


# ===========================================================================
# Reaction energy at a single site
# ===========================================================================

def reaction_energy_at_site(slab, site_idx, reagent, calc,
                            active_radius=5.0, fmax=0.05, steps=120,
                            mol_fmax=0.03, mol_steps=200):
    """
    Return dE (eV) for `reagent` reacting/adsorbing at `site_idx` of `slab`.
    Negative dE = favourable. Returns np.nan if the product cannot be built.
    """
    e_slab = _slab_energy(slab, calc)

    if reagent.reaction_type == "dissociative":
        product, center = build_capped_slab(slab, site_idx, reagent)
        if product is None:
            return np.nan
        e_mol = _mol_energy(reagent.name, reagent.build, calc, mol_fmax, mol_steps)
        e_by = _mol_energy(reagent.name + "_byproduct", reagent.byproduct,
                           calc, mol_fmax, mol_steps)
        e_prod, _ = _relax_local(product, center, calc, active_radius, fmax, steps)
        return (e_prod + e_by) - (e_slab + e_mol)

    elif reagent.reaction_type == "physisorption":
        product, center = build_physisorbed_slab(slab, site_idx, reagent)
        e_mol = _mol_energy(reagent.name, reagent.build, calc, mol_fmax, mol_steps)
        e_prod, _ = _relax_local(product, center, calc, active_radius, fmax, steps)
        return e_prod - (e_slab + e_mol)

    raise ValueError(f"unknown reaction_type {reagent.reaction_type}")


# ===========================================================================
# Screen one reagent across a surface's representative sites
# ===========================================================================

def screen_reagent_on_surface(slab, material, reagent, calc,
                              site_types=None, max_sites=4, **kw):
    """
    Evaluate a reagent at up to `max_sites` representative sites (per applicable
    site type) of a slab. Uses the exposure-filtered, representative indices
    that surface_builder.classify_sites already provides.

    Returns dict:
        {site_type: {"dE_mean", "dE_min", "n", "values": [...]}}
    plus "_overall" = the most favourable (min) dE across all sites tested.
    """
    import surface_builder as sb
    wanted = set(site_types) if site_types else set(reagent.targets)

    counts = sb.classify_sites(slab, material, exposure_filter=True)
    # Fallback: if the (strict) exposure filter left no sites of interest --
    # e.g. on a marginal/under-relaxed slab -- fall back to all classified
    # sites so the screen still returns numbers rather than silent NaNs.
    if not any(st in counts and counts[st]["indices"] for st in wanted):
        counts = sb.classify_sites(slab, material, exposure_filter=False)
    result = {}
    all_vals = []
    for st, info in counts.items():
        if wanted and st not in wanted:
            continue
        idxs = info["indices"][:max_sites]
        vals = []
        for idx in idxs:
            dE = reaction_energy_at_site(slab, idx, reagent, calc, **kw)
            if not np.isnan(dE):
                vals.append(dE)
        if vals:
            result[st] = {
                "dE_mean": float(np.mean(vals)),
                "dE_min": float(np.min(vals)),
                "n": len(vals),
                "values": [round(v, 3) for v in vals],
            }
            all_vals.extend(vals)
    result["_overall"] = float(np.min(all_vals)) if all_vals else np.nan
    result["_mean"] = float(np.mean(all_vals)) if all_vals else np.nan
    return result


def screen_reagents(surfaces_by_material, reagents, calc, max_sites=3, **kw):
    """
    Full screen: every reagent x every material, averaged over the provided
    surface ensemble.

    surfaces_by_material : {"SiO2": [slab, ...], "SiNx": [slab, ...]}
    reagents             : list of Reagent
    Returns nested dict: {reagent_name: {material: {"dE_mean","dE_min",...}}}
    """
    out = {}
    for reagent in reagents:
        out[reagent.name] = {}
        for material, slabs in surfaces_by_material.items():
            overalls, means = [], []
            per_site_agg = {}
            for slab in slabs:
                r = screen_reagent_on_surface(slab, material, reagent, calc,
                                              max_sites=max_sites, **kw)
                if not np.isnan(r["_overall"]):
                    overalls.append(r["_overall"])
                    means.append(r["_mean"])
                for st, d in r.items():
                    if st.startswith("_"):
                        continue
                    per_site_agg.setdefault(st, []).append(d["dE_mean"])
            out[reagent.name][material] = {
                "dE_min": float(np.min(overalls)) if overalls else np.nan,
                "dE_mean": float(np.mean(means)) if means else np.nan,
                "per_site": {st: round(float(np.mean(v)), 3)
                             for st, v in per_site_agg.items()},
                "n_surfaces": len(overalls),
            }
            tag = f"{out[reagent.name][material]['dE_mean']:+.2f}" \
                if means else "  n/a"
            print(f"  [energetics] {reagent.name:14s} on {material:5s}: "
                  f"dE_mean={tag} eV  dE_min="
                  f"{out[reagent.name][material]['dE_min']:+.2f} eV "
                  f"({len(overalls)} surf)", flush=True)
    return out


# ===========================================================================
# Demo: build one quick surface and screen DMATMS + pyridine on it
# ===========================================================================

if __name__ == "__main__":
    import surface_builder as sb

    print("Loading calculator...")
    calc = sb.get_calculator()
    sb.use_test_protocol()

    surfaces = {}
    for material in ["SiO2", "SiNx"]:
        print(f"\nBuilding one quick {material} surface for the demo...")
        slabs, _ = sb.build_surface_ensemble(material, calc, n_bulk=1, apply_gate=False)
        surfaces[material] = slabs[:1]

    reagents = lib.get_reagents(names=["DMATMS", "pyridine", "TMCS"])
    print("\nScreening reagents:")
    results = screen_reagents(surfaces, reagents, calc, max_sites=2)
    print("\nDone. Raw results:")
    for name, per_mat in results.items():
        print(f"  {name}: {per_mat}")
