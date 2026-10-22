"""Site-resolved reaction barriers and dose kinetics.

Barriers come from climbing-image NEB between a relaxed pre-reaction complex
and a relaxed H-X product complex. Sites are either gas-phase models (H3Si-NH2,
(H3Si)2NH, H3Si-OH) or hydrogen-capped clusters carved around representative
sites of the admitted amorphous slabs, so each barrier carries the population
weight of the local environment it stands for. Every failed endpoint, changed
connectivity or unconverged band is recorded, never silently dropped.

Barriers from one method are not chemical truth; method disagreement is
reported and later propagated as uncertainty (see process.py).
"""
from pathlib import Path
import math
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.data import atomic_masses, covalent_radii
from ase.geometry import find_mic
from ase.io import write
from ase.mep import NEB
from ase.neighborlist import NeighborList
from ase.optimize import BFGS, FIRE
from rdkit import Chem
from rdkit.Chem import AllChem
from .chemistry import ADDITION, ATTACHING, CONDENSATION, H_LENGTH, SUBSTITUTION, addition_frame, rotation
from .common import clean

KB_EV = 8.617333262e-5
H_EVS = 4.135667696e-15
DEFAULTS = {"method": "xtb", "n_images": 7, "fmax": 0.05, "neb_fmax": 0.1,
            "steps": 300, "neb_steps": 1500, "attack_distance_A": 2.8,
            "cluster_shells": 3, "orientations": 4, "barrier_orientations": 2,
            "max_image_jump_eV": 1.5, "min_barrier_eV": 0.05, "seed": 42}
STRUCTURE_FACTOR = 1.15
CAP_LENGTH = {14: 1.48, 8: 0.97, 7: 1.01, 6: 1.09}
BOND_LENGTH = {("silyl_substitution", 8): 1.65, ("silyl_substitution", 7): 1.75,
               ("acylation", 8): 1.36, ("acylation", 7): 1.38,
               ("carbonyl_addition", 8): 1.43, ("carbonyl_addition", 7): 1.47,
               ("imine_condensation", 8): 1.43, ("imine_condensation", 7): 1.47}
PATHWAYS = {"acylation": ("addition", "four_centre"),
            "silyl_substitution": ("four_centre", "backside"),
            # Concerted X-H...O=C ring (proton to O as X-C forms) or perpendicular attack.
            "carbonyl_addition": ("four_centre", "addition"),
            # Barriers are for the addition step only; the dehydration that follows
            # at NH2 sites needs its own transition state and is not computed.
            "imine_condensation": ("four_centre", "addition")}
GAS_SITES = {"SiNx": {"NH2": "[SiH3][NH2:1]", "NH_bridge": "[SiH3][NH:1][SiH3]"},
             "SiO2": {"OH": "[SiH3][OH:1]"}}


def calculator_factory(config):
    method = config.get("method", "xtb")
    if method == "xtb":
        from tblite.ase import TBLite
        return lambda: TBLite(method=config.get("xtb_method", "GFN2-xTB"), verbosity=0)
    if method == "mace":
        from .engine import calculator
        shared = calculator(config)
        return lambda: shared
    raise ValueError("method must be xtb or mace")


def bonded(atoms, factor=1.3):
    d = atoms.get_all_distances(mic=bool(atoms.pbc.any()))
    r = covalent_radii[atoms.numbers]
    out = d < factor * (r[:, None] + r[None, :])
    np.fill_diagonal(out, False)
    return out


def lone_pair_direction(atoms, site):
    neighbours = np.flatnonzero(bonded(atoms, STRUCTURE_FACTOR)[site])
    if not len(neighbours):
        raise ValueError("isolated_site")
    vectors = atoms.positions[neighbours] - atoms.positions[site]
    direction = -(vectors / np.linalg.norm(vectors, axis=1)[:, None]).sum(axis=0)
    if np.linalg.norm(direction) < 1e-3:
        raise ValueError("no_lone_pair_direction")
    return direction / np.linalg.norm(direction)


def gas_site(smiles, seed=42):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3(); params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise ValueError("could_not_embed_site_model")
    site = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() == 1)
    atoms = Atoms([a.GetSymbol() for a in mol.GetAtoms()],
                  positions=mol.GetConformer().GetPositions())
    return atoms, site, [], lone_pair_direction(atoms, site)


def carve_cluster(slab, site, shells=2):
    """H-capped cluster of all atoms within `shells` bonds of `site`.

    The outermost shell and caps are frozen so the cluster keeps the local
    amorphous geometry of the slab it came from.
    """
    # d < 1.15 (r_i + r_j) with no skin: keeps Si-Si bonds (2.35 A) but not the
    # second-neighbour Si...Si contacts (~3 A) of amorphous SiNx.
    nl = NeighborList(covalent_radii[slab.numbers] * STRUCTURE_FACTOR, skin=0., self_interaction=False, bothways=True)
    nl.update(slab)
    depth = {site: 0}
    frontier = [site]
    for level in range(1, shells + 1):
        nxt = []
        for i in frontier:
            for j in nl.get_neighbors(i)[0]:
                if j not in depth:
                    depth[j] = level; nxt.append(j)
        frontier = nxt
    # Terminal hydrogens of the outer shell belong to the cluster, not to caps.
    for i in [k for k, v in depth.items() if v == shells]:
        for j in nl.get_neighbors(i)[0]:
            if j not in depth and slab.numbers[j] == 1:
                depth[j] = shells + 1
    order = [site] + sorted(k for k in depth if k != site)
    vec, _ = find_mic(slab.positions[order] - slab.positions[site], slab.cell, slab.pbc)
    positions = {i: slab.positions[site] + v for i, v in zip(order, vec)}
    symbols, coords, frozen = [], [], []
    for n, i in enumerate(order):
        symbols.append(slab.numbers[i]); coords.append(positions[i])
        if depth[i] >= shells and i != site:
            frozen.append(n)
    for i in order:
        if slab.numbers[i] == 1:
            continue
        for j in nl.get_neighbors(i)[0]:
            if j in depth:
                continue
            (v,), _ = find_mic(slab.positions[j:j+1] - slab.positions[i], slab.cell, slab.pbc)
            symbols.append(1)
            coords.append(positions[i] + CAP_LENGTH.get(int(slab.numbers[i]), 1.1) * v / np.linalg.norm(v))
            frozen.append(len(symbols) - 1)
    cluster = Atoms(numbers=symbols, positions=coords)
    if int(cluster.numbers.sum()) % 2:
        # A dangling bond in the slab; a closed-shell barrier would be meaningless.
        raise ValueError("open_shell_cluster")
    zc = (slab.positions[:, 2].max() + slab.positions[:, 2].min()) / 2
    outward = np.array([0., 0., 1. if slab.positions[site, 2] >= zc else -1.])
    lp = lone_pair_direction(cluster, 0)
    normal = lp + outward
    normal /= np.linalg.norm(normal)
    return cluster, 0, frozen, normal


def molecule(spec, seed=42):
    mol = Chem.AddHs(Chem.MolFromSmiles(spec["smiles"]))
    params = AllChem.ETKDGv3(); params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise ValueError("could_not_embed_candidate")
    atoms = Atoms([a.GetSymbol() for a in mol.GetAtoms()],
                  positions=mol.GetConformer().GetPositions())
    anchor = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() == 1)
    leaving = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() == 2)
    if spec["template"] in ADDITION:
        return atoms, anchor, leaving, []   # map 2 is the proton acceptor; nothing leaves
    cut = Chem.RWMol(mol); cut.RemoveBond(anchor, leaving)
    lost = next(g for g in Chem.GetMolFrags(cut) if leaving in g)
    return atoms, anchor, leaving, sorted(lost)


def _perpendicular(n):
    t = np.cross(n, [1., 0., 0.])
    if np.linalg.norm(t) < 1e-3:
        t = np.cross(n, [0., 1., 0.])
    return t / np.linalg.norm(t)


def _spin(n, angle):
    k = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def endpoints(substrate, site, normal, spec, config, orientation=0, pathway=None):
    """Atom-matched reactant and product complexes for one substitution.

    Order: substrate atoms (the transferring H keeps its index), then the
    candidate's atoms. Pathways (first is the default for the template):
      addition    -- nucleophile approaches perpendicular to the carbonyl plane
      four_centre -- front side, leaving group tilted toward the surface proton
      backside    -- nucleophile opposite the leaving group
    """
    template = spec["template"]
    if template not in ATTACHING:
        raise ValueError("Barriers need a substitution or addition template")
    pathway = pathway or PATHWAYS[template][0]
    if pathway not in PATHWAYS[template]:
        raise ValueError(f"pathway {pathway} not implemented for {template}")
    mol, anchor, leaving, lost = molecule(spec, config["seed"])
    nbr = np.flatnonzero(bonded(substrate, STRUCTURE_FACTOR)[site])
    hydrogens = [i for i in nbr if substrate.numbers[i] == 1]
    if not hydrogens:
        raise ValueError("no_surface_proton")
    h = min(hydrogens, key=lambda i: substrate.get_distance(site, i))
    pos = mol.positions - mol.positions[anchor]
    spin = _spin(normal, 2 * np.pi * orientation / config["orientations"])
    if template in ADDITION:
        return _addition_endpoints(substrate, site, normal, h, mol, anchor, leaving, pos, spin,
                                   template, pathway, orientation, config)
    if pathway == "addition":
        oxygen = next(i for i in np.flatnonzero(bonded(mol)[anchor])
                      if mol.numbers[i] == 8 and i != leaving)
        axis = np.cross(pos[oxygen], pos[leaving])
        pos = pos @ rotation(axis, normal).T @ spin.T
    elif pathway == "four_centre":
        toward = substrate.positions[h] - substrate.positions[site]
        toward -= (toward @ normal) * normal
        toward = toward / np.linalg.norm(toward) if np.linalg.norm(toward) > 1e-3 else _perpendicular(normal)
        tilt = np.radians(60 + 15 * (orientation % config["orientations"]))  # 60-105 deg, same period as spin
        pos = pos @ rotation(pos[leaving], -np.cos(tilt) * normal + np.sin(tilt) * toward).T
    else:
        pos = pos @ rotation(pos[leaving], normal).T @ spin.T
    site_pos = substrate.positions[site]
    reactant = substrate.copy()
    placed = mol.copy(); placed.positions = pos + site_pos + config["attack_distance_A"] * normal
    reactant += placed
    # Product: retained group swings so the old anchor-X bond points at the site.
    retained = [i for i in range(len(mol)) if i not in lost]
    product_pos = placed.positions.copy()
    to_site = rotation(pos[leaving], -normal)
    length = BOND_LENGTH[template, int(substrate.numbers[site])]
    product_pos[retained] = pos[retained] @ to_site.T + site_pos + length * normal
    lateral = product_pos[retained].mean(axis=0) - site_pos
    lateral -= (lateral @ normal) * normal
    lateral = -lateral / np.linalg.norm(lateral) if np.linalg.norm(lateral) > 1e-3 else _perpendicular(normal)
    x_new = site_pos + 2.0 * normal + 2.6 * lateral
    product_pos[lost] = placed.positions[lost] - placed.positions[leaving] + x_new
    product = reactant.copy()
    product.positions[len(substrate):] = product_pos
    away = x_new - site_pos
    product.positions[h] = x_new - H_LENGTH[mol.get_chemical_symbols()[leaving]] * away / np.linalg.norm(away)
    n0 = len(substrate)
    return reactant, product, {"site": site, "proton": int(h), "anchor": n0 + anchor,
                               "leaving": n0 + leaving, "n_substrate": n0,
                               "molecule": mol, "lost": [n0 + i for i in lost]}


def _addition_endpoints(substrate, site, normal, h, mol, anchor, acceptor, pos, spin,
                        template, pathway, orientation, config):
    """Surface X-H across C=O: X-C forms and the proton moves to O (map 2)."""
    site_pos = substrate.positions[site]
    if pathway == "four_centre":
        # C=O leans toward the surface proton so X, H, O and C can close a ring.
        toward = substrate.positions[h] - site_pos
        toward -= (toward @ normal) * normal
        toward = toward / np.linalg.norm(toward) if np.linalg.norm(toward) > 1e-3 else _perpendicular(normal)
        tilt = np.radians(60 + 15 * (orientation % config["orientations"]))  # same period as the spin
        reactant_pos = pos @ rotation(pos[acceptor], -np.cos(tilt) * normal + np.sin(tilt) * toward).T
    else:
        # Perpendicular (Burgi-Dunitz-like) attack on the carbonyl plane.
        other = next(i for i in np.flatnonzero(bonded(mol)[anchor]) if i != acceptor)
        reactant_pos = pos @ rotation(np.cross(pos[acceptor], pos[other]), normal).T @ spin.T
    reactant = substrate.copy()
    placed = mol.copy(); placed.positions = reactant_pos + site_pos + config["attack_distance_A"] * normal
    reactant += placed
    frame, proton = addition_frame(mol.get_chemical_symbols(), pos, anchor, acceptor, H_LENGTH["O"])
    to_surface = rotation([0., 0., 1.], normal).T @ spin.T
    length = BOND_LENGTH[template, int(substrate.numbers[site])]
    product = reactant.copy()
    product.positions[len(substrate):] = frame @ to_surface + site_pos + length * normal
    product.positions[h] = proton @ to_surface + site_pos + length * normal
    n0 = len(substrate)
    return reactant, product, {"site": site, "proton": int(h), "anchor": n0 + anchor,
                               "leaving": n0 + acceptor, "n_substrate": n0, "molecule": mol,
                               "lost": [], "template": template}


def _relax(atoms, factory, frozen, fmax, steps):
    atoms = atoms.copy()
    atoms.set_constraint(FixAtoms(indices=frozen) if frozen else [])
    atoms.calc = factory()
    converged = BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
    energy = float(atoms.get_potential_energy())
    if not converged or not math.isfinite(energy):
        raise ValueError("endpoint_not_converged")
    return atoms, energy


def check_reactant(atoms, m):
    b = bonded(atoms)
    if b[m["site"], m["anchor"]]:
        raise ValueError("reactant_reacted_spontaneously")
    if not b[m["anchor"], m["leaving"]] or not b[m["site"], m["proton"]]:
        raise ValueError("reactant_connectivity_changed")


def check_product(atoms, m):
    b = bonded(atoms)
    # Substitution breaks anchor-X; addition keeps C-O (now single) and protonates O.
    keeps_anchor_bond = m.get("template") in ADDITION
    if not b[m["site"], m["anchor"]] or bool(b[m["anchor"], m["leaving"]]) != keeps_anchor_bond:
        raise ValueError("product_not_formed")
    if not b[m["proton"], m["leaving"]] or b[m["site"], m["proton"]]:
        raise ValueError("proton_transfer_not_completed")
    # The released H-X must not have re-protonated another basic atom.
    others = [i for i in np.flatnonzero(b[m["proton"]]) if i != m["leaving"]]
    if others:
        raise ValueError("byproduct_proton_captured")


def barrier(substrate, site, normal, frozen, spec, factory, config, artifact=None, orientation=0, pathway=None):
    config = {**DEFAULTS, **config}
    reactant, product, m = endpoints(substrate, site, normal, spec, config, orientation, pathway)
    reactant, e_r = _relax(reactant, factory, frozen, config["fmax"], config["steps"])
    check_reactant(reactant, m)
    product, e_p = _relax(product, factory, frozen, config["fmax"], config["steps"])
    check_product(product, m)
    images = [reactant] + [reactant.copy() for _ in range(config["n_images"])] + [product]
    for image in images:
        image.set_constraint(FixAtoms(indices=frozen) if frozen else [])
        image.calc = factory()
    neb = NEB(images, climb=False, allow_shared_calculator=True, method="improvedtangent")
    neb.interpolate("idpp", apply_constraint=True)
    FIRE(neb, logfile=None).run(fmax=0.5, steps=config["neb_steps"] // 4)
    neb.climb = True
    converged = FIRE(neb, logfile=None).run(fmax=config["neb_fmax"], steps=config["neb_steps"])
    profile = np.array([float(im.get_potential_energy()) for im in images]) - e_r
    clash = any(_clash(im) for im in images)
    if artifact:
        Path(artifact).parent.mkdir(parents=True, exist_ok=True)
        write(str(artifact), images)
    return {**classify(profile, converged, clash, config),
            "path": str(Path(artifact).resolve()) if artifact else None,
            "e_reactant_complex_eV": e_r}


def classify(profile, converged, clash, config):
    """Barrier and status from an NEB energy profile (eV, relative to reactant).

    The barrier is measured from the lowest image before the highest interior
    image, because a relaxed start can be a shallow minimum the band leaves.
    A rise under min_barrier_eV is not a resolved transition state.
    """
    config = {**DEFAULTS, **config}
    profile = np.asarray(profile, float)
    ts = int(np.argmax(profile[1:-1])) + 1
    pre = int(np.argmin(profile[:ts]))
    ea = float(profile[ts] - profile[pre])
    jump = float(np.abs(np.diff(profile)).max())
    minima = [i for i in range(1, len(profile) - 1)
              if profile[i] < profile[i - 1] - .05 and profile[i] < profile[i + 1] - .05]
    if not converged:
        status = "neb_not_converged"
    elif clash or jump > config["max_image_jump_eV"]:
        status = "band_unresolved"
    elif ea < config["min_barrier_eV"]:
        # A clean converged band that never rises by min_barrier_eV and ends downhill
        # is a fast (barrierless) reaction: real chemistry, not a failed calculation.
        # Uphill bands without a resolvable maximum stay unresolved.
        status = "barrierless" if profile[-1] - profile[pre] <= 0 else "no_barrier_resolved"
    else:
        status = "valid"
    return {"status": status, "Ea_eV": ea, "ts_image": ts, "precomplex_image": pre,
            "Ea_upper_eV": float(config["min_barrier_eV"]) if status == "barrierless" else None,
            "precomplex_shift_eV": float(profile[pre]),
            "dE_rxn_eV": float(profile[-1] - profile[pre]), "profile_eV": profile.tolist(),
            "max_image_jump_eV": jump, "image_clash": bool(clash),
            "stepwise_indication": bool(minima)}


USABLE = ("valid", "barrierless")
# A barrierless result is censored (Ea <= min_barrier_eV). It is resolved in the
# direction that weakens selectivity: fastest possible on the growth surface
# (SiO2, 0 eV), slowest still consistent on the non-growth surface (SiNx, bound).
CENSORED_AS = {"SiO2": "lower", "SiNx": "upper"}


def effective_barrier(attempt, material, config=None):
    """Barrier (eV) an attempt contributes to its site, or None if unusable."""
    config = {**DEFAULTS, **(config or {})}
    if attempt.get("status") == "valid":
        return float(attempt["Ea_eV"])
    if attempt.get("status") == "barrierless":
        return 0.0 if CENSORED_AS.get(material, "lower") == "lower" else float(config["min_barrier_eV"])
    return None


def select_best(attempts, material, config=None):
    """Lowest effective barrier over usable attempts: the site's fastest path."""
    best, best_ea = None, None
    for i, a in enumerate(attempts):
        ea = effective_barrier(a, material, config)
        if ea is not None and (best_ea is None or ea < best_ea):
            best, best_ea = {**a, "_index": i}, ea
    if best is None:
        return None
    best["Ea_eff_eV"] = best_ea
    return best


def site_fields(best):
    """Site-level fields derived from the selected attempt (None -> failed site)."""
    if best is None:
        return {"status": "failed", "Ea_eV": None, "dE_rxn_eV": None, "stepwise_indication": None,
                "barrierless": None, "best_attempt": None}
    return {"status": "valid", "Ea_eV": best["Ea_eff_eV"], "dE_rxn_eV": best["dE_rxn_eV"],
            "stepwise_indication": best["stepwise_indication"],
            "barrierless": best["status"] == "barrierless", "best_attempt": best["_index"]}


def reclassify(record, config=None):
    """Re-derive attempt statuses of a stored kinetics record under current rules."""
    config = {**DEFAULTS, **(config or {})}
    record = dict(record)
    rows = []
    for site in record["sites"]:
        site = dict(site)
        attempts = []
        for a in site.get("attempts", []):
            if a.get("profile_eV"):
                a = {**a, **classify(a["profile_eV"], a["status"] != "neb_not_converged",
                                     a.get("image_clash", False), config)}
            attempts.append(a)
        best = select_best(attempts, site["material"], config)
        previous = site.get("attempts", [])
        if site.get("best_attempt") is not None and site["best_attempt"] < len(previous):
            old = previous[site["best_attempt"]]
        else:  # records written before best_attempt existed
            old = next((a for a in previous if a.get("Ea_eV") == site.get("Ea_eV")
                        and site.get("Ea_eV") is not None), None)
        # Binding = E(pre-complex image) - E_substrate - E_molecule. Recover the reference
        # energies from the attempt that carried the stored binding, then re-evaluate on
        # the newly selected attempt's own reactant complex.
        binding = site.get("precomplex_binding_eV")
        if binding is not None and old is not None and best is not None \
                and old.get("e_reactant_complex_eV") is not None and best.get("e_reactant_complex_eV") is not None:
            references = old["e_reactant_complex_eV"] + old.get("precomplex_shift_eV", 0.) - binding
            binding = best["e_reactant_complex_eV"] + best["precomplex_shift_eV"] - references
        elif binding is not None and best is not None:
            # Legacy record without per-attempt reactant energies.
            binding += best["precomplex_shift_eV"] - (old or {}).get("precomplex_shift_eV", 0.)
        site.update(attempts=attempts, precomplex_binding_eV=binding if best else None, **site_fields(best))
        rows.append(site)
    record.update(sites=rows, summary=summarize(rows), classification_rules=
                  {k: config[k] for k in ("min_barrier_eV", "max_image_jump_eV")})
    return record


def apply_observations(record, observations):
    """Fill failed sites of a kinetics record from decision-agent observations.

    observations: stored decide.Observation dicts. Only valid barriers from the
    record's own method and candidate count; a site's barrier is the lowest over
    the pathways observed (the fastest channel caps the site). A failed site takes
    the fastest observed path; a valid site is lowered only if an observed path is
    faster, since ignoring a faster path (e.g. a barrierless oxide route found by
    repair) would overstate selectivity.
    """
    record = dict(record)
    usable = [o for o in observations if o.get("candidate") == record["candidate_id"]
              and o.get("method") == record["method"] and o.get("status") == "valid"
              and o.get("Ea_eV") is not None]
    if not usable:
        return record
    rows, repaired = [], []
    for site in record["sites"]:
        site = dict(site)
        found = [o for o in usable if o["site"] == site["label"]]
        best = min(found, key=lambda o: o["Ea_eV"]) if found else None
        if best is not None and (site["status"] != "valid" or best["Ea_eV"] < site["Ea_eV"]):
            site.update(status="valid", Ea_eV=best["Ea_eV"], dE_rxn_eV=best.get("dE_rxn_eV"),
                        barrierless=best.get("reason") == "barrierless",
                        precomplex_binding_eV=None, repaired_by={"pathway": best["pathway"],
                                                                 "source": best.get("source")})
            repaired.append(site["label"])
        rows.append(site)
    record.update(sites=rows, summary=summarize(rows), repaired_sites=repaired)
    return record


def _clash(atoms, factor=0.6):
    d = atoms.get_all_distances()
    np.fill_diagonal(d, np.inf)
    r = covalent_radii[atoms.numbers]
    return bool((d < factor * (r[:, None] + r[None, :])).any())


def relaxed_energy(atoms, factory, frozen, config):
    _, e = _relax(atoms, factory, frozen, config["fmax"], config["steps"])
    return e


def sites_for(source, manifest_report=None, atoms_by_id=None, max_sites=2, shells=2):
    """(material, label, population, substrate, site, frozen, normal) tuples."""
    out = []
    if source == "gas":
        for material, kinds in GAS_SITES.items():
            for kind, smiles in kinds.items():
                atoms, site, frozen, normal = gas_site(smiles)
                out.append({"material": material, "site_type": kind, "label": f"gas:{kind}",
                            "population": 1, "substrate": atoms, "site": site,
                            "frozen": frozen, "normal": normal})
        return out
    from .surfaces import representatives
    for surface in manifest_report["surfaces"]:
        slab = atoms_by_id[surface["id"]]
        for rep in representatives(slab, surface["material"], max_sites):
            label = f"{surface['id']}:{rep['site']}:{rep['face']}"
            try:
                cluster, site, frozen, normal = carve_cluster(slab, rep["site"], shells)
            except ValueError as exc:
                out.append({"material": surface["material"], "site_type": rep["site_type"], "label": label,
                            "surface_id": surface["id"], "population": rep["population"], "error": str(exc)})
                continue
            out.append({"material": surface["material"], "site_type": rep["site_type"],
                        "label": label, "surface_id": surface["id"], "population": rep["population"],
                        "substrate": cluster, "site": site, "frozen": frozen, "normal": normal})
    return out


def screen_kinetics(spec, sites, config, artifact_dir):
    """Barriers for one candidate at every supplied site; failures are kept."""
    config = {**DEFAULTS, **config}
    factory = calculator_factory(config)
    gas, *_ = molecule(spec, config["seed"])
    gas.center(vacuum=8)
    try:
        e_molecule = relaxed_energy(gas, factory, [], config)
    except Exception:
        e_molecule = None
    rows = []
    for s in sites:
        row = {k: s[k] for k in ("material", "site_type", "label", "population")}
        row.update(method=config["method"], attempts=[])
        if s.get("error"):
            rows.append({**row, "status": "failed", "reason": s["error"], "Ea_eV": None})
            continue
        try:
            e_substrate = relaxed_energy(s["substrate"], factory, s["frozen"], config)
        except Exception:
            e_substrate = None
        best = None
        for orientation in range(config["barrier_orientations"]):
            try:
                r = barrier(s["substrate"], s["site"], s["normal"], s["frozen"], spec, factory, config,
                            Path(artifact_dir) / f"{spec['id']}_{s['label'].replace(':', '_')}_{orientation}.extxyz",
                            orientation)
                row["attempts"].append({"orientation": orientation, **r})
            except Exception as exc:
                row["attempts"].append({"orientation": orientation, "status": "failed", "reason": str(exc)})
        best = select_best(row["attempts"], s["material"], config)
        binding = (best["e_reactant_complex_eV"] + best["precomplex_shift_eV"] - e_substrate - e_molecule
                   if best and e_substrate is not None and e_molecule is not None else None)
        row.update(precomplex_binding_eV=binding, **site_fields(best))
        rows.append(row)
    return clean({"candidate_id": spec["id"], "name": spec.get("name"), "smiles": spec["smiles"],
                  "template": spec["template"], "role": spec.get("role", "inhibitor"),
                  "method": config["method"], "mass_amu": molecular_mass(spec), "sites": rows,
                  "summary": summarize(rows),
                  "limitation": "Single-method barriers on finite models; not a measured rate."
                                + (" Addition step only; the dehydration to the imine is not computed."
                                   if spec["template"] in CONDENSATION else "")})


def summarize(rows):
    out = {}
    for material in ("SiO2", "SiNx"):
        valid = [r for r in rows if r["material"] == material and r["status"] == "valid"]
        total = sum(r["population"] for r in rows if r["material"] == material)
        if not valid:
            out[material] = None
            continue
        w = np.array([r["population"] for r in valid], float)
        ea = np.array([r["Ea_eV"] for r in valid])
        out[material] = {"Ea_mean_eV": float(np.average(ea, weights=w)), "Ea_min_eV": float(ea.min()),
                         "Ea_max_eV": float(ea.max()), "coverage": float(w.sum() / total) if total else 0.}
    if out.get("SiO2") and out.get("SiNx"):
        out["dEa_GS_minus_NGS_eV"] = out["SiO2"]["Ea_mean_eV"] - out["SiNx"]["Ea_mean_eV"]
    return out


# --- rate theory -------------------------------------------------------------

def sackur_tetrode(mass_amu, T, P_Pa):
    """Translational entropy of an ideal gas (eV/K per molecule)."""
    m = mass_amu * 1.66053906660e-27
    kB = 1.380649e-23; h = 6.62607015e-34
    v = kB * T / P_Pa
    lam3 = (h ** 2 / (2 * np.pi * m * kB * T)) ** 1.5
    return KB_EV * (np.log(v / lam3) + 2.5)


def precomplex_fraction(e_bind_eV, mass_amu, T, P_Pa):
    """Langmuir occupancy of the pre-reaction complex.

    Adsorbate entropy follows Campbell & Sellers (JACS 2012, 134, 18109):
    S_ads ~ 0.70 S_gas - 3.3 R, using translational S_gas as a lower bound.
    """
    s_gas = sackur_tetrode(mass_amu, T, P_Pa)
    s_ads = 0.70 * s_gas - 3.3 * KB_EV
    dG = e_bind_eV - T * (s_ads - s_gas)
    return float(1 / (1 + np.exp(np.clip(dG / (KB_EV * T), -700, 700))))


def rate_constant(Ea_eV, T, occupancy=1.0):
    return float(occupancy * KB_EV * T / H_EVS * np.exp(-Ea_eV / (KB_EV * T)))


def capture_probability(Ea_eV, T, dose_s, occupancy=1.0):
    return float(-np.expm1(-rate_constant(Ea_eV, T, occupancy) * dose_s))


def molecular_mass(spec):
    mol = Chem.AddHs(Chem.MolFromSmiles(spec["smiles"]))
    return float(sum(atomic_masses[a.GetAtomicNum()] for a in mol.GetAtoms()))
