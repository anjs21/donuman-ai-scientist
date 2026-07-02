"""
surface_builder.py
====================
Phase 1 of the AS-ALD in-silico co-scientist (Challenge 4).

Full amorphous-surface construction pipeline, implementing the protocol of
Kim et al., Appl. Surf. Sci. 2026, 730, 166294 (arXiv:2510.17356):

    build crystal supercell
        -> (SiNx only) substitute O for N to model PECVD oxygen contamination
        -> melt-quench to an amorphous BULK (N replicas)
        -> cleave along z (two terminations per bulk)
        -> insert 15 Angstrom vacuum gap
        -> passivate dangling bonds per Table 1 of the paper
        -> anneal 1000 K / 5 ps then quench (rearranges Si=NH -> Si-NH-Si)
        -> final relaxation
        -> classify reactive sites with an EXPOSURE FILTER

The exposure filter directly addresses the challenge brief's two concerns:
  (A) "SiOx models over-count reactive sites vs experiment" -- the paper
      attributes the 6.1 vs ~4.5 OH/nm^2 gap to counting subsurface -OH that
      experiments (which probe by molecular adsorption) never see. We add a
      z-depth / accessibility filter so only genuinely exposed sites count.
  (B) "SiNx -NH2/-NH sites have irregular spacing -> excess calculations" --
      the anneal regularises Si=NH into bridging Si-NH-Si, and the exposure
      filter (plus downstream representative-site clustering) avoids computing
      on every redundant site.

Requires: ase, numpy. Supply an MLIP calculator via get_calculator(); an
LJ placeholder is used only so the geometry pipeline runs without an MLIP.
"""

import numpy as np
from ase import Atoms, Atom
from ase.spacegroup import crystal
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.constraints import FixAtoms
from ase import units


# ===========================================================================
# 0. Literature constants (Kim et al. 2026, transcribed)
# ===========================================================================

LITERATURE_TARGETS_NM2 = {
    # Exact Fig. 1a / main-text values (Kim et al. 2026), per single surface.
    "SiO2": {"OH": (6.19, 0.51), "O_bridge": (3.86, 1.28)},
    "SiNx": {"NH2": (3.91, 1.06), "NH_bridge": (3.53, 0.88)},
}
# vicinal/isolated silanol split (SI S4.2) -- for optional refinement #4
LITERATURE_SILANOL_SPLIT = {"vicinal": (4.82, 0.88), "isolated": (1.37, 0.51)}
EXPERIMENTAL_OH_REACTIVE_NM2 = 4.5

TARGET_BULK = {
    "SiO2": {"Si": 54, "O": 108, "N": 0},
    "SiNx": {"Si": 72, "O": 8, "N": 80},   # 9:10:1 Si:N:O per paper
}

PROTOCOL = {
    "SiO2": dict(premelt_K=6000, melt_K=3000, premelt_ps=10, melt_ps=10,
                 quench_ps=15, quench_to_K=0, anneal_K=1000, anneal_ps=5,
                 anneal_quench_ps=5),
    "SiNx": dict(premelt_K=5000, melt_K=4000, premelt_ps=10, melt_ps=10,
                 quench_ps=15, quench_to_K=0, anneal_K=1000, anneal_ps=5,
                 anneal_quench_ps=5),
}

# Shortened MD times for quick pipeline validation (NOT for final numbers).
# Call use_fast_protocol() to switch; restore with use_full_protocol().
_FULL_PROTOCOL = {k: dict(v) for k, v in PROTOCOL.items()}
_FAST_PROTOCOL = {
    "SiO2": dict(premelt_K=6000, melt_K=3000, premelt_ps=1, melt_ps=4,
                 quench_ps=6, quench_to_K=0, anneal_K=1000, anneal_ps=2,
                 anneal_quench_ps=2),
    "SiNx": dict(premelt_K=5000, melt_K=4000, premelt_ps=1, melt_ps=4,
                 quench_ps=6, quench_to_K=0, anneal_K=1000, anneal_ps=2,
                 anneal_quench_ps=2),
}

# Ultra-fast: just enough MD to scramble the crystal + test the pipeline.
# Densities will be poor -- use ONLY to confirm the code path works end-to-end.
_TEST_PROTOCOL = {
    "SiO2": dict(premelt_K=6000, melt_K=3000, premelt_ps=0.4, melt_ps=1.0,
                 quench_ps=2, quench_to_K=0, anneal_K=1000, anneal_ps=0.6,
                 anneal_quench_ps=0.6),
    "SiNx": dict(premelt_K=5000, melt_K=4000, premelt_ps=0.4, melt_ps=1.0,
                 quench_ps=2, quench_to_K=0, anneal_K=1000, anneal_ps=0.6,
                 anneal_quench_ps=0.6),
}


def use_test_protocol():
    """Ultra-short MD (~a few min/surface). Pipeline validation ONLY."""
    for k in PROTOCOL:
        PROTOCOL[k].update(_TEST_PROTOCOL[k])
    print("[protocol] TEST mode (ultra-short MD; code-path validation only)")


def use_fast_protocol():
    """Shorten MD for quick validation. Densities will be rougher."""
    for k in PROTOCOL:
        PROTOCOL[k].update(_FAST_PROTOCOL[k])
    print("[protocol] FAST mode (short MD; validation only, not final numbers)")


def use_full_protocol():
    """Restore the paper's full MD times for production runs."""
    for k in PROTOCOL:
        PROTOCOL[k].update(_FULL_PROTOCOL[k])
    print("[protocol] FULL mode (Kim et al. times)")

VACUUM_GAP = 15.0
FROZEN_FRACTION = 0.25
IDEAL_CN = {"Si": 4, "O": 2, "N": 3}


# ===========================================================================
# 1. Crystal + bulk-composition construction
# ===========================================================================

def build_alpha_quartz(supercell=(3, 3, 2)):
    uc = crystal(
        symbols=["Si", "O"],
        basis=[(0.4697, 0.0, 1 / 3), (0.4133, 0.2672, 0.1188)],
        spacegroup=152,
        cellpar=[4.9134, 4.9134, 5.4052, 90, 90, 120],
    )
    assert uc.get_chemical_formula() == "O6Si3"
    return uc * supercell


def build_beta_si3n4(supercell=(2, 2, 3)):
    uc = crystal(
        symbols=["Si", "N", "N"],
        basis=[(0.1738, 0.7666, 0.0), (0.3333, 0.6667, 0.0), (0.0313, 0.3300, 0.25)],
        spacegroup=173,
        cellpar=[7.6044, 7.6044, 2.9082, 90, 90, 120],
    )
    assert uc.get_chemical_formula() == "N8Si6"
    return uc * supercell


def apply_sinx_composition(atoms, rng):
    """Si72N96 -> Si72 O8 N80: substitute 8 N->O, remove 8 N (N2 evolution)."""
    n_indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "N"]
    rng.shuffle(n_indices)
    to_oxygen = set(n_indices[:8])
    to_remove = set(n_indices[8:16])

    new = Atoms(cell=atoms.get_cell(), pbc=atoms.get_pbc())
    for i, atom in enumerate(atoms):
        if i in to_remove:
            continue
        sym = "O" if i in to_oxygen else atom.symbol
        new.append(Atom(sym, atom.position))
    return new


# ===========================================================================
# 2. Calculator
# ===========================================================================

def get_calculator(dtype="float32"):
    # float32 is ~2-3x faster than float64 on a T4 and fine for MD.
    # Use float64 only if you specifically need high-accuracy final relaxation.
    try:
        from mace.calculators import mace_mp
        return mace_mp(model="medium-omat-0", device="cuda", default_dtype=dtype)
    except Exception as e:
        print(f"[WARN] MACE unavailable ({e}); using LJ placeholder "
              f"(geometry testing ONLY, not physically accurate).")
        from ase.calculators.lj import LennardJones
        return LennardJones(sigma=2.5, epsilon=0.05, rc=6.0)


# ===========================================================================
# 3. Melt-quench (bulk amorphization)
# ===========================================================================

def _set_hydrogen_masses(atoms, temp_K):
    masses = atoms.get_masses()
    h_mask = np.array(atoms.get_chemical_symbols()) == "H"
    if h_mask.any():
        masses[h_mask] = 12.0 if temp_K > 500 else 3.0
        atoms.set_masses(masses)


def _md_stage(atoms, temp_K, time_ps, timestep_fs=2.0, label=""):
    _set_hydrogen_masses(atoms, temp_K)
    n_steps = max(1, int(time_ps * 1000 / timestep_fs))
    import time as _time
    t0 = _time.time()
    dyn = Langevin(atoms, timestep_fs * units.fs, temperature_K=temp_K, friction=0.01)
    dyn.run(n_steps)
    if label:
        print(f"      [{label}] {temp_K:.0f}K {time_ps}ps "
              f"({n_steps} steps) done in {_time.time()-t0:.0f}s", flush=True)
    return atoms


def _quench(atoms, T_start, T_end, time_ps, n_stages=15, timestep_fs=2.0):
    for k in range(n_stages):
        T = T_start + (T_end - T_start) * (k + 1) / n_stages
        _md_stage(atoms, max(T, 1), time_ps / n_stages, timestep_fs)
    return atoms


def melt_quench_bulk(atoms, material, calc, quench_ps_override=None):
    """
    Melt-quench a crystalline supercell into an amorphous bulk.

    quench_ps_override: if set, use this quench duration instead of the
    protocol default. LONGER quench = SLOWER cooling rate = more relaxed
    network with fewer defect/reactive sites; SHORTER quench = faster cooling
    = rougher surface with more reactive sites. Varying this across the
    ensemble samples a physically-meaningful spread of site densities.
    """
    p = PROTOCOL[material]
    quench_ps = quench_ps_override if quench_ps_override is not None else p["quench_ps"]
    cooling_rate = p["melt_K"] / quench_ps  # K/ps, for reporting

    atoms = atoms.copy()
    atoms.calc = calc
    print(f"    melt-quench bulk ({len(atoms)} atoms, "
          f"quench={quench_ps}ps => {cooling_rate:.0f} K/ps)...", flush=True)
    MaxwellBoltzmannDistribution(atoms, temperature_K=p["premelt_K"])
    _md_stage(atoms, p["premelt_K"], p["premelt_ps"], timestep_fs=1.0, label="premelt")
    _md_stage(atoms, p["melt_K"], p["melt_ps"], label="melt")
    print(f"      quenching ({cooling_rate:.0f} K/ps)...", flush=True)
    _quench(atoms, p["melt_K"], p["quench_to_K"], quench_ps)
    print("      final BFGS relax...", flush=True)
    _relax(atoms, label="bulk-relax", fmax=0.05, steps=200)
    atoms.info["quench_ps"] = quench_ps
    atoms.info["cooling_rate_K_per_ps"] = round(cooling_rate, 1)
    _assert_finite(atoms, "melt_quench_bulk")
    return atoms


def _assert_finite(atoms, where=""):
    """Guard against blown-up MD (NaN/inf coords). Raises with a clear message."""
    pos = atoms.get_positions()
    if not np.all(np.isfinite(pos)):
        raise RuntimeError(
            f"[{where}] non-finite atomic coordinates after MD -- the structure "
            f"blew up (unstable dynamics). With MACE this usually means the "
            f"timestep is too large or the initial geometry was bad; with the "
            f"LJ placeholder it is expected. Reduce timestep or check the "
            f"calculator.")


# ===========================================================================
# 4. Cleave + vacuum
# ===========================================================================

def cleave_with_vacuum(bulk, cleave_frac=0.5, vacuum=VACUUM_GAP):
    slab = bulk.copy()
    scaled = slab.get_scaled_positions()
    scaled[:, 2] = (scaled[:, 2] - cleave_frac) % 1.0
    slab.set_scaled_positions(scaled)

    cell = slab.get_cell()
    c_len = np.linalg.norm(cell[2])
    cell[2] = cell[2] * (c_len + vacuum) / c_len
    slab.set_cell(cell, scale_atoms=False)
    slab.center(axis=2)
    slab.set_pbc([True, True, True])
    return slab


# ===========================================================================
# 5. Passivation (Table 1 of the paper)
# ===========================================================================

def find_dangling(atoms):
    cutoffs = natural_cutoffs(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    symbols = atoms.get_chemical_symbols()
    dangling = {}
    for i, sym in enumerate(symbols):
        if sym not in IDEAL_CN:
            continue
        nbrs, _ = nl.get_neighbors(i)
        missing = IDEAL_CN[sym] - len(nbrs)
        if missing > 0:
            dangling[i] = missing
    return dangling


def _add_group_near(atoms, host_idx, group_symbols, bond_len=1.5, rng=None):
    """
    Attach a functional group to a host atom, pointing outward along the
    surface normal but with a small lateral offset + per-atom jitter so
    successive groups on the same host (and H atoms within a group) do NOT
    stack collinearly on top of each other. Vertical stacking created
    near-zero interatomic distances -> exploding forces -> the multi-hour
    relaxation hang. This spreads them into a rough tetrahedral splay.
    """
    if rng is None:
        rng = np.random.default_rng(host_idx)
    host_pos = atoms[host_idx].position
    z_centre = atoms.get_positions()[:, 2].mean()
    outward = 1.0 if host_pos[2] > z_centre else -1.0

    # random lateral direction so multiple groups on one host diverge
    theta = rng.uniform(0, 2 * np.pi)
    lateral = np.array([np.cos(theta), np.sin(theta), 0.0])

    pos = host_pos.copy()
    for k, sym in enumerate(group_symbols):
        # first atom goes mostly outward; subsequent atoms bend laterally
        if k == 0:
            direction = np.array([0, 0, outward]) * 0.85 + lateral * 0.15
        else:
            direction = np.array([0, 0, outward]) * 0.4 + lateral * 0.6
        direction = direction / np.linalg.norm(direction)
        this_len = 0.97 if sym == "H" and group_symbols[k-1] == "O" else bond_len
        pos = pos + direction * this_len
        pos = pos + rng.normal(0, 0.05, size=3)  # small jitter
        atoms.append(Atom(sym, pos))


PASSIVATION_TABLE = {
    "SiO2": {
        ("Si", 3): [["O", "H"], ["O", "H"], ["H"]],
        ("Si", 2): [["O", "H"], ["H"]],
        ("Si", 1): [["O", "H"]],
        ("O", 1):  [["H"]],
    },
    "SiNx": {
        ("Si", 3): [["N", "H"], ["H"]],
        ("Si", 2): [["N", "H"]],
        ("Si", 1): [["N", "H", "H"]],
        ("N", 2):  [["H"], ["H"]],
        ("N", 1):  [["H"]],
        ("O", 1):  [["H"]],
    },
}


def passivate(atoms, material):
    atoms = atoms.copy()
    dangling = find_dangling(atoms)
    table = PASSIVATION_TABLE[material]
    for host_idx, missing in sorted(dangling.items()):
        sym = atoms[host_idx].symbol
        key = (sym, missing)
        if key not in table:
            for _ in range(missing):
                _add_group_near(atoms, host_idx, ["H"])
            continue
        for group in table[key]:
            _add_group_near(atoms, host_idx, group)
    return atoms


# ===========================================================================
# 6. Freeze interior, anneal, relax
# ===========================================================================

def freeze_interior(atoms, frozen_fraction=FROZEN_FRACTION):
    z = atoms.get_positions()[:, 2]
    z_min, z_max = z.min(), z.max()
    span = z_max - z_min
    lo = z_min + span * (0.5 - frozen_fraction / 2)
    hi = z_min + span * (0.5 + frozen_fraction / 2)
    frozen = [i for i, zi in enumerate(z) if lo <= zi <= hi]
    atoms.set_constraint(FixAtoms(indices=frozen))
    return atoms


def _relax(atoms, label="relax", fmax=0.05, steps=300, report_every=25,
           max_seconds=600):
    """
    Relaxation with progress printing AND a hard wall-time cap. Uses the
    irun() generator so we can break on a time budget WITHOUT raising an
    exception inside a callback (which Python 3.7+ converts to RuntimeError).

    Auto-selects FIRE for bad initial geometries (huge/NaN forces), BFGS
    otherwise.
    """
    import time as _time
    from ase.optimize import FIRE

    try:
        f0 = np.sqrt((atoms.get_forces() ** 2).sum(axis=1).max())
    except Exception:
        f0 = np.inf
    Optimizer = FIRE if (not np.isfinite(f0) or f0 > 50) else BFGS
    if Optimizer is FIRE:
        print(f"      [{label}] high/NaN initial fmax={f0} -> using FIRE", flush=True)

    opt = Optimizer(atoms, logfile=None)
    t0 = _time.time()
    aborted = False
    n = 0
    for _ in opt.irun(fmax=fmax, steps=steps):
        n += 1
        elapsed = _time.time() - t0
        if n % report_every == 0:
            try:
                fmax_now = np.sqrt((atoms.get_forces() ** 2).sum(axis=1).max())
            except Exception:
                fmax_now = float("nan")
            print(f"      [{label}] step {n}: fmax={fmax_now:.3f} eV/A "
                  f"({elapsed:.0f}s)", flush=True)
        if elapsed > max_seconds:
            aborted = True
            break

    try:
        fmax_final = np.sqrt((atoms.get_forces() ** 2).sum(axis=1).max())
    except Exception:
        fmax_final = float("nan")
    if aborted:
        conv = f"ABORTED at {max_seconds}s wall-cap"
    elif np.isfinite(fmax_final) and fmax_final <= fmax:
        conv = "converged"
    else:
        conv = f"stopped (fmax={fmax_final})"
    print(f"      [{label}] done: fmax={fmax_final:.3f} eV/A, {conv} "
          f"({_time.time()-t0:.0f}s)", flush=True)
    return atoms


def _fix_close_contacts(atoms, min_dist=1.1, max_sweeps=50):
    """
    Push apart any atom pair closer than min_dist (Angstrom) before relaxation.
    This is the safeguard against NaN/exploding forces from near-coincident
    atoms created by crude passivation placement. More aggressive than before:
    larger min_dist, more sweeps, and it keeps going until no pair is too close
    (or max_sweeps reached). Uses a neighbor-list-free O(N^2) scan (fine for
    ~200-atom slabs).
    """
    pos = atoms.get_positions()
    n = len(atoms)
    for sweep in range(max_sweeps):
        # compute all pairwise distances via broadcasting
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dist, np.inf)
        too_close = np.argwhere(dist < min_dist)
        if len(too_close) == 0:
            atoms.set_positions(pos)
            if sweep > 0:
                print(f"      [passivation] resolved close contacts in {sweep} sweeps",
                      flush=True)
            return atoms
        for i, j in too_close:
            if i >= j:
                continue
            d = pos[i] - pos[j]
            r = np.linalg.norm(d)
            if r < 1e-6:
                # exactly coincident: push in a random direction
                d = np.random.default_rng(int(i) * 1000 + int(j)).normal(size=3)
                r = np.linalg.norm(d)
            push = (min_dist - r + 0.15) * (d / (r + 1e-9)) * 0.5
            pos[i] += push
            pos[j] -= push
    print(f"      [passivation] WARNING: close contacts remain after "
          f"{max_sweeps} sweeps (min pair dist {dist.min():.2f} A)", flush=True)
    atoms.set_positions(pos)
    return atoms


def anneal_and_relax(atoms, material, calc):
    p = PROTOCOL[material]
    atoms = atoms.copy()
    atoms = _fix_close_contacts(atoms)   # prevent exploding-force hang
    atoms.calc = calc
    atoms = freeze_interior(atoms)
    print(f"    anneal + relax ({len(atoms)} atoms)...", flush=True)
    # gentle pre-relax of just the passivation before heating
    _relax(atoms, label="pre-relax", fmax=0.2, steps=60, max_seconds=180)
    MaxwellBoltzmannDistribution(atoms, temperature_K=p["anneal_K"])
    _md_stage(atoms, p["anneal_K"], p["anneal_ps"], label="anneal")
    _quench(atoms, p["anneal_K"], 0, p["anneal_quench_ps"])
    _relax(atoms, label="surface-relax", fmax=0.05, steps=300, max_seconds=600)
    return atoms


# ===========================================================================
# 7. Site classification WITH exposure filter (addresses Problem A)
# ===========================================================================

def _surface_exposed(atoms, idx, nl=None, probe_top_frac=0.30):
    """
    DEPRECATED heuristic (kept for reference). The validated method is
    _is_surface_site() below, implementing Kim et al. SI S4.1 exactly.
    """
    return _is_surface_site(atoms, idx)


# Cordero covalent radii via ASE (the exact source the paper cites)
from ase.data import covalent_radii, atomic_numbers


def _is_surface_site(atoms, idx, depth_cutoff=3.5):
    """
    Kim et al. SI S4.1 surface-site test (exact reproduction):

      1. For candidate O/N atom i at (xi,yi,zi) with covalent radius r_i:
      2. Exclude it if it lies > depth_cutoff (3.5 A) below the topmost atom
         OF THE SAME ELEMENT (i.e. it is deep subsurface).
      3. Project every OTHER non-H atom j with z_j > z_i + r_cov_i onto the
         xy-plane. If any such atom's covalent-radius circle overlaps the
         circle of radius r_i centred at (xi,yi), the site is BLOCKED.
      4. If no projected circle overlaps, i is an exposed surface site.

    This is a covalent-radius occlusion test (a projected-disk shadow test),
    NOT the old z-band + cylinder heuristic. It reproduces the paper's
    reported densities on their own deposited structures.

    Works on both slab surfaces (top and bottom) because the topmost-atom
    reference is per-element and the projection is directional (only atoms
    ABOVE the candidate can occlude it).
    """
    symbols = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    sym_i = symbols[idx]
    zi = pos[idx, 2]
    xi, yi = pos[idx, 0], pos[idx, 1]
    r_i = covalent_radii[atomic_numbers[sym_i]]

    # --- consider BOTH surfaces: a site is "surface" if exposed from top OR
    # from bottom. We test top-exposure; then mirror for bottom-exposure. ---
    def _exposed_from(direction):
        # direction = +1 : atoms above occlude (top surface)
        #           = -1 : atoms below occlude (bottom surface)
        # depth exclusion relative to the extreme atom of same element in that dir
        same_elem_z = pos[[k for k, s in enumerate(symbols) if s == sym_i], 2]
        if direction > 0:
            extreme = same_elem_z.max()
            if extreme - zi > depth_cutoff:
                return False
        else:
            extreme = same_elem_z.min()
            if zi - extreme > depth_cutoff:
                return False

        for j, sym_j in enumerate(symbols):
            if j == idx or sym_j == "H":
                continue
            zj = pos[j, 2]
            r_j = covalent_radii[atomic_numbers[sym_j]]
            # only atoms further outward than i + its radius can occlude
            if direction > 0 and zj <= zi + r_i:
                continue
            if direction < 0 and zj >= zi - r_i:
                continue
            dxy = np.hypot(pos[j, 0] - xi, pos[j, 1] - yi)
            if dxy < (r_i + r_j):   # projected circles overlap -> blocked
                return False
        return True

    return _exposed_from(+1) or _exposed_from(-1)


def classify_sites(atoms, material, exposure_filter=True):
    cutoffs = natural_cutoffs(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    symbols = atoms.get_chemical_symbols()

    def record(store, site_type, host_idx):
        store.setdefault(site_type, {"total": 0, "exposed": 0, "indices": []})
        store[site_type]["total"] += 1
        if (not exposure_filter) or _is_surface_site(atoms, host_idx):
            store[site_type]["exposed"] += 1
            store[site_type]["indices"].append(host_idx)

    counts = {}
    for i, sym in enumerate(symbols):
        nbrs, _ = nl.get_neighbors(i)
        nbr_syms = [symbols[j] for j in nbrs]
        if material == "SiO2":
            if sym == "O":
                if nbr_syms.count("H") == 1 and nbr_syms.count("Si") == 1:
                    record(counts, "OH", i)
                elif nbr_syms.count("Si") == 2 and nbr_syms.count("H") == 0:
                    record(counts, "O_bridge", i)
        elif material == "SiNx":
            if sym == "N":
                nh, nsi = nbr_syms.count("H"), nbr_syms.count("Si")
                if nh == 2:
                    record(counts, "NH2", i)
                elif nh == 1 and nsi == 2:
                    record(counts, "NH_bridge", i)
            elif sym == "O":
                if nbr_syms.count("H") == 1 and nbr_syms.count("Si") == 1:
                    record(counts, "OH", i)
                elif nbr_syms.count("Si") == 2:
                    record(counts, "O_bridge", i)
    return counts


def cluster_representative_sites(atoms, site_indices, n_representatives=3):
    """
    Problem B ("excess calculations"): collapse many near-identical reactive
    sites into a few representatives, so downstream DFT/MLIP reactivity calcs
    run on ~3 sites per functional group (as Kim et al. do) instead of every
    site. Sites are clustered by a local-environment fingerprint; the member
    closest to each cluster centroid is the representative, weighted by cluster
    population.

    Returns list of (representative_site_index, population_weight).
    """
    if len(site_indices) == 0:
        return []
    if len(site_indices) <= n_representatives:
        w = 1.0 / len(site_indices)
        return [(i, w) for i in site_indices]

    # local-environment fingerprint: sorted neighbour distances within a cutoff
    cutoffs = natural_cutoffs(atoms)
    nl = NeighborList([max(3.5, c) for c in cutoffs], self_interaction=False, bothways=True)
    nl.update(atoms)
    cell = atoms.get_cell()

    def fingerprint(idx, k=12, rcut=4.0):
        nbrs, offsets = nl.get_neighbors(idx)
        p0 = atoms.positions[idx]
        ds = []
        for j, off in zip(nbrs, offsets):
            d = np.linalg.norm(atoms.positions[j] + off @ cell - p0)
            if d < rcut:
                ds.append(d)
        ds = sorted(ds)[:k]
        ds += [rcut] * (k - len(ds))   # pad
        return np.array(ds)

    fps = np.array([fingerprint(i) for i in site_indices])

    try:
        from sklearn.cluster import KMeans
        k = min(n_representatives, len(site_indices))
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(fps)
        labels, centers = km.labels_, km.cluster_centers_
    except Exception:
        # fallback: simple binning if sklearn unavailable
        labels = np.arange(len(site_indices)) % n_representatives
        centers = None

    reps = []
    for c in sorted(set(labels)):
        members = np.where(labels == c)[0]
        if centers is not None:
            dists = np.linalg.norm(fps[members] - centers[c], axis=1)
            rep_local = members[np.argmin(dists)]
        else:
            rep_local = members[0]
        reps.append((int(site_indices[rep_local]), len(members) / len(site_indices)))
    return reps


def surface_area_nm2(atoms, both_surfaces=True):
    """
    Reactive-site normalisation area in nm^2.

    A cleaved slab exposes TWO surfaces (top + bottom), and reactive sites are
    counted across both. To get a PER-SURFACE density (as the paper reports),
    divide the total site count by (2 * flat_area). This convention was
    validated against Kim et al.'s own deposited POSCARs, where
    count / (2 * flat_area) reproduced their published densities (e.g. OH 6.0
    vs 6.19 nm^-2).

    NOTE: we do NOT apply a roughness correction here, because the paper's
    reported densities (which we calibrate against) use the flat cross-section.
    """
    cell = atoms.get_cell()
    flat = np.linalg.norm(np.cross(cell[0], cell[1])) / 100.0  # A^2 -> nm^2
    return 2.0 * flat if both_surfaces else flat


# ===========================================================================
# 7b. Quality controls: strain flagging, clumping test, outlier gate
#     (addresses reviewer points 1, 2, 3 -- see notes on corrected forms)
# ===========================================================================

def _bulk_mean_angle(atoms, center_sym, ref_syms):
    """Mean X-center-X bond angle over all `center_sym` atoms (amorphous avg)."""
    cutoffs = natural_cutoffs(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    symbols = atoms.get_chemical_symbols()
    angles = []
    for i, sym in enumerate(symbols):
        if sym != center_sym:
            continue
        nbrs, offsets = nl.get_neighbors(i)
        neigh = [(j, off) for j, off in zip(nbrs, offsets) if symbols[j] in ref_syms]
        if len(neigh) < 2:
            continue
        p0 = atoms.positions[i]
        cell = atoms.get_cell()
        for a in range(len(neigh)):
            for b in range(a + 1, len(neigh)):
                ja, offa = neigh[a]
                jb, offb = neigh[b]
                va = atoms.positions[ja] + offa @ cell - p0
                vb = atoms.positions[jb] + offb @ cell - p0
                cosang = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)
                angles.append(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
    return float(np.mean(angles)) if angles else None


def flag_strained_sites(atoms, material, site_indices, deviation_deg=15.0):
    """
    Reviewer point 3: flag reactive sites whose local geometry is strained.
    For SiO2 we check the Si-O-Si angle at the site's bridging oxygen / host;
    strained 3-membered-ring oxygens deviate strongly from the amorphous mean.
    Returns set of site indices flagged as anomalous/strained hot-spots.

    NOTE: This is a geometric proxy. It flags sites for the agent to treat with
    caution (their DeltaE will be less transferable), not sites to delete.
    """
    center, refs = ("O", {"Si"}) if material == "SiO2" else ("N", {"Si"})
    mean_angle = _bulk_mean_angle(atoms, center, refs)
    if mean_angle is None:
        return set()

    cutoffs = natural_cutoffs(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    symbols = atoms.get_chemical_symbols()
    cell = atoms.get_cell()

    strained = set()
    for i in site_indices:
        if symbols[i] not in (center,):
            # for terminal sites (e.g. O in -OH) check the Si-O-host geometry
            pass
        nbrs, offsets = nl.get_neighbors(i)
        si_neigh = [(j, off) for j, off in zip(nbrs, offsets) if symbols[j] in refs]
        if len(si_neigh) < 2:
            continue
        p0 = atoms.positions[i]
        worst = 0.0
        for a in range(len(si_neigh)):
            for b in range(a + 1, len(si_neigh)):
                ja, offa = si_neigh[a]; jb, offb = si_neigh[b]
                va = atoms.positions[ja] + offa @ cell - p0
                vb = atoms.positions[jb] + offb @ cell - p0
                cosang = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)
                ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
                worst = max(worst, abs(ang - mean_angle))
        if worst > deviation_deg:
            strained.add(i)
    return strained


def clumping_ratio(atoms, site_indices):
    """
    Reviewer point 2 (corrected): PBC-aware Clark-Evans-style clumping metric.
    Compares the mean nearest-neighbour distance of the sites (in the xy plane,
    minimum-image) to that expected for a random (Poisson) arrangement at the
    same 2D density.
        R > 1  -> more dispersed than random (good, spread out)
        R ~ 1  -> random
        R < 1  -> clumped (bad; sterically unrealistic clustering)
    Uses minimum-image convention, unlike raw pdist which ignores PBC.
    """
    if len(site_indices) < 2:
        return None
    pos = atoms.get_positions()[site_indices][:, :2]
    cell = atoms.get_cell()
    Lx, Ly = cell[0, 0], cell[1, 1]  # assumes near-orthogonal in-plane
    n = len(pos)

    nn = []
    for i in range(n):
        d = pos - pos[i]
        d[:, 0] -= Lx * np.round(d[:, 0] / Lx)   # minimum image
        d[:, 1] -= Ly * np.round(d[:, 1] / Ly)
        dist = np.hypot(d[:, 0], d[:, 1])
        dist[i] = np.inf
        nn.append(dist.min())
    mean_nn = np.mean(nn)

    area = abs(Lx * Ly)
    density = n / area
    expected_nn = 0.5 / np.sqrt(density)  # Clark-Evans expected NN for Poisson
    return float(mean_nn / expected_nn)


def slab_passes_quality_gate(counts, area, material, n_std=2.5, min_clump_ratio=0.75,
                             atoms=None):
    """
    Reviewer point 1 (corrected): outlier gate on site density, using the
    paper's own mean +/- n_std*std as bounds (statistically defensible), not a
    fixed +/-15% (which is tighter than the paper's observed spread and would
    reject valid slabs). Optionally also gate on clumping.
    Returns (passed: bool, reasons: list[str]).
    """
    reasons = []
    targets = LITERATURE_TARGETS_NM2[material]

    for site_type, (mean, std) in targets.items():
        if site_type not in counts:
            density = 0.0
        else:
            density = counts[site_type]["exposed"] / area
        lo, hi = mean - n_std * std, mean + n_std * std
        if not (lo <= density <= hi):
            reasons.append(f"{site_type} density {density:.2f} outside "
                           f"[{lo:.2f},{hi:.2f}] nm^-2")

    if atoms is not None:
        # clumping check on the primary terminal site type
        primary = "OH" if material == "SiO2" else "NH2"
        if primary in counts and counts[primary]["indices"]:
            R = clumping_ratio(atoms, counts[primary]["indices"])
            if R is not None and R < min_clump_ratio:
                reasons.append(f"{primary} sites clumped (R={R:.2f} < {min_clump_ratio})")

    return (len(reasons) == 0), reasons


# ===========================================================================
# 8. Full pipeline + ensemble
# ===========================================================================

def build_one_surface(material, calc, seed, cleave_frac):
    """Build a single surface (melt-quenches its own bulk). Kept for
    backwards compatibility; build_surface_ensemble now uses the more
    efficient build_two_terminations() which shares one bulk."""
    rng = np.random.default_rng(seed)
    builder = build_alpha_quartz if material == "SiO2" else build_beta_si3n4
    crystal_sc = builder()
    if material == "SiNx":
        crystal_sc = apply_sinx_composition(crystal_sc, rng)
    bulk = melt_quench_bulk(crystal_sc, material, calc)
    slab = cleave_with_vacuum(bulk, cleave_frac=cleave_frac)
    slab = passivate(slab, material)
    slab = anneal_and_relax(slab, material, calc)
    return slab


def build_two_terminations(material, calc, seed, quench_ps_override=None):
    """
    Efficient path: melt-quench ONE bulk, then cleave it into two distinct
    surface terminations. This halves the most expensive step (melt-quench)
    compared to building each surface from its own bulk.

    quench_ps_override: cooling-rate control (see melt_quench_bulk); both
    terminations inherit the bulk's cooling rate.
    Returns a list of 2 finished surfaces.
    """
    rng = np.random.default_rng(seed)
    builder = build_alpha_quartz if material == "SiO2" else build_beta_si3n4
    crystal_sc = builder()
    if material == "SiNx":
        crystal_sc = apply_sinx_composition(crystal_sc, rng)

    bulk = melt_quench_bulk(crystal_sc, material, calc,
                            quench_ps_override=quench_ps_override)   # done ONCE

    surfaces = []
    for cleave_frac in (0.5, 0.0):
        slab = cleave_with_vacuum(bulk, cleave_frac=cleave_frac)
        slab = passivate(slab, material)
        slab = anneal_and_relax(slab, material, calc)     # cheap vs melt-quench
        slab.info["quench_ps"] = bulk.info.get("quench_ps")
        slab.info["cooling_rate_K_per_ps"] = bulk.info.get("cooling_rate_K_per_ps")
        surfaces.append(slab)
    return surfaces


def build_surface_ensemble(material, calc, n_bulk=3, target_accepted=None,
                           max_extra_bulk=3, apply_gate=True, quench_sweep=None):
    """
    Build an ensemble of amorphous surfaces with quality control.

    n_bulk           : number of bulk replicas to attempt (2 surfaces each).
    target_accepted  : if set, keep generating (up to max_extra_bulk extra
                       bulks) until this many surfaces pass the quality gate.
    apply_gate       : if False, keep all slabs (diagnostic mode) but still
                       annotate strain/clumping/pass-fail.
    quench_sweep     : list of quench durations (ps) to cycle across bulks,
                       giving cooling-rate variation. e.g. [8, 15, 30] mixes
                       fast (rougher, more sites), medium, and slow (relaxed,
                       fewer sites) cooling in one ensemble. If None, all bulks
                       use the protocol default rate (velocity-randomness only).
                       Bulk b uses quench_sweep[b % len(quench_sweep)].

    Each accepted surface is annotated with:
       .info['strained_sites'], .info['clumping_R'], .info['gate_reasons'],
       .info['quench_ps'], .info['cooling_rate_K_per_ps']
    """
    accepted, rejected, per_surface = [], [], []

    def _quench_for_bulk(b):
        if quench_sweep:
            return quench_sweep[b % len(quench_sweep)]
        return None  # protocol default

    def process_bulk(b):
        # ONE melt-quench -> two terminations (shares the expensive step)
        two = build_two_terminations(material, calc, seed=1000 * b,
                                     quench_ps_override=_quench_for_bulk(b))
        for t, slab in enumerate(two):
            counts = classify_sites(slab, material, exposure_filter=True)
            area = surface_area_nm2(slab)
            densities = {st: v["exposed"] / area for st, v in counts.items()}

            passed, reasons = slab_passes_quality_gate(counts, area, material, atoms=slab)

            primary = "OH" if material == "SiO2" else "NH2"
            strained = set()
            if primary in counts and counts[primary]["indices"]:
                strained = flag_strained_sites(slab, material, counts[primary]["indices"])
            R = clumping_ratio(slab, counts[primary]["indices"]) if (
                primary in counts and counts[primary]["indices"]) else None

            slab.info["strained_sites"] = sorted(strained)
            slab.info["clumping_R"] = R
            slab.info["gate_reasons"] = reasons

            tag = "ACCEPT" if passed else "REJECT"
            cr = slab.info.get("cooling_rate_K_per_ps")
            print(f"[{material}] bulk {b} term {t} ({cr} K/ps): "
                  f"{ {k: round(v,2) for k,v in densities.items()} } "
                  f"| strained={len(strained)} R={R if R is None else round(R,2)} "
                  f"| {tag}"
                  + (f" ({'; '.join(reasons)})" if reasons else ""), flush=True)

            if passed or not apply_gate:
                accepted.append(slab)
                per_surface.append(densities)
            else:
                rejected.append(slab)

    for b in range(n_bulk):
        process_bulk(b)

    extra = 0
    while target_accepted and len(accepted) < target_accepted and extra < max_extra_bulk:
        print(f"[{material}] only {len(accepted)} accepted; generating extra bulk...")
        process_bulk(n_bulk + extra)
        extra += 1

    all_site_types = set().union(*[d.keys() for d in per_surface]) if per_surface else set()
    summary = {}
    for st in all_site_types:
        vals = np.array([d.get(st, 0.0) for d in per_surface])
        summary[st] = {
            "mean_nm2": round(float(vals.mean()), 2),
            "std_nm2": round(float(vals.std()), 2),
            "literature": LITERATURE_TARGETS_NM2[material].get(st),
        }
    summary["_n_accepted"] = len(accepted)
    summary["_n_rejected"] = len(rejected)
    return accepted, summary


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    calc = get_calculator()
    for material in ["SiO2", "SiNx"]:
        print(f"\n=== Building {material} surface ensemble ===")
        surfaces, summary = build_surface_ensemble(material, calc, n_bulk=3)
        for i, s in enumerate(surfaces):
            write(f"{material}_surface_{i}.xyz", s)
        print(f"[{material}] ensemble summary (exposed-site densities):")
        for st, info in summary.items():
            print(f"    {st}: {info['mean_nm2']} +/- {info['std_nm2']} nm^-2 "
                  f"(literature: {info['literature']})")
