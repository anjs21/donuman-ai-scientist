"""
surface_builder.py
====================
Phase 1 of the AS-ALD in-silico co-scientist (Challenge 4).

Full amorphous-surface construction pipeline, implementing the protocol of
Kim et al., Appl. Surf. Sci. 2026, 730, 166294 (arXiv:2510.17356):

    build crystal supercell
        -> (SiNx only) substitute O for N to model PECVD oxygen contamination
        -> melt-quench to an amorphous BULK (N replicas, cooling-rate control)
        -> cleave along z (two terminations per bulk, sharing one melt-quench)
        -> insert 15 Angstrom vacuum gap
        -> passivate dangling bonds per Table 1 (splayed groups, no stacking)
        -> fix close contacts, anneal 1000 K then quench, final relaxation
        -> classify reactive sites with a covalent-radius occlusion filter

Addresses the challenge brief's two concerns:
  (A) "SiOx models over-count reactive sites vs experiment" -- _is_surface_site()
      reproduces Kim et al. SI S4.1: a covalent-radius projected-disk occlusion
      test so only genuinely exposed sites count.
  (B) "SiNx -NH2/-NH sites have irregular spacing -> excess calculations" --
      the anneal regularises Si=NH into bridging Si-NH-Si, and
      cluster_representative_sites() collapses many near-identical sites into a
      few representatives for downstream reactivity calcs.

Requires: ase, numpy (scipy/sklearn optional). Supply an MLIP calculator via
get_calculator(); an LJ placeholder runs the geometry pipeline without an MLIP.
"""

import os
import json
import hashlib
import numpy as np
from ase import Atoms, Atom
from ase.spacegroup import crystal
from ase.io import write, read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS, FIRE
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.constraints import FixAtoms
from ase.data import covalent_radii, atomic_numbers
from ase import units


# ===========================================================================
# 0. Literature constants (Kim et al. 2026, transcribed)
# ===========================================================================

LITERATURE_TARGETS_NM2 = {
    # Exact Fig. 1a / main-text values (Kim et al. 2026), per single surface.
    "SiO2": {"OH": (6.19, 0.51), "O_bridge": (3.86, 1.28)},
    "SiNx": {"NH2": (3.91, 1.06), "NH_bridge": (3.53, 0.88)},
}
# vicinal/isolated silanol split (SI S4.2) -- for optional refinement
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

BULK_CACHE_DIR = "bulk_cache"
LITERATURE_DIR = "literature"

# Wall-clock seconds between MD progress heartbeats (a long labelled stage that
# keeps emitting heartbeats is alive; one that stops is genuinely stuck). Set to
# 0 to silence per-step MD logging.
MD_HEARTBEAT_SEC = 15.0


# ===========================================================================
# 1. Bulk cache: save/load amorphous bulks to skip melt-quench on reruns
# ===========================================================================

def _cache_key(material, seed, supercell, protocol_params):
    key_data = json.dumps({
        "material": material, "seed": int(seed),
        "supercell": list(supercell),
        "protocol": {k: v for k, v in sorted(protocol_params.items())},
    }, sort_keys=True)
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


def _cache_path(material, seed, supercell, protocol_params, cache_dir=None):
    d = cache_dir or BULK_CACHE_DIR
    os.makedirs(d, exist_ok=True)
    key = _cache_key(material, seed, supercell, protocol_params)
    return os.path.join(d, f"{material}_bulk_{key}.xyz")


def save_bulk_to_cache(atoms, material, seed, supercell, protocol_params, cache_dir=None):
    path = _cache_path(material, seed, supercell, protocol_params, cache_dir)
    write(path, atoms)
    print(f"    [cache] saved bulk -> {path}", flush=True)
    return path


def load_bulk_from_cache(material, seed, supercell, protocol_params, cache_dir=None):
    path = _cache_path(material, seed, supercell, protocol_params, cache_dir)
    if os.path.isfile(path):
        atoms = read(path)
        atoms.set_pbc(True)
        print(f"    [cache] loaded bulk <- {path} ({len(atoms)} atoms)", flush=True)
        return atoms
    return None


def load_published_bulk(material, literature_dir=None):
    """Load a published amorphous bulk/slab (e.g. Kim et al. POSCARs) from the
    literature directory. Matches files whose name starts with the material."""
    d = literature_dir or LITERATURE_DIR
    if not os.path.isdir(d):
        return None
    for fname in sorted(os.listdir(d)):
        base = fname.lower().replace("poscar_", "")
        if not base.startswith(material.lower().replace("x", "")):
            continue
        path = os.path.join(d, fname)
        try:
            atoms = read(path)
        except Exception:
            continue
        atoms.set_pbc(True)
        print(f"    [literature] loaded published bulk <- {path} "
              f"({len(atoms)} atoms)", flush=True)
        return atoms
    return None


def list_cached_bulks(cache_dir=None):
    d = cache_dir or BULK_CACHE_DIR
    if not os.path.isdir(d):
        print("[cache] no cache directory found")
        return []
    files = [f for f in os.listdir(d) if f.endswith(".xyz")]
    print(f"[cache] {len(files)} cached bulks in {d}:")
    for f in sorted(files):
        size = os.path.getsize(os.path.join(d, f))
        print(f"    {f} ({size/1024:.1f} KB)")
    return files


# ===========================================================================
# 2. Crystal + bulk-composition construction
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
    """Substitute a fraction of N -> O and remove some N (N2 evolution) to model
    PECVD oxygen contamination (~9:10:1 Si:N:O)."""
    n_indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "N"]
    rng.shuffle(n_indices)
    n_sub = max(1, len(n_indices) // 12)          # ~8% -> O
    to_oxygen = set(n_indices[:n_sub])
    to_remove = set(n_indices[n_sub:2 * n_sub])   # equal number removed

    new = Atoms(cell=atoms.get_cell(), pbc=atoms.get_pbc())
    for i, atom in enumerate(atoms):
        if i in to_remove:
            continue
        sym = "O" if i in to_oxygen else atom.symbol
        new.append(Atom(sym, atom.position))
    return new


# ===========================================================================
# 3. Calculator
# ===========================================================================

def get_calculator(dtype="float32"):
    """Load MACE-MP-0 (medium-omat-0) on GPU if available, else CPU; LJ fallback
    for geometry-only testing when MACE is missing."""
    try:
        from mace.calculators import mace_mp
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        calc = mace_mp(model="medium-omat-0", device=device, default_dtype=dtype)
        print(f"[calc] MACE-MP-0 loaded on {device} (dtype={dtype})")
        if device == "cpu":
            print("[calc] *** WARNING: running on CPU. MACE MD is ~50-100x slower "
                  "than GPU. Use --mode test/fast for iteration, or a GPU for "
                  "production. ***", flush=True)
        return calc
    except Exception as e:
        print(f"[WARN] MACE unavailable ({e}); using LJ placeholder "
              f"(geometry testing ONLY, not physically accurate).")
        from ase.calculators.lj import LennardJones
        return LennardJones(sigma=2.5, epsilon=0.05, rc=6.0)


# ===========================================================================
# 4. Melt-quench (bulk amorphization) with cooling-rate control
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

    # Time-throttled progress heartbeat so long labelled stages (premelt/melt/
    # anneal) are not silent. Attached every step but only prints once per
    # MD_HEARTBEAT_SEC of wall-clock, so it stays responsive even if each MLIP
    # force call is slow -- the key signal for "slow vs genuinely stuck".
    if label and MD_HEARTBEAT_SEC > 0:
        state = {"last": t0}

        def _hb():
            now = _time.time()
            if now - state["last"] < MD_HEARTBEAT_SEC:
                return
            state["last"] = now
            step = dyn.get_number_of_steps()
            rate = step / (now - t0) if now > t0 else 0.0
            eta = (n_steps - step) / rate if rate > 0 else float("inf")
            try:
                T = atoms.get_temperature()
            except Exception:
                T = float("nan")
            print(f"      [{label}] step {step}/{n_steps} ({100.0*step/n_steps:3.0f}%) "
                  f"T={T:6.0f}K  {rate:4.1f} it/s  ETA {eta:4.0f}s "
                  f"({now-t0:.0f}s)", flush=True)

        dyn.attach(_hb, interval=1)

    dyn.run(n_steps)
    _assert_finite(atoms, label or "md")
    if label:
        print(f"      [{label}] {temp_K:.0f}K {time_ps}ps "
              f"({n_steps} steps) done in {_time.time()-t0:.0f}s", flush=True)
    return atoms


def _quench(atoms, T_start, T_end, time_ps, n_stages=15, timestep_fs=2.0):
    for k in range(n_stages):
        T = T_start + (T_end - T_start) * (k + 1) / n_stages
        _md_stage(atoms, max(T, 1), time_ps / n_stages, timestep_fs)
    return atoms


def _assert_finite(atoms, where=""):
    """Guard against blown-up MD (NaN/inf coords). Raises with a clear message
    instead of a cryptic downstream neighbour-list error."""
    if not np.all(np.isfinite(atoms.get_positions())):
        raise RuntimeError(
            f"[{where}] non-finite atomic coordinates -- the dynamics diverged "
            f"(timestep too large, bad geometry, or the LJ placeholder). Reduce "
            f"the timestep or check the calculator.")


def melt_quench_bulk(atoms, material, calc, seed=0, supercell=(3, 3, 2),
                     use_cache=True, cache_dir=None, quench_ps_override=None):
    """
    Melt-quench a crystalline supercell into an amorphous bulk, with caching.

    quench_ps_override : LONGER quench = SLOWER cooling = more relaxed network
    with fewer reactive sites; SHORTER = faster cooling = rougher surface with
    more reactive sites. Cooling rate is folded into the cache key.
    """
    p = PROTOCOL[material]
    quench_ps = quench_ps_override if quench_ps_override is not None else p["quench_ps"]
    eff_p = dict(p, quench_ps=quench_ps)
    cooling_rate = p["melt_K"] / quench_ps

    if use_cache:
        cached = load_bulk_from_cache(material, seed, supercell, eff_p, cache_dir)
        if cached is not None:
            cached.info.setdefault("quench_ps", quench_ps)
            cached.info.setdefault("cooling_rate_K_per_ps", round(cooling_rate, 1))
            return cached

    atoms = atoms.copy()
    atoms.calc = calc
    print(f"    melt-quench bulk ({len(atoms)} atoms, quench={quench_ps}ps "
          f"=> {cooling_rate:.0f} K/ps)...", flush=True)
    MaxwellBoltzmannDistribution(atoms, temperature_K=p["premelt_K"])
    _md_stage(atoms, p["premelt_K"], p["premelt_ps"], timestep_fs=1.0, label="premelt")
    _md_stage(atoms, p["melt_K"], p["melt_ps"], label="melt")
    print(f"      quenching ({cooling_rate:.0f} K/ps)...", flush=True)
    _quench(atoms, p["melt_K"], p["quench_to_K"], quench_ps)
    print("      final relax...", flush=True)
    _relax(atoms, label="bulk-relax", fmax=0.05, steps=200)
    atoms.info["quench_ps"] = quench_ps
    atoms.info["cooling_rate_K_per_ps"] = round(cooling_rate, 1)
    _assert_finite(atoms, "melt_quench_bulk")

    if use_cache:
        save_bulk_to_cache(atoms, material, seed, supercell, eff_p, cache_dir)
    return atoms


# ===========================================================================
# 5. Cleave + vacuum
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
# 6. Passivation (Table 1 of the paper) -- splayed groups (no stacking)
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
    Attach a functional group pointing outward along the surface normal but with
    a lateral offset + per-atom jitter, so successive groups on the same host
    (and H atoms within a group) do NOT stack collinearly. Vertical stacking
    created near-zero interatomic distances -> exploding forces -> hung relax.
    """
    if rng is None:
        rng = np.random.default_rng(host_idx)
    host_pos = atoms[host_idx].position
    z_centre = atoms.get_positions()[:, 2].mean()
    outward = 1.0 if host_pos[2] > z_centre else -1.0

    theta = rng.uniform(0, 2 * np.pi)
    lateral = np.array([np.cos(theta), np.sin(theta), 0.0])

    pos = host_pos.copy()
    for k, sym in enumerate(group_symbols):
        if k == 0:
            direction = np.array([0, 0, outward]) * 0.85 + lateral * 0.15
        else:
            direction = np.array([0, 0, outward]) * 0.4 + lateral * 0.6
        direction = direction / np.linalg.norm(direction)
        this_len = 0.97 if (sym == "H" and group_symbols[k - 1] == "O") else bond_len
        pos = pos + direction * this_len
        pos = pos + rng.normal(0, 0.05, size=3)
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
            for g in range(missing):
                _add_group_near(atoms, host_idx, ["H"],
                                rng=np.random.default_rng(host_idx * 17 + g))
            continue
        for g, group in enumerate(table[key]):
            _add_group_near(atoms, host_idx, group,
                            rng=np.random.default_rng(host_idx * 17 + g))
    return atoms


# ===========================================================================
# 7. Freeze interior, fix contacts, anneal, relax (with wall-time cap)
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
    Relaxation with progress printing AND a hard wall-time cap. Uses irun() so
    we can break on a time budget without raising inside a callback. Auto-picks
    FIRE for bad initial geometries (huge/NaN forces), BFGS otherwise.
    """
    import time as _time
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
    Push apart any atom pair closer than min_dist (Angstrom) before relaxation --
    the safeguard against NaN/exploding forces from near-coincident atoms created
    by crude passivation placement. O(N^2) scan (fine for ~200-atom slabs).
    """
    pos = atoms.get_positions()
    for sweep in range(max_sweeps):
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
    atoms = _fix_close_contacts(atoms)        # prevent exploding-force hang
    atoms.calc = calc
    atoms = freeze_interior(atoms)
    print(f"    anneal + relax ({len(atoms)} atoms)...", flush=True)
    _relax(atoms, label="pre-relax", fmax=0.2, steps=60, max_seconds=180)
    MaxwellBoltzmannDistribution(atoms, temperature_K=p["anneal_K"])
    _md_stage(atoms, p["anneal_K"], p["anneal_ps"], label="anneal")
    _quench(atoms, p["anneal_K"], 0, p["anneal_quench_ps"])
    _relax(atoms, label="surface-relax", fmax=0.05, steps=300, max_seconds=600)
    return atoms


# ===========================================================================
# 8. Site classification with covalent-radius occlusion filter (Problem A)
# ===========================================================================

def _is_surface_site(atoms, idx, depth_cutoff=3.5):
    """
    Kim et al. SI S4.1 surface-site test (covalent-radius projected-disk
    occlusion). A candidate O/N atom i is a surface site if, looking outward
    (top OR bottom), no non-H atom further out has a covalent-radius circle that
    overlaps i's circle in the xy projection -- and i is not > depth_cutoff below
    the outermost atom of its own element.
    """
    symbols = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    sym_i = symbols[idx]
    zi = pos[idx, 2]
    xi, yi = pos[idx, 0], pos[idx, 1]
    r_i = covalent_radii[atomic_numbers[sym_i]]
    same_elem_z = pos[[k for k, s in enumerate(symbols) if s == sym_i], 2]

    def _exposed_from(direction):
        if direction > 0:
            if same_elem_z.max() - zi > depth_cutoff:
                return False
        else:
            if zi - same_elem_z.min() > depth_cutoff:
                return False
        for j, sym_j in enumerate(symbols):
            if j == idx or sym_j == "H":
                continue
            zj = pos[j, 2]
            r_j = covalent_radii[atomic_numbers[sym_j]]
            if direction > 0 and zj <= zi + r_i:
                continue
            if direction < 0 and zj >= zi - r_i:
                continue
            dxy = np.hypot(pos[j, 0] - xi, pos[j, 1] - yi)
            if dxy < (r_i + r_j):
                return False
        return True

    return _exposed_from(+1) or _exposed_from(-1)


def _surface_exposed(atoms, idx, **kw):
    """DEPRECATED heuristic name; delegates to the validated _is_surface_site."""
    return _is_surface_site(atoms, idx)


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
    sites into a few representatives so downstream reactivity calcs run on ~3
    sites per functional group. Clusters by a local-environment fingerprint;
    returns list of (representative_site_index, population_weight).
    """
    if len(site_indices) == 0:
        return []
    if len(site_indices) <= n_representatives:
        w = 1.0 / len(site_indices)
        return [(int(i), w) for i in site_indices]

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
        ds += [rcut] * (k - len(ds))
        return np.array(ds)

    fps = np.array([fingerprint(i) for i in site_indices])
    try:
        from sklearn.cluster import KMeans
        k = min(n_representatives, len(site_indices))
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(fps)
        labels, centers = km.labels_, km.cluster_centers_
    except Exception:
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
    Reactive-site normalisation area (nm^2). A cleaved slab exposes TWO surfaces
    and sites are counted across both, so PER-SURFACE density = count/(2*flat).
    Validated against Kim et al.'s deposited POSCARs (OH ~6.0 vs 6.19). Uses the
    cross product |a x b| (correct for non-orthogonal/hexagonal cells); no
    roughness correction (the paper's densities use the flat cross-section).
    """
    cell = atoms.get_cell()
    flat = np.linalg.norm(np.cross(cell[0], cell[1])) / 100.0
    return 2.0 * flat if both_surfaces else flat


# ===========================================================================
# 8b. Quality controls: strain flagging, clumping test, outlier gate
# ===========================================================================

def _bulk_mean_angle(atoms, center_sym, ref_syms):
    cutoffs = natural_cutoffs(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    symbols = atoms.get_chemical_symbols()
    angles = []
    cell = atoms.get_cell()
    for i, sym in enumerate(symbols):
        if sym != center_sym:
            continue
        nbrs, offsets = nl.get_neighbors(i)
        neigh = [(j, off) for j, off in zip(nbrs, offsets) if symbols[j] in ref_syms]
        if len(neigh) < 2:
            continue
        p0 = atoms.positions[i]
        for a in range(len(neigh)):
            for b in range(a + 1, len(neigh)):
                ja, offa = neigh[a]; jb, offb = neigh[b]
                va = atoms.positions[ja] + offa @ cell - p0
                vb = atoms.positions[jb] + offb @ cell - p0
                cosang = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)
                angles.append(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
    return float(np.mean(angles)) if angles else None


def flag_strained_sites(atoms, material, site_indices, deviation_deg=15.0):
    """Flag reactive sites whose local X-center-X angle deviates strongly from
    the amorphous mean (strained hot-spots with less transferable dE)."""
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
    """PBC-aware Clark-Evans clumping metric (xy plane, minimum-image).
    R>1 dispersed, ~1 random, <1 clumped."""
    if len(site_indices) < 2:
        return None
    pos = atoms.get_positions()[site_indices][:, :2]
    cell = atoms.get_cell()
    Lx, Ly = cell[0, 0], cell[1, 1]
    n = len(pos)
    nn = []
    for i in range(n):
        d = pos - pos[i]
        d[:, 0] -= Lx * np.round(d[:, 0] / Lx)
        d[:, 1] -= Ly * np.round(d[:, 1] / Ly)
        dist = np.hypot(d[:, 0], d[:, 1])
        dist[i] = np.inf
        nn.append(dist.min())
    area = abs(Lx * Ly)
    density = n / area
    expected_nn = 0.5 / np.sqrt(density)
    return float(np.mean(nn) / expected_nn)


def slab_passes_quality_gate(counts, area, material, n_std=2.5, min_clump_ratio=0.75,
                             atoms=None):
    """Outlier gate on per-surface site density using the paper's own
    mean +/- n_std*std bounds. Returns (passed, reasons)."""
    reasons = []
    targets = LITERATURE_TARGETS_NM2[material]
    for site_type, (mean, std) in targets.items():
        density = counts.get(site_type, {}).get("exposed", 0) / area
        lo, hi = mean - n_std * std, mean + n_std * std
        if not (lo <= density <= hi):
            reasons.append(f"{site_type} density {density:.2f} outside "
                           f"[{lo:.2f},{hi:.2f}] nm^-2")
    if atoms is not None:
        primary = "OH" if material == "SiO2" else "NH2"
        if primary in counts and counts[primary]["indices"]:
            R = clumping_ratio(atoms, counts[primary]["indices"])
            if R is not None and R < min_clump_ratio:
                reasons.append(f"{primary} sites clumped (R={R:.2f} < {min_clump_ratio})")
    return (len(reasons) == 0), reasons


# ===========================================================================
# 9. Full pipeline + ensemble
# ===========================================================================

def build_one_surface(material, calc, seed, cleave_frac, supercell=(3, 3, 2),
                      use_cache=True, cache_dir=None, published_bulk=None,
                      quench_ps_override=None):
    """Build a single surface (kept for backwards compatibility)."""
    rng = np.random.default_rng(seed)
    if published_bulk is not None:
        bulk = published_bulk.copy()
    else:
        builder = build_alpha_quartz if material == "SiO2" else build_beta_si3n4
        crystal_sc = builder(supercell=supercell)
        if material == "SiNx":
            crystal_sc = apply_sinx_composition(crystal_sc, rng)
        bulk = melt_quench_bulk(crystal_sc, material, calc, seed=seed,
                                supercell=supercell, use_cache=use_cache,
                                cache_dir=cache_dir,
                                quench_ps_override=quench_ps_override)
    slab = cleave_with_vacuum(bulk, cleave_frac=cleave_frac)
    slab = passivate(slab, material)
    slab = anneal_and_relax(slab, material, calc)
    slab.calc = None   # detach shared calc so ase.io.write never sees stale forces
    return slab


def build_two_terminations(material, calc, seed, supercell=(3, 3, 2),
                           use_cache=True, cache_dir=None, published_bulk=None,
                           quench_ps_override=None):
    """
    Efficient path: melt-quench ONE bulk, then cleave into two terminations.
    Halves the most expensive step vs building each surface from its own bulk.
    """
    rng = np.random.default_rng(seed)
    if published_bulk is not None:
        bulk = published_bulk.copy()
        print(f"    using published bulk ({len(bulk)} atoms)", flush=True)
    else:
        builder = build_alpha_quartz if material == "SiO2" else build_beta_si3n4
        crystal_sc = builder(supercell=supercell)
        if material == "SiNx":
            crystal_sc = apply_sinx_composition(crystal_sc, rng)
        bulk = melt_quench_bulk(crystal_sc, material, calc, seed=seed,
                                supercell=supercell, use_cache=use_cache,
                                cache_dir=cache_dir,
                                quench_ps_override=quench_ps_override)

    surfaces = []
    for cleave_frac in (0.5, 0.0):
        slab = cleave_with_vacuum(bulk, cleave_frac=cleave_frac)
        slab = passivate(slab, material)
        slab = anneal_and_relax(slab, material, calc)
        slab.calc = None
        slab.info["quench_ps"] = bulk.info.get("quench_ps")
        slab.info["cooling_rate_K_per_ps"] = bulk.info.get("cooling_rate_K_per_ps")
        surfaces.append(slab)
    return surfaces


def build_surface_ensemble(material, calc, n_bulk=3, target_accepted=None,
                           max_extra_bulk=3, apply_gate=True, quench_sweep=None,
                           supercell=(3, 3, 2), use_cache=True, cache_dir=None,
                           use_published=False, literature_dir=None):
    """
    Build an ensemble of amorphous surfaces with quality control.

    quench_sweep : list of quench durations (ps) cycled across bulks for
                   cooling-rate variation, e.g. [8, 15, 30]. bulk b uses
                   quench_sweep[b % len(quench_sweep)]. None -> protocol default.
    """
    accepted, rejected, per_surface = [], [], []

    published_bulk = None
    if use_published:
        published_bulk = load_published_bulk(material, literature_dir)
        if published_bulk is None:
            print(f"[{material}] no published bulk found in "
                  f"{literature_dir or LITERATURE_DIR}/, falling back to melt-quench")

    def _quench_for_bulk(b):
        return quench_sweep[b % len(quench_sweep)] if quench_sweep else None

    def process_bulk(b):
        two = build_two_terminations(material, calc, seed=1000 * b,
                                     supercell=supercell, use_cache=use_cache,
                                     cache_dir=cache_dir, published_bulk=published_bulk,
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
            if not str(st).startswith("_"):
                print(f"    {st}: {info['mean_nm2']} +/- {info['std_nm2']} nm^-2 "
                      f"(literature: {info['literature']})")
