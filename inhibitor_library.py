"""
inhibitor_library.py
====================
Phase 2a of the AS-ALD in-silico co-scientist (Challenge 4).

A curated, extensible library of inhibitor / precursor / co-reactant molecules
drawn from the ASD literature, each with:

  * a gas-phase 3D geometry builder (approximate; the MLIP relaxes it),
  * a reaction template that tells energetics.py how the molecule interacts
    with a surface reactive site.

Two reaction types are supported:

  "dissociative" -- the molecule reacts with a protonated surface site
     (Site-H) by exchanging its leaving group for the abstracted H:

         Site-H  +  R-LG   ->   Site-R  +  H-LG

     e.g. =Si-OH + (CH3)3Si-N(CH3)2 -> =Si-O-Si(CH3)3 + HN(CH3)2
     The library supplies `fragment` (R, left on the surface, anchor atom
     first) and `byproduct` (H-LG, released to gas). Atom balance is
     guaranteed because fragment + byproduct = molecule + H.

  "physisorption" -- the whole molecule adsorbs without bond cleavage:

         Site  +  M   ->   Site...M          (dE = E_ads)

     Used for Lewis-base inhibitors (amines, pyridine) that block sites by
     coordination / H-bonding rather than covalent capping.

The problem statement (passivate SiN, grow SiOx on SiO) means a GOOD inhibitor
binds the NGS (SiNx: -NH2/-NH) strongly and the GS (SiO2: -OH) weakly. The
library spans both silane-type (OH-selective) and amine/base-type (N-selective)
chemistries so the agent can discover the contrast rather than assume it.

Only numpy + ase are required to build geometries. No GPU needed here.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple
from ase import Atoms


# ===========================================================================
# Geometry helpers (build small molecules from bond vectors)
# ===========================================================================

def _u(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _tetra_from(a):
    """Given one bond direction `a`, return the 3 remaining tetrahedral dirs."""
    a = _u(a)
    t = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _u(np.cross(a, t))
    w = _u(np.cross(a, u))
    c, s = -1.0 / 3.0, np.sqrt(1.0 - 1.0 / 9.0)
    out = []
    for k in range(3):
        ang = 2 * np.pi * k / 3
        out.append(_u(c * a + s * (np.cos(ang) * u + np.sin(ang) * w)))
    return out


def _methyl_H(c_pos, c_to_parent, ch=1.09):
    """Three H positions of a methyl whose C sits at c_pos, bonded to a parent
    atom in direction `c_to_parent`."""
    return [c_pos + _u(d) * ch for d in _tetra_from(c_to_parent)]


def _add(atoms, symbol, position):
    atoms += Atoms(symbol, positions=[np.asarray(position, float)])


# ---- building blocks -------------------------------------------------------

def _silyl(open_dir=(0, 0, -1), si_c=1.87):
    """(CH3)3Si- fragment. Anchor Si is atom index 0; its open (unfilled) bond
    points along `open_dir`. The three methyls fill the other tetrahedral
    directions."""
    open_dir = _u(open_dir)
    a = Atoms("Si", positions=[[0, 0, 0]])
    for d in _tetra_from(open_dir):
        c = _u(d) * si_c
        _add(a, "C", c)
        for h in _methyl_H(c, -_u(d)):
            _add(a, "H", h)
    return a


def _dimethylamino(n_pos, open_dir, nc=1.47):
    """-(N(CH3)2) group starting at n_pos, its open bond along open_dir (toward
    the atom it attaches to). Returns Atoms with N first."""
    open_dir = _u(open_dir)
    a = Atoms("N", positions=[n_pos])
    dirs = _tetra_from(open_dir)[:2]  # two methyls; third dir ~ lone pair
    for d in dirs:
        c = n_pos + _u(d) * nc
        _add(a, "C", c)
        for h in _methyl_H(c, -_u(d)):
            _add(a, "H", h)
    return a


# ===========================================================================
# Molecule / fragment builders
# ===========================================================================

def build_dmatms():
    """Dimethylamino-trimethylsilane, (CH3)3Si-N(CH3)2 (DMATMS)."""
    a = _silyl(open_dir=(0, 0, -1))
    si = a[0].position
    n_pos = si + np.array([0, 0, -1.0]) * 1.75
    a += _dimethylamino(n_pos, open_dir=(0, 0, 1))  # open bond back toward Si
    return a


def build_tmcs():
    """Trimethylchlorosilane, (CH3)3Si-Cl (TMCS)."""
    a = _silyl(open_dir=(0, 0, -1))
    si = a[0].position
    _add(a, "Cl", si + np.array([0, 0, -1.0]) * 2.05)
    return a


def build_bdmas():
    """Bis(dimethylamino)silane, SiH2(N(CH3)2)2 (BDMAS) -- a SiO2 precursor."""
    a = Atoms("Si", positions=[[0, 0, 0]])
    dirs = _tetra_from((0, 0, 1))
    # two H on Si
    _add(a, "H", np.array([0, 0, 1.0]) * 1.48)
    _add(a, "H", dirs[0] * 1.48)
    # two dimethylamino groups on the remaining directions
    for d in dirs[1:3]:
        n_pos = _u(d) * 1.73
        a += _dimethylamino(n_pos, open_dir=-_u(d))
    return a


def build_sicl4():
    a = Atoms("Si", positions=[[0, 0, 0]])
    for d in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]:
        _add(a, "Cl", _u(d) * 2.02)
    return a


def build_nh3():
    a = Atoms("N", positions=[[0, 0, 0]])
    for d in _tetra_from((0, 0, 1)):
        _add(a, "H", _u(d) * 1.01)
    return a


def build_dimethylamine():
    """HN(CH3)2 -- byproduct of DMATMS, and itself a small amine inhibitor."""
    a = Atoms("N", positions=[[0, 0, 0]])
    _add(a, "H", np.array([0, 0, 1.0]) * 1.01)
    for d in _tetra_from((0, 0, 1))[:2]:
        c = _u(d) * 1.47
        _add(a, "C", c)
        for h in _methyl_H(c, -_u(d)):
            _add(a, "H", h)
    return a


def build_pyridine():
    """Planar C5H5N ring (aromatic Lewis-base inhibitor)."""
    r = 1.39
    a = Atoms()
    ring = []
    for k in range(6):
        ang = np.pi / 2 + 2 * np.pi * k / 6
        ring.append(np.array([r * np.cos(ang), r * np.sin(ang), 0.0]))
    _add(a, "N", ring[0])
    for k in range(1, 6):
        _add(a, "C", ring[k])
        _add(a, "H", ring[k] * (r + 1.08) / r)  # H radially outward
    return a


def build_hcl():
    return Atoms("HCl", positions=[[0, 0, 0], [0, 0, 1.28]])


def build_water():
    return Atoms("OH2", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])


# fragments left on the surface after a dissociative reaction (anchor first)
def frag_trimethylsilyl():
    """-Si(CH3)3 to cap a surface O or N. Anchor Si is index 0; open bond -z."""
    return _silyl(open_dir=(0, 0, -1))


# ===========================================================================
# Reagent records
# ===========================================================================

@dataclass
class Reagent:
    name: str
    category: str                 # 'inhibitor' | 'precursor' | 'coreactant'
    formula: str
    reaction_type: str            # 'dissociative' | 'physisorption'
    build: Callable[[], Atoms]
    fragment: Optional[Callable[[], Atoms]] = None   # dissociative only
    byproduct: Optional[Callable[[], Atoms]] = None  # dissociative only
    anchor_bond: float = 1.65     # site-atom -> fragment anchor bond length (A)
    targets: Tuple[str, ...] = ()  # site types it is expected to attack
    volatility: str = "medium"     # qualitative screening metadata
    notes: str = ""

    def n_atoms(self):
        return len(self.build())


# The library. Add entries here to extend the screen; the agent and energetics
# engine pick them up automatically.
LIBRARY = {
    # ---- inhibitors (block the non-growth surface) ----
    "DMATMS": Reagent(
        name="DMATMS", category="inhibitor", formula="C5H15NSi",
        reaction_type="dissociative", build=build_dmatms,
        fragment=frag_trimethylsilyl, byproduct=build_dimethylamine,
        anchor_bond=1.65, targets=("OH", "NH2", "NH_bridge"),
        volatility="high",
        notes="Aminosilane inhibitor (Kim et al. 2026). Caps -OH/-NH by "
              "releasing dimethylamine; silyl group sterically blocks growth."),
    "TMCS": Reagent(
        name="TMCS", category="inhibitor", formula="C3H9ClSi",
        reaction_type="dissociative", build=build_tmcs,
        fragment=frag_trimethylsilyl, byproduct=build_hcl,
        anchor_bond=1.65, targets=("OH", "NH2"),
        volatility="high",
        notes="Chlorosilane inhibitor; caps -OH releasing HCl. Classic "
              "oxide-selective capping agent -> expect strong on SiO2."),
    "NH3": Reagent(
        name="NH3", category="inhibitor", formula="H3N",
        reaction_type="physisorption", build=build_nh3,
        targets=("OH", "NH_bridge"), volatility="high",
        notes="Small Lewis base; probes N/H-bonding blocking without covalent "
              "capping."),
    "dimethylamine": Reagent(
        name="dimethylamine", category="inhibitor", formula="C2H7N",
        reaction_type="physisorption", build=build_dimethylamine,
        targets=("OH", "NH2"), volatility="high",
        notes="Secondary amine; H-bonds to surface -OH/-NH."),
    "pyridine": Reagent(
        name="pyridine", category="inhibitor", formula="C5H5N",
        reaction_type="physisorption", build=build_pyridine,
        targets=("OH", "NH2", "NH_bridge"), volatility="medium",
        notes="Aromatic base; coordinates to Lewis-acidic / protic sites."),

    # ---- precursors (grow the SiOx film on the growth surface) ----
    "BDMAS": Reagent(
        name="BDMAS", category="precursor", formula="C4H14N2Si",
        reaction_type="dissociative", build=build_bdmas,
        fragment=frag_trimethylsilyl, byproduct=build_dimethylamine,
        anchor_bond=1.65, targets=("OH",), volatility="high",
        notes="SiO2 ALD precursor; chemisorbs on -OH releasing amine. "
              "Fragment is an approximation of the SiH2(NMe2)-anchor."),
    "SiCl4": Reagent(
        name="SiCl4", category="precursor", formula="Cl4Si",
        reaction_type="dissociative", build=build_sicl4,
        fragment=frag_trimethylsilyl, byproduct=build_hcl,
        anchor_bond=1.65, targets=("OH",), volatility="high",
        notes="Halide SiO2 precursor; chemisorbs on -OH releasing HCl."),

    # ---- co-reactant ----
    "H2O": Reagent(
        name="H2O", category="coreactant", formula="H2O",
        reaction_type="physisorption", build=build_water,
        targets=("OH",), volatility="high",
        notes="Oxidant co-reactant that re-hydroxylates the growth surface."),
}


def get_reagents(category=None, names=None):
    """Return a list of Reagent records, optionally filtered."""
    items = list(LIBRARY.values())
    if category:
        items = [r for r in items if r.category == category]
    if names:
        names = set(names)
        items = [r for r in items if r.name in names]
    return items


# ===========================================================================
# Self-test: build every molecule and report atom counts / min bond length
# ===========================================================================

if __name__ == "__main__":
    from ase.io import write
    print(f"{'reagent':16s} {'category':11s} {'type':14s} {'atoms':>5s}  min-dist(A)")
    for r in LIBRARY.values():
        mol = r.build()
        pos = mol.get_positions()
        # smallest interatomic distance -- catches overlapping-atom geometry bugs
        dmin = np.inf
        for i in range(len(mol)):
            for j in range(i + 1, len(mol)):
                dmin = min(dmin, np.linalg.norm(pos[i] - pos[j]))
        flag = "" if dmin > 0.8 else "  <-- CHECK GEOMETRY"
        print(f"{r.name:16s} {r.category:11s} {r.reaction_type:14s} "
              f"{len(mol):5d}  {dmin:6.2f}{flag}")
        if r.fragment:
            _ = r.fragment()
        if r.byproduct:
            _ = r.byproduct()
    # dump a couple to xyz for visual inspection
    write("_dmatms.xyz", build_dmatms())
    write("_pyridine.xyz", build_pyridine())
    print("\nWrote _dmatms.xyz, _pyridine.xyz for inspection.")
