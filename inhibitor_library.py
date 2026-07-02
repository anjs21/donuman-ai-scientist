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

import json
import os
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


# --- New molecule geometry builders & fragments ---

def build_isopropylamine():
    """NH2-CH(CH3)2 (isopropylamine byproduct)."""
    a = Atoms("N", positions=[[0, 0, 0]])
    _add(a, "H", [0.8, 0.6, 0])
    _add(a, "H", [-0.8, 0.6, 0])
    c_ch = np.array([0, -1.47, 0])
    _add(a, "C", c_ch)
    _add(a, "H", c_ch + [0, 0, 1.09])
    c_m1 = c_ch + [1.26, -0.89, 0]
    c_m2 = c_ch + [-1.26, -0.89, 0]
    _add(a, "C", c_m1)
    _add(a, "C", c_m2)
    for h in _methyl_H(c_m1, [-1.26, 0.89, 0]):
        _add(a, "H", h)
    for h in _methyl_H(c_m2, [1.26, 0.89, 0]):
        _add(a, "H", h)
    return a


def _safe_diisopropylamino(n_pos, z_sign):
    """Adds a N(iPr)2 group starting at n_pos. z_sign controls the direction along z."""
    a = Atoms("N", positions=[n_pos])
    rel_positions = [
        ("C", [1.4, 0.0, 0.5 * z_sign]),
        ("C", [-1.4, 0.0, 0.5 * z_sign]),
        ("H", [1.4, 0.0, 1.59 * z_sign]),
        ("H", [-1.4, 0.0, 1.59 * z_sign]),
        ("C", [1.4, 1.4, -0.1 * z_sign]),
        ("C", [1.4, -1.4, -0.1 * z_sign]),
        ("C", [-1.4, 1.4, -0.1 * z_sign]),
        ("C", [-1.4, -1.4, -0.1 * z_sign]),
    ]
    for symbol, pos in rel_positions:
        _add(a, symbol, n_pos + np.array(pos))
    for idx in [5, 6, 7, 8]:
        m_pos = a[idx].position
        _add(a, "H", m_pos + np.array([0.8, 0.5, 0.5 * z_sign]))
        _add(a, "H", m_pos + np.array([-0.8, 0.5, 0.5 * z_sign]))
        _add(a, "H", m_pos + np.array([0.0, -0.8, -0.8 * z_sign]))
    return a


def build_diisopropylamine():
    """HN(CH(CH3)2)2 (diisopropylamine byproduct)."""
    a = _safe_diisopropylamino(np.array([0.0, 0.0, 0.0]), -1)
    _add(a, "H", [0, 1.01, 0])
    return a


def build_dipas():
    """Di(isopropylamino)silane, SiH2(NH-iPr)2 (DIPAS)."""
    a = Atoms("Si", positions=[[0, 0, 0]])
    _add(a, "H", [0.0, 1.48, 0.0])
    _add(a, "H", [0.0, -1.48, 0.0])
    dirs = _tetra_from((0, 1, 0))[:2]
    for d in dirs:
        n_pos = _u(d) * 1.73
        _add(a, "N", n_pos)
        _add(a, "H", n_pos + [0, 0, 1.0])
        c_ch = n_pos + _u(d) * 1.47
        _add(a, "C", c_ch)
        _add(a, "H", c_ch + [0, 1.0, 0])
        dirs_c = _tetra_from(-_u(d))[:2]
        for dc in dirs_c:
            c_pos = c_ch + _u(dc) * 1.54
            _add(a, "C", c_pos)
            for h in _methyl_H(c_pos, -_u(dc)):
                _add(a, "H", h)
    return a


def frag_monoisopropylaminosilyl():
    """-SiH2(NH-iPr) fragment left on surface by DIPAS."""
    a = Atoms("Si", positions=[[0, 0, 0]])
    _add(a, "H", [1.0, 1.0, 0.5])
    _add(a, "H", [-1.0, 1.0, 0.5])
    n_pos = np.array([0, 0, -1.73])
    _add(a, "N", n_pos)
    _add(a, "H", n_pos + [0.8, 0.6, 0])
    c_ch = n_pos + [0, -1.47, 0]
    _add(a, "C", c_ch)
    _add(a, "H", c_ch + [0, 0, 1.09])
    c_m1 = c_ch + [1.26, -0.89, 0]
    c_m2 = c_ch + [-1.26, -0.89, 0]
    _add(a, "C", c_m1)
    _add(a, "C", c_m2)
    for h in _methyl_H(c_m1, [-1.26, 0.89, 0]):
        _add(a, "H", h)
    for h in _methyl_H(c_m2, [1.26, 0.89, 0]):
        _add(a, "H", h)
    return a


def build_bdipads():
    """1,2-bis(diisopropylamino)disilane, (iPr2N)SiH2-SiH2(NiPr2) (BDIPADS)."""
    a = Atoms("Si", positions=[[0, 0, 1.175]])
    _add(a, "Si", [0, 0, -1.175])
    _add(a, "H", [1.0, 1.0, 1.675])
    _add(a, "H", [-1.0, -1.0, 1.675])
    _add(a, "H", [1.0, -1.0, -1.675])
    _add(a, "H", [-1.0, 1.0, -1.675])
    n1_pos = np.array([0, 0, 1.175 + 1.73])
    a += _safe_diisopropylamino(n1_pos, -1)
    n2_pos = np.array([0, 0, -1.175 - 1.73])
    a += _safe_diisopropylamino(n2_pos, 1)
    return a


def frag_bdipads():
    """-SiH2-SiH2(NiPr2) fragment left on surface by BDIPADS."""
    a = Atoms("Si", positions=[[0, 0, 0]])
    _add(a, "H", [1.0, 1.0, 0.5])
    _add(a, "H", [-1.0, 1.0, 0.5])
    si2_pos = np.array([0, 0, -2.35])
    _add(a, "Si", si2_pos)
    _add(a, "H", si2_pos + [1.0, -1.0, -0.5])
    _add(a, "H", si2_pos + [-1.0, -1.0, -0.5])
    n_pos = si2_pos + np.array([0, 0, -1.73])
    a += _safe_diisopropylamino(n_pos, 1)
    return a


def _build_alkyl_trichlorosilane(n_carbons):
    a = Atoms("Si", positions=[[0, 0, 0]])
    dirs = _tetra_from((0, 0, -1))
    for d in dirs:
        _add(a, "Cl", _u(d) * 2.02)
    for i in range(n_carbons):
        y_off = 0.889 * (i % 2)
        z_off = -1.87 - i * 1.257
        pos = np.array([0.0, y_off, z_off])
        _add(a, "C", pos)
        if i == n_carbons - 1:
            _add(a, "H", pos + [0.89, 0.5, 0.5])
            _add(a, "H", pos + [-0.89, 0.5, 0.5])
            _add(a, "H", pos + [0.0, -0.889, -0.8])
        else:
            _add(a, "H", pos + [1.02, 0.0, 0.0])
            _add(a, "H", pos + [-1.02, 0.0, 0.0])
    return a


def _frag_alkyl_dichlorosilyl(n_carbons):
    a = Atoms("Si", positions=[[0, 0, 0]])
    dirs = _tetra_from((0, 0, -1))
    _add(a, "Cl", _u(dirs[0]) * 2.02)
    _add(a, "Cl", _u(dirs[1]) * 2.02)
    for i in range(n_carbons):
        y_off = 0.889 * (i % 2)
        z_off = -1.87 - i * 1.257
        pos = np.array([0.0, y_off, z_off])
        _add(a, "C", pos)
        if i == n_carbons - 1:
            _add(a, "H", pos + [0.89, 0.5, 0.5])
            _add(a, "H", pos + [-0.89, 0.5, 0.5])
            _add(a, "H", pos + [0.0, -0.889, -0.8])
        else:
            _add(a, "H", pos + [1.02, 0.0, 0.0])
            _add(a, "H", pos + [-1.02, 0.0, 0.0])
    return a


def build_ets():
    """Ethyltrichlorosilane, (CH3CH2)SiCl3 (ETS)."""
    return _build_alkyl_trichlorosilane(2)


def frag_ets():
    """Ethyldichlorosilyl fragment."""
    return _frag_alkyl_dichlorosilyl(2)


def build_odts():
    """Octadecyltrichlorosilane, (C18H37)SiCl3 (ODTS)."""
    return _build_alkyl_trichlorosilane(18)


def frag_odts():
    """Octadecyldichlorosilyl fragment."""
    return _frag_alkyl_dichlorosilyl(18)


def build_odpa():
    """Octadecylphosphonic acid, C18H37-PO(OH)2 (ODPA)."""
    a = Atoms("P", positions=[[0, 0, 0]])
    _add(a, "O", [0, 1.48, 0])
    o1 = np.array([1.1, -1.1, 0])
    o2 = np.array([-1.1, -1.1, 0])
    _add(a, "O", o1)
    _add(a, "O", o2)
    _add(a, "H", o1 + [0.7, -0.7, 0])
    _add(a, "H", o2 + [-0.7, -0.7, 0])
    for i in range(18):
        y_off = 0.889 * (i % 2)
        z_off = -1.8 - i * 1.257
        pos = np.array([0.0, y_off, z_off])
        _add(a, "C", pos)
        if i == 17:
            _add(a, "H", pos + [0.89, 0.5, 0.5])
            _add(a, "H", pos + [-0.89, 0.5, 0.5])
            _add(a, "H", pos + [0.0, -0.889, -0.8])
        else:
            _add(a, "H", pos + [1.02, 0.0, 0.0])
            _add(a, "H", pos + [-1.02, 0.0, 0.0])
    return a


def build_uda():
    """Undecylaldehyde, C10H21-CHO (UDA)."""
    a = Atoms("C", positions=[[0, 0, 0]])
    _add(a, "O", [0, 1.22, 0])
    _add(a, "H", [-0.94, -0.54, 0])
    for i in range(10):
        y_off = -1.1 + 0.889 * (i % 2)
        z_off = -1.257 - i * 1.257
        pos = np.array([0.0, y_off, z_off])
        _add(a, "C", pos)
        if i == 9:
            _add(a, "H", pos + [0.89, 0.5, 0.5])
            _add(a, "H", pos + [-0.89, 0.5, 0.5])
            _add(a, "H", pos + [0.0, -0.889, -0.8])
        else:
            _add(a, "H", pos + [1.02, 0.0, 0.0])
            _add(a, "H", pos + [-1.02, 0.0, 0.0])
    return a


def build_hacac():
    """Acetylacetone, CH3-CO-CH2-CO-CH3 (hacac)."""
    a = Atoms("C", positions=[[0, 0, 0]])
    _add(a, "H", [0, -0.6, 0.89])
    _add(a, "H", [0, -0.6, -0.89])
    _add(a, "C", [-1.25, 0.3, 0])
    _add(a, "O", [-1.25, 1.52, 0])
    c1 = np.array([-2.5, -0.5, 0])
    _add(a, "C", c1)
    for h in _methyl_H(c1, [1.25, 0.8, 0]):
        _add(a, "H", h)
    _add(a, "C", [1.25, 0.3, 0])
    _add(a, "O", [1.25, 1.52, 0])
    c5 = np.array([2.5, -0.5, 0])
    _add(a, "C", c5)
    for h in _methyl_H(c5, [-1.25, 0.8, 0]):
        _add(a, "H", h)
    return a


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
# The library. Add entries here to extend the screen; the agent and energetics
# engine pick them up automatically.
DEFAULT_LIBRARY = {
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
    "ETS": Reagent(
        name="ETS", category="inhibitor", formula="C2H5Cl3Si",
        reaction_type="dissociative", build=build_ets,
        fragment=frag_ets, byproduct=build_hcl,
        anchor_bond=1.65, targets=("OH", "NH2"),
        volatility="medium",
        notes="Ethyltrichlorosilane inhibitor; caps -OH/-NH releasing HCl."),
    "ODTS": Reagent(
        name="ODTS", category="inhibitor", formula="C18H37Cl3Si",
        reaction_type="dissociative", build=build_odts,
        fragment=frag_odts, byproduct=build_hcl,
        anchor_bond=1.65, targets=("OH", "NH2"),
        volatility="low",
        notes="Octadecyltrichlorosilane inhibitor; caps -OH/-NH releasing HCl. Form hydrophobic SAM."),
    "ODPA": Reagent(
        name="ODPA", category="inhibitor", formula="C18H39O3P",
        reaction_type="physisorption", build=build_odpa,
        targets=("OH", "NH2"), volatility="low",
        notes="Octadecylphosphonic acid inhibitor; coordinates to oxide/nitride surfaces to form SAM."),
    "UDA": Reagent(
        name="UDA", category="inhibitor", formula="C11H22O",
        reaction_type="physisorption", build=build_uda,
        targets=("OH", "NH2"), volatility="low",
        notes="Undecylaldehyde inhibitor; physisorbs/coordinates via carbonyl group."),
    "hacac": Reagent(
        name="hacac", category="inhibitor", formula="C5H8O2",
        reaction_type="physisorption", build=build_hacac,
        targets=("OH", "NH2"), volatility="medium",
        notes="Acetylacetone ligand; chelates/coordinates to surface protic/Lewis sites."),

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
    "DIPAS": Reagent(
        name="DIPAS", category="precursor", formula="C6H18N2Si",
        reaction_type="dissociative", build=build_dipas,
        fragment=frag_monoisopropylaminosilyl, byproduct=build_isopropylamine,
        anchor_bond=1.65, targets=("OH",), volatility="high",
        notes="Di(isopropylamino)silane precursor; chemisorbs on -OH releasing isopropylamine."),
    "BDIPADS": Reagent(
        name="BDIPADS", category="precursor", formula="C12H30N2Si2",
        reaction_type="dissociative", build=build_bdipads,
        fragment=frag_bdipads, byproduct=build_diisopropylamine,
        anchor_bond=1.65, targets=("OH",), volatility="medium",
        notes="1,2-bis(diisopropylamino)disilane precursor; chemisorbs on -OH releasing diisopropylamine."),

    # ---- co-reactant ----
    "H2O": Reagent(
        name="H2O", category="coreactant", formula="H2O",
        reaction_type="physisorption", build=build_water,
        targets=("OH",), volatility="high",
        notes="Oxidant co-reactant that re-hydroxylates the growth surface."),
}

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reagents_config.json")

def _load_reagents_config():
    default_config = {
        name: {
            "enabled": True,
            "category": r.category,
            "volatility": r.volatility,
            "notes": r.notes
        }
        for name, r in DEFAULT_LIBRARY.items()
    }
    if not os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "w") as f:
                json.dump(default_config, f, indent=2)
        except OSError:
            pass
        return default_config
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default_config

_config = _load_reagents_config()
LIBRARY = {}
for name, r in DEFAULT_LIBRARY.items():
    cfg_entry = _config.get(name, {})
    if cfg_entry.get("enabled", True):
        r.category = cfg_entry.get("category", r.category)
        r.volatility = cfg_entry.get("volatility", r.volatility)
        r.notes = cfg_entry.get("notes", r.notes)
        LIBRARY[name] = r



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
