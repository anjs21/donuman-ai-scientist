"""Data-driven candidates: intact adsorption, one Si--Cl/Si--N substitution, or
one acyl substitution (C(=O)--X + surface N-H/O-H -> amide/ester + HX).

The agent supplies structures, not executable Python. Only neutral closed-shell
molecules and the explicitly supported mechanisms enter screening.
"""
from collections import Counter
import re
import numpy as np
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from .lit import quote_in
from .common import digest

ALLOWED_ELEMENTS = {"H", "C", "N", "O", "Si", "Cl", "F", "P", "S"}
# Substitution templates: anchor element bonded to the surface heteroatom and
# the leaving atoms that leave as H-X.
SUBSTITUTION = {"silyl_substitution": ("Si", {"Cl", "N"}),
                "acylation": ("C", {"Cl", "O", "N"})}
# Addition templates: surface X-H adds across [C:1]=[O:2]; X bonds to the
# carbonyl carbon and the proton moves to the oxygen (aldehyde/ketone + Si-NH2
# -> hemiaminal, + Si-OH -> hemiacetal). Nothing leaves; a later dehydration to
# an imine is a separate step and is not modelled.
ADDITION = {"carbonyl_addition": ("C", "O")}
# Templates that form one covalent bond to a surface heteroatom and consume its proton.
ATTACHING = {*SUBSTITUTION, *ADDITION}
TEMPLATES = {"physisorption", *ATTACHING}
H_LENGTH = {"Cl": 1.28, "N": 1.02, "O": 0.97}
SP3_TILT = np.radians(180 - 109.47)   # C-O against the surface normal once C-X forms
# A quote must be a real passage, not a word that occurs in any text.
MIN_QUOTE_WORDS = 5


def validate_candidate(spec, evidence=None, require_evidence=False):
    spec = dict(spec)
    name = spec.get("name", "")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9 _().+-]{1,80}", name):
        raise ValueError("name must be 1-80 simple printable characters")
    mol = Chem.MolFromSmiles(spec.get("smiles", ""))
    if mol is None or mol.GetNumAtoms() == 0 or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("Need one connected, valid molecular SMILES")
    if mol.GetNumAtoms() > 80:
        raise ValueError("Hackathon limit: 80 heavy atoms")
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise ValueError("Charged/radical chemistry requires a different validated template")
    if not {a.GetSymbol() for a in mol.GetAtoms()} <= ALLOWED_ELEMENTS:
        raise ValueError("Unsupported element")
    maps = [a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetAtomMapNum()]
    if len(maps) != len(set(maps)):
        raise ValueError("Atom map numbers must be unique")
    template = spec.get("template")
    if template not in TEMPLATES:
        raise ValueError("Unsupported mechanism: needs a validated reaction template")
    if template in SUBSTITUTION:
        element, leaving_elements = SUBSTITUTION[template]
        anchor = next((a for a in mol.GetAtoms() if a.GetAtomMapNum() == 1), None)
        leaving = next((a for a in mol.GetAtoms() if a.GetAtomMapNum() == 2), None)
        if anchor is None or leaving is None or anchor.GetSymbol() != element:
            raise ValueError(f"Substitution needs [{element}:1] and a leaving atom mapped :2")
        bond = mol.GetBondBetweenAtoms(anchor.GetIdx(), leaving.GetIdx())
        if not bond or bond.GetBondType() != Chem.BondType.SINGLE:
            raise ValueError("Maps 1 and 2 must have a single bond")
        if leaving.GetSymbol() not in leaving_elements:
            raise ValueError("Leaving group not implemented for this template")
        if template == "acylation" and not any(
                b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(anchor).GetSymbol() == "O"
                for b in anchor.GetBonds()):
            raise ValueError("Acylation needs a carbonyl carbon mapped [C:1]")
        cut = Chem.RWMol(Chem.AddHs(mol))
        cut.RemoveBond(anchor.GetIdx(), leaving.GetIdx())
        if len(Chem.GetMolFrags(cut)) != 2:
            raise ValueError("Leaving group must separate into exactly two components")
    if template in ADDITION:
        anchor = next((a for a in mol.GetAtoms() if a.GetAtomMapNum() == 1), None)
        acceptor = next((a for a in mol.GetAtoms() if a.GetAtomMapNum() == 2), None)
        if anchor is None or acceptor is None or anchor.GetSymbol() != "C" or acceptor.GetSymbol() != "O":
            raise ValueError("Carbonyl addition needs the carbonyl carbon [C:1] and its oxygen [O:2]")
        bond = mol.GetBondBetweenAtoms(anchor.GetIdx(), acceptor.GetIdx())
        if not bond or bond.GetBondType() != Chem.BondType.DOUBLE:
            raise ValueError("Maps 1 and 2 must be the C=O double bond")
        if any(n.GetSymbol() not in {"C", "H"} for n in anchor.GetNeighbors() if n.GetIdx() != acceptor.GetIdx()):
            # Acyl halides, esters, anhydrides, amides react by substitution: use acylation.
            raise ValueError("Carbonyl addition is for aldehydes/ketones (carbon or H substituents only)")
    role = spec.get("role", "inhibitor")
    if role not in {"inhibitor", "precursor"}:
        raise ValueError("role must be inhibitor or precursor")
    refs = spec.get("evidence", [])
    if require_evidence and not refs:
        raise ValueError("Agent candidates require evidence and an explicit hypothesis")
    for ref in refs:
        if not isinstance(ref, dict):
            raise ValueError("Each evidence reference needs id and quote")
        item = (evidence or {}).get(ref.get("id"))
        quote = ref.get("quote") or ""
        # Matched on the readable text (escapes, subscripts, typographic quotes normalised).
        if item is None or not quote or not quote_in(quote, item["text"]):
            raise ValueError("Evidence quote is absent from the retrieved source record "
                             "(quotes must be verbatim, at least 12 characters)")
        if len(quote.split()) < MIN_QUOTE_WORDS:
            raise ValueError(f"Evidence quote must be at least {MIN_QUOTE_WORDS} words of the source")
    if require_evidence and (not spec.get("hypothesis") or not spec.get("falsification")):
        raise ValueError("State a hypothesis and a falsifying observation")
    identity = Chem.Mol(mol)
    for a in identity.GetAtoms():
        a.SetAtomMapNum(0)
    canonical = Chem.MolToSmiles(identity, canonical=True)
    spec.update({"canonical_smiles": canonical, "role": role,
                 "molecular_weight": Descriptors.MolWt(mol),
                 "smiles": Chem.MolToSmiles(mol, canonical=True)})
    spec["id"] = "cand_" + digest({"smiles": spec["smiles"], "template": template,
                                  "role": role})[:16]
    return spec


def rotation(source, target):
    # Copy: callers pass rows of position arrays, which in-place normalisation would move.
    source, target = np.array(source, float), np.array(target, float)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    c = float(np.clip(source @ target, -1, 1))
    if c > 1 - 1e-9:
        return np.eye(3)
    if c < -1 + 1e-9:
        axis = np.cross(source, [1., 0., 0.])
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(source, [0., 1., 0.])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    v = np.cross(source, target)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def _branches(symbols, positions, anchor):
    """{neighbour of anchor: atoms reached from it without passing the anchor}."""
    from ase.data import atomic_numbers, covalent_radii
    r = covalent_radii[[atomic_numbers[s] for s in symbols]]
    d = np.linalg.norm(positions[:, None] - positions[None], axis=2)
    adjacent = (d < 1.3 * (r[:, None] + r[None])) & ~np.eye(len(symbols), dtype=bool)
    out = {}
    for n in np.flatnonzero(adjacent[anchor]):
        seen, stack = {int(n)}, [int(n)]
        while stack:
            for j in np.flatnonzero(adjacent[stack.pop()]):
                if j != anchor and int(j) not in seen:
                    seen.add(int(j)); stack.append(int(j))
        out[int(n)] = sorted(seen)
    return out


def addition_frame(symbols, positions, anchor, acceptor, h_length=0.97):
    """Product-side placement for carbonyl addition, in a frame where the new
    surface bond runs from the anchor (origin) along -z.

    The carbonyl carbon becomes sp3: each substituent branch (O and the two
    others) is turned rigidly onto a tetrahedral direction pointing away from
    the surface (C=O at azimuth 0), and the transferred proton goes on O
    pointing up. Returns (positions, proton position).
    """
    pos = np.asarray(positions, float) - positions[anchor]
    branches = _branches(symbols, pos, anchor)
    if acceptor not in branches or len(branches) != 3:
        raise ValueError("carbonyl carbon must have exactly three substituents")
    up = np.cos(SP3_TILT)
    def direction(azimuth):
        return np.array([np.sin(SP3_TILT) * np.cos(azimuth), np.sin(SP3_TILT) * np.sin(azimuth), up])
    others = [n for n in branches if n != acceptor]
    best, best_score = None, -np.inf
    for order in (others, others[::-1]):          # either assignment of the two other branches
        trial = pos.copy()
        for n, azimuth in zip([acceptor, *order], (0., 2 * np.pi / 3, 4 * np.pi / 3)):
            idx = branches[n]
            trial[idx] = pos[idx] @ rotation(pos[n], direction(azimuth)).T
        rest = [i for i in range(len(pos)) if i != anchor]
        d = np.linalg.norm(trial[rest][:, None] - trial[rest][None], axis=2) + np.eye(len(rest)) * 9
        score = d.min()                            # keep branches from colliding
        if score > best_score:
            best, best_score = trial, score
    tilt_up = np.array([0., 0., 1.]) + 0.35 * direction(0.)
    proton = best[acceptor] + h_length * tilt_up / np.linalg.norm(tilt_up)
    return best, proton


def geometries(spec, n_conformers=3, seed=42):
    mol = Chem.AddHs(Chem.MolFromSmiles(spec["smiles"]))
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.pruneRmsThresh = 0.25
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params))
    if not ids:
        raise ValueError("RDKit could not embed this candidate")
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    anchor = next((a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() == 1), 0)
    output = []
    for cid in ids:
        # Embedding is a starting geometry, not a molecular energy prediction.
        pos = np.array(mol.GetConformer(cid).GetPositions())
        whole = Atoms(symbols, positions=pos)
        if spec["template"] == "physisorption":
            indices = [anchor] + [i for i in range(len(symbols)) if i != anchor]
            fragment = whole[indices]
            frag_pos = fragment.positions - fragment.positions[0]
            outward = frag_pos.mean(axis=0)
            if np.linalg.norm(outward) > 1e-6:
                frag_pos = frag_pos @ rotation(outward, [0, 0, 1]).T
            fragment.positions = frag_pos
            byproduct = None
        elif spec["template"] in ADDITION:
            # The whole molecule stays; the surface proton joins it on the carbonyl O.
            acceptor = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() == 2)
            indices = [anchor] + [i for i in range(len(symbols)) if i != anchor]
            frame, proton = addition_frame(symbols, pos, anchor, acceptor)
            fragment = Atoms([symbols[i] for i in indices] + ["H"],
                             positions=np.vstack([frame[indices], proton]))
            byproduct = None
            lhs = Counter(symbols); lhs["H"] += 1
            if lhs != Counter(fragment.get_chemical_symbols()):
                raise ValueError("Reaction is not element-balanced")
        else:
            leaving = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() == 2)
            cut = Chem.RWMol(mol)
            cut.RemoveBond(anchor, leaving)
            groups = Chem.GetMolFrags(cut)
            retained = next(g for g in groups if anchor in g)
            lost = next(g for g in groups if leaving in g)
            indices = [anchor] + [i for i in retained if i != anchor]
            fragment = whole[indices]
            bond_dir = pos[leaving] - pos[anchor]
            fragment.positions = (fragment.positions - pos[anchor]) @ rotation(bond_dir, [0, 0, -1]).T
            byproduct = whole[list(lost)]
            h_length = H_LENGTH[symbols[leaving]]
            byproduct += Atoms("H", positions=[pos[leaving] - h_length * bond_dir / np.linalg.norm(bond_dir)])
            lhs = Counter(symbols); lhs["H"] += 1
            rhs = Counter(fragment.get_chemical_symbols()) + Counter(byproduct.get_chemical_symbols())
            if lhs != rhs:
                raise ValueError("Reaction is not element-balanced")
        inverse = {old: new for new, old in enumerate(indices)}
        bonds = [(inverse[b.GetBeginAtomIdx()], inverse[b.GetEndAtomIdx()])
                 for b in mol.GetBonds()
                 if b.GetBeginAtomIdx() in inverse and b.GetEndAtomIdx() in inverse]
        if spec["template"] in ADDITION:
            bonds.append((inverse[acceptor], len(indices)))   # new O-H
        molecule_bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
        output.append({"molecule": whole, "fragment": fragment,
                       "byproduct": byproduct, "bonds": bonds,
                       "molecule_bonds": molecule_bonds})
    return output
