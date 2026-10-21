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
from .common import digest

ALLOWED_ELEMENTS = {"H", "C", "N", "O", "Si", "Cl", "F", "P", "S"}
# Substitution templates: anchor element bonded to the surface heteroatom and
# the leaving atoms that leave as H-X.
SUBSTITUTION = {"silyl_substitution": ("Si", {"Cl", "N"}),
                "acylation": ("C", {"Cl", "O", "N"})}
TEMPLATES = {"physisorption", *SUBSTITUTION}
H_LENGTH = {"Cl": 1.28, "N": 1.02, "O": 0.97}


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
        if item is None or not ref.get("quote") or ref["quote"] not in item["text"]:
            raise ValueError("Evidence quote is absent from the retrieved source record")
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
    source, target = np.asarray(source, float), np.asarray(target, float)
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
        molecule_bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
        output.append({"molecule": whole, "fragment": fragment,
                       "byproduct": byproduct, "bonds": bonds,
                       "molecule_bonds": molecule_bonds})
    return output
