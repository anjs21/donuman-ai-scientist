"""Deterministic analog design around a substitution candidate.

The planner asks for analogs of a molecule in a direction ("more_reactive",
"less_reactive", "bulkier" or "all"); this module enumerates concrete,
validated structures by three edits that map onto kinetic levers:

    leaving group   swap X in anchor--X for another implemented leaving group
    sterics         grow a methyl on the retained group to ethyl/iPr/tBu
    electronics     put F / CF3 / OMe on an aromatic C-H of the retained group

Each analog carries descriptors (anchor Gasteiger charge, heavy atoms within
two bonds of the anchor, approximate pKa of the released H-X) relative to the
parent. These are heuristics for choosing what to compute, not predictions of
barriers. Output is molecular data only; nothing here is executed by a model.
"""
from rdkit import Chem
from rdkit.Chem import AllChem
from .chemistry import ATTACHING, SUBSTITUTION, validate_candidate

ATTACH = 9  # atom-map number of the dummy that joins a leaving group to the anchor

# (label, fragment SMILES with the leaving atom mapped :2 and the joint [*:9],
#  approximate aqueous pKa of the released H-X; textbook values, for ordering only)
LEAVING_GROUPS = {
    "acylation": [
        ("chloride", "[Cl:2][*:9]", -7.0),
        ("N-hydroxysuccinimide ester", "O=C1CCC(=O)N1[O:2][*:9]", 6.0),
        ("pentafluorophenyl ester", "Fc1c(F)c(F)c([O:2][*:9])c(F)c1F", 5.5),
        ("acetate (anhydride)", "CC(=O)[O:2][*:9]", 4.8),
        ("imidazolide", "[*:9][n:2]1ccnc1", 14.5),
    ],
    "silyl_substitution": [
        ("chloride", "[Cl:2][*:9]", -7.0),
        ("imidazolyl", "[*:9][n:2]1ccnc1", 14.5),
        ("dimethylamino", "C[N:2](C)[*:9]", 36.0),
        ("diethylamino", "CC[N:2](CC)[*:9]", 36.0),
        ("diisopropylamino", "CC(C)[N:2](C(C)C)[*:9]", 36.0),
        ("tert-butylamino", "CC(C)(C)[NH:2][*:9]", 36.0),
    ],
}
STERIC = [("ethyl", 1), ("isopropyl", 2), ("tert-butyl", 3)]   # carbons added to a methyl
AROMATIC = [("fluoro", "F", "withdrawing"), ("trifluoromethyl", "C(F)(F)F", "withdrawing"),
            ("methoxy", "OC", "donating")]
DIRECTIONS = {"more_reactive", "less_reactive", "bulkier", "all"}


def _atom(mol, mapnum):
    return next(a for a in mol.GetAtoms() if a.GetAtomMapNum() == mapnum)


def _parts(mol):
    """(retained group with a [*:9] on the anchor, leaving-atom symbol)."""
    anchor, leaving = _atom(mol, 1), _atom(mol, 2)
    bond = mol.GetBondBetweenAtoms(anchor.GetIdx(), leaving.GetIdx())
    pieces = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=True)
    for frag in Chem.GetMolFrags(pieces, asMols=True, sanitizeFrags=True):
        if any(a.GetAtomMapNum() == 1 for a in frag.GetAtoms()):
            for a in frag.GetAtoms():
                if a.GetAtomicNum() == 0:
                    a.SetIsotope(0); a.SetAtomMapNum(ATTACH)
            return frag, leaving.GetSymbol()
    raise ValueError("anchor fragment not found")


def descriptors(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.ComputeGasteigerCharges(mol)
    anchor = _atom(mol, 1)
    near = {anchor.GetIdx()}
    for _ in range(2):
        near |= {n.GetIdx() for i in list(near) for n in mol.GetAtomWithIdx(i).GetNeighbors()}
    return {"anchor_charge": round(float(anchor.GetDoubleProp("_GasteigerCharge")), 3),
            "heavy_atoms_within_2_bonds": sum(1 for i in near if mol.GetAtomWithIdx(i).GetAtomicNum() > 1) - 1}


def _swap_leaving_groups(mol, template):
    retained, symbol = _parts(mol)
    for label, fragment, pka in LEAVING_GROUPS[template]:
        joined = Chem.molzip(retained, Chem.MolFromSmiles(fragment))
        yield joined, f"leaving group -> {label}", {"leaving_group": label, "pKa_HX_approx": pka}


def _grow_methyls(mol):
    retained_atoms = _retained_indices(mol)
    for atom in mol.GetAtoms():
        if (atom.GetIdx() in retained_atoms and atom.GetSymbol() == "C" and not atom.GetIsAromatic()
                and atom.GetDegree() == 1 and atom.GetTotalNumHs() == 3):
            for label, extra in STERIC:
                rw = Chem.RWMol(mol)
                for _ in range(extra):
                    rw.AddBond(atom.GetIdx(), rw.AddAtom(Chem.Atom(6)), Chem.BondType.SINGLE)
                yield rw.GetMol(), f"methyl -> {label} (atom {atom.GetIdx()})", {"steric_edit": label}


def _substitute_aromatics(mol):
    retained_atoms = _retained_indices(mol)
    for atom in mol.GetAtoms():
        if atom.GetIdx() in retained_atoms and atom.GetIsAromatic() and atom.GetSymbol() == "C" \
                and atom.GetTotalNumHs() == 1:
            for label, group, kind in AROMATIC:
                rw = Chem.RWMol(mol)
                sub = Chem.MolFromSmiles(group)
                offset = rw.GetNumAtoms()
                for a in sub.GetAtoms():
                    rw.AddAtom(Chem.Atom(a.GetAtomicNum()))
                for b in sub.GetBonds():
                    rw.AddBond(offset + b.GetBeginAtomIdx(), offset + b.GetEndAtomIdx(), b.GetBondType())
                rw.AddBond(atom.GetIdx(), offset, Chem.BondType.SINGLE)
                yield rw.GetMol(), f"aromatic C{atom.GetIdx()}-H -> {label}", {"electronic_edit": kind}


def _retained_indices(mol):
    anchor, leaving = _atom(mol, 1), _atom(mol, 2)
    rw = Chem.RWMol(mol)
    rw.RemoveBond(anchor.GetIdx(), leaving.GetIdx())
    return next(set(g) for g in Chem.GetMolFrags(rw) if anchor.GetIdx() in g)


def leaving_pka(mol):
    """Approximate pKa of the H-X the parent releases (same textbook scale as LEAVING_GROUPS)."""
    anchor, leaving = _atom(mol, 1), _atom(mol, 2)
    if leaving.GetSymbol() == "Cl":
        return -7.0
    if leaving.GetIsAromatic():
        return 14.5
    if leaving.GetSymbol() == "N":
        return 36.0
    other = [n for n in leaving.GetNeighbors() if n.GetIdx() != anchor.GetIdx()]
    if any(n.GetSymbol() == "N" for n in other):
        return 6.0                       # N-hydroxy imide ester
    if any(n.GetIsAromatic() for n in other):
        return 5.5                       # aryl ester
    if any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(n).GetSymbol() == "O"
           for n in other for b in n.GetBonds()):
        return 4.8                       # anhydride
    return 16.0                          # alkoxide


def _wanted(direction, edit, parent):
    if direction == "all":
        return True
    if direction == "bulkier":
        return "steric_edit" in edit
    lg = edit.get("pKa_HX_approx")
    if direction == "more_reactive":
        return (lg is not None and lg < parent["pKa_HX_approx"]) or edit.get("electronic_edit") == "withdrawing"
    # less_reactive: poorer leaving group, more bulk, or donating substituents
    return ((lg is not None and lg > parent["pKa_HX_approx"]) or "steric_edit" in edit
            or edit.get("electronic_edit") == "donating")


def enumerate_analogs(spec, direction="all", limit=8, known=()):
    """Validated analog specs of a substitution candidate, deduplicated."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {sorted(DIRECTIONS)}")
    template = spec["template"]
    if template not in ATTACHING:
        raise ValueError("Analog design needs a substitution or addition template")
    mol = Chem.MolFromSmiles(spec["smiles"])
    # Addition templates have no leaving group: only steric and electronic edits apply.
    swaps = template in SUBSTITUTION
    parent = {"pKa_HX_approx": leaving_pka(mol) if swaps else None, **descriptors(spec["smiles"])}
    seen = set(known) | {Chem.CanonSmiles(Chem.MolToSmiles(_unmapped(mol)))}
    out = []
    generators = [_grow_methyls(mol), _substitute_aromatics(mol)]
    if swaps:
        generators.insert(0, _swap_leaving_groups(mol, template))
    for generator in generators:
        for child, change, edit in generator:
            try:
                Chem.SanitizeMol(child)
                smiles = Chem.MolToSmiles(child)
                identity = Chem.CanonSmiles(Chem.MolToSmiles(_unmapped(child)))
            except Exception:
                continue
            if identity in seen or not _wanted(direction, edit, parent):
                continue
            name = f"{spec['name'][:40]} analog {len(out) + 1}"
            try:
                checked = validate_candidate({"name": name, "smiles": smiles, "template": template,
                                              "role": "inhibitor"})
            except ValueError:
                continue
            seen.add(identity)
            d = descriptors(checked["smiles"])
            out.append({"name": name, "smiles": checked["smiles"], "canonical_smiles": checked["canonical_smiles"],
                        "template": template, "parent": spec["name"], "change": change, **edit,
                        "anchor_charge": d["anchor_charge"], "anchor_charge_parent": parent["anchor_charge"],
                        "heavy_atoms_within_2_bonds": d["heavy_atoms_within_2_bonds"],
                        "heavy_atoms_within_2_bonds_parent": parent["heavy_atoms_within_2_bonds"]})
            if len(out) >= limit:
                return out
    return out


def _unmapped(mol):
    mol = Chem.Mol(mol)
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    return mol


def as_evidence_text(parent, analogs, direction):
    """One quotable line per analog; the planner cites a line when proposing it."""
    lines = [f"Computed analog design for {parent['name']} (direction {direction}); "
             "structural heuristics only, not measured or computed performance."]
    for a in analogs:
        extra = (f"; pKa(HX) approx {a['pKa_HX_approx']}" if "pKa_HX_approx" in a else "")
        lines.append(f"{a['name']}: {a['smiles']} via {a['change']}; anchor charge "
                     f"{a['anchor_charge']} vs parent {a['anchor_charge_parent']}; heavy atoms near anchor "
                     f"{a['heavy_atoms_within_2_bonds']} vs {a['heavy_atoms_within_2_bonds_parent']}{extra}.")
    return "\n".join(lines)
