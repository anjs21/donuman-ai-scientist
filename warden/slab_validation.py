"""Finite-slab diagnostics, without equating slab envelope and bulk density.

Source-specific preparation values are not acceptance intervals. No central
window is selected as a pass/fail test, and no parent bulk is invented.
"""
from collections import Counter
import numpy as np
from ase.data import atomic_masses

PROFILE = 'finite_slab_screening_v1'
REFERENCE = {
    'source': 'https://arxiv.org/html/2510.17356v1#S3.SS1',
    'section': '3.1 Surface modeling',
    'bulk_preparation_density_gcc': {'SiO2': 2.2, 'SiNx': 2.6},
    'note': 'Source-specific preparation values, not universal slab-density acceptance thresholds.',
}


def density_diagnostics(atoms, material):
    z = atoms.positions[:, 2]
    area = float(np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1])))
    heavy = atoms.numbers != 1
    if not heavy.any() or not (atoms.numbers == 14).any() or area <= 0:
        raise ValueError('Need a Si-containing slab with positive in-plane area')
    extent = float(np.ptp(z[heavy]))
    if extent <= 0:
        raise ValueError('Slab heavy-atom extent must be positive')
    # Use physical elemental masses even if MD used inflated hydrogen masses.
    masses = atomic_masses[atoms.numbers]
    lo, hi = float(z[atoms.numbers == 14].min()), float(z[atoms.numbers == 14].max())
    if hi <= lo:
        raise ValueError('A finite-thickness silicon network is required')
    windows = []
    for trim in (.10, .20, .25, .30):
        left, right = lo + trim*(hi-lo), hi - trim*(hi-lo)
        selected = (z >= left) & (z < right)
        windows.append({'fraction_trimmed_each_side': trim,
                        'window_thickness_A': right-left,
                        'atom_count': int(selected.sum()),
                        'density_gcc': float(masses[selected].sum() / (area*(right-left)) * 1.66053906660)})
    return {'status': 'diagnostic_only', 'bulk_density_validated': False,
            'parent_bulk_evidence': 'not_supplied_or_verified',
            'envelope_density_gcc': float(masses.sum() / (area*extent) * 1.66053906660),
            'heavy_atom_extent_A': extent, 'silicon_extent_A': hi-lo,
            'area_A2': area, 'central_windows': windows,
            'bulk_preparation_reference_gcc': REFERENCE['bulk_preparation_density_gcc'][material],
            'reference': REFERENCE,
            'interpretation': 'No bulk-like plateau or bulk density is certified by these thin-slab windows.'}


def identify_composition(atoms, declared_material):
    counts = Counter(atoms.get_chemical_symbols())
    reasons = []
    if set(counts) - {'Si', 'O', 'N', 'H'}:
        reasons.append('Unsupported elements in substrate; only Si/O/N/H are supported')
    if not counts['Si']:
        reasons.append('Substrate has no Si')
    if declared_material == 'SiO2':
        if not counts['O'] or counts['N']:
            reasons.append('Declared SiO2 requires oxygen and no nitrogen in this profile')
        family = 'hydroxylated_silicon_oxide' if counts['H'] else 'silicon_oxide'
    else:
        if not counts['N']:
            reasons.append('Declared SiNx requires nitrogen')
        family = 'silicon_oxynitride' if counts['O'] else 'silicon_nitride'
        if counts['H']:
            family = 'hydrogenated_' + family
    return {'formula': atoms.get_chemical_formula(), 'element_counts': dict(sorted(counts.items())),
            'observed_family': family, 'declared_material': declared_material,
            'assignment_method': 'elemental composition checked against declared class; no independent phase identification',
            'validated_amorphous_phase': False}, reasons


def finite_slab_structure(atoms, material, legacy_structure):
    """Keep coordination/bond checks; density comparison is explicitly inapplicable."""
    report = {k: (dict(v) if isinstance(v, dict) else v)
              for k, v in legacy_structure.items()}
    diagnostic = density_diagnostics(atoms, material)
    report['density_gcc'] = {
        'value': diagnostic['envelope_density_gcc'], 'target': None, 'tol': None,
        'pass': None, 'required_for_screening': False,
        'status': 'not_comparable_to_bulk_reference',
        'bulk_preparation_reference_gcc': diagnostic['bulk_preparation_reference_gcc'],
        'source': REFERENCE['source'],
        'note': 'Not passed, not validated: density is diagnostic for a finite passivated slab.'}
    required = [v for k, v in report.items() if not k.startswith('_') and k != 'density_gcc']
    report['_passed'] = sum(bool(v['pass']) for v in required)
    report['_total'] = len(required)
    report['_ok'] = all(v['pass'] for v in required)
    report['_scope'] = 'finite_slab_screening_only'
    return report, diagnostic
