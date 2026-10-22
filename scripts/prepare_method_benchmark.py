"""Select frozen NH3 orientation pairs; compare independently supplied reference energies.

No quantum calculation, new material claim, or candidate ranking is performed here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def select(report):
    candidates = [c for c in report['candidates'] if c['name'] == 'NH3']
    if len(candidates) != 1:
        raise ValueError('Expected exactly one NH3 candidate')
    candidate = candidates[0]
    if candidate['template'] != 'physisorption':
        raise ValueError('This benchmark supports NH3 physisorption only')
    results = [r for r in report['results'] if r['candidate_id'] == candidate['id'] and r.get('mode') == 'mace']
    if len(results) != 1:
        raise ValueError('Expected exactly one MACE result for NH3')
    result = results[0]
    pairs = []
    for material in ('SiO2', 'SiNx'):
        eligible = []
        for site in result['sites']:
            if site['material'] != material or site['status'] != 'valid':
                continue
            attempts = [a for a in site['attempts'] if a['status'] == 'valid' and finite(a.get('energy_eV')) and a.get('geometry')]
            if len(attempts) < 2:
                continue
            attempts.sort(key=lambda a: (a['energy_eV'], a['geometry']))
            spread = attempts[-1]['energy_eV'] - attempts[0]['energy_eV']
            if spread > 0:
                eligible.append((spread, site, attempts))
        if not eligible:
            raise ValueError(f'No valid distinct orientation pair for {material}')
        spread, site, attempts = sorted(eligible, key=lambda x: (-x[0], x[1]['surface_id'], x[1]['site']))[0]
        pairs.append({
            'material': material, 'surface_id': site['surface_id'], 'site': site['site'],
            'site_type': site['site_type'], 'face': site['face'], 'mace_gap_eV': spread,
            'members': [{'label': label, 'attempt': attempt} for label, attempt in [('low', attempts[0]), ('high', attempts[-1])]],
        })
    return pairs, result['provenance']


def prepare(args):
    report_path = Path(args.report).resolve()
    pairs, provenance = select(json.loads(report_path.read_text()))
    output = Path(args.out).resolve()
    if output.exists():
        raise ValueError(f'Refusing to overwrite existing directory: {output}')
    manifest = {
        'schema': 'nh3_orientation_benchmark_v1', 'scientific_success': False,
        'purpose': 'targeted method-disagreement diagnostic, not a held-out accuracy benchmark',
        'selection': 'largest valid NH3 placement spread per material; low/high energy endpoints',
        'source_report_sha256': digest(report_path), 'model_sha256': provenance['model_sha256'],
        'geometry_exported': not args.select_only, 'pairs': [],
        'limitations': ['Provisional surface models; material identity and phase not independently validated.',
                       'Placement spread is sampling sensitivity, not calibrated predictive uncertainty.',
                       'Reference energies must use the same frozen periodic geometries and consistent settings.',
                       'These four biased stress cases cannot establish general chemical accuracy.'],
    }
    # Validate all inputs before creating an output directory.
    exports = []
    for pair in pairs:
        source = next(s for s in provenance['surface_admission']['surfaces'] if s['id'] == pair['surface_id'])
        pair['surface_sha256'] = source['sha256']
        pair['cases'] = []
        for member in pair.pop('members'):
            attempt = member['attempt']
            name = f"{pair['material']}_site{pair['site']}_{member['label']}"
            path = Path(attempt['geometry'])
            if args.repo_root:
                # Relocate recorded absolute cluster paths without guessing basenames.
                parts = path.parts
                if 'runs' not in parts:
                    raise ValueError(f'Cannot relocate non-runs path: {path}')
                path = Path(args.repo_root).resolve().joinpath(*parts[parts.index('runs'):])
            case = {'id': name, 'label': member['label'], 'source_geometry': str(path),
                    'mace_screening_energy_eV': attempt['energy_eV'],
                    'orientation': attempt['orientation'], 'conformer': attempt['conformer']}
            if not args.select_only:
                from ase.io import read
                import numpy as np
                surface_path = Path(source['path'])
                if args.repo_root:
                    parts = surface_path.parts
                    if 'runs' not in parts:
                        raise ValueError('Surface path has no runs component')
                    surface_path = Path(args.repo_root).resolve().joinpath(*parts[parts.index('runs'):])
                if digest(surface_path) != source['sha256']:
                    raise ValueError(f'Surface checksum mismatch: {surface_path}')
                slab, atoms = read(surface_path), read(path)
                if len(atoms) != len(slab) + 4 or atoms[:len(slab)].get_chemical_symbols() != slab.get_chemical_symbols():
                    raise ValueError(f'Unexpected substrate/adsorbate ordering: {path}')
                if atoms[-4:].get_chemical_formula() != 'H3N':
                    raise ValueError(f'Expected intact NH3 atom counts: {path}')
                if not np.isfinite(atoms.positions).all() or not np.allclose(atoms.cell, slab.cell) or not np.array_equal(atoms.pbc, slab.pbc):
                    raise ValueError(f'Nonfinite coordinates or changed cell: {path}')
                symbols = atoms[-4:].get_chemical_symbols()
                n = len(slab) + symbols.index('N')
                distances = [atoms.get_distance(n, len(slab) + i, mic=True) for i, s in enumerate(symbols) if s == 'H']
                if not all(0.8 < d < 1.3 for d in distances):
                    raise ValueError(f'NH3 connectivity requires review: {path}: {distances}')
                atoms.calc = None
                atoms.set_constraint()
                atoms.info.clear()
                filename = name + '.extxyz'
                case.update(source_geometry_sha256=digest(path), geometry=filename)
                exports.append((filename, atoms))
            pair['cases'].append(case)
        manifest['pairs'].append(pair)
    output.mkdir(parents=True)
    if exports:
        from ase.io import write
        for filename, atoms in exports:
            write(output / filename, atoms, format='extxyz')
        for pair in manifest['pairs']:
            for case in pair['cases']:
                case['geometry_sha256'] = digest(output / case['geometry'])
    (output / 'cases.json').write_text(json.dumps(manifest, indent=2) + '\n')
    template = {'method': {'code': '', 'version': '', 'functional': '', 'dispersion': '',
                          'basis_or_cutoff': '', 'pseudopotentials': '', 'kpoints': '',
                          'electronic_tolerance': '', 'periodicity_and_dipole': '',
                          'convergence_evidence': '', 'frozen_geometry': True},
                'cases': [{'id': c['id'], 'geometry_sha256': c.get('geometry_sha256', ''),
                           'energy_eV': None, 'converged': False}
                          for p in manifest['pairs'] for c in p['cases']]}
    (output / 'reference-template.json').write_text(json.dumps(template, indent=2) + '\n')
    for pair in manifest['pairs']:
        print(f"{pair['material']} site {pair['site']}: gap {pair['mace_gap_eV']:.6f} eV")
    print('Manifest:', output / 'cases.json')


def compare(args):
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    if not manifest['geometry_exported']:
        raise ValueError('Export and hash the actual geometries before comparing energies')
    reference = json.loads(Path(args.reference).read_text())
    method = reference['method']
    if method.get('frozen_geometry') is not True:
        raise ValueError('Reference calculations must be frozen-geometry single points')
    for key in ('code', 'version', 'functional', 'dispersion', 'basis_or_cutoff', 'pseudopotentials', 'kpoints', 'electronic_tolerance', 'periodicity_and_dipole', 'convergence_evidence'):
        if not isinstance(method.get(key), str) or not method[key].strip():
            raise ValueError(f'Missing method metadata: {key}')
    cases = reference['cases']
    energies = {c['id']: c for c in cases}
    expected = {c['id'] for p in manifest['pairs'] for c in p['cases']}
    if len(energies) != len(cases) or set(energies) != expected:
        raise ValueError('Reference case IDs must match the manifest exactly without duplicates')
    rows = []
    for pair in manifest['pairs']:
        values = {}
        for case in pair['cases']:
            entry = energies[case['id']]
            actual = digest(manifest_path.parent / case['geometry'])
            if actual != case['geometry_sha256'] or entry['geometry_sha256'] != actual:
                raise ValueError(f"Geometry checksum mismatch: {case['id']}")
            if entry.get('converged') is not True or not finite(entry.get('energy_eV')):
                raise ValueError(f"Missing finite converged energy: {case['id']}")
            values[case['label']] = entry['energy_eV']
        gap = values['high'] - values['low']
        rows.append({'material': pair['material'], 'site': pair['site'],
                     'mace_high_minus_low_eV': pair['mace_gap_eV'], 'reference_high_minus_low_eV': gap,
                     'gap_disagreement_eV': gap - pair['mace_gap_eV'],
                     'reference_prefers_mace_low': gap > 0})
    result = {'scientific_success': False, 'scope': manifest['purpose'], 'method': method,
              'manifest_sha256': digest(manifest_path), 'reference_sha256': digest(args.reference), 'comparisons': rows}
    out = Path(args.out)
    if out.exists():
        raise ValueError(f'Refusing to overwrite {out}')
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--report', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--repo-root', help='Relocate recorded absolute paths under this repository root')
    p.add_argument('--select-only', action='store_true', help='Inspect selections without reading geometries; cannot compare this manifest')
    p.set_defaults(function=prepare)
    p = commands.add_parser('compare')
    p.add_argument('--manifest', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--out', required=True)
    p.set_defaults(function=compare)
    args = parser.parse_args()
    try:
        args.function(args)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f'Benchmark stopped: {exc}\n')


if __name__ == '__main__':
    main()
