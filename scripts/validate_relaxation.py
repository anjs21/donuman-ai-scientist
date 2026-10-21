"""GPU relaxation diagnostics. Does not create screening or ranking records.

Run from the repository root: python -m scripts.validate_relaxation --help
The reference slabs may fail admission; access here is explicitly diagnostic.
"""
import argparse
import json
import os
from pathlib import Path
from importlib.metadata import version

import numpy as np
from ase.io import write
from ase.optimize import BFGS

from warden.__main__ import configuration
from warden.chemistry import validate_candidate, geometries
from warden.common import file_hash, write_json
from warden.engine import calculator, graph, overlaps
from warden.surfaces import load_manifest


def optimize_case(name, atoms, calc, output, expected_bonds, steps, fmax):
    """Return numerical and identity checks, saving failed geometries too."""
    atoms = atoms.copy()
    atoms.set_constraint()  # Explicit, unconstrained fixed-cell diagnostic.
    atoms.calc = calc
    initial_cell = atoms.cell.array.copy()
    initial_formula = atoms.get_chemical_formula()
    row = {'name': name, 'formula': initial_formula, 'n_atoms': len(atoms),
           'status': 'failed', 'fmax_threshold_eV_A': fmax}
    path = output / name.replace('/', '_')
    try:
        if overlaps(atoms):
            raise ValueError('Initial atom overlap')
        initial_energy = float(atoms.get_potential_energy())
        initial_forces = atoms.get_forces()
        if not np.isfinite(initial_energy) or not np.isfinite(initial_forces).all():
            raise ValueError('Nonfinite initial energy/forces')
        row['initial_energy_eV'] = initial_energy
        row['initial_max_force_eV_A'] = float(np.linalg.norm(initial_forces, axis=1).max())
        # One central-difference check of the largest force component.
        i, axis = np.unravel_index(np.abs(initial_forces).argmax(), initial_forces.shape)
        positions = atoms.positions.copy()
        displacement = 1e-3
        energies = []
        try:
            for sign in (1., -1.):
                displaced = positions.copy()
                displaced[i, axis] += sign * displacement
                atoms.set_positions(displaced)
                energies.append(float(atoms.get_potential_energy()))
        finally:
            atoms.set_positions(positions)
        numerical_force = -(energies[0] - energies[1]) / (2 * displacement)
        error = abs(numerical_force - initial_forces[i, axis])
        row['force_difference_eV_A'] = float(error)
        if not np.isfinite(error) or error > 0.005:
            raise ValueError('Energy/force central-difference check failed')
        optimizer = BFGS(atoms, logfile=str(path) + '.log', maxstep=0.2)
        def progress():
            if optimizer.nsteps % 25 == 0:
                residual = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
                print(f'{name}: step={optimizer.nsteps}, fmax={residual:.5f}', flush=True)
        optimizer.attach(progress, interval=1)
        converged = bool(optimizer.run(fmax=fmax, steps=steps))
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        residual = float(np.linalg.norm(forces, axis=1).max())
        row.update(optimizer_converged=converged, steps=optimizer.nsteps,
                   final_energy_eV=energy, final_max_force_eV_A=residual,
                   energy_change_eV=energy-initial_energy,
                   fixed_cell=bool(np.allclose(initial_cell, atoms.cell.array)),
                   connectivity_preserved=(graph(atoms) == expected_bonds) if expected_bonds is not None else None)
        if not np.isfinite(energy) or not np.isfinite(forces).all():
            raise ValueError('Nonfinite optimized energy/forces')
        if not converged or residual > fmax * 1.001:
            raise ValueError('Optimization did not meet force threshold')
        if overlaps(atoms):
            raise ValueError('Optimized atom overlap')
        if row['connectivity_preserved'] is False:
            raise ValueError('Molecular connectivity changed under the distance-based identity check')
        if not row['fixed_cell'] or atoms.get_chemical_formula() != initial_formula:
            raise ValueError('Cell/composition unexpectedly changed')
        row['status'] = 'pass'
    except Exception as exc:
        row['reason'] = str(exc)
    finally:
        # No calculator results in the exported geometry; status is in the report.
        geometry = Path(str(path) + '.extxyz')
        write(geometry, atoms.copy(), format='extxyz')
        row['geometry'] = str(geometry.resolve())
    return row, atoms.copy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='config/v2-r2scan.local.json')
    p.add_argument('--manifest', default='examples/surfaces-reference.json')
    p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--fmax', type=float, default=0.05)
    args = p.parse_args()
    if args.steps < 1 or args.fmax <= 0:
        p.error('Positive steps and fmax are required')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('This diagnostic must run in your GPU allocation')
    out = Path(args.out).resolve()
    if (out/'relaxation_report.json').exists():
        raise RuntimeError('Output already contains a report. Use a new output directory.')
    out.mkdir(parents=True, exist_ok=True)
    config = configuration(args.config)
    settings = config['engine']
    if settings.get('device') != 'cuda':
        raise ValueError('This GPU diagnostic requires engine.device=cuda')
    report = {'scientific_success': False, 'purpose': 'fixed_cell_relaxation_diagnostic',
              'torch': torch.__version__, 'mace': version('mace-torch'),
              'gpu': torch.cuda.get_device_name(0),
              'checkpoint_sha256': file_hash(settings['model_path']),
              'steps_limit_per_case': args.steps, 'force_threshold_eV_A': args.fmax,
              'surface_density_gate_modified': False,
              'results': [], 'status': 'running'}
    def save(): write_json(out/'relaxation_report.json', report)
    save()
    calc = calculator(settings)
    # No free retained fragments: those are not gas references in this reaction.
    for filename in ('examples/seeds-v2.json', 'examples/precursors-v2.json'):
        for raw in json.loads(Path(filename).read_text()):
            candidate = validate_candidate(raw)
            geo = geometries(candidate, 1, 42)[0]
            for part in ('molecule', 'byproduct'):
                if geo[part] is None:
                    continue
                atoms = geo[part].copy()
                atoms.set_pbc(False)
                atoms.center(vacuum=8.)
                expected = ({tuple(sorted(b)) for b in geo['molecule_bonds']}
                            if part == 'molecule' else graph(atoms))
                row, _ = optimize_case(candidate['name']+'/'+part, atoms, calc, out,
                                       expected, args.steps, args.fmax)
                report['results'].append(row); save()
                print(json.dumps(row), flush=True)
    input_report, surfaces = load_manifest(args.manifest, require_valid=False)
    write_json(out/'input_admission.json', input_report)
    manifest_rows = []
    for surface in input_report['surfaces']:
        name = 'surface/'+surface['id']
        row, relaxed = optimize_case(name, surfaces[surface['id']], calc, out,
                                     None, args.steps, args.fmax)
        row['input_admitted'] = surface['valid']
        report['results'].append(row); save()
        print(json.dumps(row), flush=True)
        if row['status'] == 'pass':
            manifest_rows.append({'id': surface['id']+'_relaxed', 'material': surface['material'],
                'path': row['geometry'], 'format': 'extxyz', 'stage': 'passivated_slab',
                'provenance': {'type': 'diagnostic_fixed_cell_relaxation',
                    'input_sha256': surface['sha256'], 'input_provenance': surface['provenance'],
                    'checkpoint_sha256': report['checkpoint_sha256'],
                    'note': 'Relaxation alone does not establish surface validity.'}})
    report['relaxation_checks_passed'] = all(r['status']=='pass' for r in report['results'])
    if len(manifest_rows) == len(input_report['surfaces']):
        manifest = out/'relaxed_surfaces.json'
        write_json(manifest, {'surfaces': manifest_rows})
        try:
            admission, _ = load_manifest(manifest, require_valid=False)
            write_json(out/'relaxed_admission.json', admission)
            report['relaxed_surfaces_admitted'] = admission['valid']
        except Exception as exc:
            report['relaxed_surfaces_admitted'] = False
            report['admission_error'] = str(exc)
    else:
        report['relaxed_surfaces_admitted'] = False
    report['status'] = ('technical_checks_passed' if report['relaxation_checks_passed']
                        and report['relaxed_surfaces_admitted'] else 'needs_review')
    save()
    print('FINAL:', report['status'], '| report:', out/'relaxation_report.json', flush=True)
    print('No candidate ranking or experimental success is claimed.', flush=True)
    return 0 if report['status']=='technical_checks_passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
