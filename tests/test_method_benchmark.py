"""Behavior checks with synthetic data, not scientific validation."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

spec = importlib.util.spec_from_file_location('benchmark', Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_method_benchmark.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def report():
    sites = []
    for material in ('SiO2', 'SiNx'):
        for site, gap in [(1, 0.2), (2, 0.7)]:
            sites.append({'material': material, 'surface_id': material, 'site': site,
                          'face': 'top', 'site_type': 'OH', 'status': 'valid',
                          'attempts': [{'status': 'valid', 'energy_eV': energy,
                                        'geometry': f'{material}_{site}_{i}.extxyz',
                                        'orientation': i, 'conformer': 0}
                                       for i, energy in enumerate([-1., -1. + gap])]})
    return {'candidates': [{'id': 'NH3', 'name': 'NH3', 'template': 'physisorption'}],
            'results': [{'candidate_id': 'NH3', 'mode': 'mace', 'sites': sites,
                         'provenance': {'model_sha256': 'test', 'surface_admission': {
                             'surfaces': [{'id': m, 'sha256': 'test'} for m in ('SiO2', 'SiNx')]}}}]}


class BenchmarkTests(unittest.TestCase):
    def test_selects_largest_valid_spread(self):
        data = report()
        data['results'][0]['sites'][0]['attempts'].append({'status': 'invalid', 'energy_eV': -100., 'geometry': 'bad'})
        pairs, _ = benchmark.select(data)
        self.assertEqual([p['site'] for p in pairs], [2, 2])
        self.assertAlmostEqual(pairs[0]['mace_gap_eV'], 0.7)

    def test_missing_second_material_stops(self):
        data = report()
        data['results'][0]['sites'] = data['results'][0]['sites'][:2]
        with self.assertRaises(ValueError):
            benchmark.select(data)

    def test_ambiguous_result_stops(self):
        data = report()
        data['results'].append(copy.deepcopy(data['results'][0]))
        with self.assertRaises(ValueError):
            benchmark.select(data)

    def test_compare_rejects_missing_energy_and_modified_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cases = []
            for label in ('low', 'high'):
                path = root / (label + '.extxyz')
                path.write_text('synthetic fixture ' + label)
                cases.append({'id': label, 'label': label, 'geometry': path.name,
                              'geometry_sha256': benchmark.digest(path)})
            manifest = {'geometry_exported': True, 'purpose': 'test only',
                        'pairs': [{'material': 'test', 'site': 1, 'mace_gap_eV': 0.7, 'cases': cases}]}
            mp = root / 'cases.json'
            mp.write_text(json.dumps(manifest))
            method = {key: 'test' for key in ('code', 'version', 'functional', 'dispersion', 'basis_or_cutoff', 'pseudopotentials', 'kpoints', 'electronic_tolerance', 'periodicity_and_dipole', 'convergence_evidence')}
            method['frozen_geometry'] = True
            reference = {'method': method, 'cases': [{**c, 'energy_eV': e, 'converged': True} for c, e in zip(cases, [-10., -10.2])]}
            rp = root / 'reference.json'
            args = types.SimpleNamespace(manifest=mp, reference=rp, out=root / 'comparison.json')
            reference['cases'][0]['energy_eV'] = None
            rp.write_text(json.dumps(reference))
            with self.assertRaisesRegex(ValueError, 'finite converged'):
                benchmark.compare(args)
            reference['cases'][0]['energy_eV'] = -10.
            rp.write_text(json.dumps(reference))
            benchmark.compare(args)
            row = json.loads(args.out.read_text())['comparisons'][0]
            self.assertAlmostEqual(row['reference_high_minus_low_eV'], -0.2)
            self.assertFalse(row['reference_prefers_mace_low'])
            (root / 'low.extxyz').write_text('modified')
            args.out = root / 'other.json'
            with self.assertRaisesRegex(ValueError, 'checksum'):
                benchmark.compare(args)


if __name__ == '__main__':
    unittest.main()
