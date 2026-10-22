"""Prepare a pinned, explicitly provisional smoke screen from the reviewed upload.

This script accepts only the exact relaxed geometries reviewed in this update.
It does not claim original author data provenance or bulk/experimental validity.
"""
import argparse
import json
from pathlib import Path

from warden.__main__ import configuration
from warden.common import file_hash, write_json
from warden.surfaces import load_manifest
from warden.slab_validation import PROFILE

REVIEWED = {
    'SiO2': '7beef8468980d61a53d1995d2be697a4e3cd0a64b9b17f4d00110e995f437014',
    'SiNx': 'cacae0779d47a7702a87f376a9c00c9b2be303e801bdc1e9d5c5527818399830',
}
REVIEWED_CHECKPOINT = '8f147ecffa1d06a696e5648b13b81abda9c96fda9d0fb3faba3cd3fb27d9bba9'


def prepare(relaxation_dir, config_path, out):
    relaxation_dir, out = Path(relaxation_dir).resolve(), Path(out).resolve()
    if (out/'campaign.sqlite').exists():
        raise ValueError('This output already has a campaign database. Use a new --out.')
    old = json.loads((relaxation_dir/'relaxation_report.json').read_text())
    if not old.get('relaxation_checks_passed') or old.get('checkpoint_sha256') != REVIEWED_CHECKPOINT:
        raise ValueError('Expected the successful reviewed relaxation run with the recorded checkpoint')
    config = configuration(config_path)
    checkpoint = config['engine'].get('model_path', '')
    if file_hash(checkpoint) != REVIEWED_CHECKPOINT:
        raise ValueError('Checkpoint differs from the one used for the reviewed relaxed geometries')
    if config['engine'].get('device') != 'cuda':
        raise ValueError('Set engine.device=cuda in your local config for this GPU smoke test')
    entries = []
    for material, checksum in REVIEWED.items():
        path = relaxation_dir/f'surface_{material}_reference.extxyz'
        if file_hash(path) != checksum:
            raise ValueError(f'{material}: geometry differs from the reviewed upload')
        matching = [r for r in old['results'] if r.get('name') == 'surface/'+material+'_reference']
        if len(matching) != 1 or matching[0].get('status') != 'pass' or matching[0].get('final_max_force_eV_A', 1e9) > .05:
            raise ValueError(f'{material}: missing/failed convergence record')
        entries.append({'id': material+'_reviewed_relaxed', 'material': material,
            'path': str(path), 'format': 'extxyz', 'stage': 'passivated_slab',
            'expected_sha256': checksum,
            'provenance': {'source': 'user-supplied relaxation_49543552 archive',
                'checkpoint_sha256': old['checkpoint_sha256'],
                'relaxation_report_sha256': file_hash(relaxation_dir/'relaxation_report.json'),
                'original_input': 'Structures bundled in donuman-ai-scientist/zenodo_structures',
                'original_author_data_provenance': 'unverified',
                'parent_bulk_validation': 'not_supplied',
                'scope': 'provisional numerical screening only; not validated reference materials'}})
    out.mkdir(parents=True, exist_ok=True)
    manifest = out/'surfaces.json'
    write_json(manifest, {'validation_profile': PROFILE, 'surfaces': entries})
    admission, _ = load_manifest(manifest, require_valid=False)
    write_json(out/'surface_validation.json', admission)
    if not admission['valid']:
        raise ValueError('Finite-slab screening checks failed; inspect surface_validation.json')
    # A bounded integration test. One representative per face/site-type, not one total.
    config['engine'].update(max_sites=1, n_conformers=1, orientations=2, steps=400,
                            fmax=.05, min_site_success_fraction=.8)
    write_json(out/'config.json', config)
    seeds = json.loads(Path('examples/seeds-v2.json').read_text())
    selected = [c for c in seeds if c['name'] in {'DMATMS', 'NH3'}]
    if len(selected) != 2:
        raise ValueError('Expected DMATMS and NH3 control definitions')
    write_json(out/'candidates.json', selected)
    print('Prepared:', out)
    print('Surface screening admission: PASS | bulk density remains unverified')
    print('Candidates: DMATMS and NH3 controls; no literature-performance assertion')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--relaxation-dir', default='runs/relaxation_49543552')
    p.add_argument('--config', default='config/v2-r2scan.local.json')
    p.add_argument('--out', default='runs/reference_smoke_v1')
    a = p.parse_args()
    prepare(a.relaxation_dir, a.config, a.out)


if __name__ == '__main__':
    main()
