"""Extract pinned acquisition inputs from an existing real screening report."""
import argparse
from pathlib import Path
from warden.common import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default='runs/reference_smoke_v2/report.json')
    parser.add_argument('--out', default='runs/adaptive_inputs_v1')
    args = parser.parse_args()
    report = read_json(args.report)
    results = [r for r in report['results'] if r.get('mode') == 'mace']
    if not results:
        parser.error('No MACE seed results')
    provenance = results[0]['provenance']
    if any(r['provenance']['surface_manifest'] != provenance['surface_manifest'] or
           r['provenance']['model_sha256'] != provenance['model_sha256'] for r in results):
        parser.error('Seed results must use one surface manifest and checkpoint')
    admission = provenance['surface_admission']
    entries = [{'id': s['id'], 'material': s['material'], 'path': s['path'],
                'expected_sha256': s['sha256'], 'stage': 'passivated_slab', 'provenance': s['provenance']}
               for s in admission['surfaces']]
    config = dict(provenance['config'])
    objective = config.pop('objective', {})
    output = Path(args.out)
    if output.exists():
        parser.error('Use a new output directory; existing inputs are not overwritten')
    write_json(output / 'surfaces.json', {'validation_profile': admission['validation_profile'], 'surfaces': entries})
    write_json(output / 'config.json', {'engine': config, 'objective': objective})
    print('Prepared pinned inputs:', output)
    print('Run the acquisition plan command to validate admission and candidate identities.')


if __name__ == '__main__':
    main()
