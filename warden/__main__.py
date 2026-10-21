"""Reproducible v2 entry points. Run from the repository root."""
import argparse
import sys
from pathlib import Path
from .common import read_json, write_json, digest
from .store import Store


def configuration(path):
    path = Path(path).resolve()
    config = read_json(path)
    engine = config.setdefault('engine', {})
    if engine.get('model_path'):
        engine['model_path'] = str((path.parent / engine['model_path']).resolve())
    return config


def add_evidence(store, path):
    for row in read_json(path):
        if not row.get('text') or not row.get('source'):
            raise ValueError('Evidence needs text and source')
        store.evidence(row['text'], row['source'], row.get('kind', 'curated_summary'))


def screen_command(args):
    from .chemistry import validate_candidate
    from .engine import DEFAULTS, calculation_key, calculator, screen
    from .surfaces import load_manifest
    from .report import export
    config = configuration(args.config)
    settings = {**DEFAULTS, **config.get('engine', {}), 'objective': config.get('objective', {})}
    if not 0 <= args.fidelity <= 4:
        raise ValueError('fidelity must be 0..4')
    for field in ('max_sites', 'n_conformers', 'orientations', 'steps'):
        if not isinstance(settings[field], int) or settings[field] < 1:
            raise ValueError(field + ' must be a positive integer')
    for field in ('fmax', 'active_radius_A', 'energy_review_limit_eV'):
        if settings[field] <= 0:
            raise ValueError(field + ' must be positive')
    if not 0 < settings['min_site_success_fraction'] <= 1:
        raise ValueError('min_site_success_fraction must be in (0,1]')
    settings['max_sites'] += args.fidelity
    settings['n_conformers'] += args.fidelity
    report, atoms = load_manifest(args.manifest)
    store = Store(args.db)
    try:
        ids = args.ids or []
        if args.candidates:
            for raw in read_json(args.candidates):
                candidate = validate_candidate(raw)
                store.put('candidate', candidate['id'], candidate)
                ids.append(candidate['id'])
        if not ids:
            raise ValueError('Provide --candidates or --ids')
        calc = None
        for cid in ids:
            candidate = store.get('candidate', cid)
            if candidate is None:
                raise ValueError('Unknown candidate: ' + cid)
            key, signature = calculation_key(candidate, report, settings)
            if store.get('result', key):
                store.event('cache_hit', {'id': key, 'candidate_id': cid})
                continue
            if calc is None:
                calc = calculator(settings)
            store.event('calculation_started', {'id': key, 'candidate_id': cid})
            try:
                result = screen(candidate, report, atoms, settings, calc,
                                Path(args.db).resolve().parent / 'geometries' / key)
            except Exception as exc:
                result = {'candidate_id': cid, 'mode': 'mace', 'status': 'failed',
                          'reason': str(exc), 'dE_GS': None, 'dE_NGS': None}
            result.update(id=key, fidelity=args.fidelity, role=candidate['role'], provenance=signature)
            store.put('result', key, result)
            store.event('calculation_finished', {'id': key, 'status': result['status']})
            export(store, Path(args.db).resolve().parent)
        export(store, Path(args.db).resolve().parent)
    finally:
        store.close()


def main(argv=None):
    p = argparse.ArgumentParser(description='WARDEN v2: evidence-driven AS-ALD screening')
    commands = p.add_subparsers(dest='command', required=True)
    d = commands.add_parser('demo', help='Synthetic offline control-flow test; no scientific predictions')
    d.add_argument('--out', default='runs/demo')
    v = commands.add_parser('validate-surfaces')
    v.add_argument('--manifest', required=True); v.add_argument('--out', default='runs/surface_validation.json')
    c = commands.add_parser('validate-candidates')
    c.add_argument('--candidates', required=True); c.add_argument('--out', default='runs/candidates.json')
    e = commands.add_parser('add-evidence')
    e.add_argument('--db', required=True); e.add_argument('--file', required=True)
    h = commands.add_parser('import-results', help='Import old values as unverified history only')
    h.add_argument('--db', required=True); h.add_argument('--file', required=True)
    b = commands.add_parser('bond-tools', help='Discover actual MCP tool names and schemas')
    b.add_argument('--config', required=True); b.add_argument('--out', default='runs/bond_tools.json')
    r = commands.add_parser('report')
    r.add_argument('--db', required=True); r.add_argument('--out', default='runs/report')
    for name in ('screen', 'campaign'):
        x = commands.add_parser(name)
        x.add_argument('--db', required=True); x.add_argument('--manifest', required=True)
        x.add_argument('--config', default='config/v2.json')
        if name == 'screen':
            group = x.add_mutually_exclusive_group(required=True)
            group.add_argument('--candidates'); group.add_argument('--ids', nargs='+')
            x.add_argument('--fidelity', type=int, default=0)
        else:
            x.add_argument('--bond'); x.add_argument('--evidence')
    args = p.parse_args(argv)
    try:
        if args.command == 'demo':
            from .demo import execute
            state = execute(args.out)
            print('SYNTHETIC DEMO:', state['status'], '| report:', str(Path(args.out) / 'report.md'))
            return 0 if state['status'] == 'finished' else 2
        if args.command == 'validate-surfaces':
            from .surfaces import load_manifest
            report, _ = load_manifest(args.manifest, require_valid=False)
            write_json(args.out, report)
            print('Surface admission:', 'PASS' if report['valid'] else 'FAIL', '|', args.out)
            return 0 if report['valid'] else 2
        if args.command == 'validate-candidates':
            from .chemistry import validate_candidate, geometries
            checked = [validate_candidate(c) for c in read_json(args.candidates)]
            for candidate in checked:
                geometries(candidate, 1, 42)
            write_json(args.out, checked)
            print('Validated', len(checked), 'candidate definitions and reaction atom balances.')
            return 0
        if args.command == 'bond-tools':
            from .literature import Bond
            write_json(args.out, Bond(read_json(args.config)).tools())
            print('Tool inventory:', args.out)
            return 0
        if args.command == 'screen':
            screen_command(args); return 0
        store = Store(args.db)
        try:
            if args.command == 'add-evidence':
                add_evidence(store, args.file)
            elif args.command == 'import-results':
                rows = read_json(args.file)
                if not isinstance(rows, list):
                    raise ValueError('Expected a JSON list; see examples/historical_format.json')
                for row in rows:
                    if not row.get('candidate_id') or not row.get('source'):
                        raise ValueError('Historical records need candidate_id and source')
                    row.update(mode='legacy_unverified', status='unverified', scientific_success=False)
                    store.put('historical', digest(row), row)
            elif args.command == 'campaign':
                from .campaign import run, subprocess_runner
                from .planner import Ollama
                from .literature import Bond
                from .surfaces import load_manifest
                config = configuration(args.config)
                report, _ = load_manifest(args.manifest)
                config['surface_fingerprint'] = report['fingerprint']
                bond = Bond(read_json(args.bond)) if args.bond else None
                config['literature_config'] = bond.config if bond else None
                if args.evidence:
                    add_evidence(store, args.evidence)
                if any(e['kind'] == 'synthetic_demo' for e in store.all('evidence')):
                    raise ValueError('Synthetic evidence cannot be used in a live campaign; use a fresh DB')
                runner = subprocess_runner(Path(args.db).resolve(), Path(args.manifest).resolve(),
                                           Path(args.config).resolve(),
                                           config.get('campaign', {}).get('worker_timeout_s', 1800))
                state = run(store, Ollama(config['llm']), runner, config, bond)
                print('Campaign:', state['status'], '| scientific_success: false')
            from .report import export
            export(store, args.out if args.command == 'report' else Path(args.db).resolve().parent)
        finally:
            store.close()
        return 0
    except Exception as exc:
        print('ERROR:', exc, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
