"""Reproducible v2 entry points. Run from the repository root."""
import argparse
import json
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
    ids = []
    for row in read_json(path):
        if not row.get('text') or not row.get('source'):
            raise ValueError('Evidence needs text and source')
        ids.append(store.evidence(row['text'], row['source'], row.get('kind', 'curated_summary')))
        src = row['source'] if isinstance(row['source'], dict) else {'source': row['source']}
        # IDs hash text+source+kind, so the same file gives the same IDs in every database.
        print(ids[-1], src.get('tool', ''), json.dumps(src.get('arguments', src.get('source', '')))[:100])
    return ids


def lit_brief_command(args):
    from . import kinetics as kin
    from .lit import lit_brief, brief_markdown
    config = read_json(args.config)
    rules = {**kin.DEFAULTS, **config.get('kinetics', {})}
    from .lit import gather
    store = Store(args.db)
    try:
        candidates = {c['id']: c for c in store.all('candidate')}
        records, skipped = gather([kin.reclassify(r, rules) for r in store.all('kinetics')],
                                  store.all('observation'), candidates, rules)
    finally:
        store.close()
    brief = lit_brief(records)
    if skipped:
        print(f"note: {skipped} agent observations skipped (unknown site material or candidate)")
    out = Path(args.out); (out / 'queries').mkdir(parents=True, exist_ok=True)
    write_json(out / 'brief.json', brief)
    (out / 'brief.md').write_text(brief_markdown(brief))
    for i, q in enumerate(brief['questions'], 1):
        # Starting arguments for `warden bond-call`; check the live schema from `bond-tools`
        # and add fields it requires (e.g. a source or limit) before calling.
        write_json(out / 'queries' / f'q{i:02d}.json', {'query': q['query']})
    print(f"Literature brief: {out / 'brief.md'} | {len(brief['questions'])} questions, "
          f"{len(brief['method_actions'])} method actions")
    return 0


def lit_import_command(args):
    from .lit import import_candidates
    store = Store(args.db)
    try:
        evidence = {e['id']: e for e in store.all('evidence')}
        known = {r.get('canonical_smiles') for r in store.all('candidate')}
        known |= {r.get('canonical_smiles') for r in store.all('literature_candidate')}
        from .chemistry import validate_candidate
        for k in store.all('kinetics'):
            try:
                known.add(validate_candidate({'name': 'x', 'smiles': k['smiles'], 'template': k['template']})['canonical_smiles'])
            except (ValueError, KeyError):
                pass
        known.discard(None)
        library, rejected = import_candidates(read_json(args.candidates), evidence, known)
        for spec in library:
            store.put('literature_candidate', spec['id'], spec)
        store.event('literature_import', {'accepted': [s['name'] for s in library], 'rejected': rejected})
    finally:
        store.close()
    write_json(args.out, library)
    for r in rejected:
        print(f"rejected {r['name']}: {r['reason']}")
    print(f"Library: {args.out} | {len(library)} accepted "
          f"({sum(s['literature_status'] == 'tested' for s in library)} literature-tested), {len(rejected)} rejected")
    return 0 if library else 2


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
            print('Screening:', candidate['name'], '| fidelity:', args.fidelity, flush=True)
            try:
                result = screen(candidate, report, atoms, settings, calc,
                                Path(args.db).resolve().parent / 'geometries' / key)
            except Exception as exc:
                result = {'candidate_id': cid, 'mode': 'mace', 'status': 'failed',
                          'reason': str(exc), 'dE_GS': None, 'dE_NGS': None}
            result.update(id=key, fidelity=args.fidelity, role=candidate['role'], provenance=signature)
            store.put('result', key, result)
            print('Result:', candidate['name'], result['status'], flush=True)
            store.event('calculation_finished', {'id': key, 'status': result['status']})
            export(store, Path(args.db).resolve().parent)
        export(store, Path(args.db).resolve().parent)
    finally:
        store.close()


def kinetics_command(args):
    from .chemistry import validate_candidate
    from .common import file_hash
    from . import kinetics as kin
    config = configuration(args.config)
    settings = {**kin.DEFAULTS, **config.get('kinetics', {})}
    if args.method:
        settings['method'] = args.method
    if settings['method'] == 'mace':
        settings = {**config.get('engine', {}), **settings}
    store = Store(args.db)
    try:
        ids = args.ids or []
        if args.candidates:
            for raw in read_json(args.candidates):
                candidate = validate_candidate(raw)
                store.put('candidate', candidate['id'], candidate)
                ids.append(candidate['id'])
        admitted, fingerprint = True, 'gas_models'
        if args.sites == 'manifest':
            from .surfaces import load_manifest
            report, atoms = load_manifest(args.manifest, require_valid=not args.allow_unadmitted)
            admitted, fingerprint = report['valid'], report['fingerprint']
            sites = kin.sites_for('manifest', report, atoms, settings.get('max_sites', 2),
                                  settings['cluster_shells'])
        else:
            sites = kin.sites_for('gas')
        for cid in ids:
            candidate = store.get('candidate', cid)
            if candidate is None:
                raise ValueError('Unknown candidate: ' + cid)
            if candidate['template'] not in kin.SUBSTITUTION:
                store.event('kinetics_skipped', {'candidate_id': cid, 'reason': 'no reaction to compute a barrier for'})
                continue
            key = digest({'candidate': {k: candidate[k] for k in ('smiles', 'template')},
                          'sites': fingerprint, 'settings': settings,
                          'code': file_hash(kin.__file__)})
            if store.get('kinetics', key):
                store.event('cache_hit', {'id': key, 'candidate_id': cid}); continue
            store.event('kinetics_started', {'id': key, 'candidate_id': cid, 'method': settings['method']})
            result = kin.screen_kinetics(candidate, sites, settings,
                                         Path(args.db).resolve().parent / 'paths' / key)
            result.update(id=key, sites_source=args.sites, surface_admission=admitted,
                          surface_fingerprint=fingerprint)
            store.put('kinetics', key, result)
            store.event('kinetics_finished', {'id': key, 'candidate_id': cid,
                                              'summary': result['summary']})
            print(candidate['name'], '->', result['summary'].get('dEa_GS_minus_NGS_eV'))
    finally:
        store.close()


def assess_command(args):
    from dataclasses import fields
    from .assess import assess, markdown
    from .process import ProcessParams
    config = configuration(args.config)
    known = {f.name for f in fields(ProcessParams)}
    params = ProcessParams(**{k: v for k, v in config.get('process', {}).items() if k in known})
    options = dict(config.get('design', {}))
    for key in ('temperatures', 'doses', 'intervals'):
        if key in options:
            options[key] = tuple(options[key])
    store = Store(args.db)
    try:
        from .kinetics import DEFAULTS, reclassify
        rules = {**DEFAULTS, **config.get('kinetics', {})}
        # Stored NEB profiles are re-judged under the current rules; no recomputation.
        records = [reclassify(r, rules) for r in store.all('kinetics')]
        controls = read_json(args.controls) if args.controls else []
        result = assess(records, controls, params, options)
        store.put('assessment', 'latest', result)
        store.event('assessment', {'label': result['label'],
                                   'top': result['ranking'][0]['name'] if result['ranking'] else None})
    finally:
        store.close()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'assessment.json', result)
    (out / 'assessment.md').write_text(markdown(result))
    print('Assessment:', result['label'], '|', out / 'assessment.md')


def decide_command(args):
    from dataclasses import asdict
    from . import decide as dc
    from . import kinetics as kin
    from .chemistry import validate_candidate
    config = configuration(args.config)
    cfg = {**dc.DEFAULTS, **config.get('decide', {})}
    store = Store(args.db)
    try:
        if args.sites == 'manifest':
            from .surfaces import load_manifest
            report, atoms = load_manifest(args.manifest, require_valid=not args.allow_unadmitted)
            sites = kin.sites_for('manifest', report, atoms, config.get('kinetics', {}).get('max_sites', 2),
                                  config.get('kinetics', {}).get('cluster_shells', 3))
        else:
            sites = kin.sites_for('gas')
        candidates = {c['id']: c for c in store.all('candidate') if c['template'] in kin.SUBSTITUTION}
        library = []
        evidence = {e['id']: e for e in store.all('evidence')}
        for raw in (read_json(args.library) if args.library else []):
            # Literature-backed entries (from lit-import) carry evidence references,
            # re-checked against this database's evidence.
            spec = validate_candidate(raw, evidence)
            if spec['template'] in kin.SUBSTITUTION and spec['id'] not in candidates:
                library.append(spec)
        if args.candidates:
            for raw in read_json(args.candidates):
                spec = validate_candidate(raw)
                store.put('candidate', spec['id'], spec); candidates[spec['id']] = spec
        belief = dc.Belief(sites, candidates, cfg)
        rules = {**kin.DEFAULTS, **config.get('kinetics', {})}
        seeds = dc.observations_from_kinetics([kin.reclassify(r, rules) for r in store.all('kinetics')
                                               if r['candidate_id'] in candidates], rules)
        for o in seeds + [dc.Observation(**r) for r in store.all('observation')]:
            if o.candidate in candidates and any(s['label'] == o.site for s in belief.sites):
                belief.add(o)
        methods = args.methods
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            agent = dc.Agent(belief, methods, library, cfg)
            action, shortlist = agent.choose(args.budget)
            result = {"pick": None, "pcs": belief.pcs(), "spent": 0., "budget": args.budget,
                      "stop": "dry_run", "scores": belief.scores(), "trace": [],
                      "methods": belief.methods.report(), "failures": belief.failures.report()}
            result["next"] = [f"{a.kind} {candidates.get(a.candidate, {}).get('name', a.extra.get('name', ''))} "
                              f"{a.site or ''} {a.pathway or ''} {a.method or ''}: value/cost {a.score():.3g}, "
                              f"P(success) {a.p_success:.2f}" for a in shortlist]
        else:
            executor = dc.LiveExecutor(sites, config.get('kinetics', {}), str(out / 'paths'), cfg['methods'])

            def log(record):
                store.event('decision', record)
                if record.get('result'):
                    store.put('observation', digest(record['result'] | {'step': record['step']}), record['result'])
                print(record['step'], record['action']['kind'], record['action'].get('site'),
                      record['action'].get('method'), '->', (record.get('result') or {}).get('status', record.get('added')))
            result = dc.run(belief, executor, args.budget, library, methods, cfg, log)
            result["budget"] = args.budget
            result["stop"] = next((t["stop"] for t in reversed(result["trace"]) if "stop" in t), "max_steps")
            for cid, spec in belief.candidates.items():
                store.put('candidate', cid, spec)
        result["names"] = {c: s.get('name', c) for c, s in belief.candidates.items()}
        result["contention"] = belief.contention() if len(belief.candidates) > 1 else {}
        if belief.candidates:
            scores = belief.scores()
            result["pick"] = result.get("pick") or max(scores, key=scores.get)
            result["robustness"] = dc.site_robustness(belief)
            result["process"] = dc.posterior_design(belief, result["pick"],
                                                    kin.molecular_mass(belief.candidates[result["pick"]]))
        result["note"] = ("Decisions use a hierarchical Gaussian belief with stated priors; "
                          "method biases are learned only from results paired with the reference method.")
        write_json(out / 'decisions.json', result)
        (out / 'decisions.md').write_text(dc.markdown(result))
        print('Decision report:', out / 'decisions.md')
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
    bc = commands.add_parser('bond-call', help='Call an allowlisted literature tool and save campaign evidence')
    bc.add_argument('--config', required=True); bc.add_argument('--tool', required=True)
    bc.add_argument('--arguments', required=True, help='Path to a JSON object matching the discovered tool schema')
    bc.add_argument('--out', required=True, help='New evidence JSON file; existing files are never overwritten')
    lb = commands.add_parser('lit-brief', help='Turn computed outcomes into literature questions (no LLM)')
    lb.add_argument('--db', required=True); lb.add_argument('--config', required=True)
    lb.add_argument('--out', default='runs/literature')
    li = commands.add_parser('lit-import', help='Validate literature-backed candidates into a decide library')
    li.add_argument('--db', required=True); li.add_argument('--candidates', required=True)
    li.add_argument('--out', required=True, help='Library JSON for `warden decide --library`')
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
            x.add_argument('--bond'); x.add_argument('--evidence'); x.add_argument('--controls')
    k = commands.add_parser('kinetics', help='Site-resolved reaction barriers (xTB or MACE NEB)')
    k.add_argument('--db', required=True); k.add_argument('--config', default='config/v3.json')
    group = k.add_mutually_exclusive_group(required=True)
    group.add_argument('--candidates'); group.add_argument('--ids', nargs='+')
    k.add_argument('--sites', choices=['gas', 'manifest'], default='gas')
    k.add_argument('--manifest'); k.add_argument('--method', choices=['xtb', 'mace'])
    k.add_argument('--allow-unadmitted', action='store_true',
                   help='Diagnostic only: results are excluded from ranking')
    a = commands.add_parser('assess', help='Control-gated, uncertainty-aware process design and ranking')
    a.add_argument('--db', required=True); a.add_argument('--config', default='config/v3.json')
    a.add_argument('--controls'); a.add_argument('--out', default='runs/assess')
    dd = commands.add_parser('decide', help='Evidence-acquisition agent: choose and run the next calculations')
    dd.add_argument('--db', required=True); dd.add_argument('--config', default='config/v3.json')
    dd.add_argument('--sites', choices=['gas', 'manifest'], default='gas'); dd.add_argument('--manifest')
    dd.add_argument('--allow-unadmitted', action='store_true')
    dd.add_argument('--candidates'); dd.add_argument('--library')
    dd.add_argument('--methods', nargs='+', default=['xtb'], choices=['xtb', 'mace', 'dft'])
    dd.add_argument('--budget', type=float, default=10.); dd.add_argument('--out', default='runs/decide')
    dd.add_argument('--dry-run', action='store_true', help='Only report what the agent would compute next')
    bm = commands.add_parser('benchmark', help='Equal-budget policy comparison on synthetic worlds')
    bm.add_argument('--worlds', type=int, default=30); bm.add_argument('--budget', type=float, default=60.)
    bm.add_argument('--methods', nargs='+', default=['xtb', 'dft']); bm.add_argument('--out', default='runs/benchmark')
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
            print('Surface screening admission:', 'PASS' if report['valid'] else 'FAIL', '|', args.out)
            print('Scope:', report['admission_scope'], '| bulk density validated:', report['bulk_density_validated'])
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
        if args.command == 'bond-call':
            from datetime import datetime, timezone
            from .literature import Bond
            from .common import dumps
            destination = Path(args.out)
            if destination.exists():
                raise ValueError('Evidence output already exists; choose a new --out path')
            arguments = read_json(args.arguments)
            if not isinstance(arguments, dict):
                raise ValueError('Tool arguments must be a JSON object')
            result = Bond(read_json(args.config)).call(args.tool, arguments)
            evidence = [{'kind': 'bond_result', 'text': dumps(result),
                         'source': {'provider': 'BOND', 'tool': args.tool, 'arguments': arguments,
                                    'retrieved_at': datetime.now(timezone.utc).isoformat()}}]
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('x') as output:
                import json
                json.dump(evidence, output, indent=2, allow_nan=False)
                output.write('\n')
            print('Literature evidence:', destination)
            return 0
        if args.command == 'lit-brief':
            return lit_brief_command(args)
        if args.command == 'lit-import':
            return lit_import_command(args)
        if args.command == 'screen':
            screen_command(args); return 0
        if args.command == 'kinetics':
            if args.sites == 'manifest' and not args.manifest:
                raise ValueError('--sites manifest needs --manifest')
            kinetics_command(args); return 0
        if args.command == 'assess':
            assess_command(args); return 0
        if args.command == 'decide':
            if 'dft' in args.methods:
                raise ValueError('No DFT backend is configured here; benchmark it with `warden benchmark`')
            decide_command(args); return 0
        if args.command == 'benchmark':
            from .benchmark import benchmark
            r = benchmark(args.worlds, args.budget, methods=tuple(args.methods))
            write_json(Path(args.out) / 'benchmark.json', r)
            for row in r['summary']:
                print(f"{row['policy']:22s} P(correct) {row['p_correct']:.2f}  regret {row['mean_regret_decades']:.2f}  "
                      f"spent {row['mean_spent']:.1f}  failures {row['mean_failures']:.1f}")
            return 0
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
                                           config.get('campaign', {}).get('worker_timeout_s', 1800),
                                           config.get('campaign', {}).get('kinetics', False),
                                           Path(args.controls).resolve() if args.controls else None)
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
