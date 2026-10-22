"""Budgeted, deterministic MACE acquisition baseline; no LLM or DFT required.

Site observations do not replace whole-candidate screening averages.
"""
import argparse
import math
import random
import time
from pathlib import Path
from .common import digest, file_hash, read_json, write_json


class BudgetStop(BaseException):
    """Escape legacy broad Exception handlers without treating a budget as chemistry."""


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def actions(results, candidates, inventory, observations, poses=2, threshold=.25, max_site_actions=2):
    """Explicit hand-set priorities, not learned probabilities or error estimates."""
    done = {o['action']['id'] for o in observations}
    seen = {r['candidate_id'] for r in results}
    available, reviews = [], []
    for candidate in candidates:
        cid = candidate['id']
        if cid not in seen:
            available.append({'kind': 'screen_candidate', 'candidate_id': cid,
                              'priority': 1., 'reason': 'Untested molecule from the supplied pool'})
            continue
        matches = [r for r in results if r['candidate_id'] == cid]
        if len(matches) != 1:
            raise ValueError('Ambiguous candidate results; supply one chosen result per molecule')
        result = matches[0]
        if result.get('status') != 'complete':
            reviews.append({'candidate_id': cid, 'reason': 'Incomplete/failed screen needs diagnosis; no automatic retry'})
            continue
        visits = sum(o['action']['kind'] == 'sample_site' and o['action']['candidate_id'] == cid for o in observations)
        if visits >= max_site_actions:
            continue
        # New observations affect later priorities without changing seed averages.
        change = 0.
        hold = False
        for observation in observations:
            action = observation['action']
            if action['candidate_id'] != cid or action['kind'] != 'sample_site':
                continue
            energies = [a['energy_eV'] for a in observation.get('attempts', []) if a.get('status') == 'valid' and number(a.get('energy_eV'))]
            original = next((s for s in result.get('sites', []) if s['surface_id'] == action['surface_id'] and s['site'] == action['site']), None)
            if energies and original and number(original.get('energy_eV')):
                change = max(change, abs(min(energies) - original['energy_eV']))
            if not energies:
                hold = True
                reviews.append({'candidate_id': cid, 'site': action['site'], 'surface_id': action['surface_id'],
                                'reason': 'Adaptive site action produced no valid measurement; inspect observation'})
            elif sum(a.get('status') != 'valid' for a in observation.get('attempts', [])) / len(observation['attempts']) > .2:
                hold = True
                reviews.append({'candidate_id': cid, 'site': action['site'], 'surface_id': action['surface_id'],
                                'reason': 'Adaptive site action has more than 20% failures; further site actions held'})
        if hold:
            continue
        sampled = {(s['surface_id'], s['site']) for s in result.get('sites', [])}
        for site in result.get('sites', []):
            attempts = site.get('attempts', [])
            valid = [a for a in attempts if a.get('status') == 'valid' and number(a.get('energy_eV'))]
            failures = len(attempts) - len(valid)
            if failures and failures / len(attempts) > .2:
                reviews.append({'candidate_id': cid, 'surface_id': site['surface_id'], 'site': site['site'],
                                'reason': 'More than 20% placement failures; inspect identities/clashes before retrying'})
                continue
            spread = max((a['energy_eV'] for a in valid), default=0) - min((a['energy_eV'] for a in valid), default=0)
            if len(valid) > 1 and spread >= threshold:
                available.append({'kind': 'sample_site', 'candidate_id': cid,
                    'surface_id': site['surface_id'], 'site': site['site'],
                    'priority': spread / threshold, 'reason': 'Placement spread exceeds the sampling-review threshold',
                    'seed_spread_eV': spread, 'novel_site': False, 'poses': poses})
        for surface_id, sites in inventory.items():
            for site in sites:
                if (surface_id, site['site']) not in sampled:
                    available.append({'kind': 'sample_site', 'candidate_id': cid,
                        'surface_id': surface_id, 'site': site['site'], 'priority': .75 + min(change / threshold, 2.),
                        'reason': 'Exposed site absent from the selected candidate screen',
                        'observed_site_minimum_change_eV': change,
                        'novel_site': True, 'poses': poses})
    for action in available:
        action['id'] = digest({k: action[k] for k in ('kind', 'candidate_id', 'surface_id', 'site', 'poses') if k in action})[:20]
    return sorted([a for a in available if a['id'] not in done], key=lambda a: (-a['priority'], a['id'])), reviews


def choose(available, policy, seed, completed):
    if not available:
        return None
    if policy == 'random':
        return random.Random(seed + completed).choice(sorted(available, key=lambda a: a['id']))
    if policy == 'fixed':
        # A fixed library/site order; deliberately ignores result-dependent priorities.
        return min(available, key=lambda a: (a['kind'] != 'screen_candidate', a['candidate_id'],
                                            a.get('surface_id', ''), a.get('site', -1)))
    return available[0]


class Ledger:
    def __init__(self, state, path, calls, seconds, clock=time.monotonic):
        self.state, self.path, self.calls, self.seconds, self.clock = state, path, calls, seconds, clock
        self.initial = state['elapsed_seconds']
        self.start = clock()

    def save(self):
        self.state['elapsed_seconds'] = self.initial + self.clock() - self.start
        write_json(self.path, self.state)

    def before_call(self):
        if self.state['calculator_calls'] >= self.calls or self.initial + self.clock() - self.start >= self.seconds:
            raise BudgetStop('calculator-call or elapsed-time budget exhausted')
        # Charge before execution; a crash cannot erase an invoked evaluation.
        self.state['calculator_calls'] += 1
        self.save()


def counted_calculator(backend, ledger):
    from ase.calculators.calculator import Calculator, all_changes

    class Counted(Calculator):
        implemented_properties = backend.implemented_properties

        def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            ledger.before_call()
            try:
                backend.calculate(atoms, properties, system_changes)
                self.results = backend.results.copy()
            finally:
                ledger.save()

    return Counted()


def sample_site(action, candidate, surface, slab, settings, calc, destination, previous_orientations, record):
    """One site, bounded new starting poses; same chemistry checks as the screen."""
    from ase.io import write
    from .chemistry import geometries
    from .engine import freeze_region, relax, place, check_product
    destination.mkdir(parents=True, exist_ok=True)
    # Match the seed's gas conformer/reference protocol; products use its first conformer.
    geos = geometries(candidate, settings['n_conformers'], settings['seed'])
    gas, byproducts, failures = [], [], []
    for geo in geos:
        try:
            molecule = geo['molecule'].copy(); molecule.center(vacuum=8)
            _, energy, _ = relax(molecule, calc, settings, geo['molecule_bonds'])
            gas.append(energy)
            if geo['byproduct'] is not None:
                from .engine import graph
                by = geo['byproduct'].copy(); by.center(vacuum=8)
                _, energy, _ = relax(by, calc, settings, graph(by))
                byproducts.append(energy)
        except Exception as exc:
            failures.append(str(exc))
    if not gas or (candidate['template'] == 'silyl_substitution' and not byproducts):
        raise ValueError('gas_reference_failed: ' + str(failures))
    e_gas, e_by, geo = min(gas), min(byproducts) if byproducts else 0., geos[0]
    bare = slab.copy()
    freeze_region(bare, slab.positions[action['site']], settings['active_radius_A'], len(bare))
    bare, e_bare, _ = relax(bare, calc, settings)
    record.update(gas_reference_eV=e_gas, byproduct_reference_eV=e_by, bare_reference_eV=e_bare,
                  gas_reference_failures=failures)
    # A non-grid azimuthal offset gives new starting poses relative to the seed grid.
    # Geometries can still converge to an already observed basin; that is recorded.
    old = max(1, previous_orientations)
    for i in range(action['poses']):
        attempt = {'pose': i}
        try:
            product, mapped, count = place(bare, action['site'], geo, candidate['template'], i, action['poses'])
            import numpy as np
            angle = np.pi / old
            rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            anchor = product.positions[count].copy()
            product.positions[count:] = (product.positions[count:] - anchor) @ rot.T + anchor
            freeze_region(product, bare.positions[action['site']], settings['active_radius_A'], count)
            product, energy, force = relax(product, calc, settings)
            check_product(product, count, mapped, geo, candidate['template'])
            delta = energy + e_by - e_bare - e_gas
            if abs(delta) > settings['energy_review_limit_eV']:
                raise ValueError('energy_requires_review_not_a_universal_physics_cutoff')
            path = destination / f'pose_{i}.extxyz'
            write(path, product)
            attempt.update(status='valid', energy_eV=delta, max_force=force, geometry=str(path.resolve()))
        except Exception as exc:
            attempt.update(status='failed', reason=str(exc))
        record['attempts'].append(attempt)
        write_json(destination / 'progress.json', record)
    record['status'] = 'observed' if any(a['status'] == 'valid' for a in record['attempts']) else 'failed'
    record['surface_id'] = surface['id']
    record['material'] = surface['material']


def prepare_context(args):
    from .__main__ import configuration
    from .chemistry import validate_candidate
    from .engine import DEFAULTS
    from .surfaces import load_manifest, representatives
    report = read_json(args.report)
    candidates = {c['id']: validate_candidate(c) for c in report['candidates']}
    if any(key != c['id'] for key, c in candidates.items()):
        raise ValueError('Candidate IDs differ from current chemistry validation')
    seed_ids = set(candidates)
    for raw in read_json(args.pool) if args.pool else []:
        candidate = validate_candidate(raw)
        identity = (candidate['canonical_smiles'], candidate['template'], candidate['role'])
        if identity not in {(c['canonical_smiles'], c['template'], c['role']) for c in candidates.values()}:
            candidates[candidate['id']] = candidate
    candidates = {cid: c for cid, c in candidates.items() if c['role'] == 'inhibitor'}
    surface_report, slabs = load_manifest(args.manifest)
    config = configuration(args.config)
    settings = {**DEFAULTS, **config['engine'], 'objective': config.get('objective', {})}
    for name in ('steps', 'max_sites', 'n_conformers', 'orientations'):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    for name in ('fmax', 'active_radius_A', 'energy_review_limit_eV'):
        if not number(settings[name]) or settings[name] <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if not 0 < settings['min_site_success_fraction'] <= 1:
        raise ValueError('Invalid minimum site coverage')
    results = [r for r in report['results'] if r.get('mode') == 'mace' and r['candidate_id'] in candidates]
    model_hashes = set()
    for result in results:
        provenance = result['provenance']
        if provenance['surface_manifest'] != surface_report['fingerprint']:
            raise ValueError('Seed result and current surface manifest differ')
        model_hashes.add(provenance['model_sha256'])
    if len(model_hashes) != 1:
        raise ValueError('Need seed results from exactly one recorded MACE checkpoint')
    inventory = {s['id']: representatives(slabs[s['id']], s['material'], len(slabs[s['id']]))
                 for s in surface_report['surfaces']}
    signature = {'report_sha256': file_hash(args.report), 'manifest_fingerprint': surface_report['fingerprint'],
                 'pool': list(candidates.values()), 'settings': settings,
                 'model_sha256': next(iter(model_hashes)), 'policy': args.policy, 'seed': args.seed,
                 'poses': args.poses, 'max_calls': args.max_calls, 'max_seconds': args.max_seconds,
                 'code': file_hash(__file__)}
    # Transitive engine/chemistry/surface versions are also locked on execution.
    return results, candidates, surface_report, slabs, settings, inventory, signature, seed_ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'run'])
    parser.add_argument('--report', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--pool', help='Optional curated candidate JSON; not autonomous molecule generation')
    parser.add_argument('--out', required=True)
    parser.add_argument('--policy', choices=['adaptive', 'random', 'fixed'], default='adaptive')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--poses', type=int, default=2)
    parser.add_argument('--max-actions', type=int, default=1, help='Actions per invocation; rerun to continue')
    parser.add_argument('--max-calls', type=int, default=2000, help='Total backend evaluations across resumes')
    parser.add_argument('--max-seconds', type=float, default=3600, help='Total wall seconds across run invocations; checked between evaluations')
    args = parser.parse_args(argv)
    if args.poses < 1 or args.poses > 8 or args.max_actions < 1 or args.max_calls < 1 or not number(args.max_seconds) or args.max_seconds <= 0:
        parser.error('Positive budgets required; poses must be 1..8')
    destination = Path(args.out).resolve(); destination.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (destination / '.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another process owns this acquisition directory')
        try:
            results, candidates, surfaces, slabs, settings, inventory, signature, _ = prepare_context(args)
            state_path = destination / 'state.json'
            state = read_json(state_path) if state_path.exists() else {
                'signature': signature, 'calculator_calls': 0, 'elapsed_seconds': 0., 'observations': [],
                'pending': None, 'scientific_success': False, 'reference_validation': 'pending'}
            if state['signature'] != signature:
                raise ValueError('Inputs, code or budgets changed; use a new output directory')
            # A process killed by Slurm consumed a charged prefix; avoid an uncharged retry.
            if state.get('pending'):
                if args.command == 'plan':
                    raise ValueError('Interrupted action found; run once to recover its charged prefix')
                pending = state['pending']
                progress = destination / pending['id'] / 'progress.json'
                observation = read_json(progress) if progress.exists() else {'action': pending, 'attempts': []}
                observation['status'] = 'interrupted_process'
                state['observations'].append(observation)
                state['pending'] = None
                state['elapsed_seconds'] = args.max_seconds  # unknown crash time: fail closed
                write_json(state_path, state)
            calc, ledger = None, None
            for index in range(args.max_actions + 1):
                updated = results + [o['result'] for o in state['observations'] if o.get('result')]
                available, reviews = actions(updated, list(candidates.values()), inventory, state['observations'], args.poses)
                action = choose(available, args.policy, args.seed, len(state['observations']))
                exhausted = state['calculator_calls'] >= args.max_calls or state['elapsed_seconds'] >= args.max_seconds
                plan = {'scientific_success': False, 'reference_validation': 'pending',
                        'status': 'budget_exhausted' if exhausted else 'ready' if action else 'no_actions',
                        'next_action': None if exhausted else action, 'manual_reviews': reviews,
                        'calculator_calls_used': state['calculator_calls'], 'elapsed_seconds': state['elapsed_seconds'],
                        'eligible_actions': len(available), 'policy': args.policy,
                        'limitation': 'Hand-set sampling/exploration priorities; no calibrated uncertainty or DFT accuracy claim.'}
                write_json(destination / 'next_action.json', plan)
                write_json(state_path, state)
                if args.command == 'plan' or index == args.max_actions or exhausted or action is None:
                    print(__import__('json').dumps(plan, indent=2)); break
                if calc is None:
                    from .engine import calculator, calculation_key
                    if file_hash(settings['model_path']) != signature['model_sha256']:
                        raise ValueError('Configured checkpoint differs from seed results')
                    _, execution_signature = calculation_key(candidates[action['candidate_id']], surfaces, settings)
                    code_lock = digest({k: execution_signature[k] for k in ('code', 'engine', 'versions')})
                    if state.get('execution_code_lock', code_lock) != code_lock:
                        raise ValueError('Calculation code/dependencies changed; use a new output directory')
                    state['execution_code_lock'] = code_lock
                    ledger = Ledger(state, state_path, args.max_calls, args.max_seconds)
                    try:
                        calc = counted_calculator(calculator(settings), ledger)
                    finally:
                        ledger.save()
                record = {'action': action, 'status': 'running', 'attempts': [],
                          'calls_before': state['calculator_calls'], 'seconds_before': state['elapsed_seconds'],
                          'scientific_success': False, 'reference_validation': 'pending'}
                state['pending'] = action; ledger.save()
                directory = destination / action['id']; directory.mkdir(exist_ok=True)
                write_json(directory / 'progress.json', record)
                print('Executing:', action, flush=True)
                try:
                    candidate = candidates[action['candidate_id']]
                    if action['kind'] == 'screen_candidate':
                        from .engine import screen, calculation_key
                        key, provenance = calculation_key(candidate, surfaces, settings)
                        result = screen(candidate, surfaces, slabs, settings, calc, directory)
                        result.update(id=key, fidelity=0, provenance=provenance)
                        record.update(status='observed', result=result)
                    else:
                        surface = next(s for s in surfaces['surfaces'] if s['id'] == action['surface_id'])
                        original = next(r for r in updated if r['candidate_id'] == action['candidate_id'])
                        orientation_count = original['provenance']['config']['orientations']
                        task_settings = {**settings, **original['provenance']['config'],
                                         'model_path': settings['model_path'], 'device': settings['device']}
                        record['calculation_settings'] = task_settings
                        sample_site(action, candidate, surface, slabs[surface['id']], task_settings, calc,
                                    directory, orientation_count, record)
                except BudgetStop as exc:
                    record.update(status='budget_interrupted', reason=str(exc))
                except Exception as exc:
                    record.update(status='failed', reason=str(exc))
                finally:
                    ledger.save()
                    record.update(calculator_calls=state['calculator_calls'] - record['calls_before'],
                                  elapsed_seconds=state['elapsed_seconds'] - record['seconds_before'])
                    write_json(directory / 'observation.json', record)
                    state['observations'].append(record)
                    state['pending'] = None; ledger.save()
            return 0
        except Exception as exc:
            parser.exit(2, f'Acquisition stopped: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
