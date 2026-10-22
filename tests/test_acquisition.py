"""Policy/budget tests. Simulated calculations cannot establish chemical accuracy."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from warden import acquisition as acq
from warden.common import read_json, write_json, file_hash


def inputs():
    candidates = [{'id': c} for c in ('known', 'other', 'new')]
    results = []
    for cid, spread in [('known', .7), ('other', .4)]:
        results.append({'candidate_id': cid, 'status': 'complete', 'provenance': {'config': {'orientations': 4}},
                        'sites': [{'surface_id': 'surface', 'site': 1, 'status': 'valid', 'energy_eV': -1.,
                                   'attempts': [{'status': 'valid', 'energy_eV': e} for e in [-1., -1. + spread]]}]})
    return candidates, results, {'surface': [{'site': 1}, {'site': 2}]}


def test_policy_responds_to_actual_feedback_without_repeating_site():
    candidates, results, inventory = inputs()
    before = copy.deepcopy(results)
    available, _ = acq.actions(results, candidates, inventory, [])
    first = acq.choose(available, 'adaptive', 42, 0)
    assert first['kind'] == 'sample_site' and first['candidate_id'] == 'known'
    observations = [{'action': first, 'attempts': [{'status': 'valid', 'energy_eV': -1.5}]}]
    available, _ = acq.actions(results, candidates, inventory, observations)
    second = acq.choose(available, 'adaptive', 42, 1)
    assert second['candidate_id'] == 'known' and second['site'] == 2
    assert first['id'] not in {a['id'] for a in available}
    assert results == before  # no silent rewriting of seed candidate averages


def test_failure_heavy_site_is_reviewed_and_not_automatically_retried():
    candidates, results, inventory = inputs()
    results[0]['sites'][0]['attempts'].append({'status': 'failed', 'reason': 'identity changed'})
    available, reviews = acq.actions(results, candidates, inventory, [])
    assert reviews
    assert not any(a.get('site') == 1 and a['candidate_id'] == 'known' for a in available)


def test_failed_adaptive_observation_holds_further_site_actions():
    candidates, results, inventory = inputs()
    available, _ = acq.actions(results, candidates, inventory, [])
    first = acq.choose(available, 'adaptive', 42, 0)
    available, reviews = acq.actions(results, candidates, inventory,
        [{'action': first, 'status': 'failed', 'attempts': [{'status': 'failed'}]}])
    assert not any(a['candidate_id'] == 'known' for a in available)
    assert any('no valid measurement' in r['reason'] for r in reviews)


def test_failed_candidate_is_not_treated_as_a_chemical_negative_or_retried():
    candidates, results, inventory = inputs()
    results[0]['status'] = 'failed'
    available, reviews = acq.actions(results, candidates, inventory, [])
    assert not any(a['candidate_id'] == 'known' for a in available)
    assert any(r['candidate_id'] == 'known' for r in reviews)


def test_policies_reproducible_and_screen_result_consumes_pool_entry():
    candidates, results, inventory = inputs()
    available, _ = acq.actions(results, candidates, inventory, [])
    assert acq.choose(available, 'fixed', 42, 0)['candidate_id'] == 'new'
    assert acq.choose(available, 'random', 42, 3) == acq.choose(list(reversed(available)), 'random', 42, 3)
    results.append({'candidate_id': 'new', 'status': 'failed'})
    available, _ = acq.actions(results, candidates, inventory, [])
    assert not any(a['kind'] == 'screen_candidate' for a in available)


def test_ambiguous_candidate_results_refused():
    candidates, results, inventory = inputs()
    results.append(copy.deepcopy(results[0]))
    with pytest.raises(ValueError, match='Ambiguous'):
        acq.actions(results, candidates, inventory, [])


class Harmonic(Calculator):
    implemented_properties = ['energy', 'forces']

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {'energy': float(np.square(atoms.positions).sum()), 'forces': -2 * atoms.positions}


def test_backend_evaluations_counted_once_for_cached_energy_and_forces(tmp_path):
    state = {'calculator_calls': 0, 'elapsed_seconds': 0.}
    ledger = acq.Ledger(state, tmp_path / 'state.json', 2, 100)
    atoms = Atoms('H', positions=[[1, 0, 0]])
    atoms.calc = acq.counted_calculator(Harmonic(), ledger)
    atoms.get_forces(); atoms.get_potential_energy()
    assert state['calculator_calls'] == 1
    atoms.positions[0, 0] = 2; atoms.get_forces()
    atoms.positions[0, 0] = 3
    with pytest.raises(acq.BudgetStop):
        atoms.get_forces()
    assert read_json(tmp_path / 'state.json')['calculator_calls'] == 2


def test_time_budget_checked_between_evaluations(tmp_path):
    state = {'calculator_calls': 0, 'elapsed_seconds': 9.}
    now = [100.]
    ledger = acq.Ledger(state, tmp_path / 'state.json', 20, 10, clock=lambda: now[0])
    now[0] = 101.1
    with pytest.raises(acq.BudgetStop):
        ledger.before_call()
    assert state['calculator_calls'] == 0


def test_budget_stop_escapes_broad_legacy_exception_handlers(tmp_path):
    from warden.engine import relax
    state = {'calculator_calls': 0, 'elapsed_seconds': 0.}
    calc = acq.counted_calculator(Harmonic(), acq.Ledger(state, tmp_path / 'state.json', 0, 100))
    with pytest.raises(acq.BudgetStop):
        relax(Atoms('H', positions=[[1, 0, 0]]), calc, {'fmax': .01, 'steps': 5})


def test_resumed_loop_uses_new_measurements_and_preserves_budget(tmp_path):
    from warden import engine
    candidates, results, inventory = inputs()
    model = tmp_path / 'model'; model.write_text('test fixture, not a model')
    signature = {'model_sha256': file_hash(model)}
    settings = {'model_path': str(model), 'device': 'cpu'}
    surfaces = {'surfaces': [{'id': 'surface', 'material': 'test'}]}
    slabs = {'surface': Atoms('H')}
    context = (results, {c['id']: c for c in candidates}, surfaces, slabs, settings, inventory, signature, set())

    def observed(action, candidate, surface, slab, settings, calc, destination, old, record):
        atoms = Atoms('H', positions=[[1, 0, 0]]); atoms.calc = calc; atoms.get_forces()
        record.update(status='observed', attempts=[{'status': 'valid', 'energy_eV': -1.5}])

    def screened(candidate, surfaces, slabs, settings, calc, destination):
        atoms = Atoms('H', positions=[[2, 0, 0]]); atoms.calc = calc; atoms.get_forces()
        return {'candidate_id': candidate['id'], 'status': 'complete', 'sites': []}

    argv = ['run', '--report', 'unused', '--manifest', 'unused', '--config', 'unused',
            '--out', str(tmp_path / 'run'), '--max-calls', '3', '--max-seconds', '100']
    with patch.object(acq, 'prepare_context', return_value=context), \
         patch.object(engine, 'calculator', return_value=Harmonic()), \
         patch.object(engine, 'calculation_key', return_value=('test', {'code': [], 'engine': 'test', 'versions': {}})), \
         patch.object(acq, 'sample_site', side_effect=observed), patch.object(engine, 'screen', side_effect=screened), \
         contextlib.redirect_stdout(io.StringIO()):
        assert acq.main(argv) == 0
        state = read_json(tmp_path / 'run/state.json')
        assert state['calculator_calls'] == 1
        assert read_json(tmp_path / 'run/next_action.json')['next_action']['site'] == 2
        assert acq.main(argv) == 0
        assert acq.main(argv) == 0
    state = read_json(tmp_path / 'run/state.json')
    assert state['calculator_calls'] == 3
    assert len(state['observations']) == 3
    assert len({o['action']['id'] for o in state['observations']}) == 3
    assert read_json(tmp_path / 'run/next_action.json')['status'] == 'budget_exhausted'


def test_killed_process_recovery_does_not_reset_unknown_wall_time(tmp_path):
    candidates, results, inventory = inputs()
    signature = {'fixture': True}
    context = (results, {c['id']: c for c in candidates}, {}, {}, {}, inventory, signature, set())
    state = {'signature': signature, 'calculator_calls': 3, 'elapsed_seconds': 4., 'observations': [],
             'pending': {'id': 'interrupted', 'kind': 'sample_site', 'candidate_id': 'known', 'site': 1, 'surface_id': 'surface'}}
    output = tmp_path / 'run'; output.mkdir()
    write_json(output / 'state.json', state)
    with patch.object(acq, 'prepare_context', return_value=context), contextlib.redirect_stdout(io.StringIO()):
        acq.main(['run', '--report', 'unused', '--manifest', 'unused', '--config', 'unused', '--out', str(output), '--max-seconds', '100'])
    state = read_json(output / 'state.json')
    assert state['elapsed_seconds'] == 100
    assert state['calculator_calls'] == 3
    assert state['observations'][0]['status'] == 'interrupted_process'
    assert read_json(output / 'next_action.json')['status'] == 'budget_exhausted'
