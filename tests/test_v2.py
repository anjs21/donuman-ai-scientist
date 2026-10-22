"""Regression tests for scientific gates and the bounded control loop."""
import json
from collections import Counter
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from ase import Atoms
from warden.chemistry import validate_candidate, geometries, rotation
from warden.metrics import evaluate, diagnose
from warden.store import Store
from warden.report import export
from warden.demo import execute, FixturePlanner, DEMO_QUOTE
from warden.campaign import run
from selectivity_model import GrowthParams, selectivity_curve

ROOT = Path(__file__).resolve().parents[1]


def test_target_is_not_clipped_to_budget():
    r = evaluate(-2.6, -3.0)
    assert r['n_target'] > 200
    assert not r['target_reached'] and not r['model_pass']
    assert r['selectivity_at_target'] is None


@pytest.mark.parametrize('gs,ngs', [(None, -1), (-1, None), (float('nan'), -1)])
def test_missing_energy_is_unknown(gs, ngs):
    r = evaluate(gs, ngs)
    assert r['status'] == 'insufficient_evidence' and not r['model_pass']


def test_zero_growth_not_perfect_selectivity():
    r = selectivity_curve(-100, -100, GrowthParams())
    assert np.isnan(r['selectivity']).all()
    assert not r['meets_target']


def test_target_uses_exact_thickness():
    r = selectivity_curve(0., -.8, GrowthParams())
    thk_ngs = r['gpc_NGS_A'] * max(0., r['n_target'] - r['n_delay_NGS'])
    assert r['selectivity_at_target'] == pytest.approx((100-thk_ngs)/(100+thk_ngs))
    assert r['target_reached']


def test_precursor_diagnosis_handles_no_metrics():
    assert diagnose([{'status':'complete','dE_GS':0.,'dE_NGS':-1.,'metrics':None}])


SEEDS = json.loads((ROOT/'examples/seeds-v2.json').read_text()) + json.loads((ROOT/'examples/precursors-v2.json').read_text())

@pytest.mark.parametrize('spec', SEEDS, ids=[s['name'] for s in SEEDS])
def test_structure_and_reaction_balance(spec):
    c = validate_candidate(spec)
    g = geometries(c, 1, 42)[0]
    original = Counter(g['molecule'].get_chemical_symbols())
    retained = Counter(g['fragment'].get_chemical_symbols())
    if g['byproduct'] is not None:
        original['H'] += 1
        retained += Counter(g['byproduct'].get_chemical_symbols())
    assert original == retained
    if spec['name'] == 'BDMAS':
        assert Counter(g['fragment'].get_chemical_symbols())['N'] == 1
    if spec['name'] == 'SiCl4':
        assert Counter(g['fragment'].get_chemical_symbols()) == Counter({'Si':1,'Cl':3})


@pytest.mark.parametrize('smiles,template', [('[NH4+]','physisorption'), ('C.C','physisorption'),
    ('C[Si](C)(C)Cl','silyl_substitution'), ('C','unimplemented')])
def test_unsupported_chemistry_rejected(smiles,template):
    with pytest.raises(ValueError):
        validate_candidate({'name':'bad','smiles':smiles,'template':template})


def test_fake_evidence_quote_rejected():
    candidate = {**SEEDS[0], 'hypothesis':'h', 'falsification':'f', 'evidence':[{'id':'ev','quote':'invented'}]}
    with pytest.raises(ValueError, match='absent'):
        validate_candidate(candidate, {'ev':{'text':'actual source'}}, True)


def test_trivial_evidence_quote_rejected():
    source = {'ev':{'text':'Aminosilanes react with surface silanols at low temperature.'}}
    candidate = {**SEEDS[0], 'hypothesis':'h', 'falsification':'f', 'evidence':[{'id':'ev','quote':'a'}]}
    with pytest.raises(ValueError, match='at least'):
        validate_candidate(candidate, source, True)
    candidate['evidence'] = [{'id':'ev','quote':'Aminosilanes react with surface silanols'}]
    assert validate_candidate(candidate, source, True)['id']


def test_planner_sees_newest_evidence(tmp_path):
    from warden.campaign import context, EVIDENCE_ITEMS
    store = Store(tmp_path/'c.sqlite')
    ids = [store.evidence(f'record {i}', f'source {i}') for i in range(EVIDENCE_ITEMS + 18)]
    shown = [e['id'] for e in context(store, {'step':0}, [])['evidence']]
    assert shown == ids[::-1][:EVIDENCE_ITEMS]


def test_planner_refuses_prompt_larger_than_context():
    from warden.planner import Ollama
    planner = Ollama({'host':'http://127.0.0.1:9', 'model':'mock', 'num_ctx':1000})
    with pytest.raises(ValueError, match='num_ctx'):
        planner.plan({'evidence':['x' * 5000]})


@pytest.mark.parametrize('target', [[0,0,-1], [0,0,1], [1,2,3]])
def test_rotation_preserves_handedness(target):
    target = np.array(target, dtype=float); target /= np.linalg.norm(target)
    r = rotation([0,0,1], target)
    assert np.linalg.det(r) == pytest.approx(1)
    assert np.allclose(r @ [0,0,1], target)


def test_offline_feedback_and_resume(tmp_path):
    state = execute(tmp_path)
    first = json.loads((tmp_path/'report.json').read_text())
    assert state['status'] == 'finished'
    assert len(first['results']) == 2
    assert len(first['candidates']) == 2
    assert {c['name'] for c in first['candidates']} == {'demo_methyltrichlorosilane','demo_propyltrichlorosilane'}
    assert not first['ranked_real_screening_results']
    assert not first['scientific_success']
    assert any(d['issue']=='growth_surface_poisoning_in_surrogate' for d in first['diagnoses'])
    execute(tmp_path)
    second = json.loads((tmp_path/'report.json').read_text())
    assert len(second['events']) == len(first['events'])


def test_premature_finish_consumes_budget(tmp_path):
    class Premature:
        def plan(self, ctx): return {'action':'finish','reason':'unsupported claim'}
    s=Store(tmp_path/'db')
    state=run(s,Premature(),lambda *a: pytest.fail('should not compute'),{'campaign':{'max_steps':2}})
    assert state['status']=='budget_exhausted' and state['step']==2
    assert not state['submitted']
    s.close()


def test_critic_rejection_does_not_register(tmp_path):
    class Reject(FixturePlanner):
        def critique(self, action,ctx): return {'approve':False,'reason':'unsupported'}
    s=Store(tmp_path/'db'); s.evidence(DEMO_QUOTE,'fixture','synthetic_demo')
    run(s,Reject(),lambda *a: pytest.fail('should not compute'),{'campaign':{'max_steps':1}},mode='synthetic_demo')
    assert not s.all('candidate')
    s.close()


def test_timeout_resume_preserves_candidate_budget(tmp_path):
    s=Store(tmp_path/'db'); s.evidence(DEMO_QUOTE,'fixture','synthetic_demo')
    calls=[]
    def timeout(ids,fidelity):
        calls.append(ids); raise subprocess.TimeoutExpired('worker',1)
    config={'campaign':{'max_steps':1,'max_candidates':1}}
    state=run(s,FixturePlanner(),timeout,config,mode='synthetic_demo')
    assert state['status']=='paused' and len(state['submitted'])==1
    state=run(s,FixturePlanner(),lambda ids, f: calls.append(ids),config,mode='synthetic_demo')
    assert len(state['submitted'])==1 and calls[0]==calls[1]
    assert state['status']=='budget_exhausted'
    s.close()


def test_failed_refinement_prevents_stale_ranking(tmp_path):
    s=Store(tmp_path/'db')
    base={'candidate_id':'a','mode':'mace','role':'inhibitor','status':'complete','fidelity':0,'metrics':evaluate(0,-2)}
    s.put('result','old',base)
    s.put('result','new',{**base,'fidelity':1,'status':'failed','metrics':None})
    report=export(s,tmp_path)
    assert report['ranked_real_screening_results']==[]
    s.close()


def test_reference_surfaces_do_not_bypass_admission():
    from warden.surfaces import load_manifest
    report,_=load_manifest(ROOT/'examples/surfaces-reference.json',False)
    assert not report['valid']
    with pytest.raises(ValueError,match='admission failed'):
        load_manifest(ROOT/'examples/surfaces-reference.json')


def test_engine_rejects_unconverged_optimization(monkeypatch):
    from warden import engine
    from ase.calculators.singlepoint import SinglePointCalculator
    class FailedOptimizer:
        def __init__(self,*a,**kw): pass
        def run(self,**kw): return False
    atoms=Atoms('He',positions=[[0,0,0]])
    calc=SinglePointCalculator(atoms,energy=0.,forces=np.zeros((1,3)))
    monkeypatch.setattr(engine,'BFGS',FailedOptimizer)
    with pytest.raises(ValueError,match='not_converged'):
        engine.relax(atoms,calc,engine.DEFAULTS)


def test_freeze_does_not_freeze_adsorbate():
    from warden.engine import freeze_region
    atoms=Atoms('OOH',positions=[[0,0,0],[8,0,0],[0,0,3]],cell=[20,20,20],pbc=[1,1,0])
    freeze_region(atoms,np.zeros(3),5,2)
    assert list(atoms.constraints[0].index)==[1]


def test_mcp_inventory_allowlist_and_call(tmp_path):
    from warden.literature import Bond
    server=tmp_path/'mock_mcp.py'
    server.write_text('''from mcp.server.fastmcp import FastMCP
mcp=FastMCP("test-only")
@mcp.tool()
def search_papers(query: str) -> str:
    return "MOCK literature response for " + query
if __name__ == "__main__":
    mcp.run(transport="stdio")
''')
    bond=Bond({'transport':'stdio','command':sys.executable,'args':[str(server)],'allowed_tools':['search_papers']})
    assert bond.tools()[0]['name']=='search_papers'
    assert 'MOCK literature' in json.dumps(bond.call('search_papers',{'query':'AS-ALD'}))
    # AsyncExitStack wraps request errors in an exception group.
    with pytest.raises(Exception): bond.call('unapproved_tool',{})
    with pytest.raises(Exception): bond.call('search_papers',{'query':42})


def test_cli_demo(tmp_path):
    result=subprocess.run([sys.executable,'-m','warden','demo','--out',str(tmp_path)],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0, result.stderr
    assert 'SYNTHETIC DEMO: finished' in result.stdout


def test_ollama_adapter_uses_structured_output_and_releases_model():
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread
    from warden.planner import Ollama
    captured=[]
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            captured.append(request)
            decision={'approve':True,'reason':'mock review'} if 'approve' in request['format']['properties'] else {'action':'finish','reason':'mock stop'}
            payload=json.dumps({'message':{'content':json.dumps(decision)}}).encode()
            self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(payload)
        def log_message(self,*a): pass
    server=HTTPServer(('127.0.0.1',0),Handler)
    thread=Thread(target=server.serve_forever,daemon=True); thread.start()
    try:
        planner=Ollama({'host':f'http://127.0.0.1:{server.server_port}','model':'mock'})
        assert planner.plan({})['action']=='finish'
        assert planner.critique({}, {'evidence':[],'diagnoses':[]})['approve']
        assert all(r['keep_alive']==0 and r['format']['type']=='object' for r in captured)
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_live_entrypoint_rejects_bad_surfaces_before_loading_mace(tmp_path):
    result=subprocess.run([sys.executable,'-m','warden','screen','--db',str(tmp_path/'db'),
        '--manifest','examples/surfaces-reference.json','--candidates','examples/seeds-v2.json'],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==2
    assert 'Surface admission failed' in result.stderr
    assert not (tmp_path/'db').exists()


def test_cache_depends_on_checkpoint_contents_and_sampling(tmp_path,monkeypatch):
    from warden import engine
    model=tmp_path/'mace.model'; model.write_bytes(b'model version one')
    monkeypatch.setattr(engine.importlib.metadata,'version',lambda p:'mock-version')
    config={'model_path':str(model),'n_conformers':1}
    candidate=validate_candidate(SEEDS[0]); surface={'fingerprint':'surfaces-one'}
    key,_=engine.calculation_key(candidate,surface,config)
    model.write_bytes(b'model version two')
    new,_=engine.calculation_key(candidate,surface,config)
    sampled,_=engine.calculation_key(candidate,surface,{**config,'n_conformers':2})
    assert len({key,new,sampled})==3


@pytest.mark.parametrize('index',[0,1,4,5])
def test_surface_reaction_is_element_balanced(index):
    from warden.engine import place
    candidate=validate_candidate(SEEDS[index]); geo=geometries(candidate,1,42)[0]
    # Minimal geometry fixture, not an admitted research slab.
    slab=Atoms('SiOH',positions=[[0,0,0],[0,0,1.6],[0,0,2.6]],cell=[30,30,30],pbc=[1,1,0])
    product,_,_=place(slab,1,geo,candidate['template'],0,2)
    lhs=Counter(slab.get_chemical_symbols())+Counter(geo['molecule'].get_chemical_symbols())
    rhs=Counter(product.get_chemical_symbols())+Counter(geo['byproduct'].get_chemical_symbols())
    assert lhs==rhs
