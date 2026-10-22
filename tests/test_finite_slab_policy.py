"""The finite-slab profile is explicit screening admission, not bulk validation."""
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write
from warden.common import file_hash, write_json
from warden.slab_validation import PROFILE, density_diagnostics, identify_composition
from warden.surfaces import load_manifest

ROOT=Path(__file__).resolve().parents[1]
FIXTURES=ROOT/'tests/fixtures/relaxed_review'

def manifest(tmp_path,profile=PROFILE):
    rows=[]
    for material in ('SiO2','SiNx'):
        f=tmp_path/f'{material}.extxyz'
        f.write_bytes((FIXTURES/f'surface_{material}_reference.extxyz').read_bytes())
        rows.append({'id':material,'material':material,'path':str(f),'format':'extxyz',
                     'stage':'passivated_slab','expected_sha256':file_hash(f),
                     'provenance':'User-supplied fixture; numerical tests only'})
    data={'surfaces':rows}
    if profile is not None:data['validation_profile']=profile
    p=tmp_path/'manifest.json';write_json(p,data);return p

def mutate(p,fn):
    d=json.loads(p.read_text());f=Path(d['surfaces'][0]['path']);a=read(f)
    fn(a);write(f,a,format='extxyz');d['surfaces'][0]['expected_sha256']=file_hash(f);write_json(p,d)

def test_reviewed_pair_is_not_bulk_validated(tmp_path):
    r,_=load_manifest(manifest(tmp_path))
    assert r['valid'] and r['admission_scope']=='numerical_screening_only'
    assert not r['bulk_density_validated'] and not r['scientific_success']
    for s in r['surfaces']:
        assert s['structural']['density_gcc']['pass'] is None
        assert s['structural']['density_gcc']['target'] is None
        assert s['structural']['_total']==3 and s['warnings']
        assert s['scientific_validity']=='unverified'
        assert not s['material_identity']['validated_amorphous_phase']
    assert r['surfaces'][1]['material_identity']['observed_family']=='hydrogenated_silicon_oxynitride'

def test_legacy_not_silently_overridden(tmp_path):
    r,_=load_manifest(manifest(tmp_path,None),False)
    assert not r['valid'] and 'structural: density_gcc' in r['surfaces'][1]['reasons']

def test_policy_changes_fingerprint(tmp_path):
    p=manifest(tmp_path);r,_=load_manifest(p,False)
    d=json.loads(p.read_text());d.pop('validation_profile');write_json(p,d)
    old,_=load_manifest(p,False)
    assert r['fingerprint']!=old['fingerprint']

def test_profile_requires_pin(tmp_path):
    p=manifest(tmp_path);d=json.loads(p.read_text());d['surfaces'][0].pop('expected_sha256');write_json(p,d)
    with pytest.raises(ValueError,match='pin expected_sha256'):load_manifest(p)

def test_mutation_rejected(tmp_path):
    p=manifest(tmp_path);d=json.loads(p.read_text());f=Path(d['surfaces'][0]['path']);f.write_bytes(f.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='checksum differs'):load_manifest(p)

def test_unknown_profile(tmp_path):
    with pytest.raises(ValueError,match='Unknown surface validation_profile'):load_manifest(manifest(tmp_path,'skip_all'))

def test_clashes_remain_rejected(tmp_path):
    p=manifest(tmp_path);mutate(p,lambda a:a.positions.__setitem__(1,a.positions[0]))
    r,_=load_manifest(p,False)
    assert not r['valid'] and 'severe atom overlap' in r['surfaces'][0]['reasons']

def test_vacuum_remains_required(tmp_path):
    p=manifest(tmp_path)
    def fn(a):
        c=a.cell.array.copy();c[2,2]=np.ptp(a.positions[:,2])+5;a.set_cell(c)
    mutate(p,fn);r,_=load_manifest(p,False)
    assert not r['valid'] and 'less than 10 Angstrom vacuum' in r['surfaces'][0]['reasons']

def test_disconnected_species_rejected(tmp_path):
    p=manifest(tmp_path)
    def fn(a):
        a+=Atoms('H',positions=[[0,0,a.positions[:,2].max()+5]])
        c=a.cell.array.copy();c[2,2]+=10;a.set_cell(c)
    mutate(p,fn);r,_=load_manifest(p,False)
    assert not r['valid'] and any('disconnected' in x for x in r['surfaces'][0]['reasons'])

def test_composition_crosschecks_declaration():
    a=read(FIXTURES/'surface_SiNx_reference.extxyz');identity,reasons=identify_composition(a,'SiO2')
    assert reasons and not identity['validated_amorphous_phase']

def test_density_diagnostic_uses_physical_masses():
    a=read(FIXTURES/'surface_SiNx_reference.extxyz');d=density_diagnostics(a,'SiNx')
    assert d['bulk_preparation_reference_gcc']==2.6 and not d['bulk_density_validated']
    assert d['envelope_density_gcc']==pytest.approx(2.1354148568,rel=1e-5)
    assert d['status']=='diagnostic_only' and len(d['central_windows'])==4
    a.set_masses(np.full(len(a),99.))
    assert density_diagnostics(a,'SiNx')['envelope_density_gcc']==d['envelope_density_gcc']

def test_prepare_exact_reviewed_geometries(tmp_path,monkeypatch):
    prep=importlib.import_module('scripts.prepare_reference_smoke');model=tmp_path/'model';model.write_bytes(b'mock')
    original=prep.file_hash
    monkeypatch.setattr(prep,'file_hash',lambda p:prep.REVIEWED_CHECKPOINT if Path(p)==model else original(p))
    c=tmp_path/'config.json';write_json(c,{'engine':{'model_path':str(model),'device':'cuda'}})
    out=tmp_path/'screen';monkeypatch.chdir(ROOT);prep.prepare(FIXTURES,c,out)
    assert json.loads((out/'surface_validation.json').read_text())['valid']
    assert json.loads((out/'config.json').read_text())['engine']['n_conformers']==1
    assert {c['name'] for c in json.loads((out/'candidates.json').read_text())}=={'DMATMS','NH3'}

def test_prepare_rejects_foreign_checkpoint(tmp_path):
    prep=importlib.import_module('scripts.prepare_reference_smoke');model=tmp_path/'model';model.write_bytes(b'wrong')
    c=tmp_path/'config.json';write_json(c,{'engine':{'model_path':str(model),'device':'cuda'}})
    with pytest.raises(ValueError,match='Checkpoint differs'):prep.prepare(FIXTURES,c,tmp_path/'screen')
