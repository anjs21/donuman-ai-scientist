"""Create an explicit manifest of all generated slabs; validation remains separate."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--surface-dir', required=True)
p.add_argument('--model', required=True)
p.add_argument('--out', default='examples/surfaces-live.json')
a=p.parse_args()
root=Path(a.surface_dir).resolve(); model=Path(a.model).resolve(); out=Path(a.out).resolve()
sha=hashlib.sha256(model.read_bytes()).hexdigest()
rows=[]
for material in ('SiO2','SiNx'):
    files=sorted(root.glob(material+'_v2_*.xyz'))
    if not files: raise SystemExit('No generated '+material+' slabs. Inspect the build report.')
    for i,f in enumerate(files):
        rows.append({'id':f'{material}_{i}', 'material':material, 'path':str(f), 'format':'extxyz',
                     'stage':'passivated_slab', 'provenance':{'builder':'run_surface_builder.py',
                     'checkpoint_sha256':sha, 'build_report':str(root/'build_report.txt'),
                     'note':'Generated structures; not admitted until validate-surfaces succeeds.'}})
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'surfaces':rows},indent=2)+'\n')
print(out)
