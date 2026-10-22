#!/bin/bash
#SBATCH --job-name=warden_screen
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=warden_screen_%j.out
#SBATCH --error=warden_screen_%j.err
set -euo pipefail
cd /ocean/projects/mat250016p/dvarghes/on_the_fly/donuman-ai-scientist
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
source .venv-clean/bin/activate
nvidia-smi
python -u -m warden screen \
  --db runs/reference_smoke_v1/campaign.sqlite \
  --manifest runs/reference_smoke_v1/surfaces.json \
  --config runs/reference_smoke_v1/config.json \
  --candidates runs/reference_smoke_v1/candidates.json
python -u - <<'PY'
import json
from pathlib import Path
r = json.loads(Path('runs/reference_smoke_v1/report.json').read_text())
names = {c['id']: c['name'] for c in r['candidates']}
complete = 0
for result in r['results']:
    print(names[result['candidate_id']], result['status'],
          'GS=', result.get('dE_GS'), 'NGS=', result.get('dE_NGS'), flush=True)
    complete += result['status'] == 'complete'
print('Complete candidates:', complete, '/ 2')
print('Scope: numerical smoke screen; bulk validation and scientific accuracy remain unresolved.')
if complete != 2:
    raise SystemExit('Some calculations are unresolved. Inspect report.json; do not infer chemical rejection.')
PY
