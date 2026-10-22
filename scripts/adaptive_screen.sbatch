#!/bin/bash
#SBATCH --job-name=warden_adaptive
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=warden_adaptive_%j.out
#SBATCH --error=warden_adaptive_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
source .venv-clean/bin/activate
nvidia-smi
python -u -m warden.acquisition run \
  --report runs/reference_smoke_v2/report.json \
  --manifest runs/adaptive_inputs_v1/surfaces.json \
  --config runs/adaptive_inputs_v1/config.json \
  --pool examples/seeds-v2.json \
  --out "${ACQ_OUT:-runs/adaptive_v1}" \
  --policy "${ACQ_POLICY:-adaptive}" \
  --poses 2 --max-actions "${ACQ_ACTIONS:-1}" --max-calls 2000 --max-seconds 3600
