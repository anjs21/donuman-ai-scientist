#!/bin/bash
#SBATCH --job-name=warden_relax
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=warden_relax_%j.out
#SBATCH --error=warden_relax_%j.err
set -euo pipefail
cd /ocean/projects/mat250016p/dvarghes/on_the_fly/donuman-ai-scientist
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
source .venv-clean/bin/activate
nvidia-smi
python -u -m scripts.validate_relaxation \
  --config config/v2-r2scan.local.json \
  --manifest examples/surfaces-reference.json \
  --out "runs/relaxation_${SLURM_JOB_ID}" \
  --steps 400 --fmax 0.05
