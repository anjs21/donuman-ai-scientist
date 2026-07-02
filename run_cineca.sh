#!/bin/bash
#SBATCH --job-name=ai_scientist
#SBATCH --output=ai_scientist_%j.out
#SBATCH --error=ai_scientist_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --account=iscrc_mnlp26

set -e

echo "=== Job started at: $(date) ==="

# Load modules
echo "=== Loading modules on compute node ==="
module purge
module load python/3.11.7
module load cuda/12.1

# Explicitly configure MACE cache location (pointing to the shared home directory where the model was pre-downloaded)
export XDG_CACHE_HOME="$HOME/.cache"
export HF_HUB_OFFLINE=1  # Prevent Hugging Face or other hub packages from trying to hit the internet
export REQ_OFFLINE=1

# Activate virtual environment
echo "=== Activating virtual environment ==="
source venv/bin/activate

# Print GPU info to confirm CUDA is working on the compute node
echo "=== GPU Information ==="
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Run the pipeline
# Defaulting to "test" mode if no argument is passed. Change to "fast" or "full" as needed.
MODE=${1:-test}
echo "=== Running pipeline in ${MODE} mode ==="
python3 run_pipeline.py --mode "${MODE}"

echo "=== Pipeline completed successfully ==="
echo "=== Job finished at: $(date) ==="
