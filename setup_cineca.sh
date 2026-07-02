#!/bin/bash
# setup_cineca.sh
# Set up the Python virtual environment and cache the MACE model on the Cineca Leonardo login node.

set -e

# Load modules
echo "=== Loading Cineca modules ==="
module purge
module load python/3.11.7
module load cuda/12.1

# Create virtual environment
echo "=== Creating virtual environment ==="
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment 'venv' created."
else
    echo "Virtual environment 'venv' already exists."
fi

# Activate virtual environment
source venv/bin/activate
pip install --upgrade pip

# Install dependencies (using the PyTorch index for CUDA 12.1 to match Leonardo's GPU nodes)
echo "=== Installing dependencies ==="
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Pre-download and cache MACE model
echo "=== Pre-downloading and caching MACE model (medium-omat-0) ==="
# Set cache location explicitly
export XDG_CACHE_HOME="$HOME/.cache"
python3 -c "
from mace.calculators import mace_mp
print('Downloading and caching medium-omat-0 model...')
calc = mace_mp(model='medium-omat-0', device='cpu')
print('Model successfully cached.')
"

echo "=== Setup complete! ==="
echo "You can now submit your job to the compute nodes using: sbatch run_cineca.sh"
