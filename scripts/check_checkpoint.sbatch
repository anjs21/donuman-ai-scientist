#!/bin/bash
#SBATCH --job-name=warden_checkpoint
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=warden_checkpoint_%j.out
#SBATCH --error=warden_checkpoint_%j.err

set -euo pipefail

cd /ocean/projects/mat250016p/dvarghes/on_the_fly/donuman-ai-scientist

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

source .venv-clean/bin/activate

nvidia-smi

python -u - <<'PY'
import json
import os
from pathlib import Path
from importlib.metadata import version

import numpy as np
import torch

from warden.__main__ import configuration
from warden.chemistry import validate_candidate, geometries
from warden.common import file_hash, write_json
from warden.engine import calculator
from warden.surfaces import load_manifest

out = Path("runs") / f"checkpoint_{os.environ['SLURM_JOB_ID']}"
out.mkdir(parents=True, exist_ok=True)

metadata = {
    "purpose": "checkpoint_execution_diagnostic",
    "scientific_success": False,
    "torch": torch.__version__,
    "cuda_build": torch.version.cuda,
    "mace": version("mace-torch"),
}
print(json.dumps(metadata, indent=2), flush=True)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable inside the GPU allocation")

metadata["gpu"] = torch.cuda.get_device_name(0)
metadata["capability"] = list(torch.cuda.get_device_capability(0))
metadata["compiled_architectures"] = torch.cuda.get_arch_list()
print("GPU:", metadata["gpu"], flush=True)
print("Compiled architectures:", metadata["compiled_architectures"], flush=True)

# Exercise a GPU kernel before loading the checkpoint.
x = torch.ones((32, 32), device="cuda", dtype=torch.float64)
assert torch.isfinite(x @ x).all().item()
torch.cuda.synchronize()
print("GPU tensor calculation: PASS", flush=True)

config = configuration("config/v2-r2scan.local.json")
settings = config["engine"]
metadata["checkpoint_sha256"] = file_hash(settings["model_path"])

calc = calculator(settings)
print("Checkpoint loaded: PASS", flush=True)

# Diagnostic access only: this does not override surface admission.
surface_report, surfaces = load_manifest(
    "examples/surfaces-reference.json",
    require_valid=False,
)
write_json(out / "surface_admission.json", surface_report)

cases = []
for row in surface_report["surfaces"]:
    cases.append((
        "surface/" + row["id"],
        surfaces[row["id"]].copy(),
        {
            "material": row["material"],
            "admitted": row["valid"],
            "admission_reasons": row["reasons"],
        },
    ))

for filename in (
    "examples/seeds-v2.json",
    "examples/precursors-v2.json",
):
    for raw in json.loads(Path(filename).read_text()):
        candidate = validate_candidate(raw)
        geo = geometries(candidate, n_conformers=1, seed=42)[0]
        for part in ("molecule", "fragment", "byproduct"):
            if geo[part] is None:
                continue
            # Whole molecule and fragment are equivalent for physisorption.
            if part == "fragment" and candidate["template"] == "physisorption":
                continue
            atoms = geo[part].copy()
            atoms.set_pbc(False)
            atoms.center(vacuum=8.0)
            cases.append((
                candidate["name"] + "/" + part,
                atoms,
                {"role": candidate["role"], "template": candidate["template"]},
            ))

results = []
for name, atoms, extra in cases:
    row = {
        "name": name,
        "formula": atoms.get_chemical_formula(),
        "n_atoms": len(atoms),
        **extra,
    }
    try:
        atoms.calc = calc
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        if not np.isfinite(energy) or not np.isfinite(forces).all():
            raise ValueError("Nonfinite energy or forces")
        row.update(
            status="pass",
            energy_eV=energy,
            max_force_eV_A=float(np.linalg.norm(forces, axis=1).max()),
        )
    except Exception as exc:
        row.update(status="failed", error=str(exc))

    results.append(row)
    write_json(out / "checkpoint_report.json", {
        **metadata,
        "results": results,
    })
    print(json.dumps(row), flush=True)

failed = [r for r in results if r["status"] != "pass"]
print("Report:", out / "checkpoint_report.json", flush=True)
print("Failed cases:", len(failed), "/", len(results), flush=True)

if failed:
    raise SystemExit(2)

print("CHECKPOINT EXECUTION TEST: PASS", flush=True)
print("Surface admission and chemical accuracy remain separate checks.", flush=True)
PY
