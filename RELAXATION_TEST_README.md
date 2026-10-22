# WARDEN GPU relaxation diagnostic

Extract this ZIP in the root of your existing donuman-ai-scientist checkout.
It adds scripts/validate_relaxation.py and scripts/validate_relaxation.sbatch.
It does not replace the screening code or change any admission thresholds.

Prerequisites: your working .venv-clean with MACE 0.3.16, V100-compatible PyTorch,
and config/v2-r2scan.local.json pointing to the trusted local checkpoint.
The batch file uses the repository path and GPU request supplied in this chat.
Adjust those fields if using it elsewhere; add your account directive if required.

Run from the repository root:

    sbatch scripts/validate_relaxation.sbatch

Outputs (JOBID is the Slurm job number):

- warden_relax_JOBID.out / .err
- runs/relaxation_JOBID/relaxation_report.json
- input_admission.json
- relaxed_surfaces.json and relaxed_admission.json if both slab optimizations pass
- Individual optimization .log files and final .extxyz geometries, including failed cases

The job requests two hours, with a maximum of 400 BFGS steps per case and a
0.05 eV/Angstrom force threshold. It runs ten gas-reference starting geometries
(six molecules, four associated byproduct starts) and two reference slabs.
Repeated byproducts deliberately retain their different starting geometries.
Only one conformer is requested per parent; this is a bounded diagnostic,
not a final conformational search. Free retained fragments are not gas references.

A one-coordinate finite-difference force check uses a 0.001 Angstrom step and
0.005 eV/Angstrom error tolerance. This is a limited implementation check,
not validation of all force components or of chemical accuracy. Molecular
connectivity is checked with the existing distance-based heuristic. All atoms
are mobile and cell vectors remain fixed for slab diagnostics; this is explicitly
different from the local frozen-substrate relaxations used during screening.
The nitride's density threshold is not modified. Optimization may change surface
chemistry or retain the density failure; final structural admission remains required.

Exit 0 means these numerical/identity/admission checks passed. Exit 2 and
status=needs_review are expected if any gas identity, force, convergence or slab
admission check fails. Inspect the report before deciding on repairs. Failed
optimization geometries must not be treated as accepted structures. A wall-time
termination can leave a partial report with status=running; inspect the logs.

No ranking, adsorption-energy conclusion, experimental success, or validated
material identification is produced by this diagnostic. Finite forces/energies
and a relaxed local minimum do not establish MLIP accuracy. Slab admission may
still need revision based on source provenance and appropriate reference data,
not an arbitrary change made to obtain PASS.

Local helper checks used an artificial spring calculator to verify optimization,
rejection of inconsistent forces, and rejection of changed connectivity. Batch
shell syntax was checked. Live MACE/GPU relaxation has not been run by the assistant.
