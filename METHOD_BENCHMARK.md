# First method-reliability benchmark

The existing GPU and relaxation checks establish execution and numerical convergence.
They do not validate MACE interaction/reaction energies or AS-ALD performance.
This package prepares an explicitly limited next test: does an independent electronic
structure method agree with MACE about the relative energies of two NH3 placements
on each provisional surface?

The preparation script selects the site with the largest valid placement spread on
each material and exports its lowest- and highest-energy accepted geometries.
On the supplied refined report these are SiO2 site 32 (0.707120 eV gap) and the
nominal SiNx site 62 (0.255584 eV gap). The latter substrate includes oxygen and H;
its report label does not prove pure silicon nitride or an independently validated phase.

These are four diagnostic stress cases, deliberately selected using MACE results.
They are not an unbiased held-out benchmark, a calibrated uncertainty estimate,
or a demonstration of autonomous discovery.

## Run order on Bridges-2

From the repository root, after extracting this package there:

```bash
source .venv-clean/bin/activate
python -m unittest discover -s tests -p test_method_benchmark.py -v
python scripts/prepare_method_benchmark.py prepare \
  --report runs/reference_smoke_v2/report.json \
  --repo-root . \
  --out runs/method_benchmark_v1
python -m json.tool runs/method_benchmark_v1/cases.json
```

Preparation is a small CPU task, requiring ASE and NumPy already in the environment.
It reads the original geometry files on your cluster, checks the recorded slab
checksum, atom ordering, NH3 bond lengths and cell, and exports full frozen
periodic structures without their old calculator results or optimization constraints.
No cap atoms, cluster boundaries, structural optimizations, or fabricated reference
energies are introduced. Existing output directories are not overwritten.

For selection alone when the source geometries are unavailable, use `--select-only`
with a different output directory. Such a manifest cannot be used for comparison.

## Reference calculation step

Run four independently converged electronic-structure single-point calculations,
one for each exported `.extxyz`. Keep atom coordinates and cells frozen and use
the same functional, pseudopotentials, cutoff/basis, k-point and electrostatic
treatment across each low/high pair. Treat surface periodicity, vacuum and dipole
corrections explicitly. A periodic r2SCAN calculation is a natural initial test
against this r2SCAN checkpoint; its settings still require convergence checks.
An additional dispersion or different functional is a separate sensitivity test.

This package does not launch a DFT engine. The executable, pseudopotentials and
converged input settings have not yet been established for this project. It does
not silently substitute a small capped cluster for the full periodic system.
Reference calculations are not yet included or claimed complete.

Copy `reference-template.json` to `reference-results.json`, enter actual total
energies in eV, mark only converged cases true, and fill method/provenance fields
(`none` is explicit for a deliberately omitted correction). Keep geometry hashes.
Then run:

```bash
python scripts/prepare_method_benchmark.py compare \
  --manifest runs/method_benchmark_v1/cases.json \
  --reference runs/method_benchmark_v1/reference-results.json \
  --out runs/method_benchmark_v1/comparison.json
```

The script checks case IDs, finite energies, convergence declarations and geometry
hashes. It cannot verify the truth of manually entered energies or metadata;
retain engine inputs, outputs and convergence studies with the reference results.

Within a pair the composition and original gas/slab references are identical, so
the saved screening-energy difference equals the complex-total-energy difference.
Compare `E(high) - E(low)` within each method. Never subtract a DFT total energy
from a MACE total energy. No gas-phase or cluster approximation is needed for this
relative-placement test. Differences include substrate relaxation already present
in the saved geometries; this is not a pure molecular rotation or adsorption-energy
accuracy test. Near-degenerate reference gaps need numerical-error review before
declaring a preferred orientation.

## What follows this diagnostic

1. If reference ordering changes, investigate the corresponding geometry and method
   sensitivity before using MACE to discard chemical candidates. If it agrees,
   extend validation to independent sites and balanced reaction energies.
2. Build a reference set spanning molecules, surface sites and reaction families;
   reserve independent cases for evaluation and keep literature priors separate
   from computed observations. Four NH3 cases do not validate DMATMS substitution.
3. Add a deterministic next-calculation policy with explicit cost: another pose,
   another site, a new molecule, a reaction-path check or a reference calculation.
   Placement spread can trigger review; it is not a trained error model.
4. Compare policies at equal measured compute budgets against random sampling and
   fixed-library screening, including a deterministic acquisition baseline before
   attributing gains to an LLM. Use independent reference decisions for evaluation.

The proposed research contribution is demonstrating improved next-calculation
decisions under a budget, with method failures tracked explicitly. It has not yet
been demonstrated. Literature retrieval and a larger agent count alone do not
establish that contribution.
