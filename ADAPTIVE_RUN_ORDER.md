# MACE-only acquisition baseline

This additive update selects and runs the next calculation from prior screening
results. It does not change the existing engine or overwrite earlier reports.
DFT validation is deferred: every acquisition output records
`reference_validation: pending` and `scientific_success: false`.

## Implemented scope

- Screen an untested inhibitor from a supplied, chemically validated candidate pool.
- Sample two bounded new starting poses at one previously sampled, variable site.
- Sample a previously unsampled exposed site on the same admitted slab.
- Record gas/byproduct and bare-surface references, geometry/identity failures,
  backend evaluation counts, elapsed time and accepted geometry files.
- Use new site-minimum changes to prioritize further site checks, and hold further
  site actions for a candidate after a failure-heavy adaptive observation.
- Continue from committed observations, without repeating a completed action.

This is a deterministic acquisition baseline, with transparent hand-set priorities.
It does not train a model, generate new molecules, run an LLM or call BOND. The
pool is user-supplied chemistry, not an autonomous discovery claim. The next
extension is a literature-backed proposal component evaluated against this baseline.

Priorities: placement spread / 0.25 eV for existing-site reviews, 1.0 for an
untested pool molecule, and 0.75 plus a capped observation-change term for a new
site. Placement spread is sampling variability, not calibrated prediction error.
At most two site actions per candidate are allowed in this initial bounded policy.
This prevents indefinite site refinement from consuming the entire exploration
budget. No predicted growth/selectivity threshold certifies a successful material.

## Install and check

Place the ZIP in the repository root on Bridges-2:

```bash
source .venv-clean/bin/activate
unzip -n WARDEN_adaptive_workflow.zip
python -m pytest tests/test_acquisition.py -q
python -m pytest tests -q
```

The archive contains only new files. If you previously installed a file with the
same name, `unzip -n` preserves it; resolve that version explicitly before running.
No new dependencies are added to the existing WARDEN environment.

## Prepare inputs and inspect the first choice (login node)

```bash
python -m scripts.prepare_adaptive_inputs \
  --report runs/reference_smoke_v2/report.json \
  --out runs/adaptive_inputs_v1

python -m warden.acquisition plan \
  --report runs/reference_smoke_v2/report.json \
  --manifest runs/adaptive_inputs_v1/surfaces.json \
  --config runs/adaptive_inputs_v1/config.json \
  --pool examples/seeds-v2.json \
  --out runs/adaptive_v1 \
  --poses 2 --max-actions 1 --max-calls 2000 --max-seconds 3600
```

The helper extracts the original pinned paths, checkpoint settings and admission
profile from your report. Planning verifies current surface admission/checksums
and candidate definitions but does not load MACE or run GPU calculations.
The supplied report selects NH3 at SiO2 site 32 with seed spread 0.707120 eV.
It flags DMATMS sites 134, 75 and 146 for review because of placement failures.
The exact output can change if you supply a different report.

The pool currently adds the other existing seed definitions. Review their
chemistry before treating them as useful hypotheses. New reaction mechanisms are
still limited to the templates already implemented by WARDEN.

## Run one action (GPU job)

Submit from the repository root:

```bash
sbatch scripts/adaptive_screen.sbatch
```

The job uses the existing V100/PyTorch/MACE environment. By default it runs one
action, then writes the next decision. For the first choice it revisits one NH3
site with two new azimuthal starting poses, using the original gas-conformer,
force-tolerance and local-relaxation settings. Different starts may converge to
the same basin; a new starting pose is not a guaranteed new physical state.
The small task still includes gas-reference and bare-site optimization overhead.

Budgets persist across invocations: 2,000 backend evaluations and 3,600 elapsed
run seconds. An evaluation is charged before calling MACE, including evaluations
that fail; repeated cached ASE energy/force reads are not charged again. The
elapsed budget includes model initialization and orchestration after execution
starts, excludes login-node planning, and is checked between backend evaluations.
A running evaluation may therefore overrun the elapsed threshold. The two-hour
Slurm limit provides scheduling headroom; it is not a runtime prediction.

If a budget is exhausted mid-action, completed targeted-site attempts are retained
with `budget_interrupted`. A whole-candidate screen interrupted inside the legacy
engine may leave geometry artifacts without a complete candidate result. It is
not counted as a chemically failed candidate. Inspect the status rather than
assuming exit code zero means a completed scientific calculation.

## Inspect and continue

```bash
python -m json.tool runs/adaptive_v1/next_action.json

python - <<'PY'
import json
from pathlib import Path
s = json.loads(Path('runs/adaptive_v1/state.json').read_text())
print('Backend evaluations:', s['calculator_calls'])
print('Elapsed run seconds:', s['elapsed_seconds'])
for o in s['observations']:
    a = o['action']
    print(a['kind'], a['candidate_id'], a.get('surface_id'), a.get('site'), o['status'])
    for attempt in o.get('attempts', []):
        print(' ', attempt)
PY
```

Each action has a subdirectory containing `observation.json`, progress and accepted
geometries. Targeted measurements remain separate from the seed candidate averages:
one new site/pose does not establish a new population-weighted candidate ranking.
The next-action plan incorporates these measurements and explicitly lists reviews.

If the plan is ready and the remaining budget permits, resubmit the same job:

```bash
sbatch scripts/adaptive_screen.sbatch
```

One process may own an output directory at a time. Inputs, code and budget settings
are locked for the run; changes require a fresh output directory. Seed runs are
shared initial information and are not included in the new acquisition budget.
If Slurm kills a process, its last charged prefix remains. Recovery conservatively
exhausts the elapsed budget because uncommitted crash time cannot be established;
it does not silently reset the budget or rerun the interrupted action.

## Baselines and the next development milestone

`--policy random` chooses reproducibly among the same admissible actions.
`--policy fixed` uses a fixed order among those actions, preferring untested
molecules. Both retain the shared validity/review gates; this is not yet a fully
independent fixed-library benchmark or a general demonstration of superiority.

For a preliminary policy comparison, create separate plans/output directories from
the same seed report, pool, settings, hardware and budgets. Then set the matching
policy/output in the batch job. For example, after preparing its matching plan:

```bash
sbatch --export=ALL,ACQ_POLICY=random,ACQ_OUT=runs/random_v1,ACQ_ACTIONS=10 \
  scripts/adaptive_screen.sbatch
```

Use the same number of actions per invocation for each comparison to control
model-reload overhead. A backend call on different-sized systems costs different
amounts; report actual wall time as well as evaluation counts. Report intermediate
budget stops and validity failures. Do not compare rankings produced from unequal
site sampling as if they were equally validated candidate outcomes.

Next implement an independent withheld set of MACE sites/placements and explicit
evaluation metrics, followed by literature-backed structured proposals. Better
MACE-only search efficiency can be measured; physical accuracy and experimental
AS-ALD performance remain separate future validations.

## Local verification

66 tests passed, including 11 new policy/budget/recovery tests. A read-only plan
was tested on the supplied real refined report and admitted relaxed slabs. GPU
execution of this new acquisition runner has not yet been tested with the user's
checkpoint; unit loop tests use an explicitly simulated calculator, not physical
reference energies.
