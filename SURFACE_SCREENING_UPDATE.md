# Reviewed finite-slab screening update

Base: dominicvarghese/donuman-ai-scientist, improve-agent-workflow,
commit 92b0d52. This update was prepared locally; it has not been pushed.

## What the uploaded structures establish

The uploaded archive is the relaxation_49543552 diagnostic. Both geometry hashes
match the files referenced in its relaxation report. All twelve uploaded
relaxation records passed the numerical tests; this is user-executed MACE data,
not an assistant-run GPU calculation.

| Quantity | Oxide | Oxynitride (workflow class SiNx) |
|---|---:|---:|
| Formula | H30O118Si54 | H59N99O8Si72 |
| Connected atom networks (distance heuristic) | 1 | 1 |
| Minimum interatomic distance, Angstrom | 0.9597 | 1.0036 |
| RMS displacement from bundled input, Angstrom | 0.0904 | 0.1772 |
| Maximum displacement from bundled input, Angstrom | 0.3618 | 1.1686 |
| Final maximum force, eV/Angstrom | 0.04534 | 0.04490 |
| Envelope density, g/cm3 | 2.13450 | 2.13541 |
| Central-window density range, g/cm3 | 2.30795–2.49443 | 2.65206–2.75776 |
| Exposed sites | OH 24, O_bridge 17 | NH2 19, NH_bridge 14, O_bridge 6 |

The substrate graph is connected, but connectivity is not identical to the
starting input: the oxide has two removed/two added distance-defined bonds;
the nitride has four removed/one added. No claim of unchanged slab chemistry
is made. Small changes in distances around bond cutoffs can change these counts.
The graphs and composition are not independent proof of amorphous phase identity.

## Correction to the scientific admission policy

A finite, terminated slab's envelope density is not the density of its parent
periodic bulk. The earlier 3.0 +/- 0.2 nitride comparison is not matched to the
PECVD-like preparation in the cited paper. Section 3.1 specifies parent bulk
preparation densities 2.2 (oxide) and 2.6 (nitride), with oxygen incorporated in
the nitride; it does not supply a universal acceptance interval for this slab
estimator. Source: https://arxiv.org/html/2510.17356v1#S3.SS1

This update adds the explicit `finite_slab_screening_v1` manifest profile:

- Envelope density and all four central windows remain reported as diagnostics.
- Density has `pass: null`, no target/tolerance, and is NOT marked validated.
- `bulk_density_validated` remains false. Parent bulk and original author-file
  provenance are not supplied/verified and remain unresolved.
- Bond/coordination, vacuum, overlaps, connected network, supported composition,
  exposed-site quality checks, and pinned file checksums remain enforced.
- A permitted input is labelled `admission_scope: numerical_screening_only`.
  `scientific_success` remains false. This permits a provisional pipeline smoke
  screen; it does not certify a valid research surface or experimental material.
- The old manifest without a profile retains its old strict behavior. The old
  comparison is retained for reproducibility, not endorsed for these slabs.
- No window is chosen to make an input pass. No density threshold is loosened.
- Profile changes and policy code now enter the calculation/cache signature.

The material labels are checked against elemental composition. The file called
SiNx contains oxygen and hydrogen and is therefore labelled hydrogenated
silicon oxynitride in the report. This does not independently identify its phase.

## Candidate clarification

The seed formerly called ETS is explicitly renamed ETS_pristine, with a
chemical-form note. The paper principally uses hydrolyzed ETS (Methods,
paragraph following the MD description). No hydrolyzed structure is guessed
or silently substituted here. The next smoke screen uses DMATMS and NH3 to
exercise the two implemented mechanisms. Those are controls, not recommended
inhibitors or a reproduction of the paper's full chemistry.

The model is MACE-MATPES-r2SCAN; the cited DFT work uses PBE with D3(BJ).
Its numerical reaction energies cannot be treated as a same-method error
benchmark for the current uncorrected MLIP. Dispersion, model transferability,
barriers, finite-temperature behavior, multi-step chemistry, and precursor/
inhibitor compatibility still require separate work.

## Apply and run

The ZIP includes an installer that checks the hashes of the changed files
against commit 92b0d52 before copying. It refuses conflicting local edits and
backs up overwritten files under runs/surface_update_backup/. It makes no git
commit and performs no push. Existing local configs, checkpoint, and results
are not replaced. A reapplication of identical updated files is accepted.

From the directory containing your repository and downloaded ZIP:

    unzip WARDEN_surface_validation_update.zip -d warden-surface-update
    cd donuman-ai-scientist
    python ../warden-surface-update/apply_update.py --repo .

Activate .venv-clean, then:

    python -m pytest -q
    python -m scripts.prepare_reference_smoke

Expected: 51 tests pass. The prepare command uses
runs/relaxation_49543552 and config/v2-r2scan.local.json. It verifies the exact
reviewed geometry and checkpoint hashes and writes:

- runs/reference_smoke_v1/surfaces.json
- runs/reference_smoke_v1/surface_validation.json
- runs/reference_smoke_v1/config.json
- runs/reference_smoke_v1/candidates.json

If the configuration was lost, recreate it using config/v2.json, your checkpoint's
absolute path, device cuda, and dtype float64. The preparation script requires
the same checkpoint used for the uploaded run. It does not load the model.
Inspect surface_validation.json: screening admission should pass, while bulk
density validation remains false. Old relaxed_admission.json is not overwritten.

Submit the bounded GPU integration test:

    sbatch scripts/reference_screen.sbatch

The scheduler configuration matches the supplied V100 job and project path.
Add your account directive if needed. The job allows two hours and evaluates
DMATMS and NH3, one conformer request and two orientations, with up to 400
optimizer steps per geometry. max_sites=1 is per face and site type, not a
single site total. This is a limited integration test, not adequate ensemble
or conformational sampling for scientific conclusions.

Candidate calculations print their names and surface/site progress. A result
may still fail a reaction-identity check, convergence, or population-coverage
gate. Those are unresolved computations, not evidence that an inhibitor is
chemically unsuitable. The job exits nonzero if either candidate lacks a
complete calculation. Completed calculations are cached if the same job is
resubmitted; failed records are also cached, so a scientifically motivated
sampling/configuration change and a new run are needed for a retry.

Read:

    cat warden_screen_JOBID.out
    cat warden_screen_JOBID.err
    python -m json.tool runs/reference_smoke_v1/report.json

Send those results before expanding the candidate set or invoking the planner.
If both calculations complete, the next stage is a broader site/conformer
sensitivity screen and a live planner/critic run. A successful two-candidate
smoke screen does not establish that the autonomous campaign or scientific
ranking works. No new adsorption-energy or successful inhibitor claim is made
by this update.

## Verification performed locally

- 51 unit/integration tests passed (38 existing + 13 added policy/preparation tests).
- Recomputed composition, distances, connectivity, density windows, and site
  checks directly from the uploaded relaxed geometries.
- Trial starting placements, before any new relaxation: oxide DMATMS 4/4 and
  NH3 4/4 without an initial clash; oxynitride DMATMS 7/8 and NH3 8/8. The
  clashing DMATMS start is rejected by the existing engine, not forced through.
- The exact reviewed pair passes only the explicit finite-slab screening
  profile. Legacy admission still rejects the nitride density comparison.
- Tested that changed hashes, unsupported profiles, clashes, insufficient
  vacuum, detached atoms, and foreign checkpoints remain rejected.
- No new MACE, GPU adsorption/interaction calculation, or live LLM/MCP campaign
  was run by the assistant. Local preparation tests use a mock checkpoint hash
  solely to exercise plumbing; the actual user command verifies real bytes.
