# WARDEN v2 — scripts and run order

This bundle updates `anjs21/donuman-ai-scientist` at commit
`54cb64c5a25a4c624e183253b6514af24d859020`. Run commands from the extracted
`warden_updated` folder. The supported entrypoint is `python -m warden`.
The old `app.py` and `run_pipeline.py` stop with a migration message.
No changes have been pushed to your GitHub repository.

## 1. Create the environment

Python **3.12** was used for verification. Linux/macOS shell:

```bash
cd warden_updated
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-v2.txt
```

Windows: use `py -3.12 -m venv .venv`, then `.venv\Scripts\Activate.ps1`.
All subsequent `python` commands are the same. Do not install the old
`requirements.txt` for this workflow.

## 2. Check the implementation without GPU, LLM, or credentials

```bash
python -m pytest -q
python -m warden demo --out runs/demo
```

Open `runs/demo/report.md` and `runs/demo/report.json`.
Expected: `SYNTHETIC DEMO: finished`, two candidates, two results, and a
`growth_surface_poisoning_in_surrogate` diagnosis followed by an analog proposal.
`ranked_real_screening_results` must be empty. The demo's planner and energies
are fixtures: it tests feedback, persistence, and reporting, not LLM quality or
molecular performance. Running it again reuses its state.

## 3. Validate the structure-based candidate definitions

```bash
python -m warden validate-candidates --candidates examples/seeds-v2.json
python -m warden validate-candidates --candidates examples/precursors-v2.json --out runs/precursors.json
```

This embeds the molecules and checks reaction element balance. The seed
inhibitors are DMATMS, ETS, pyridine, and ammonia. The precursor file contains
BDMAS and SiCl4 with their own correct retained fragments. Precursors are kept
out of inhibitor rankings. These definitions are starting controls, not a list
that constrains the agent's new proposals.

## 4. Check surface admission before installing/running MACE

```bash
python -m warden validate-surfaces --manifest examples/surfaces-reference.json --out runs/reference_validation.json
```

**Expected exit code: 2 (FAIL).** The bundled oxide passes these checks; the
bundled nitride has an estimated slab density of about 2.135 g/cm³, outside the
current 3.0 ± 0.2 gate. Inspect the JSON. This is an operational admission
failure under these heuristics, not proof that a published structure is wrong:
composition, surface termination, finite thickness, and the density estimator
need scientific review. Do not lower a gate merely to make screening pass.

The live screen needs admitted SiO2 **and** SiNx slabs. It never silently falls
back to the `dominic` files or to an invalid structure.

## 5. Set up a real checkpoint and surfaces

```bash
python -m pip install -r requirements-mlip.txt
```

Obtain a trusted MACE checkpoint suitable for your elements and chemistry.
The original project uses `medium-omat-0`; that is a starting baseline, not a
validated surface-reaction potential. See the official model documentation:
https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html

Save the checkpoint as `models/mace.model`, or edit `engine.model_path` in
`config/v2.json` (paths are relative to that config file). Set `engine.device`
to `cuda` if your PyTorch/GPU environment supports it. Select a suitable model
head/checkpoint if your model requires special multihead settings; this minimal
adapter does not configure arbitrary model-specific heads.

**Option A — use existing, reviewed slabs:** create `examples/surfaces-live.json`
using `examples/surfaces-reference.json` as the schema. Record actual provenance,
use passivated slabs with z-normal and sufficient vacuum, and list every replica
explicitly. Relative structure paths are relative to the manifest. Use at least
one admitted slab per material for a smoke screen; multiple independent replicas
are needed to assess sensitivity.

**Option B — build new slabs:** this is a long, unverified live compute step.
Use the same local checkpoint, a fresh output directory, and disable old caches.
For example, with CUDA:

```bash
python run_surface_builder.py --material both --mode full --n-bulk 3 --target-accepted 4 --max-extra-bulk 2 --dtype float64 --device cuda --model-path models/mace.model --no-cache --outdir surfaces_v2 --prefix v2
python scripts/make_manifest.py --surface-dir surfaces_v2 --model models/mace.model --out examples/surfaces-live.json
```

For CPU, replace `cuda` with `cpu`; MD may be very slow. The inherited surface
construction protocol is still a research component. It now stops on failed
relaxation, but generated surfaces are not guaranteed to pass admission. Read
`surfaces_v2/build_report.txt`; repair the physical preparation protocol if needed.
`--mode test` is only a construction smoke test and is not a source of validated
research surfaces. Do not use `--no-gate` to obtain research inputs.

For either option, admission is mandatory:

```bash
python -m warden validate-surfaces --manifest examples/surfaces-live.json --out runs/live_surface_validation.json
```

Proceed only after PASS. No real energy calculations were performed in this
bundle because a checkpoint and admitted surface pair were not available.

## 6. Run baseline screening

```bash
python -m warden screen --db runs/live/campaign.sqlite --manifest examples/surfaces-live.json --config config/v2.json --candidates examples/seeds-v2.json
```

Inspect `runs/live/report.json`, `report.md`, and `geometries/`. Failure records
are useful outputs: a command completing does not imply successful chemistry.
The report labels complete calculations, failed optimizations, insufficient site
coverage, and uncalibrated growth-proxy predictions separately.

Optional: import your actual earlier results into the same campaign DB:

```bash
python -m warden import-results --db runs/live/campaign.sqlite --file path/to/your_historical_results.json
```

Use the list format in `examples/historical_format.json`; replace its placeholder
record with actual eV values and provenance. Historical values motivate
revalidation and never enter the real ranking. No values were extracted from
your PDF and represented as newly verified calculations.

## 7. Configure evidence and the LLM

Start Ollama in another terminal if it is not already running:

```bash
ollama serve
```

Then, in the project environment:

```bash
ollama pull qwen2.5:7b
```

Edit `llm.model` and `llm.host` in `config/v2.json` if you use another installed
model or endpoint. Model choice affects proposal quality; the adapter's request
format was tested with a mock server, not a live model.

Create `examples/my-evidence.json` as a list of records:

```json
[
  {
    "kind": "literature",
    "source": "DOI or paper URL; section/table and retrieval date",
    "text": "A source-supported excerpt or clearly labelled curated summary."
  }
]
```

Use real literature and your validated observations. The supplied
`examples/evidence-v2.json` contains project context only; it is not evidence
that any molecule performs well. Do not confuse exact quote matching with
verification of a paper's conclusions. An analog hypothesis must remain labelled
as an extrapolation.

## 8. Optional BOND MCP setup

The MCP adapter discovers tools and their JSON schemas. It does not assume a
BOND paper-search function name. The example endpoint/bridge must be checked
against your current Valency account instructions; live authentication and
provider behavior have not been verified here.

Install Node.js/npm for the OAuth-capable `mcp-remote` bridge, then:

```bash
cp config/bond.example.json config/bond.json
python -m warden bond-tools --config config/bond.json --out runs/bond_tools.json
```

Complete the provider's normal sign-in if the bridge asks. Inspect the returned
inventory. Put the exact **read-only literature search/read tool names** you
intend to permit into `allowed_tools` in `config/bond.json`; it starts empty.
Do not allow tools that send messages, publish, purchase, or mutate external data.
The campaign refuses an empty inventory/allowlist. An existing bearer-token
HTTP configuration is illustrated in `config/bond-http.example.json`; tokens
belong in environment variables, never in the JSON or repository.

## 9. Run the feedback-driven campaign

With curated evidence only:

```bash
python -m warden campaign --db runs/live/campaign.sqlite --manifest examples/surfaces-live.json --config config/v2.json --evidence examples/my-evidence.json
```

Or with BOND retrieval as well:

```bash
python -m warden campaign --db runs/live/campaign.sqlite --manifest examples/surfaces-live.json --config config/v2.json --evidence examples/my-evidence.json --bond config/bond.json
```

The loop reads current results → diagnoses failures → chooses retrieval,
new structures, or refinement → runs a separate critique pass → validates
structures and evidence references → executes a MACE worker → records results
→ replans. New candidates are molecular data, not model-generated executable
code. Supported actions are bounded by `campaign` budgets in the config.
`refine` increases representative sites and requested conformers, not the
underlying model's physical fidelity. RDKit may prune duplicate conformers.

A worker timeout pauses the campaign; repeat the same command to resume.
Completed candidate calculations are reused. The Ollama model is asked to
unload between calls, and MACE runs in a separate worker process. This reduces
GPU contention but does not guarantee adequate VRAM for your setup.

Use a new DB when changing campaign configuration or surface contents. A
finished/budget-exhausted campaign remains finished on rerun. For another campaign, use a fresh DB and repeat the baseline screen. The CLI
does not silently reset budgets or copy results between different configurations.

## 10. Export and inspect the final records

```bash
python -m warden report --db runs/live/campaign.sqlite --out runs/final
```

Read `runs/final/report.md` and `report.json`. Check geometry outcomes and source
provenance, not just a scalar rank. The agent is not allowed to certify success:
`scientific_success` remains false until an independent validated workflow is
implemented. A `model_pass` is only an uncalibrated surrogate result.

## Files to inspect first

| File | Responsibility |
|---|---|
| `warden/__main__.py` | CLI and run-stage coordination |
| `warden/campaign.py` | Bounded planner/critic/compute feedback and resume |
| `warden/planner.py` | Ollama structured actions and critique |
| `warden/chemistry.py` | Dynamic SMILES registry, conformers, balanced reaction fragments |
| `warden/engine.py` | MACE calculations, convergence/geometry checks, site sampling, cache provenance |
| `warden/surfaces.py` | Explicit surface admission and representative exposed sites |
| `warden/literature.py` | MCP inventory, schema validation, allowlisted calls |
| `warden/metrics.py`, `selectivity_model.py` | Growth-proxy gates; no target-cycle clipping |
| `warden/store.py`, `warden/report.py` | Persistent records, trace, deterministic ranking |
| `tests/test_v2.py` | Executable regression and mocked integration checks |

## Scope of this implementation

This is a testable research prototype, not a completed autonomous discovery
system. It supports neutral closed-shell molecules and two reaction mechanisms:
intact adsorption or one proton exchange at a Si–Cl/Si–N bond. Multidentate,
charged, radical, multi-step, or arbitrary reaction pathways are rejected.
No barriers, finite-temperature free energies, pressure-dependent kinetics,
volatility measurements, inhibitor regeneration, or precursor/inhibitor cycle
compatibility are calculated. Broad chemical validity of the pretrained MACE
potential still needs DFT/experimental benchmarking. Pairwise distance rules
for connectivity and the slab admission thresholds are heuristics. Reported
site/placement variation is not calibrated epistemic uncertainty. The ranking
is a feasibility-first heuristic, not Bayesian optimization or a trained
acquisition model. Planner and critic use separate calls to the same configured
LLM, so their errors can be correlated.

Old energetics, hand-built reagent code, notebooks, and `agent/` are retained
for comparison and are not called by the v2 workflow. Their legacy behavior
is not claimed to have been fully repaired. See `VERIFICATION.md` for what was
actually executed.
