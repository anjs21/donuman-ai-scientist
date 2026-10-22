# Changes from the linked repository

- Added the `warden` package, CLI, configurations, evidence/candidate/manifest
  schemas, offline demo, run guide, and regression tests.
- Replaced the fixed-candidate path in the supported workflow with SMILES-based
  proposals and explicitly supported reaction templates, including correct
  BDMAS and SiCl4 fragments in the new candidate definitions.
- Added persistent campaign state, structured planner actions, a critique pass,
  source quote checks, candidate deduplication, calculation budgets, and resume.
- Added an MCP client with tool discovery, allowlisting, and schema validation.
- Added converged-only serial calculations, geometry checks, representative site
  sampling, per-surface coverage gates, and content/version-aware cache keys.
- Corrected target-thickness evaluation: no clipped cycle count, no perfect
  selectivity at zero growth, and missing energies remain unknown.
- Corrected selected legacy ranking and surface geometry issues; seeded serial
  surface MD; stopped fallback calculators and failed surface relaxation.
- Added a local checkpoint argument to the surface builder and a manifest helper.
- Disabled the old UI/pipeline launch routes in this bundle to avoid bypassing
  v2 gates. Retained their source and other legacy modules for comparison.

Scientific scope and remaining limitations are documented in RUN_ORDER.md.

# v3 additions

- `warden/kinetics.py`: climbing-image NEB barriers on gas-phase site models and
  on H-capped clusters carved from admitted slabs; pre-complex binding, Eyring
  rates with Langmuir occupancy; guards for unconverged/discontinuous bands,
  changed connectivity, re-captured protons and open-shell clusters.
- `acylation` reaction template in candidate validation, MACE placement and the planner.
- `warden/process.py`: multi-cycle model with redosing, cap degradation,
  irreversible nucleation and steric shielding; Monte Carlo robust schedule
  design, dose windows, barrier targets, input sensitivity, calibration.
- `warden/assess.py`: control gate, method consensus, baselines and lift.
- CLI `kinetics` and `assess`; campaign runs them when `campaign.kinetics` is set.
- Surface validation reports coordination defects and Si-Si bonds.
- `warden/decide.py`: evidence-acquisition agent (hierarchical belief, method bias
  learning, empirical-Bayes failure model, knowledge-gradient + relevance values,
  confident stopping), live xTB/MACE executor, decision reports; CLI `decide`.
- `warden/benchmark.py`: equal-budget synthetic benchmark vs four baselines; CLI `benchmark`.
- Selectable reaction pathways in barrier endpoints; `process.stress_test`.
